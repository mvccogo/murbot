"""Bounded, independent live speech streams for each Discord utterance.

Local Whisper uses rolling re-decodes for partial hypotheses and one final
decode of the utterance. Deepgram sends PCM over a native duplex WebSocket.
The existing upload API providers remain final-only; uploads are never polled.
"""
import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass, field

import aiohttp
import numpy as np

from bot.config import config

logger = logging.getLogger("VoiceBot.StreamingSTT")


@dataclass(frozen=True)
class AudioStreamFrame:
    turn_id: int
    user_id: int
    username: str
    audio: np.ndarray
    is_final: bool = False
    discard: bool = False
    started_at: float = 0.0


@dataclass(frozen=True)
class TranscriptEvent:
    turn_id: int
    user_id: int
    username: str
    text: str
    is_final: bool
    started_at: float = 0.0


@dataclass
class _Utterance:
    first: AudioStreamFrame
    frames: list = field(default_factory=list)
    sample_count: int = 0
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    ended: asyncio.Event = field(default_factory=asyncio.Event)
    outgoing: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=256))
    discard: bool = False
    last_text: str = ""
    task: asyncio.Task | None = None


class StreamingSpeechRoom:
    """Owned by one guild session; feed() and close() run on the event loop."""

    def __init__(self, provider, on_transcript):
        self.provider = provider
        self.on_transcript = on_transcript
        self.active: dict[int, _Utterance] = {}
        self.retired = deque(maxlen=256)
        self.closed = False

    def feed(self, frame: AudioStreamFrame):
        if self.closed or frame.turn_id in self.retired:
            return
        state = self.active.get(frame.turn_id)
        if state is None:
            if len(self.active) >= max(1, config.stt_max_streams):
                logger.warning("STT stream limit reached; dropping turn %s", frame.turn_id)
                self.retired.append(frame.turn_id)
                return
            state = _Utterance(frame)
            self.active[frame.turn_id] = state
            state.task = asyncio.create_task(self._run(state))
            state.task.add_done_callback(lambda task, s=state: self._completed(s))
        if state.ended.is_set():
            return
        audio = np.asarray(frame.audio, dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            self._abort(state)
            return
        if state.sample_count + len(audio) > (config.max_utterance_seconds + 1) * 16000:
            self._abort(state)
            return
        if len(audio):
            audio = np.clip(audio, -1, 1).copy()
            state.frames.append(audio)
            state.sample_count += len(audio)
        if self.provider.provider == "deepgram":
            try:
                if len(audio):
                    state.outgoing.put_nowait((audio * 32767).astype("<i2").tobytes())
                if frame.is_final:
                    state.outgoing.put_nowait(None)
            except asyncio.QueueFull:
                logger.warning("Streaming STT send queue full; dropping turn %s", frame.turn_id)
                self._abort(state)
                return
        if frame.is_final:
            state.discard = frame.discard
            state.ended.set()
        state.changed.set()

    def _abort(self, state):
        state.discard = True
        state.ended.set()
        state.changed.set()
        state.task.cancel()

    def _completed(self, state):
        self.active.pop(state.first.turn_id, None)
        self.retired.append(state.first.turn_id)

    async def _emit(self, state, text, final=False):
        if self.closed or state.discard:
            return
        text = text.strip()
        if not final and (not text or text == state.last_text):
            return
        state.last_text = text
        first = state.first
        await self.on_transcript(TranscriptEvent(
            first.turn_id, first.user_id, first.username, text, final, first.started_at,
        ))

    async def _run(self, state):
        try:
            # Bounds failed connections/inference and missing end markers.
            async with asyncio.timeout(config.max_utterance_seconds + 30):
                if self.provider.provider == "deepgram":
                    await self._deepgram(state)
                else:
                    await self._whisper(state)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("STT stream failed for turn %s", state.first.turn_id)

    async def _whisper(self, state):
        last_decode = 0.0
        loop = asyncio.get_running_loop()
        while True:
            await state.changed.wait()
            state.changed.clear()
            if state.discard or self.closed:
                return
            if not state.ended.is_set():
                if self.provider.provider != "local":
                    continue  # REST upload providers are final-only.
                if state.sample_count < max(300, config.stt_stream_min_audio_ms) * 16:
                    continue
                delay = last_decode + max(100, config.stt_stream_interval_ms) / 1000 - loop.time()
                if delay > 0:
                    try:
                        await asyncio.wait_for(state.ended.wait(), delay)
                    except asyncio.TimeoutError:
                        pass
            # Snapshot after pacing, coalescing audio received during inference.
            state.changed.clear()
            final = state.ended.is_set()
            if state.discard or not state.frames:
                return
            audio = np.concatenate(state.frames)
            if not final:
                window_samples = int(max(1, config.stt_stream_window_seconds) * 16000)
                audio = audio[-window_samples:]
                text = await self.provider.transcribe(audio, partial=True)
            else:
                text = await self.provider.transcribe(audio)
            last_decode = loop.time()
            # If end arrived during a partial decode, still do the final decode.
            if not state.discard:
                await self._emit(state, text, final)
            if final:
                return

    async def _deepgram(self, state):
        params = {
            "model": config.deepgram_model, "language": config.deepgram_language,
            "encoding": "linear16", "sample_rate": "16000", "channels": "1",
            "interim_results": "true", "punctuate": "true", "endpointing": "false",
        }
        headers = {"Authorization": f"Token {config.deepgram_api_key}"}
        timeout = aiohttp.ClientTimeout(total=None, connect=10)
        async with aiohttp.ClientSession(timeout=timeout) as client:
            async with client.ws_connect(config.deepgram_ws_url, params=params, headers=headers, heartbeat=5) as ws:
                async def send():
                    while True:
                        chunk = await state.outgoing.get()
                        if chunk is None:
                            await ws.send_json({"type": "CloseStream"})
                            return
                        await ws.send_bytes(chunk)

                # A failed sender cancels receiving immediately, and vice versa.
                async with asyncio.TaskGroup() as group:
                    sender = group.create_task(send())
                    committed = await self._receive_deepgram(ws, state)
                    if not state.ended.is_set():
                        raise RuntimeError("STT socket closed before the utterance ended")
                    await sender
                    await self._emit(state, " ".join(committed), final=True)

    async def _receive_deepgram(self, ws, state):
        committed = []
        seen = set()
        async for message in ws:
            if message.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError("STT WebSocket receive failed")
            if message.type != aiohttp.WSMsgType.TEXT:
                continue
            data = json.loads(message.data)
            if data.get("type") == "Error":
                raise RuntimeError("Streaming STT provider returned an error")
            if data.get("type") != "Results":
                continue
            alternatives = data.get("channel", {}).get("alternatives", [])
            text = alternatives[0].get("transcript", "").strip() if alternatives else ""
            if data.get("is_final"):
                # Interim results replace a hypothesis; final segments append
                # once. Neither provider flag alone dispatches an LLM turn.
                key = (data.get("start"), data.get("duration"), text)
                if text and key not in seen:
                    seen.add(key)
                    committed.append(text)
                partial = " ".join(committed)
            else:
                partial = " ".join([*committed, text])
            await self._emit(state, partial)
        return committed

    def close(self):
        self.closed = True
        for state in list(self.active.values()):
            state.task.cancel()

    async def aclose(self):
        tasks = [state.task for state in self.active.values()]
        self.close()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.active.clear()
