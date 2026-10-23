"""Per-speaker Opus -> PCM -> VAD -> bounded utterance delivery."""
import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import discord
import numpy as np
from discord.ext import voice_recv

from bot.config import config
from bot.voice.transport import decrypt_voice_frame
from bot.voice.vad import SileroVoiceDetector
from bot.providers.streaming_stt import AudioStreamFrame

logger = logging.getLogger("VoiceBot.Sink")


@dataclass
class UserAudioTrack:
    user_id: int
    username: str
    vad: SileroVoiceDetector
    decoder: discord.opus.Decoder = field(default_factory=discord.opus.Decoder)
    frames: list = field(default_factory=list)
    pending: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    preroll: deque = field(default_factory=lambda: deque(maxlen=8))
    speaking: bool = False
    consecutive: int = 0
    speech_samples: int = 0
    sample_count: int = 0
    last_speech: float = 0.0
    last_packet: float = 0.0
    decode_errors: int = 0
    turn_id: int = 0
    stream_offset: int = 0
    started_at: float = 0.0

    def clear(self):
        self.frames.clear()
        self.preroll.clear()
        self.pending = np.empty(0, dtype=np.float32)
        self.speaking = False
        self.consecutive = self.speech_samples = self.sample_count = 0
        self.last_speech = 0.0
        self.stream_offset = 0
        self.vad.reset()


class MultiUserAudioSink(voice_recv.AudioSink):
    def __init__(self, bot_user, player_controller, on_speech_turn_callback,
                 on_audio_stream_frame=None, on_stream_cleanup=None):
        super().__init__()
        self.bot_user = bot_user
        self.player_controller = player_controller
        self.on_speech_turn_callback = on_speech_turn_callback
        self.on_audio_stream_frame = on_audio_stream_frame
        self.on_stream_cleanup = on_stream_cleanup
        self._turn_counter = 0
        self.user_tracks = {}
        self.loop = asyncio.get_running_loop()
        self._lock = threading.RLock()
        self._running = True
        self._tasks = set()
        self._silence_monitor_task = self.loop.create_task(self._monitor_silence_loop())

    def wants_opus(self):
        return True

    def write(self, user, data):
        # PacketRouter invokes this on its own thread. The monitor takes the same
        # lock before draining buffers; asyncio/player state stays on the loop.
        with self._lock:
            if not self._running or data is None:
                return
            try:
                vc = self.voice_client
                user_id = user.id if user else vc._get_id_from_ssrc(data.packet.ssrc)
                if not user_id or user_id == self.bot_user.id or getattr(user, "bot", False):
                    return
                payload = data.opus
                if not payload:
                    return  # PLC/fake packets have no payload.
                payload = decrypt_voice_frame(vc, user_id, payload)
                if not payload:
                    return
                if user_id not in self.user_tracks:
                    self.user_tracks[user_id] = UserAudioTrack(
                        user_id, getattr(user, "display_name", str(user_id)),
                        SileroVoiceDetector(config.vad_threshold),
                    )
                track = self.user_tracks[user_id]
                track.username = getattr(user, "display_name", track.username)
                try:
                    pcm = track.decoder.decode(payload, fec=False)
                except discord.opus.OpusError:
                    track.decode_errors += 1
                    if track.decode_errors <= 3 or track.decode_errors % 100 == 0:
                        logger.warning("Opus decode failed for %s (%s failures); check DAVE/transport", user_id, track.decode_errors)
                    return
                now = time.monotonic()
                # A new talk spurt must not inherit recurrent state from minutes ago.
                if not track.speaking and now - track.last_packet > 1:
                    track.clear()
                track.last_packet = now
                samples = SileroVoiceDetector.discord_to_16k_mono(pcm)
                track.pending = np.concatenate((track.pending, samples))
                while len(track.pending) >= 512:
                    window = track.pending[:512].copy()
                    track.pending = track.pending[512:]
                    speech = track.vad.predict_chunk(window) >= track.vad.threshold
                    if speech:
                        track.consecutive += 1
                        track.last_speech = now
                    else:
                        track.consecutive = 0
                    if not track.speaking:
                        track.preroll.append(window)
                        if track.consecutive >= max(2, config.interruption_frames_threshold):
                            track.speaking = True
                            self._turn_counter += 1
                            track.turn_id = self._turn_counter
                            track.started_at = now
                            track.frames.extend(track.preroll)
                            track.sample_count = len(track.preroll) * 512
                            track.speech_samples = track.consecutive * 512
                            track.preroll.clear()
                    else:
                        track.frames.append(window)
                        track.sample_count += 512
                        if speech:
                            track.speech_samples += 512
                    if track.speaking and len(track.frames) - track.stream_offset >= 3:
                        self._stream(track)
                    if track.sample_count >= config.max_utterance_seconds * 16000:
                        self._finish(track)
            except Exception:
                logger.exception("Audio receive failure")

    def _finish(self, track):
        meaningful = track.speech_samples >= 16000 * 0.16
        if track.speaking and self.on_audio_stream_frame:
            self._stream(track, final=True, discard=not meaningful)
        elif track.speaking and meaningful:
            audio = np.concatenate(track.frames)
            self.loop.call_soon_threadsafe(self._dispatch, track.user_id, track.username, audio)
        track.clear()

    def _stream(self, track, final=False, discard=False):
        if self.on_audio_stream_frame is None:
            return
        frames = track.frames[track.stream_offset:]
        audio = np.concatenate(frames) if frames else np.empty(0, dtype=np.float32)
        track.stream_offset = len(track.frames)
        event = AudioStreamFrame(track.turn_id, track.user_id, track.username, audio, final, discard, track.started_at)
        self.loop.call_soon_threadsafe(self._deliver_stream, event)

    def _deliver_stream(self, event):
        if self._running:
            self.on_audio_stream_frame(event)

    def _dispatch(self, user_id, username, audio):
        if not self._running:
            return
        if len(self._tasks) >= 8:
            logger.warning("Speech backlog full; dropping turn from %s", user_id)
            return
        task = self.loop.create_task(self.on_speech_turn_callback(user_id, username, audio))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _monitor_silence_loop(self):
        try:
            while self._running:
                await asyncio.sleep(0.05)
                now = time.monotonic()
                with self._lock:
                    for user_id, track in list(self.user_tracks.items()):
                        if track.speaking and now - track.last_speech >= config.silence_duration_ms / 1000:
                            self._finish(track)
                        if now - track.last_packet > 60:
                            del self.user_tracks[user_id]
        except asyncio.CancelledError:
            return

    def cleanup(self):
        with self._lock:
            self._running = False
            self.user_tracks.clear()
        self.loop.call_soon_threadsafe(self._cancel_tasks)

    def speaking_snapshot(self):
        now = time.monotonic()
        with self._lock:
            return {user_id for user_id, track in self.user_tracks.items()
                    if track.speaking and now - track.last_speech <= config.silence_duration_ms / 1000}

    def _cancel_tasks(self):
        if self.on_stream_cleanup:
            self.on_stream_cleanup()
        self._silence_monitor_task.cancel()
        for task in self._tasks:
            task.cancel()
