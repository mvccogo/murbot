import asyncio
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import aiohttp
import numpy as np
from bot.config import config
from bot.providers.streaming_stt import AudioStreamFrame, StreamingSpeechRoom, TranscriptEvent


def frame(turn=1, user=1, count=12000, final=False, discard=False):
    return AudioStreamFrame(turn, user, str(user), np.full(count, 0.05, dtype=np.float32), final, discard)


async def drain(room):
    await asyncio.wait_for(asyncio.gather(*(s.task for s in room.active.values())), 2)
    await asyncio.sleep(0)


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_partial_arrives_before_end_and_final_runs_once(self):
        events = []
        provider = NS(provider="local", transcribe=AsyncMock(side_effect=["Murbot", "Murbot, como vai?"]))
        arrived = asyncio.Event()
        async def callback(event):
            events.append(event)
            arrived.set()
        room = StreamingSpeechRoom(provider, callback)
        try:
            room.feed(frame())
            await asyncio.wait_for(arrived.wait(), 1)
            self.assertFalse(events[0].is_final)
            self.assertEqual(events[0].text, "Murbot")
            room.feed(frame(count=4000, final=True))
            await drain(room)
            self.assertEqual([e.is_final for e in events], [False, True])
            self.assertEqual(provider.transcribe.call_args_list[0].kwargs, {"partial": True})
            self.assertEqual(len(provider.transcribe.call_args_list[1].args[0]), 16000)
            room.feed(frame(final=True))
            self.assertEqual(len(room.active), 0)
        finally:
            await room.aclose()

    async def test_end_during_inference_is_not_lost(self):
        started, release = asyncio.Event(), asyncio.Event()
        async def transcribe(audio, **kwargs):
            if kwargs.get("partial"):
                started.set()
                await release.wait()
                return "Murbot"
            return "Murbot, oi"
        events = []
        room = StreamingSpeechRoom(NS(provider="local", transcribe=transcribe), AsyncMock(side_effect=events.append))
        room.feed(frame())
        await asyncio.wait_for(started.wait(), 1)
        room.feed(frame(count=4000, final=True))
        release.set()
        await drain(room)
        self.assertEqual([e.is_final for e in events], [False, True])

    async def test_rest_provider_uploads_only_at_end(self):
        provider = NS(provider="cloud", transcribe=AsyncMock(return_value="Murbot, oi"))
        callback = AsyncMock()
        room = StreamingSpeechRoom(provider, callback)
        room.feed(frame())
        await asyncio.sleep(0)
        provider.transcribe.assert_not_called()
        room.feed(frame(count=4000, final=True))
        await drain(room)
        provider.transcribe.assert_awaited_once()
        self.assertTrue(callback.call_args.args[0].is_final)

    async def test_limits_discard_cleanup_and_speaker_isolation(self):
        release = asyncio.Event()
        async def transcribe(audio, **kwargs):
            await release.wait()
            return "Murbot"
        callback = AsyncMock()
        room = StreamingSpeechRoom(NS(provider="local", transcribe=transcribe), callback)
        with patch.object(config, "stt_max_streams", 2):
            room.feed(frame(turn=1, user=1))
            room.feed(frame(turn=2, user=2))
            room.feed(frame(turn=3, user=3))
            self.assertEqual(len(room.active), 2)
            self.assertIsNot(room.active[1].frames, room.active[2].frames)
            room.feed(frame(turn=1, count=0, final=True, discard=True))
            room.feed(frame(turn=2, count=0, final=True, discard=True))
            release.set()
            await drain(room)
        callback.assert_not_called()
        room.feed(frame(turn=4))
        await room.aclose()
        room.feed(frame(turn=5))
        self.assertEqual(len(room.active), 0)

    async def test_partial_name_cancels_once_without_generating_reply(self):
        from bot import main
        session = {"player": Mock()}
        main.guild_sessions[777] = session
        with patch.object(main, "cancel_reply", new_callable=AsyncMock) as cancel, patch.object(main, "generate_reply", new_callable=AsyncMock) as reply:
            try:
                await main.route_transcript(777, 1, "Alice", "pessoal olhem", turn_id=1, is_final=False)
                cancel.assert_not_called()
                await main.route_transcript(777, 1, "Alice", "Murbot", turn_id=1, is_final=False)
                await main.route_transcript(777, 1, "Alice", "Murbot, me ajuda", turn_id=1, is_final=False)
                cancel.assert_awaited_once()
                reply.assert_not_called()
                await main.route_transcript(777, 1, "Alice", "Murbot, me ajuda?", turn_id=1)
                await asyncio.sleep(0)
                reply.assert_awaited_once()
            finally:
                main.guild_sessions.pop(777)

    async def test_quiet_partial_stays_quiet_and_stale_result_cannot_interrupt(self):
        from bot import main
        session = {"player": Mock()}
        main.guild_sessions[778] = session
        with patch.object(main, "cancel_reply", new_callable=AsyncMock) as cancel, patch.object(main, "generate_reply", new_callable=AsyncMock) as reply:
            try:
                await main.route_transcript(778, 1, "Alice", "Murbot, fica quieto", turn_id=2, is_final=False)
                await main.route_transcript(778, 1, "Alice", "Murbot", turn_id=2)
                reply.assert_not_called()
                count = cancel.await_count
                await main.route_transcript(778, 2, "Bob", "Murbot", turn_id=1, is_final=False)
                self.assertEqual(cancel.await_count, count)
            finally:
                main.guild_sessions.pop(778)

    async def test_name_in_partial_survives_final_revision_but_not_another_turn(self):
        from bot import main
        session = {"player": Mock()}
        main.guild_sessions[779] = session
        with patch.object(main, "cancel_reply", new_callable=AsyncMock), patch.object(main, "generate_reply", new_callable=AsyncMock) as reply:
            try:
                await main.route_transcript(779, 1, "Alice", "Murbot, me diga", turn_id=1, is_final=False)
                await main.route_transcript(779, 1, "Alice", "Me diga a capital do Brasil", turn_id=1)
                await asyncio.sleep(0)
                reply.assert_awaited_once()
                await main.route_transcript(779, 2, "Bob", "Pessoal, olhem isso", turn_id=2)
                await asyncio.sleep(0)
                self.assertEqual(reply.await_count, 1)
                await main.route_transcript(779, 1, "Alice", "Murbot", turn_id=3, is_final=False)
                await main.route_transcript(779, 1, "Alice", "", turn_id=3)
                self.assertEqual(reply.await_count, 1)
            finally:
                main.guild_sessions.pop(779)

    async def test_repeated_streamed_requests_reach_playback_and_keep_listening(self):
        from bot import main
        from bot.voice.player import VoicePlaybackController
        class VoiceClient:
            def __init__(self):
                self.listening = True
                self.playing = False
                self.sources = []
            def stop_playing(self):
                self.playing = False
            def stop(self):
                self.listening = False
                self.playing = False
            def is_playing(self):
                return self.playing
            def play(self, source, *, after=None):
                self.sources.append(source)
                self.playing = True
        replies = []
        class Engine:
            async def stream_reply_sentences(self, username, user_id, text, token):
                replies.append(text)
                yield "Uma resposta."
        vc = VoiceClient()
        session = {"player": VoicePlaybackController(vc), "engine": Engine()}
        main.guild_sessions[780] = session
        partial_arrived = asyncio.Event()
        async def callback(event):
            await main.route_transcript(780, event.user_id, event.username, event.text, turn_id=event.turn_id, is_final=event.is_final)
            if not event.is_final:
                partial_arrived.set()
        provider = NS(provider="local", transcribe=AsyncMock(side_effect=[
            "Murbot, me diga", "Me diga a capital do Brasil.",
            "Murbot, outra pergunta", "Outra pergunta, como as estrelas se formam?",
        ]))
        room = StreamingSpeechRoom(provider, callback)
        async def stream_pcm(text):
            yield b"\x01" * 3840
        try:
            with patch.object(main, "tts_provider", NS(stream_pcm=stream_pcm)):
                for turn in (1, 2):
                    partial_arrived.clear()
                    room.feed(frame(turn=turn))
                    await asyncio.wait_for(partial_arrived.wait(), 1)
                    room.feed(frame(turn=turn, count=4000, final=True))
                    await drain(room)
                    await asyncio.wait_for(session["current_turn_task"], 1)
                    self.assertEqual(len(replies), turn)
                    self.assertEqual(len(vc.sources), turn)
                    self.assertTrue(vc.listening)
                self.assertFalse(room.closed)
                self.assertFalse(room.active)
        finally:
            await room.aclose()
            await main.cancel_reply(session)
            main.guild_sessions.pop(780)

    async def test_native_websocket_streams_pcm_and_combines_final_segments(self):
        events = []
        interim_arrived = asyncio.Event()
        class Socket:
            def __init__(self):
                self.messages = asyncio.Queue()
                self.audio = []
                self.controls = []
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            def __aiter__(self):
                return self
            async def __anext__(self):
                data = await self.messages.get()
                if data is None:
                    raise StopAsyncIteration
                return NS(type=aiohttp.WSMsgType.TEXT, data=json.dumps(data))
            def result(self, text, final, start=0):
                return {"type": "Results", "is_final": final, "start": start, "duration": 1, "channel": {"alternatives": [{"transcript": text}]}}
            async def send_bytes(self, data):
                self.audio.append(data)
                await self.messages.put(self.result("Murbot", False))
                await self.messages.put(self.result("Murbot", True))
            async def send_json(self, data):
                self.controls.append(data)
                await self.messages.put(self.result("Murbot", True))  # Duplicate final segment.
                await self.messages.put(self.result("fica quieto", True, start=1))
                await self.messages.put(None)
        socket = Socket()
        class Client:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            def ws_connect(self, *args, **kwargs):
                self.options = kwargs
                return socket
        client = Client()
        async def callback(event):
            events.append(event)
            if not event.is_final:
                interim_arrived.set()
        with patch("bot.providers.streaming_stt.aiohttp.ClientSession", return_value=client):
            room = StreamingSpeechRoom(NS(provider="deepgram"), callback)
            room.feed(frame())
            await asyncio.wait_for(interim_arrived.wait(), 1)
            self.assertFalse(any(e.is_final for e in events))
            room.feed(frame(count=0, final=True))
            await drain(room)
        self.assertEqual(len(socket.audio[0]), 12000 * 2)
        self.assertEqual(socket.controls, [{"type": "CloseStream"}])
        self.assertEqual(events[-1].text, "Murbot fica quieto")
        self.assertEqual(sum(e.is_final for e in events), 1)
        self.assertEqual(client.options["params"]["sample_rate"], "16000")


if __name__ == "__main__":
    unittest.main()
