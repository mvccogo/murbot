import asyncio
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import httpx
import numpy as np
from bot.config import config
from bot.providers.elevenlabs import PCMStreamResampler, ElevenLabsSpeechProvider
from bot.providers.tts import TextToSpeechProvider
from bot.voice.player import VoicePlaybackController


class PCMTests(unittest.TestCase):
    def test_arbitrary_network_boundaries_preserve_audio(self):
        audio = (np.sin(2*np.pi*400*np.arange(2400)/24000)*6000).astype("<i2").tobytes()
        full = PCMStreamResampler()
        expected = full.feed(audio) + full.finish()
        fragmented = PCMStreamResampler()
        pieces = [fragmented.feed(audio[start:start+137]) for start in range(0, len(audio), 137)]
        actual = b"".join(pieces) + fragmented.finish()
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 4800 * 4)

    def test_odd_trailing_byte_is_rejected(self):
        converter = PCMStreamResampler()
        converter.feed(b"\x00")
        with self.assertRaises(ValueError):
            converter.finish()


class StreamingTTSTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_pcm_arrives_before_response_finishes_and_close_cancels_http(self):
        class Stream(httpx.AsyncByteStream):
            closed = False
            finished = False
            async def __aiter__(self):
                yield b"\x10\x00" * 2400
                await asyncio.Event().wait()
                self.finished = True
            async def aclose(self):
                self.closed = True
        response_stream = Stream()
        requests = []
        async def handler(request):
            requests.append(request)
            return httpx.Response(200, stream=response_stream)
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = ElevenLabsSpeechProvider(client)
        try:
            with patch.object(config, "elevenlabs_api_key", "test-key"), patch.object(config, "elevenlabs_voice_id", "voice123"), patch.object(config, "elevenlabs_model", "eleven_v4_turbo"):
                chunks = provider.stream_pcm("Olá!")
                pcm = await asyncio.wait_for(anext(chunks), 1)
                self.assertGreater(len(pcm), 0)
                self.assertFalse(response_stream.finished)
                request = requests[0]
                self.assertEqual(request.headers["xi-api-key"], "test-key")
                self.assertEqual(request.url.params["output_format"], "pcm_24000")
                self.assertEqual(json.loads(request.content)["language_code"], "pt")
                await chunks.aclose()
                self.assertTrue(response_stream.closed)
        finally:
            await provider.aclose()

    async def test_api_error_does_not_retry_or_fall_back(self):
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(401, json={"detail": "no"})))
        provider = ElevenLabsSpeechProvider(client)
        try:
            with patch.object(config, "elevenlabs_api_key", "test-key"), patch.object(config, "elevenlabs_voice_id", "voice123"):
                with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
                    _ = [chunk async for chunk in provider.stream_pcm("Olá!")]
        finally:
            await provider.aclose()

    async def test_outer_tts_stream_closes_inner_provider(self):
        closed = []
        async def stream(text):
            try:
                yield b"pcm"
                await asyncio.Event().wait()
            finally:
                closed.append(True)
        provider = TextToSpeechProvider.__new__(TextToSpeechProvider)
        provider.provider = "elevenlabs"
        provider.elevenlabs = NS(stream_pcm=stream)
        chunks = provider.stream_pcm("Bom, olá.")
        self.assertEqual(await anext(chunks), b"pcm")
        await chunks.aclose()
        self.assertEqual(closed, [True])

    async def test_barge_in_cancels_inflight_tts_and_flushes_playback(self):
        from bot import main
        closed, streaming = asyncio.Event(), asyncio.Event()
        async def stream(text):
            try:
                yield b"\x01" * 3840
                streaming.set()
                await asyncio.Event().wait()
            finally:
                closed.set()
        class Engine:
            async def stream_reply_sentences(self, *args):
                yield "Olá."
        vc = Mock()
        vc.is_playing.return_value = False
        session = {"player": VoicePlaybackController(vc), "engine": Engine()}
        with patch.object(main, "tts_provider", NS(stream_pcm=stream)):
            task = asyncio.create_task(main.generate_reply(session, 1, "Ana", "Murbot, oi"))
            session["current_turn_task"] = task
            await asyncio.wait_for(streaming.wait(), 1)
            source = session["player"].current_source
            await main.cancel_reply(session)
            self.assertTrue(closed.is_set())
            self.assertTrue(task.cancelled())
            self.assertEqual(source.read(), b"")
            vc.play.assert_called_once()
