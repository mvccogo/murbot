import asyncio
import io
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from scipy.io import wavfile
from bot.config import config
from bot.voice.addressing import classify_address
from bot.voice.transport import decrypt_voice_frame, OPUS_SILENCE
from bot.voice.player import VoicePlaybackController, InterruptibleAudioSource
from bot.voice.vad import SileroVoiceDetector
from bot.providers.stt import SpeechToTextProvider
from bot.providers.tts import TextToSpeechProvider


class AudioTests(unittest.TestCase):
    def test_addressing(self):
        for phrase, result in [
            ("pessoal, que horas são?", "ignore"),
            ("MURBOT, que horas são?", "reply"),
            ("mur bot, pode ajudar?", "reply"),
            ("murbot, fica quieto por favor", "stop"),
            ("Murbot, silêncio!", "stop"),
            ("murbot, be quiet", "stop"),
            ("amurbot, oi", "ignore"),
            ("para de falar", "ignore"),
        ]:
            with self.subTest(phrase=phrase):
                self.assertEqual(classify_address(phrase, "murbot", "mur bot"), result)

    def test_downsampling_preserves_frequency_and_level(self):
        mono = (np.sin(2 * np.pi * 1000 * np.arange(4800) / 48000) * 10000).astype(np.int16)
        pcm = np.column_stack((mono, mono)).tobytes()
        output = SileroVoiceDetector.discord_to_16k_mono(pcm)
        self.assertEqual(len(output), 1600)
        peak_hz = np.argmax(np.abs(np.fft.rfft(output))) * 16000 / len(output)
        self.assertEqual(peak_hz, 1000)
        self.assertAlmostEqual(float(np.max(output)), 10000/32768, delta=0.01)

    def test_dave_is_decrypted_before_opus(self):
        import davey
        session = NS(ready=True, decrypt=Mock(return_value=b"opus"))
        vc = NS(_connection=NS(dave_protocol_version=1, dave_session=session))
        self.assertEqual(decrypt_voice_frame(vc, 123, b"encrypted"), b"opus")
        session.decrypt.assert_called_once_with(123, davey.MediaType.audio, b"encrypted")
        session.decrypt.side_effect = ValueError("bad key")
        self.assertIsNone(decrypt_voice_frame(vc, 123, b"bad"))
        session.ready = False
        self.assertIsNone(decrypt_voice_frame(vc, 123, b"not ready"))
        self.assertEqual(decrypt_voice_frame(vc, 123, OPUS_SILENCE), OPUS_SILENCE)
        vc._connection.dave_protocol_version = 0
        self.assertEqual(decrypt_voice_frame(vc, 123, b"plain"), b"plain")

    def test_tts_decode_returns_discord_frames(self):
        buffer = io.BytesIO()
        wavfile.write(buffer, 24000, np.zeros(2400, dtype=np.int16))
        pcm = TextToSpeechProvider.decode_to_discord_pcm(buffer.getvalue())
        self.assertEqual(len(pcm), 4800 * 4)

    def test_source_drops_late_tts_after_interrupt(self):
        source = InterruptibleAudioSource()
        source.write_pcm(b"x" * 3840)
        source.interrupt()
        source.write_pcm(b"y" * 3840)
        self.assertEqual(source.read(), b"")


class AsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_playback_token_survives_old_cancellation(self):
        vc = Mock()
        vc.is_playing.return_value = False
        player = VoicePlaybackController(vc)
        _, old = player.start_new_session()
        _, new = player.start_new_session()
        await asyncio.sleep(0)
        self.assertTrue(old.is_set())
        self.assertFalse(new.is_set())

    async def test_stt_rejects_invalid_and_silent_audio_without_inference(self):
        provider = SpeechToTextProvider.__new__(SpeechToTextProvider)
        provider.provider = "local"
        provider.local_model = Mock()
        provider._transcribe_local = AsyncMock(return_value="murbot")
        for audio in (np.zeros(16000), np.ones(20), np.full(16000, np.nan), np.zeros((2,16000))):
            self.assertEqual(await provider.transcribe(audio), "")
        provider._transcribe_local.assert_not_awaited()
        with patch.object(config, "debug_audio_dir", ""):
            audio = np.full(16000, 0.02, dtype=np.float32)
            self.assertEqual(await provider.transcribe(audio), "murbot")
        np.testing.assert_allclose(provider._transcribe_local.call_args.args[0], audio)

    async def test_local_stt_filters_low_confidence_and_repetitive_segments(self):
        provider = SpeechToTextProvider.__new__(SpeechToTextProvider)
        provider._executor = ThreadPoolExecutor(max_workers=1)
        provider._inference_lock = threading.Lock()
        def segment(text, probability=0.1, confidence=-0.2, compression=1):
            return NS(text=text, no_speech_prob=probability, avg_logprob=confidence, compression_ratio=compression)
        provider.local_model = Mock()
        provider.local_model.transcribe.return_value = (iter([
            segment("Murbot, oi."), segment("random outro", probability=0.9),
            segment("bad", confidence=-2), segment("repeat", compression=5),
        ]), None)
        try:
            with patch.object(config, "stt_hotwords", ""):
                self.assertEqual(await provider._transcribe_local(np.ones(16000)), "Murbot, oi.")
            options = provider.local_model.transcribe.call_args.kwargs
            self.assertTrue(options["vad_filter"])
            self.assertNotIn("initial_prompt", options)
            self.assertIn(config.bot_name, options["hotwords"])
            provider.local_model.transcribe.return_value = (iter([segment("Murbot, oi.")]), None)
            with patch.object(config, "stt_hotwords", "Murbot, Mur bot"):
                self.assertEqual(await provider._transcribe_local(np.ones(16000), partial=True), "Murbot, oi.")
            options = provider.local_model.transcribe.call_args.kwargs
            self.assertEqual(options["hotwords"], "Murbot, Mur bot")
            self.assertFalse(options["vad_filter"])
        finally:
            provider._executor.shutdown()

    async def test_background_speech_does_not_cancel_and_quiet_does(self):
        from bot import main
        session = {"stt_lock": asyncio.Lock(), "player": Mock()}
        prior = asyncio.create_task(asyncio.Event().wait())
        session["current_turn_task"] = prior
        main.guild_sessions[999] = session
        stt = NS(transcribe=AsyncMock(side_effect=["pessoal, olhem isso", "Murbot, fica quieto"]))
        try:
            with patch.object(main, "stt_provider", stt), patch.object(config, "bot_name", "murbot"):
                await main.handle_speech_turn(999, 1, "Alice", np.ones(16000))
                self.assertFalse(prior.done())
                session["player"].stop_playback.assert_not_called()
                await main.handle_speech_turn(999, 2, "Bob", np.ones(16000))
                self.assertTrue(prior.cancelled())
                self.assertIsNone(session["current_turn_task"])
                session["player"].stop_playback.assert_called_once()
        finally:
            main.guild_sessions.pop(999)
            prior.cancel()
            await asyncio.gather(prior, return_exceptions=True)

    async def test_stream_yields_before_completion_and_closes_on_cancel(self):
        from bot.brain.llm import LLMConversationEngine
        class Stream:
            closed = False
            def __aiter__(self):
                return self
            async def __anext__(self):
                self.count = getattr(self, "count", 0) + 1
                if self.count > 2:
                    raise StopAsyncIteration
                return NS(choices=[NS(delta=NS(content="Olá. " if self.count == 1 else "Tudo bem?", tool_calls=None))])
            async def close(self):
                self.closed = True
        stream = Stream()
        engine = LLMConversationEngine.__new__(LLMConversationEngine)
        engine.history = []
        engine._build_system_prompt = Mock(return_value="test")
        create = AsyncMock(return_value=stream)
        engine.client = NS(chat=NS(completions=NS(create=create)))
        event = asyncio.Event()
        generator = engine.stream_reply_sentences("Alice", 1, "Murbot, oi", event)
        self.assertEqual(await anext(generator), "Olá.")
        self.assertEqual(stream.count, 1)
        event.set()
        self.assertEqual([s async for s in generator], [])
        self.assertTrue(stream.closed)
        self.assertEqual(engine.history[0]["content"], "[Alice / 1]: Murbot, oi")
        self.assertIn("Resposta interrompida", engine.history[1]["content"])
        self.assertTrue(create.call_args.kwargs["stream"])

    async def test_two_speakers_have_independent_vad_and_preroll(self):
        from bot.voice.audio_sink import MultiUserAudioSink, UserAudioTrack
        callback = AsyncMock()
        class Detector:
            def __init__(self, threshold):
                self.threshold = threshold
            def predict_chunk(self, chunk):
                return 0.99
            def reset(self):
                pass
            discord_to_16k_mono = staticmethod(lambda _: np.full(512, 0.1, dtype=np.float32))
        def make_track(user_id, username, vad):
            decoder = Mock()
            decoder.decode.return_value = b"pcm"
            return UserAudioTrack(user_id, username, vad, decoder=decoder)
        with patch("bot.voice.audio_sink.SileroVoiceDetector", Detector), patch("bot.voice.audio_sink.UserAudioTrack", side_effect=make_track):
            sink = MultiUserAudioSink(NS(id=99), Mock(), callback)
            sink._voice_client = NS()
            try:
                with patch.object(config, "interruption_frames_threshold", 2):
                    for _ in range(12):
                        for uid in (1, 2):
                            sink.write(NS(id=uid, display_name=str(uid), bot=False), NS(opus=OPUS_SILENCE))
                a, b = sink.user_tracks.values()
                self.assertIsNot(a.vad, b.vad)
                self.assertEqual(a.sample_count, 12 * 512)
                self.assertEqual(b.sample_count, 12 * 512)
                sink._finish(a)
                self.assertEqual(b.sample_count, 12 * 512)
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                callback.assert_awaited_once()
                audio = callback.call_args.args[2]
                self.assertEqual(len(audio), 12 * 512)
                np.testing.assert_allclose(audio, 0.1)
                sink.player_controller.stop_playback.assert_not_called()
            finally:
                sink.cleanup()
                await asyncio.sleep(0)

    async def test_sink_streams_audio_before_end_without_duplicate_samples(self):
        from bot.voice.audio_sink import MultiUserAudioSink, UserAudioTrack
        class Detector:
            threshold = 0.5
            def __init__(self, threshold):
                pass
            def predict_chunk(self, chunk):
                return 0.99
            def reset(self):
                pass
            discord_to_16k_mono = staticmethod(lambda _: np.full(512, 0.1, dtype=np.float32))
        def make_track(user_id, username, vad):
            decoder = Mock()
            decoder.decode.return_value = b"pcm"
            return UserAudioTrack(user_id, username, vad, decoder=decoder)
        events = []
        legacy = AsyncMock()
        with patch("bot.voice.audio_sink.SileroVoiceDetector", Detector), patch("bot.voice.audio_sink.UserAudioTrack", side_effect=make_track):
            sink = MultiUserAudioSink(NS(id=99), Mock(), legacy, events.append)
            sink._voice_client = NS()
            try:
                with patch.object(config, "interruption_frames_threshold", 2):
                    for _ in range(12):
                        sink.write(NS(id=1, display_name="Alice", bot=False), NS(opus=OPUS_SILENCE))
                await asyncio.sleep(0)
                self.assertTrue(events)
                self.assertFalse(any(e.is_final for e in events))
                sink._finish(sink.user_tracks[1])
                await asyncio.sleep(0)
                self.assertTrue(events[-1].is_final)
                self.assertEqual(sum(len(e.audio) for e in events), 12 * 512)
                self.assertEqual(len({e.turn_id for e in events}), 1)
                legacy.assert_not_called()
            finally:
                sink.cleanup()
                await asyncio.sleep(0)


if __name__ == "__main__":
    unittest.main()
