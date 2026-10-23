"""HTTP streaming TTS: 24 kHz mono PCM -> continuous Discord stereo PCM."""
from fractions import Fraction
import re

import av
import httpx

from bot.config import config


class PCMStreamResampler:
    def __init__(self, rate=24000):
        self.rate = rate
        self.samples_read = 0
        self.pending = b""
        self.resampler = av.AudioResampler(format="s16", layout="stereo", rate=48000)

    @staticmethod
    def _bytes(frames):
        return b"".join(bytes(frame.planes[0])[:frame.samples * 4] for frame in frames)

    def feed(self, data):
        data = self.pending + data
        usable = len(data) // 2 * 2
        self.pending = data[usable:]
        if not usable:
            return b""
        frame = av.AudioFrame(format="s16", layout="mono", samples=usable // 2)
        frame.sample_rate = self.rate
        frame.time_base = Fraction(1, self.rate)
        frame.pts = self.samples_read
        frame.planes[0].update(data[:usable])
        self.samples_read += frame.samples
        return self._bytes(self.resampler.resample(frame))

    def finish(self):
        if self.pending:
            raise ValueError("ElevenLabs returned a truncated PCM sample")
        return self._bytes(self.resampler.resample(None))


class ElevenLabsSpeechProvider:
    def __init__(self, client=None):
        self.client = client

    async def stream_pcm(self, text):
        if not config.elevenlabs_api_key:
            raise ValueError("Set ELEVENLABS_API_KEY or ELEVEN_LABS_API_KEY")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", config.elevenlabs_voice_id):
            raise ValueError("Set ELEVENLABS_VOICE_ID to an available ElevenLabs voice ID")
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=httpx.Timeout(20, connect=10))
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{config.elevenlabs_voice_id}/stream"
        payload = {
            "text": text, "model_id": config.elevenlabs_model,
            "voice_settings": {"speed": config.elevenlabs_speed},
        }
        if config.elevenlabs_language and config.elevenlabs_model != "eleven_multilingual_v2":
            payload["language_code"] = config.elevenlabs_language
        converter = PCMStreamResampler()
        async with self.client.stream(
            "POST", url, headers={"xi-api-key": config.elevenlabs_api_key},
            params={"output_format": "pcm_24000"}, json=payload,
        ) as response:
            if response.status_code != 200:
                # Never put the API key or server response body in voice/logs.
                raise RuntimeError(f"ElevenLabs TTS returned HTTP {response.status_code}; check model, voice, key permissions and credits")
            async for data in response.aiter_bytes():
                pcm = converter.feed(data)
                if pcm:
                    yield pcm
            tail = converter.finish()
            if tail:
                yield tail

    async def aclose(self):
        if self.client:
            await self.client.aclose()
