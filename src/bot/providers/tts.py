import io
import asyncio
import re
import logging
import av
import edge_tts
from openai import AsyncOpenAI
from bot.config import config
from bot.providers.elevenlabs import ElevenLabsSpeechProvider

logger = logging.getLogger("VoiceBot.TTS")

class TextToSpeechProvider:
    """
    Unified Text-To-Speech supporting:
    1. Kokoro TTS (SOTA 82M open-source model running via Docker GPU with natural tones and human pauses).
    2. Edge-TTS (Free studio-quality neural pt-BR).
    3. OpenAI TTS (Cloud API).
    
    Converts text sentences directly to 48kHz 16-bit stereo PCM bytes,
    ready for instant streaming playback into Discord voice clients.
    """
    def __init__(self):
        self.provider = config.tts_provider
        self.edge_voice = config.tts_voice
        self.kokoro_voice = config.kokoro_voice
        self.kokoro_speed = config.kokoro_speed
        self.rate = config.tts_rate

        self.kokoro_client = AsyncOpenAI(
            base_url=config.local_tts_base_url,
            api_key="kokoro-local", timeout=20.0, max_retries=0,
        )
        self.openai_client = None
        self.elevenlabs = ElevenLabsSpeechProvider()

    @staticmethod
    def decode_to_discord_pcm(encoded_audio_bytes: bytes) -> bytes:
        """
        Decodes any encoded audio (MP3, OGG, WAV, AAC) in memory directly to
        48kHz 16-bit stereo PCM bytes using PyAV in ~10 milliseconds.
        """
        if not encoded_audio_bytes:
            return b""
        
        container = av.open(io.BytesIO(encoded_audio_bytes))
        resampler = av.AudioResampler(format="s16", layout="stereo", rate=48000)
        pcm_out = bytearray()

        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                pcm_out.extend(bytes(resampled.planes[0])[: resampled.samples * 4])

        for resampled in resampler.resample(None):
            pcm_out.extend(bytes(resampled.planes[0])[: resampled.samples * 4])

        container.close()
        return bytes(pcm_out)

    def _prepare_text_prosody(self, text: str) -> str:
        """
        Cleans markdown and enhances punctuation for natural human pauses,
        breathing intervals, and conversational cadence in Kokoro.
        """
        # Remove code blocks and inline code
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(r"`.*?`", "", text)
        # Remove markdown bold/italics
        text = re.sub(r"[*_~#>]", "", text)
        # Remove markdown links, keep text
        text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
        # Remove raw URLs
        text = re.sub(r"https?://\S+", "", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        # Preserve the author's punctuation; injected ellipses turn ordinary
        # conversational transitions into hesitant, exaggerated pauses.
        return text

    async def synthesize(self, text: str) -> bytes:
        """
        Synthesizes Portuguese text to 48kHz stereo PCM bytes.
        """
        processed_text = self._prepare_text_prosody(text)
        if not processed_text:
            return b""

        if self.provider == "elevenlabs":
            return b"".join([chunk async for chunk in self.elevenlabs.stream_pcm(processed_text)])
        elif self.provider == "kokoro":
            return await self._synthesize_kokoro(processed_text)
        elif self.provider == "openai":
            if self.openai_client is None:
                self.openai_client = AsyncOpenAI(
                    api_key=config.tts_api_key, base_url=config.tts_base_url,
                    timeout=30.0, max_retries=0,
                )
            return await self._synthesize_openai(processed_text)
        else:
            return await self._synthesize_edge(processed_text)

    async def stream_pcm(self, text: str):
        if self.provider == "elevenlabs":
            processed = self._prepare_text_prosody(text)
            if processed:
                chunks = self.elevenlabs.stream_pcm(processed)
                try:
                    async for chunk in chunks:
                        yield chunk
                finally:
                    await chunks.aclose()
        else:
            pcm = await self.synthesize(text)
            if pcm:
                yield pcm

    async def _synthesize_kokoro(self, text: str) -> bytes:
        """
        Synthesizes speech using the local Kokoro-FastAPI Docker container.
        Falls back automatically to Edge-TTS if the container is offline.
        """
        try:
            response = await self.kokoro_client.audio.speech.create(
                model="kokoro",
                voice=self.kokoro_voice,
                input=text,
                speed=self.kokoro_speed,
                response_format="mp3"
            )
            mp3_bytes = response.content
            return await asyncio.to_thread(self.decode_to_discord_pcm, mp3_bytes)
        except Exception as e:
            logger.warning(
                f"Kokoro TTS service at {config.local_tts_base_url} unavailable ({e}). "
                "Falling back smoothly to Edge-TTS..."
            )
            return await self._synthesize_edge(text)

    async def _synthesize_edge(self, text: str) -> bytes:
        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.edge_voice,
                rate=self.rate
            )
            mp3_data = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_data.extend(chunk["data"])

            if not mp3_data:
                return b""

            return await asyncio.to_thread(self.decode_to_discord_pcm, bytes(mp3_data))
        except Exception as e:
            logger.error(f"Edge-TTS synthesis error: {e}")
            return b""

    async def _synthesize_openai(self, text: str) -> bytes:
        try:
            response = await self.openai_client.audio.speech.create(
                model=config.tts_model,
                voice=config.api_tts_voice,
                input=text,
                response_format="mp3"
            )
            mp3_bytes = response.content
            return await asyncio.to_thread(self.decode_to_discord_pcm, mp3_bytes)
        except Exception as e:
            logger.error(f"OpenAI TTS synthesis error: {e}")
            return b""
