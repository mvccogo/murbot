import io
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
import numpy as np
from scipy.io import wavfile
from openai import AsyncOpenAI
from bot.config import config

logger = logging.getLogger("VoiceBot.STT")

class SpeechToTextProvider:
    """
    Unified Speech-to-Text supporting both:
    1. Local Faster-Whisper running on GPU (RTX 2070 CUDA) or CPU.
    2. Cloud Whisper (Groq Whisper Turbo ~100ms or OpenAI).
    """
    def __init__(self):
        self.provider = config.active_stt_provider
        self.local_model = None
        self.cloud_client = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
        self._inference_lock = threading.Lock()
        self._dll_handles = []

        if self.provider == "local":
            self._init_local_whisper()
        elif self.provider == "deepgram":
            if not config.deepgram_api_key:
                raise ValueError("STT_PROVIDER=deepgram requires DEEPGRAM_API_KEY")
        else:
            self._init_cloud_whisper()

    def _setup_cuda_dlls(self):
        import os, sys, site
        from pathlib import Path
        try:
            candidates = set(site.getsitepackages())
            candidates.add(sys.prefix)
            candidates.add(str(Path(sys.prefix) / "Lib" / "site-packages"))
            ws_venv = Path(__file__).resolve().parents[3] / ".venv" / "Lib" / "site-packages"
            if ws_venv.exists():
                candidates.add(str(ws_venv))

            for sp in candidates:
                for p in Path(sp).glob("nvidia/*/bin"):
                    if p.is_dir():
                        try:
                            self._dll_handles.append(os.add_dll_directory(str(p)))
                        except Exception:
                            pass
                        os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
        except Exception as e:
            logger.debug(f"Error registering CUDA DLLs: {e}")

    def _init_local_whisper(self):
        self._setup_cuda_dlls()
        try:
            from faster_whisper import WhisperModel
            logger.info(
                f"Loading local Faster-Whisper model '{config.local_whisper_model}' "
                f"on {config.local_whisper_device} ({config.local_whisper_compute})..."
            )
            self.local_model = WhisperModel(
                config.local_whisper_model,
                device=config.local_whisper_device,
                compute_type=config.local_whisper_compute
            )
            # Warm up with a short silent buffer to confirm CUDA is working
            import numpy as np
            warmup = np.zeros(8000, dtype=np.float32)
            list(self.local_model.transcribe(warmup, language="pt", beam_size=1)[0])
            logger.info("Local Faster-Whisper loaded and warmed up on GPU successfully.")
        except Exception as e:
            logger.warning(
                f"Failed to load Faster-Whisper on {config.local_whisper_device}: {e}. "
                "Falling back to CPU int8..."
            )
            from faster_whisper import WhisperModel
            self.local_model = WhisperModel(
                config.local_whisper_model,
                device="cpu",
                compute_type="int8"
            )

    def _init_cloud_whisper(self):
        logger.info(f"Configuring API Whisper via {config.stt_base_url}...")
        self.cloud_client = AsyncOpenAI(
            api_key=config.stt_api_key,
            base_url=config.stt_base_url,
            timeout=30.0,
            max_retries=0,
        )

    async def transcribe(self, audio_16k_mono: np.ndarray, *, partial: bool = False) -> str:
        """
        Transcribes a 16kHz float32 mono audio array to Brazilian Portuguese text.
        """
        audio_16k_mono = np.asarray(audio_16k_mono, dtype=np.float32)
        if audio_16k_mono.ndim != 1 or not np.isfinite(audio_16k_mono).all():
            return ""
        if len(audio_16k_mono) == 0:
            return ""

        # Remove audio too short (<0.3s) or silent (< -50dB RMS) to avoid hallucinating noise
        if len(audio_16k_mono) < 16000 * 0.3:
            return ""

        rms = float(np.sqrt(np.mean(audio_16k_mono**2)))
        if rms < config.stt_min_rms:
            return ""

        # Preserve microphone levels. Peak normalization amplifies quiet noise.
        audio_16k_mono = np.clip(audio_16k_mono, -1.0, 1.0)
        logger.debug("STT input duration=%.2fs rms=%.5f peak=%.5f", len(audio_16k_mono)/16000, rms, float(np.max(np.abs(audio_16k_mono))))
        if config.debug_audio_dir and not partial:
            from pathlib import Path
            from uuid import uuid4
            directory = Path(config.debug_audio_dir)
            directory.mkdir(parents=True, exist_ok=True)
            wavfile.write(directory / f"{uuid4().hex}.wav", 16000, (audio_16k_mono * 32767).astype(np.int16))

        if self.provider == "local" and self.local_model:
            if partial:
                return await self._transcribe_local(audio_16k_mono, partial=True)
            return await self._transcribe_local(audio_16k_mono)
        elif self.provider == "deepgram":
            raise RuntimeError("Deepgram requires a live audio stream, not a WAV upload")
        else:
            return await self._transcribe_cloud(audio_16k_mono)

    async def _transcribe_local(self, audio_16k_mono: np.ndarray, *, partial: bool = False) -> str:
        import asyncio
        loop = asyncio.get_running_loop()

        def _run_inference():
            # Supply vocabulary rather than a canned sentence: uncommon wake
            # names otherwise become ordinary Portuguese names in transcription.
            hotwords = config.stt_hotwords.strip() or ", ".join(dict.fromkeys(
                word.strip() for word in [config.bot_name, *config.bot_aliases.split(",")] if word.strip()
            ))
            segments, _ = self.local_model.transcribe(
                audio_16k_mono,
                language=None if config.stt_language == "auto" else config.stt_language,
                beam_size=1 if partial else 2,
                best_of=1 if partial else 2,
                temperature=0.0,
                vad_filter=not partial,
                vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 200},
                condition_on_previous_text=False,
                hotwords=hotwords or None,
            )
            text = " ".join(segment.text for segment in segments
                            if segment.no_speech_prob < 0.6
                            and segment.avg_logprob >= -1.0
                            and segment.compression_ratio <= 2.4).strip()
            return text

        # Lock lives inside the worker: cancelling the await cannot release it
        # while CTranslate2 is still executing on the GPU.
        def serialized_inference():
            with self._inference_lock:
                return _run_inference()
        return await loop.run_in_executor(self._executor, serialized_inference)

    async def _transcribe_cloud(self, audio_16k_mono: np.ndarray) -> str:
        # Convert float32 to int16 PCM wav file in memory
        int16_audio = (np.clip(audio_16k_mono, -1.0, 1.0) * 32767).astype(np.int16)
        buffer = io.BytesIO()
        wavfile.write(buffer, 16000, int16_audio)
        buffer.seek(0)
        buffer.name = "audio.wav"

        try:
            transcription = await self.cloud_client.audio.transcriptions.create(
                model=config.cloud_whisper_model,
                file=buffer,
                **({} if config.stt_language == "auto" else {"language": config.stt_language}),
                response_format="text"
            )
            return transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
        except Exception as e:
            logger.error(f"Cloud STT transcription failed: {e}")
            return ""
