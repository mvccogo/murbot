import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class BotConfig:
    # Persistent runtime logging (size-based rotation).
    log_dir: str = os.getenv("LOG_DIR", "logs")
    log_level: str = os.getenv("LOG_LEVEL", "DEBUG")
    log_console_level: str = os.getenv("LOG_CONSOLE_LEVEL", "INFO")
    log_max_bytes: int = int(os.getenv("LOG_MAX_BYTES", "10485760"))
    log_backup_count: int = int(os.getenv("LOG_BACKUP_COUNT", "10"))
    # Discord
    discord_token: str = os.getenv("DISCORD_TOKEN", "")
    command_prefix: str = os.getenv("DISCORD_PREFIX", "!")
    bot_name: str = os.getenv("BOT_NAME", "murbot")
    bot_aliases: str = os.getenv("BOT_ALIASES", "mur bot")
    conversation_followup_seconds: float = float(os.getenv("CONVERSATION_FOLLOWUP_SECONDS", "20"))
    conversation_semantic_followups: bool = os.getenv("CONVERSATION_SEMANTIC_FOLLOWUPS", "true").lower() == "true"
    
    # Mode selection: "cloud" or "local"
    # Selects the LLM endpoint only; STT and TTS are independent.
    mode: str = os.getenv("MODE", "cloud").lower()
    
    # LLM Settings
    llm_provider: str = os.getenv("LLM_PROVIDER", "cloud").lower()  # cloud, local, ollama, groq, openai, deepseek
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
    llm_model: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    llm_tools_enabled: bool = os.getenv("LLM_TOOLS_ENABLED", "true").lower() == "true"
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEP_SEEK_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    
    local_llm_base_url: str = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
    local_llm_model: str = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:3b")
    
    # STT Settings (Speech-To-Text)
    stt_provider: str = os.getenv("STT_PROVIDER", "local").lower()  # local (faster-whisper) or cloud (groq/openai)
    local_whisper_model: str = os.getenv("LOCAL_WHISPER_MODEL", "large-v3-turbo")
    local_whisper_device: str = os.getenv("LOCAL_WHISPER_DEVICE", "cuda")  # cuda or cpu
    local_whisper_compute: str = os.getenv("LOCAL_WHISPER_COMPUTE", "float16")  # float16, int8_float16, int8
    cloud_whisper_model: str = os.getenv("CLOUD_WHISPER_MODEL", "whisper-large-v3-turbo")
    stt_base_url: str = os.getenv("STT_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"))
    stt_api_key: str = os.getenv("STT_API_KEY", os.getenv("OPENAI_API_KEY", "local-stt"))
    stt_language: str = os.getenv("STT_LANGUAGE", "pt")  # "auto" for multilingual
    stt_hotwords: str = os.getenv("STT_HOTWORDS", "")  # Empty uses bot name and configured aliases.
    stt_min_rms: float = float(os.getenv("STT_MIN_RMS", "0.003"))
    debug_audio_dir: str = os.getenv("DEBUG_AUDIO_DIR", "")
    stt_log_transcripts: bool = os.getenv("STT_LOG_TRANSCRIPTS", "true").lower() == "true"
    stt_streaming_enabled: bool = os.getenv("STT_STREAMING_ENABLED", "true").lower() == "true"
    stt_stream_interval_ms: int = int(os.getenv("STT_STREAM_INTERVAL_MS", "320"))
    stt_stream_min_audio_ms: int = int(os.getenv("STT_STREAM_MIN_AUDIO_MS", "640"))
    stt_stream_window_seconds: float = float(os.getenv("STT_STREAM_WINDOW_SECONDS", "6"))
    stt_max_streams: int = int(os.getenv("STT_MAX_STREAMS", "8"))
    deepgram_api_key: str = os.getenv("DEEPGRAM_API_KEY", "")
    deepgram_ws_url: str = os.getenv("DEEPGRAM_WS_URL", "wss://api.deepgram.com/v1/listen")
    deepgram_model: str = os.getenv("DEEPGRAM_MODEL", "nova-3")
    deepgram_language: str = os.getenv("DEEPGRAM_LANGUAGE", "pt-BR")
    
    # TTS Settings (Text-To-Speech)
    tts_provider: str = os.getenv("TTS_PROVIDER", "kokoro").lower()  # kokoro, edge, openai
    tts_voice: str = os.getenv("TTS_VOICE", "pt-BR-FranciscaNeural")  # Fallback edge voice
    tts_rate: str = os.getenv("TTS_RATE", "+10%")
    tts_base_url: str = os.getenv("TTS_BASE_URL", "https://api.openai.com/v1")
    tts_api_key: str = os.getenv("TTS_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    tts_model: str = os.getenv("TTS_MODEL", "tts-1")
    api_tts_voice: str = os.getenv("API_TTS_VOICE", "nova")
    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVEN_LABS_API_KEY", "")
    elevenlabs_voice_id: str = os.getenv("ELEVENLABS_VOICE_ID", "")
    elevenlabs_model: str = os.getenv("ELEVENLABS_MODEL", "eleven_v4_turbo")
    elevenlabs_language: str = os.getenv("ELEVENLABS_LANGUAGE", "pt")
    elevenlabs_speed: float = float(os.getenv("ELEVENLABS_SPEED", "1.0"))
    
    # Kokoro SOTA TTS Settings
    local_tts_base_url: str = os.getenv("LOCAL_TTS_BASE_URL", "http://localhost:8880/v1")
    kokoro_voice: str = os.getenv("KOKORO_VOICE", "pf_dora")  # pf_dora, pm_alex, pm_santa
    kokoro_speed: float = float(os.getenv("KOKORO_SPEED", "1.0"))
    
    # Voice Activity Detection (VAD) & Barge-In
    vad_threshold: float = float(os.getenv("VAD_THRESHOLD", "0.5"))
    interruption_enabled: bool = os.getenv("INTERRUPTION_ENABLED", "true").lower() == "true"
    interruption_frames_threshold: int = int(os.getenv("INTERRUPTION_FRAMES_THRESHOLD", "1"))  # 32ms onset
    silence_duration_ms: int = int(os.getenv("SILENCE_DURATION_MS", "750"))  # ms of silence to mark end of turn
    max_utterance_seconds: float = float(os.getenv("MAX_UTTERANCE_SECONDS", "20"))
    
    # Paths
    memories_dir: str = os.getenv("MEMORIES_DIR", "data/memories")
    skills_dir: str = os.getenv("SKILLS_DIR", "data/skills")

    @property
    def active_llm_base_url(self) -> str:
        if self.mode == "local" or self.llm_provider in ("local", "ollama"):
            return self.local_llm_base_url
        if self.llm_provider == "deepseek":
            return self.deepseek_base_url
        return self.openai_base_url

    @property
    def active_llm_api_key(self) -> str:
        if self.mode == "local" or self.llm_provider in ("local", "ollama"):
            return "ollama-or-local"
        if self.llm_provider == "deepseek":
            return self.deepseek_api_key
        return self.openai_api_key

    @property
    def active_llm_model(self) -> str:
        if self.mode == "local" or self.llm_provider in ("local", "ollama"):
            return self.local_llm_model
        if self.llm_provider == "deepseek":
            return self.deepseek_model
        return self.llm_model

    @property
    def active_stt_provider(self) -> str:
        return self.stt_provider

config = BotConfig()
