import logging
import numpy as np
import torch
import silero_vad
from scipy import signal

logger = logging.getLogger("VoiceBot.VAD")

class SileroVoiceDetector:
    """
    Silero VAD processor for real-time speech detection and barge-in.
    Converts Discord 48kHz stereo PCM to 16kHz mono, runs Silero VAD,
    and returns speech probabilities and active speech frames.
    """
    def __init__(self, threshold: float = 0.55):
        self.threshold = threshold
        self.sample_rate = 16000
        logger.info("Initializing Silero VAD model...")
        self.model = silero_vad.load_silero_vad()
        self.model.eval()
        self.reset()
        logger.info("Silero VAD initialized successfully.")

    def reset(self):
        """Reset internal model state between speakers or audio resets."""
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()

    @staticmethod
    def discord_to_16k_mono(pcm_bytes: bytes) -> np.ndarray:
        """
        Converts Discord 48kHz 16-bit stereo PCM bytes to 16kHz float32 mono array.
        3840 bytes (20ms at 48kHz stereo) -> 320 samples at 16kHz mono.
        Applies polyphase anti-aliasing filtering.
        """
        if not pcm_bytes:
            return np.empty(0, dtype=np.float32)
        
        # Read int16 stereo samples
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        if len(audio_int16) % 2 != 0:
            audio_int16 = audio_int16[: len(audio_int16) - 1]
            
        stereo = audio_int16.reshape(-1, 2)
        left = stereo[:, 0].astype(np.float32) / 32768.0
        right = stereo[:, 1].astype(np.float32) / 32768.0

        # Avoid cutting amplitude in half if user has a mono microphone where one channel is silent/noise
        left_energy = float(np.mean(left**2)) if len(left) > 0 else 0.0
        right_energy = float(np.mean(right**2)) if len(right) > 0 else 0.0
        if left_energy > 3.0 * right_energy:
            mono = left
        elif right_energy > 3.0 * left_energy:
            mono = right
        else:
            mono = (left + right) / 2.0

        # Downsample from 48kHz to 16kHz with anti-aliasing FIR filter (decimate 3:1)
        mono_16k = signal.resample_poly(mono, 1, 3).astype(np.float32)
        return mono_16k

    def predict_chunk(self, chunk_512: np.ndarray) -> float:
        """
        Runs VAD on a 512-sample float32 chunk (32ms at 16kHz).
        Returns probability of human speech (0.0 to 1.0).
        """
        if len(chunk_512) != 512:
            return 0.0
        
        tensor = torch.from_numpy(chunk_512).float()
        with torch.no_grad():
            prob = self.model(tensor, self.sample_rate).item()
        return prob

    def is_speech(self, chunk_512: np.ndarray) -> bool:
        """Returns True if speech probability exceeds threshold."""
        prob = self.predict_chunk(chunk_512)
        return prob >= self.threshold
