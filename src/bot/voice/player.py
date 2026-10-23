import time
import asyncio
import logging
import threading
import discord

logger = logging.getLogger("VoiceBot.Player")

class InterruptibleAudioSource(discord.AudioSource):
    """
    A streaming AudioSource for Discord that pulls 20ms frames (3840 bytes)
    from an internal thread-safe buffer.
    Supports instant clearing/cancellation when an interruption (barge-in) occurs.
    """
    FRAME_SIZE = 3840  # 48kHz, 16-bit, stereo: 960 samples * 2 channels * 2 bytes

    def __init__(self):
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._finished = False
        self._is_interrupted = False
        self._has_written_audio = False
        self._frames_played = 0

    @property
    def is_actively_playing(self) -> bool:
        """Returns True only if actual audible speech audio has been written and is playing."""
        with self._lock:
            if self._is_interrupted:
                return False
            if self._finished and len(self._buffer) == 0:
                return False
            return self._has_written_audio and len(self._buffer) > 0

    def write_pcm(self, pcm_bytes: bytes):
        """Append 48kHz 16-bit stereo PCM audio to the streaming buffer."""
        if self._is_interrupted or not pcm_bytes:
            return
        with self._lock:
            if self._is_interrupted or self._finished:
                return
            self._buffer.extend(pcm_bytes)
            self._has_written_audio = True

    @property
    def buffered_seconds(self):
        with self._lock:
            return len(self._buffer) / 192000

    def mark_finished(self):
        """Signals that no more audio will be added to this stream."""
        with self._lock:
            self._finished = True

    def interrupt(self):
        """Immediately flushes the buffer and halts playback within 0ms."""
        with self._lock:
            self._is_interrupted = True
            self._buffer.clear()
            self._finished = True
        logger.info("InterruptibleAudioSource: Playback interrupted and flushed.")

    def read(self) -> bytes:
        """Called by Discord every 20ms to retrieve the next audio frame."""
        with self._lock:
            if self._is_interrupted:
                return b""

            if len(self._buffer) >= self.FRAME_SIZE:
                frame = bytes(self._buffer[:self.FRAME_SIZE])
                del self._buffer[:self.FRAME_SIZE]
                self._frames_played += 1
                return frame
            elif self._finished and len(self._buffer) > 0:
                # Pad remaining bytes to full frame size with silence
                frame = bytes(self._buffer)
                frame += b"\x00" * (self.FRAME_SIZE - len(frame))
                self._buffer.clear()
                self._frames_played += 1
                return frame
            elif self._finished:
                return b""
            else:
                # Buffer underrun during active synthesis; output 20ms silence frame
                return b"\x00" * self.FRAME_SIZE

    def is_opus(self) -> bool:
        return False

    @property
    def is_pending(self) -> bool:
        with self._lock:
            return not self._is_interrupted and (not self._finished or bool(self._buffer))


class VoicePlaybackController:
    """
    Manages active playback sessions, audio queues, and instant barge-in cancellation.
    """
    def __init__(self, voice_client: discord.VoiceClient):
        self.vc = voice_client
        self.current_source: InterruptibleAudioSource | None = None
        self._cancel_token = asyncio.Event()
        self._playback_started_at: float = 0.0
        self.playback_finished_at: float = 0.0
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.get_event_loop()

    @property
    def is_session_active(self) -> bool:
        return self.current_source is not None and (
            self.current_source.is_pending or self.vc.is_playing()
        )

    @property
    def is_playing(self) -> bool:
        """
        True ONLY if the bot is actively emitting audible sound to Discord.
        Returns False during silent buffering or LLM generation.
        """
        if self.current_source is None or not self.current_source.is_actively_playing:
            return False
        return self.vc is not None and self.vc.is_playing()

    def stop_playback(self, is_barge_in: bool = True):
        """Stops current audio and raises cancel token in a thread-safe manner."""
        token = self._cancel_token
        try:
            on_loop = asyncio.get_running_loop() is self.loop
        except RuntimeError:
            on_loop = False
        if on_loop:
            token.set()
        elif self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(token.set)
        else:
            token.set()

        if self.current_source:
            self.current_source.interrupt()
            self.current_source = None

        if self.vc:
            if hasattr(self.vc, "stop_playing"):
                self.vc.stop_playing()
            elif self.vc.is_playing():
                self.vc.stop()

        self._playback_started_at = 0.0
        if is_barge_in:
            logger.info("VoicePlaybackController: Playback interrupted by user speech (barge-in).")

    def ensure_playing(self):
        """Starts Discord voice client playback with the current source if not already active."""
        if not self.vc or not self.current_source:
            return

        if not self.vc.is_playing():
            try:
                self._play_source()
                self._playback_started_at = time.time()
                logger.info("VoicePlaybackController: Started streaming audio to Discord.")
            except discord.ClientException:
                if hasattr(self.vc, "stop_playing"):
                    self.vc.stop_playing()
                elif self.vc.is_playing():
                    self.vc.stop()
                try:
                    self._play_source()
                    self._playback_started_at = time.time()
                except Exception as e:
                    logger.error(f"VoicePlaybackController: Error starting playback: {e}")
            except Exception as e:
                logger.error(f"VoicePlaybackController: Error starting playback: {e}")

    def _play_source(self):
        source = self.current_source
        def after(error):
            if not self.loop.is_closed():
                self.loop.call_soon_threadsafe(self._playback_finished, source, error)
        self.vc.play(source, after=after)

    def _playback_finished(self, source, error):
        if source is not self.current_source:
            return  # Completion of an interrupted source cannot affect a new reply.
        self.playback_finished_at = time.monotonic()
        if error:
            logger.error("Discord playback failed: %s", error)

    def start_new_session(self) -> tuple[InterruptibleAudioSource, asyncio.Event]:
        """
        Prepares a new playback session.
        Returns the new audio source and the active session cancel token.
        """
        self.stop_playback(is_barge_in=False)
        self._cancel_token = asyncio.Event()
        self.current_source = InterruptibleAudioSource()
        self._playback_started_at = 0.0

        return self.current_source, self._cancel_token
