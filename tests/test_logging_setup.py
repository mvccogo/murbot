import io
import logging
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bot.logging_setup import RedactingFormatter, ConsoleNoiseFilter, configure_logging


class LoggingTests(unittest.TestCase):
    def test_console_filter_only_hides_routine_rtcp_sender_reports(self):
        noise_filter = ConsoleNoiseFilter()
        def record(level, message):
            return logging.LogRecord("discord.ext.voice_recv.reader", level, __file__, 1, message, (), None)
        self.assertFalse(noise_filter.filter(record(logging.INFO, "Received unexpected rtcp packet: type=200, SenderReportPacket")))
        self.assertTrue(noise_filter.filter(record(logging.WARNING, "Received unexpected rtcp packet: type=200, SenderReportPacket")))
        self.assertTrue(noise_filter.filter(record(logging.INFO, "Voice disconnected")))

    def test_redaction_covers_credentials_and_exception_text(self):
        formatter = RedactingFormatter(["configured-secret-value"])
        try:
            raise ValueError("configured-secret-value token=another-secret")
        except ValueError:
            record = logging.LogRecord("test", logging.ERROR, __file__, 1,
                                       "Authorization: Bearer bearer-secret api_key='query-secret'", (), sys.exc_info())
        text = formatter.format(record)
        for secret in ("configured-secret-value", "another-secret", "bearer-secret", "query-secret"):
            self.assertNotIn(secret, text)
        self.assertIn("ValueError", text)
        self.assertIn("[REDACTED]", text)

    def test_queued_debug_logging_rotates_and_flushes_without_leaking_keys(self):
        root = logging.getLogger()
        old_handlers, old_level = root.handlers[:], root.level
        old_hook, old_thread_hook = sys.excepthook, threading.excepthook
        reader = logging.getLogger("discord.ext.voice_recv.reader")
        old_reader_level = reader.level
        try:
            with tempfile.TemporaryDirectory() as folder, patch("sys.stderr", new_callable=io.StringIO) as console, patch.dict(os.environ, {"TEST_API_KEY": "test-configured-secret"}):
                runtime = configure_logging(NS(log_dir=folder, log_level="DEBUG", log_console_level="INFO", log_max_bytes=1024, log_backup_count=2))
                try:
                    logger = logging.getLogger("VoiceBot.LogTest")
                    for index in range(60):
                        logger.debug("rotation %s %s", index, "x" * 90)
                    logger.debug("file-only-marker test-configured-secret")
                    logger.info("console-marker")
                    try:
                        raise RuntimeError("test-configured-secret")
                    except RuntimeError:
                        logger.exception("exception-marker")
                finally:
                    runtime.close()
                    runtime.close()  # Shutdown is idempotent.
                files = list(Path(folder).glob("murbot.log*"))
                self.assertEqual(len(files), 3)
                text = "\n".join(path.read_text(encoding="utf-8") for path in files)
                self.assertIn("file-only-marker", text)
                self.assertIn("exception-marker", text)
                self.assertIn("RuntimeError", text)
                self.assertIn("[REDACTED]", text)
                self.assertNotIn("test-configured-secret", text + console.getvalue())
                self.assertNotIn("file-only-marker", console.getvalue())
                self.assertIn("console-marker", console.getvalue())
        finally:
            root.handlers[:] = old_handlers
            root.setLevel(old_level)
            reader.setLevel(old_reader_level)
            sys.excepthook, threading.excepthook = old_hook, old_thread_hook
            logging.captureWarnings(False)


if __name__ == "__main__":
    unittest.main()
