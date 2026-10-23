"""Queued console/file logging with bounded retention and credential redaction."""
import atexit
import logging
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
import os
from pathlib import Path
import queue
import re
import sys
import threading


class RedactingFormatter(logging.Formatter):
    def __init__(self, secrets=()):
        super().__init__("%(asctime)s [%(levelname)s] %(name)s [%(threadName)s]: %(message)s")
        self.secrets = sorted({value for value in secrets if len(value) >= 6}, key=len, reverse=True)

    def format(self, record):
        text = super().format(record)
        for secret in self.secrets:
            text = text.replace(secret, "[REDACTED]")
        text = re.sub(r"(?i)(bearer\s+)[\w.\-]+", r"\1[REDACTED]", text)
        return re.sub(
            r"(?i)((?:api[_-]?key|xi-api-key|authorization|password|secret|token)[\"']?\s*[:=]\s*[\"']?)([^\s\"'&,}]+)",
            r"\1[REDACTED]", text,
        )


class ConsoleNoiseFilter(logging.Filter):
    def filter(self, record):
        # Sender reports are routine control packets, not failed speech frames.
        # Preserve them in the rotating file without flooding the voice console.
        return not (record.name == "discord.ext.voice_recv.reader"
                    and record.levelno == logging.INFO
                    and record.getMessage().startswith("Received unexpected rtcp packet: type=200,"))


class LoggingRuntime:
    def __init__(self, listener, handlers, queue_handler):
        self.listener = listener
        self.handlers = handlers
        self.queue_handler = queue_handler
        self.closed = False

    def close(self):
        if self.closed:
            return
        self.closed = True
        logging.getLogger().removeHandler(self.queue_handler)
        self.listener.stop()  # Drain queued records before closing files.
        for handler in self.handlers:
            handler.close()
        self.queue_handler.close()


def configure_logging(config):
    directory = Path(config.log_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if config.log_max_bytes < 1024 or config.log_backup_count < 1:
        raise ValueError("LOG_MAX_BYTES must be at least 1024 and LOG_BACKUP_COUNT at least 1")
    secrets = [value for key, value in os.environ.items()
               if re.search(r"(?:KEY|TOKEN|PASSWORD|SECRET)", key, re.I)]
    formatter = RedactingFormatter(secrets)
    file_handler = RotatingFileHandler(
        directory / "murbot.log", maxBytes=config.log_max_bytes,
        backupCount=config.log_backup_count, encoding="utf-8", errors="backslashreplace",
    )
    file_handler.setLevel(config.log_level.upper())
    console = logging.StreamHandler()
    console.addFilter(ConsoleNoiseFilter())
    console.setLevel(config.log_console_level.upper())
    for handler in (file_handler, console):
        handler.setFormatter(formatter)
    queue_handler = QueueHandler(queue.Queue())
    root = logging.getLogger()
    # Replace previously configured handlers rather than duplicating output.
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    root.setLevel(min(file_handler.level, console.level))
    root.addHandler(queue_handler)
    # Keep verbose reception logs in the file as well as other dependencies.
    logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.NOTSET)
    logging.captureWarnings(True)
    listener = QueueListener(queue_handler.queue, file_handler, console, respect_handler_level=True)
    listener.start()
    runtime = LoggingRuntime(listener, (file_handler, console), queue_handler)
    atexit.register(runtime.close)

    def exception_hook(kind, value, traceback):
        if issubclass(kind, KeyboardInterrupt):
            sys.__excepthook__(kind, value, traceback)
        else:
            logging.getLogger("VoiceBot.Uncaught").critical("Uncaught exception", exc_info=(kind, value, traceback))

    sys.excepthook = exception_hook
    threading.excepthook = lambda args: exception_hook(args.exc_type, args.exc_value, args.exc_traceback)
    logging.getLogger("VoiceBot.Logging").info(
        "Recording logs to %s; rotation=%s bytes, backups=%s, file_level=%s, console_level=%s",
        file_handler.baseFilename, config.log_max_bytes, config.log_backup_count,
        config.log_level, config.log_console_level,
    )
    return runtime
