"""Compatibility adapter for voice-recv 0.5.2's missing DAVE receive layer.

Keep this at the raw Opus boundary. Remove when upgrading to a receiver that
already decrypts DAVE; applying decryption twice is not supported.
"""
import logging

logger = logging.getLogger("VoiceBot.Transport")
OPUS_SILENCE = b"\xf8\xff\xfe"


def decrypt_voice_frame(voice_client, user_id: int, payload: bytes) -> bytes | None:
    state = getattr(voice_client, "_connection", None)
    if not getattr(state, "dave_protocol_version", 0) or payload == OPUS_SILENCE:
        return payload
    session = getattr(state, "dave_session", None)
    if session is None or not session.ready:
        return None  # Never decode ciphertext while MLS keys are negotiating.
    import davey
    try:
        return session.decrypt(user_id, davey.MediaType.audio, payload)
    except Exception:
        logger.debug("Dropping undecryptable DAVE frame from user %s", user_id)
        return None
