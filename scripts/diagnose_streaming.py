"""Replay a known local Kokoro phrase in real time through streaming STT.

No Discord connection. Uses the configured STT provider; Deepgram sends audio
to that provider and requires its own API key.
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import httpx
import numpy as np
from diagnose_audio import load_audio
from bot.config import config
from bot.providers.stt import SpeechToTextProvider
from bot.providers.streaming_stt import AudioStreamFrame, StreamingSpeechRoom
from bot.voice.addressing import classify_address


async def main():
    phrase = "Murbot, me explique como as estrelas se formam e por que elas brilham no céu durante a noite."
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(config.local_tts_base_url.rstrip("/") + "/audio/speech", json={
            "model": "kokoro", "voice": config.kokoro_voice, "input": phrase,
            "response_format": "wav", "speed": 1.0,
        })
        response.raise_for_status()
        audio = load_audio(response.content)
    provider = await asyncio.to_thread(SpeechToTextProvider)
    if provider.provider not in ("local", "deepgram"):
        raise RuntimeError("Select local or deepgram to test partial transcription")
    start = time.monotonic()
    ended = False
    early_name = False
    finals = []
    async def callback(event):
        nonlocal early_name
        action = classify_address(event.text, config.bot_name, config.bot_aliases)
        print(f"{time.monotonic()-start:.2f}s {'FINAL' if event.is_final else 'PARTIAL'} [{action}]: {event.text}", flush=True)
        if not event.is_final and not ended and action != "ignore":
            early_name = True
        if event.is_final:
            finals.append(event)
    room = StreamingSpeechRoom(provider, callback)
    try:
        print(f"Sending {len(audio)/16000:.2f}s audio in 96ms chunks", flush=True)
        for offset in range(0, len(audio), 1536):
            room.feed(AudioStreamFrame(1, 1, "Diagnostic", audio[offset:offset+1536]))
            await asyncio.sleep(1536/16000)
        ended = True
        print(f"{time.monotonic()-start:.2f}s INPUT ENDED", flush=True)
        room.feed(AudioStreamFrame(1, 1, "Diagnostic", np.empty(0, dtype=np.float32), True))
        await asyncio.wait_for(asyncio.gather(*(s.task for s in room.active.values())), 30)
        if not early_name:
            raise RuntimeError("No addressed partial arrived before the input ended")
        if len(finals) != 1:
            raise RuntimeError("Expected exactly one final transcript")
        print("PASS: name recognized while audio was still arriving; one final transcript.", flush=True)
    finally:
        await room.aclose()
        provider._executor.shutdown(wait=True)
        if provider.cloud_client:
            await provider.cloud_client.close()


if __name__ == "__main__":
    asyncio.run(main())
