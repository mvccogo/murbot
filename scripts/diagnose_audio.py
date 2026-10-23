"""Exercise configured local audio services without connecting to Discord.

python scripts/diagnose_audio.py --synthesize
python scripts/diagnose_audio.py --wav data/debug_audio/<clip>.wav
"""
import argparse
import asyncio
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import av
import httpx
import numpy as np
from bot.config import config
from bot.providers.stt import SpeechToTextProvider
from bot.voice.addressing import classify_address


def load_audio(data):
    samples = []
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
    with av.open(io.BytesIO(data)) as container:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                samples.append(out.to_ndarray().flatten())
        for out in resampler.resample(None):
            samples.append(out.to_ndarray().flatten())
    return np.concatenate(samples).astype(np.float32)


async def run(args):
    if args.synthesize:
        phrase = "Murbot, me diga qual é a capital do Brasil."
        print("Known input:", phrase)
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(config.local_tts_base_url.rstrip("/") + "/audio/speech", json={
                "model": "kokoro", "voice": config.kokoro_voice,
                "input": phrase, "response_format": "wav", "speed": 1.0,
            })
            response.raise_for_status()
            audio = load_audio(response.content)
    else:
        audio = load_audio(Path(args.wav).read_bytes())
    print(f"Audio: {len(audio)/16000:.2f}s, RMS={np.sqrt(np.mean(audio**2)):.5f}, peak={np.max(np.abs(audio)):.5f}")
    provider = await asyncio.to_thread(SpeechToTextProvider)
    try:
        text = await provider.transcribe(audio)
        print("Transcript:", text)
        print("Routing:", classify_address(text, config.bot_name, config.bot_aliases))
        if args.synthesize and ("brasil" not in text.casefold() or not text):
            raise RuntimeError("Known phrase failed STT check")
    finally:
        provider._executor.shutdown(wait=True)
        if provider.cloud_client:
            await provider.cloud_client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthesize", action="store_true")
    group.add_argument("--wav")
    asyncio.run(run(parser.parse_args()))
