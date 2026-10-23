"""Inspect available voices/models, or --speak one short neutral streaming test."""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import httpx
from bot.config import config
from bot.providers.elevenlabs import ElevenLabsSpeechProvider


async def main(args):
    if not config.elevenlabs_api_key:
        raise RuntimeError("ElevenLabs key is not configured")
    if args.speak:
        provider = ElevenLabsSpeechProvider()
        start = time.monotonic()
        total = chunks = 0
        try:
            async for pcm in provider.stream_pcm("Olá! Eu sou o Murbot. Podemos conversar em português."):
                if chunks == 0:
                    print(f"First PCM: {time.monotonic()-start:.2f}s", flush=True)
                chunks += 1
                total += len(pcm)
            if not total:
                raise RuntimeError("No audio received")
            print(f"PASS: {chunks} PCM chunks, {total/192000:.2f}s audio, {time.monotonic()-start:.2f}s request", flush=True)
        finally:
            await provider.aclose()
        return
    async with httpx.AsyncClient(timeout=20, headers={"xi-api-key": config.elevenlabs_api_key}) as client:
        if args.library:
            response = await client.get("https://api.elevenlabs.io/v1/shared-voices", params={"language": "pt", "locale": "pt-BR", "page_size": 5, "use_cases": "conversational", "sort": "cloned_by_count"})
            print(json.dumps({"http": response.status_code, "voices": [{"id": v["voice_id"], "name": v["name"], "accent": v.get("accent"), "language": v.get("language"), "free_users_allowed": v.get("free_users_allowed")} for v in response.json().get("voices", [])]}))
            return
        models, voices = await asyncio.gather(client.get("https://api.elevenlabs.io/v1/models"), client.get("https://api.elevenlabs.io/v2/voices", params={"page_size": 100}))
        output = {"models_http": models.status_code, "voices_http": voices.status_code}
        if models.status_code == 200:
            output["tts_models"] = [{"id": m["model_id"], "portuguese": any(l["language_id"] == "pt" for l in m.get("languages", []))} for m in models.json() if m.get("can_do_text_to_speech")]
        if voices.status_code == 200:
            items = voices.json().get("voices", [])
            portuguese = [v for v in items if any(w in json.dumps(v.get("labels", {})).casefold() for w in ("brazil", "portug", '"pt"')) or any(l.get("language") == "pt" for l in v.get("verified_languages", []))]
            output["portuguese_compatible_voices"] = [{"id": v["voice_id"], "name": v["name"], "labels": v.get("labels", {})} for v in portuguese[:20]]
            if not portuguese:
                output["available_voice_examples"] = [{"id": v["voice_id"], "name": v["name"], "labels": v.get("labels", {})} for v in items[:8]]
        print(json.dumps(output, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speak", action="store_true", help="Generate one short test phrase (uses API credits)")
    parser.add_argument("--library", action="store_true", help="Read public Brazilian Portuguese conversational voices")
    asyncio.run(main(parser.parse_args()))
