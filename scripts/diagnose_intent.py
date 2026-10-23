"""Check active LLM intent decisions without Discord, STT, TTS, or tools."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bot.brain.llm import LLMConversationEngine


async def main():
    engine = LLMConversationEngine()
    engine.history = [
        {"role": "user", "content": "[Ana / 1]: Murbot, explica uma build de Phantom Assassin."},
        {"role": "assistant", "content": "Depois da Battle Fury, você pode fazer BKB para proteção mágica. A ordem depende da partida."},
    ]
    cases = [
        ("João, qual item você comprou?", "ignore"),
        ("Galera, essa BKB é muito boa nesse jogo.", "ignore"),
        ("Ah, entendi, legal.", "ignore"),
        ("Você pode explicar melhor por que recomendou BKB?", "reply"),
        ("E depois da BKB, o que você recomenda para essa build?", "reply"),
    ]
    try:
        for text, expected in cases:
            actual = await engine.classify_followup("Ana", 1, text)
            print(f"{actual}: {text}", flush=True)
            if actual != expected:
                raise RuntimeError(f"Expected {expected}, got {actual}")
        print("PASS: follow-ups and side conversations separated in all examples.", flush=True)
    finally:
        await engine.client.close()


if __name__ == "__main__":
    asyncio.run(main())
