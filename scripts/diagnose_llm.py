"""Stream a neutral test reply through the configured local LLM, without Discord."""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bot.config import config
from bot.brain.llm import LLMConversationEngine

async def main():
    if config.mode != "local" and config.llm_provider not in ("local", "ollama"):
        raise RuntimeError("This diagnostic is restricted to the local LLM configuration")
    engine = LLMConversationEngine()
    try:
        print("Model:", config.active_llm_model)
        async for sentence in engine.stream_reply_sentences(
            "Audio test", 0, "Murbot, responda apenas com uma saudação curta em português. Não use ferramentas.", asyncio.Event()
        ):
            print("Sentence:", sentence)
        if not engine.history:
            raise RuntimeError("LLM did not complete successfully; inspect the error above")
    finally:
        await engine.client.close()

if __name__ == "__main__":
    asyncio.run(main())
