"""Check local model tool use with a simulated Discord room; --web also checks public web access."""
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bot.config import config
from bot.brain.llm import LLMConversationEngine
from bot.brain.tools import execute_tool


def simulated_context():
    channel = NS(id=10, name="Sala de teste", permissions_for=lambda member: NS(view_channel=True))
    members = [NS(id=uid, name=name, display_name=name, bot=is_bot,
                  voice=NS(channel=channel)) for uid, name, is_bot in [(1, "Ana", False), (2, "Beto", False), (3, "Murbot", True)]]
    channel.members = members
    channel.voice_states = {m.id: m.voice for m in members}
    guild = NS(id=100, name="Servidor de teste", me=members[2], channels=[channel],
               get_member=lambda uid: next((m for m in members if m.id == uid), None))
    vc = NS(is_connected=lambda: True, channel=channel, guild=guild)
    return {"voice_client": vc}


async def main(args):
    if config.mode != "local" and config.llm_provider not in ("local", "ollama") and not args.api:
        raise RuntimeError("Pass --api to test the configured API provider")
    engine = LLMConversationEngine(tool_context=simulated_context())
    calls = []
    async def record(name, arguments, context):
        calls.append(name)
        return await execute_tool(name, arguments, context)
    try:
        print("Provider:", config.llm_provider, "Model:", config.active_llm_model, flush=True)
        if args.api:
            greeting = " ".join([part async for part in engine.stream_reply_sentences(
                "Ana", 1, "Murbot, responda só com uma saudação curta em português.", asyncio.Event())])
            if not engine.history:
                raise RuntimeError("API greeting failed; inspect the error above")
            print("Greeting:", greeting, flush=True)
        with patch("bot.brain.llm.execute_tool", side_effect=record):
            text = " ".join([part async for part in engine.stream_reply_sentences(
                "Ana", 1, "Murbot, quem está na sala agora? Diga os nomes das pessoas e bots presentes.", asyncio.Event())])
        print("Tool calls:", calls, flush=True)
        print("Reply:", text, flush=True)
        if "discord_list_room_members" not in calls and "discord_room_info" not in calls:
            raise RuntimeError("The model did not consult a live room tool")
        names_correct = all(name.casefold() in text.casefold() for name in ("Ana", "Beto"))
        bot_identified = "murbot" in text.casefold() or re.search(r"\beu\b.*\bbot\b|\bbot\b.*\beu\b", text.casefold())
        if not names_correct or not bot_identified:
            raise RuntimeError("The model did not correctly summarize the simulated room")
        print("PASS: configured LLM used Discord tools and summarized the simulated room.", flush=True)
        if args.web:
            search = json.loads(await execute_tool("web_search", {"query": "Python official documentation", "max_results": 2}, {}))
            page = json.loads(await execute_tool("browse_webpage", {"url": "https://www.python.org/"}, {}))
            if not search.get("results") or not page.get("text"):
                raise RuntimeError("Web diagnostic returned no usable result")
            print("PASS: search returned sources and Python.org was fetched; title:", page["title"], flush=True)
        if args.retrieval:
            calls.clear()
            with patch("bot.brain.llm.execute_tool", side_effect=record):
                text = " ".join([part async for part in engine.stream_reply_sentences(
                    "Ana", 1, "Murbot, leia https://www.python.org/ e diga o que é Python segundo essa página.", asyncio.Event())])
            print("Retrieval calls:", calls, flush=True)
            print("Retrieval reply:", text, flush=True)
            if not calls or calls[0] != "browse_webpage" or "python" not in text.casefold():
                raise RuntimeError("Automatic page retrieval did not produce an answer")
            print("PASS: automatic retrieval reached the configured model before its answer.", flush=True)
    finally:
        await engine.client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web", action="store_true")
    parser.add_argument("--api", action="store_true", help="Use the configured paid API for greeting, follow-up, and tool-call checks")
    parser.add_argument("--retrieval", action="store_true", help="Test automatic page retrieval followed by a real local LLM answer")
    asyncio.run(main(parser.parse_args()))
