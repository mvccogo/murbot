import asyncio
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bot.brain.retrieval import lookup_plan, retrieve
from bot.brain.llm import LLMConversationEngine
from bot.config import config


class Stream:
    def __init__(self, text):
        self.text = iter(text)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return NS(choices=[NS(delta=NS(content=next(self.text), tool_calls=None))])
        except StopIteration:
            raise StopAsyncIteration

    async def close(self):
        self.closed = True


class RetrievalTests(unittest.IsolatedAsyncioTestCase):
    def test_requests_and_explicit_opt_out(self):
        self.assertEqual(lookup_plan("Leia https://example.com/page."), ("browse_webpage", {"url": "https://example.com/page"}))
        self.assertEqual(lookup_plan("Veja x.com/alguem/status/123")[1]["url"], "https://x.com/alguem/status/123")
        self.assertEqual(lookup_plan("Murbot, pesquisa o lançamento do jogo")[0], "web_search")
        for text in ("Olá, Murbot", "Quem está na sala?", "Não pesquise, explique isso", "Don't search the internet"):
            self.assertIsNone(lookup_plan(text))

    async def test_failed_page_searches_and_reads_an_alternate_source(self):
        execute = AsyncMock(side_effect=[
            "Erro ao executar browse_webpage: HTTP 403",
            json.dumps({"results": [{"url": "https://example.com/post"}, {"url": "https://other.org/copy"}]}),
            json.dumps({"text": "Public alternative source. " * 10}),
        ])
        attempts = await retrieve("browse_webpage", {"url": "https://example.com/post?token=secret"}, execute, {}, asyncio.Event())
        self.assertEqual([a[0] for a in attempts], ["browse_webpage", "web_search", "browse_webpage"])
        self.assertEqual(attempts[1][1]["query"], "example.com  post")
        self.assertEqual(attempts[2][1]["url"], "https://other.org/copy")
        self.assertNotIn("secret", attempts[1][1]["query"])

    async def test_empty_or_challenge_page_also_uses_search(self):
        for text in ("", "Just a moment: verify you are human. " * 10):
            execute = AsyncMock(side_effect=[json.dumps({"text": text}), '{"results": []}'])
            attempts = await retrieve("browse_webpage", {"url": "https://example.com"}, execute, {}, asyncio.Event())
            self.assertEqual([a[0] for a in attempts], ["browse_webpage", "web_search"])

    async def test_search_tries_next_source_after_blocked_result(self):
        execute = AsyncMock(side_effect=[
            json.dumps({"results": [{"url": "https://blocked.org"}, {"url": "https://public.org"}]}),
            json.dumps({"http_status": 403, "error": "HTTP 403"}),
            json.dumps({"text": "Verified public source. " * 10}),
        ])
        attempts = await retrieve("web_search", {"query": "build"}, execute, {}, asyncio.Event())
        self.assertEqual([a[0] for a in attempts], ["web_search", "browse_webpage", "browse_webpage"])
        self.assertEqual(attempts[-1][1]["url"], "https://public.org")

    async def test_failed_page_is_cached_for_this_turn(self):
        from bot.brain.tools import execute_tool
        context = {}
        with patch("bot.brain.tools.browse_webpage", new_callable=AsyncMock, side_effect=ValueError("O site retornou HTTP 403.")) as page:
            first = await execute_tool("browse_webpage", {"url": "https://example.com"}, context)
            second = await execute_tool("browse_webpage", {"url": "https://example.com"}, context)
        page.assert_awaited_once()
        self.assertEqual(first, second)
        self.assertEqual(json.loads(first)["http_status"], 403)

    async def test_rejected_url_is_not_forwarded_to_search(self):
        for url in ("http://127.0.0.1/private", "https://user:secret@example.com/private"):
            execute = AsyncMock(return_value="URL rejeitada")
            attempts = await retrieve("browse_webpage", {"url": url}, execute, {}, asyncio.Event())
            self.assertEqual(len(attempts), 1)

    async def test_cancellation_does_not_start_fallback(self):
        cancel = asyncio.Event()
        async def stop(*args):
            cancel.set()
            return "Erro HTTP 403"
        execute = AsyncMock(side_effect=stop)
        attempts = await retrieve("browse_webpage", {"url": "https://example.com"}, execute, {}, cancel)
        self.assertEqual(len(attempts), 1)

    def engine(self, *streams):
        engine = LLMConversationEngine.__new__(LLMConversationEngine)
        engine.history = []
        engine._build_system_prompt = Mock(return_value="test")
        engine.client = NS(chat=NS(completions=NS(create=AsyncMock(side_effect=streams))))
        return engine

    async def test_search_happens_before_model_can_refuse(self):
        engine = self.engine(Stream(["Segundo a fonte, aqui está a resposta."]))
        async def search(*args):
            engine.client.chat.completions.create.assert_not_awaited()
            return '{"results": []}'
        with patch.object(config, "llm_tools_enabled", True), patch("bot.brain.llm.execute_tool", side_effect=search) as execute:
            reply = [s async for s in engine.stream_reply_sentences("Ana", 1, "Pesquise novidades do Python", asyncio.Event())]
        execute.assert_awaited_once()
        self.assertTrue(reply)
        messages = engine.client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(messages[-1]["role"], "tool")

    async def test_access_refusal_is_replaced_by_a_real_lookup_and_retry(self):
        first = Stream(["Não tenho acesso ", "a informações em tempo real. "])
        engine = self.engine(first, Stream(["Encontrei informações na fonte consultada."]))
        with patch.object(config, "llm_tools_enabled", True), patch("bot.brain.llm.execute_tool", new_callable=AsyncMock, return_value='{"results": []}') as execute:
            reply = [s async for s in engine.stream_reply_sentences("Ana", 1, "Me conte sobre esse lançamento", asyncio.Event())]
        execute.assert_awaited_once()
        self.assertEqual(reply, ["Encontrei informações na fonte consultada."])
        self.assertTrue(first.closed)
        self.assertNotIn("Não tenho", engine.history[-1]["content"])

    async def test_disabled_tools_and_user_opt_out_never_run_preflight(self):
        for enabled, text in ((False, "Pesquise Python"), (True, "Não pesquise na internet, explique Python")):
            engine = self.engine(Stream(["Uma linguagem de programação."]))
            with patch.object(config, "llm_tools_enabled", enabled), patch("bot.brain.llm.execute_tool", new_callable=AsyncMock) as execute:
                _ = [s async for s in engine.stream_reply_sentences("Ana", 1, text, asyncio.Event())]
            execute.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
