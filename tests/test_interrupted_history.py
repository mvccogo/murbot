import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bot.brain.llm import LLMConversationEngine


class InterruptedHistoryTests(unittest.IsolatedAsyncioTestCase):
    def engine(self, create):
        engine = LLMConversationEngine.__new__(LLMConversationEngine)
        engine.history = []
        engine._build_system_prompt = Mock(return_value="test")
        engine.client = NS(chat=NS(completions=NS(create=create)))
        return engine

    async def test_cancel_during_retrieval_retains_original_request_for_continue(self):
        started = asyncio.Event()
        async def wait(*args):
            started.set()
            await asyncio.Event().wait()
        engine = self.engine(AsyncMock())
        async def consume():
            return [s async for s in engine.stream_reply_sentences("Ana", 1, "Pesquise uma build para Phantom Assassin", asyncio.Event())]
        with patch("bot.brain.llm.execute_tool", side_effect=wait):
            task = asyncio.create_task(consume())
            await asyncio.wait_for(started.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(len(engine.history), 1)
        self.assertIn("Phantom Assassin", engine.history[0]["content"])

    async def test_closing_during_tts_retains_only_emitted_text_with_interruption_marker(self):
        class Stream:
            closed = False
            def __aiter__(self):
                return self
            async def __anext__(self):
                return NS(choices=[NS(delta=NS(content="Vou consultar as fontes. ", tool_calls=None))])
            async def close(self):
                self.closed = True
        stream = Stream()
        engine = self.engine(AsyncMock(return_value=stream))
        reply = engine.stream_reply_sentences("Ana", 1, "Me ajude com a build", asyncio.Event())
        self.assertEqual(await anext(reply), "Vou consultar as fontes.")
        await reply.aclose()
        self.assertTrue(stream.closed)
        self.assertEqual(len(engine.history), 2)
        self.assertIn("build", engine.history[0]["content"])
        self.assertIn("Resposta interrompida", engine.history[1]["content"])
        self.assertIn("Vou consultar", engine.history[1]["content"])


if __name__ == "__main__":
    unittest.main()
