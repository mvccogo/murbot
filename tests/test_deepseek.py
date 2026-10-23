import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bot.config import BotConfig, config
from bot.brain.llm import LLMConversationEngine


class DeepSeekTests(unittest.IsolatedAsyncioTestCase):
    def test_provider_routes_independently_of_speech_services(self):
        settings = BotConfig(mode="cloud", llm_provider="deepseek", deepseek_api_key="test-key",
                             deepseek_base_url="https://api.deepseek.com", deepseek_model="deepseek-flash")
        self.assertEqual(settings.active_llm_api_key, "test-key")
        self.assertEqual(settings.active_llm_base_url, "https://api.deepseek.com")
        self.assertEqual(settings.active_llm_model, "deepseek-flash")
        speech = (settings.active_stt_provider, settings.tts_provider)
        settings.mode = "local"
        self.assertEqual(settings.active_llm_model, settings.local_llm_model)
        self.assertEqual(settings.active_llm_api_key, "ollama-or-local")
        settings.mode = "cloud"
        self.assertEqual(settings.active_llm_model, "deepseek-flash")
        self.assertEqual((settings.active_stt_provider, settings.tts_provider), speech)

    async def test_non_thinking_option_is_sent_with_streamed_tools(self):
        class Stream:
            def __init__(self):
                self.sent = False
            def __aiter__(self):
                return self
            async def __anext__(self):
                if self.sent:
                    raise StopAsyncIteration
                self.sent = True
                return NS(choices=[NS(delta=NS(content="Oi, Ana!", tool_calls=None))])
            async def close(self):
                pass
        engine = LLMConversationEngine.__new__(LLMConversationEngine)
        engine.history = []
        engine._build_system_prompt = Mock(return_value="test")
        create = AsyncMock(return_value=Stream())
        engine.client = NS(chat=NS(completions=NS(create=create)))
        with patch.object(config, "mode", "cloud"), patch.object(config, "llm_provider", "deepseek"), patch.object(config, "llm_tools_enabled", True):
            reply = [s async for s in engine.stream_reply_sentences("Ana", 1, "Oi, Murbot", asyncio.Event())]
        self.assertEqual(reply, ["Oi, Ana!"])
        self.assertEqual(create.call_args.kwargs["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertTrue(create.call_args.kwargs["stream"])
        self.assertTrue(create.call_args.kwargs["tools"])


if __name__ == "__main__":
    unittest.main()
