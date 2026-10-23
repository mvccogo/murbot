import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bot.brain.llm import LLMConversationEngine
from bot.voice.conversation import ConversationFocus
from bot.config import config


class IntentTests(unittest.IsolatedAsyncioTestCase):
    async def test_classifier_requires_confident_valid_decision_and_never_uses_tools(self):
        engine = LLMConversationEngine.__new__(LLMConversationEngine)
        engine.history = [{"role": "user", "content": "Murbot, explica BKB"}]
        for decision, expected in (({"action": "reply", "confidence": .98}, "reply"),
                                   ({"action": "reply", "confidence": .7}, "ignore"),
                                   ({"action": "stop", "confidence": .99}, "stop"),
                                   ({"action": "reply", "confidence": "1"}, "ignore")):
            create = AsyncMock(return_value=NS(choices=[NS(message=NS(content=json.dumps(decision)))]))
            engine.client = NS(chat=NS(completions=NS(create=create)))
            self.assertEqual(await engine.classify_followup("Ana", 1, "Explica melhor?"), expected)
            self.assertNotIn("tools", create.call_args.kwargs)
            self.assertFalse(create.call_args.kwargs["stream"])
        engine.client.chat.completions.create = AsyncMock(side_effect=TimeoutError())
        self.assertEqual(await engine.classify_followup("Ana", 1, "Oi"), "ignore")

    async def test_background_and_partials_keep_reply_but_clear_followup_can_interrupt(self):
        from bot import main
        classifier = AsyncMock(side_effect=["ignore", "reply"])
        session = {"player": NS(is_session_active=True, playback_finished_at=0),
                   "conversation_focus": ConversationFocus(1, 100),
                   "engine": NS(classify_followup=classifier)}
        main.guild_sessions[799] = session
        with patch.object(config, "conversation_semantic_followups", True), patch.object(config, "conversation_followup_seconds", 20), patch.object(main, "cancel_reply", new_callable=AsyncMock) as cancel, patch.object(main, "generate_reply", new_callable=AsyncMock) as reply:
            try:
                await main.route_transcript(799, 1, "Ana", "Galera, olha isso", turn_id=1, is_final=False)
                classifier.assert_not_awaited()
                cancel.assert_not_awaited()
                await main.route_transcript(799, 1, "Ana", "Galera, olha isso", turn_id=1)
                cancel.assert_not_awaited()
                await main.route_transcript(799, 1, "Ana", "Você pode explicar melhor?", turn_id=2)
                await asyncio.sleep(0)
                cancel.assert_awaited_once()
                reply.assert_awaited_once()
            finally:
                main.guild_sessions.pop(799)

    async def test_slow_classification_cannot_delay_or_undo_explicit_stop(self):
        from bot import main
        started, finish = asyncio.Event(), asyncio.Event()
        async def slow(*args):
            started.set()
            await finish.wait()
            return "reply"
        session = {"player": NS(is_session_active=True, playback_finished_at=0),
                   "conversation_focus": ConversationFocus(1, 100),
                   "engine": NS(classify_followup=slow)}
        main.guild_sessions[798] = session
        with patch.object(config, "conversation_semantic_followups", True), patch.object(config, "conversation_followup_seconds", 20), patch.object(main, "cancel_reply", new_callable=AsyncMock) as cancel, patch.object(main, "generate_reply", new_callable=AsyncMock) as reply:
            pending = asyncio.create_task(main.route_transcript(798, 1, "Ana", "Explica melhor?", turn_id=1))
            try:
                await asyncio.wait_for(started.wait(), 1)
                await asyncio.wait_for(main.route_transcript(798, 1, "Ana", "Para, Murbot", turn_id=2), 1)
                cancel.assert_awaited_once()
                finish.set()
                await pending
                reply.assert_not_awaited()
            finally:
                finish.set()
                await pending
                main.guild_sessions.pop(798)


if __name__ == "__main__":
    unittest.main()
