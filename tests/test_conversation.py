import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bot.config import config
from bot.voice.conversation import ConversationFocus
from bot.voice.player import VoicePlaybackController


class FocusTests(unittest.TestCase):
    def test_focus_expires_and_is_speaker_scoped(self):
        focus = ConversationFocus()
        self.assertFalse(focus.permits(1, 100, 20))
        focus.accept(1, 100)
        self.assertTrue(focus.permits(1, 119, 20))
        self.assertFalse(focus.permits(2, 119, 20))
        self.assertFalse(focus.permits(1, 121, 20))
        self.assertFalse(focus.permits(1, 110, 0, playback_active=True))
        focus.clear()
        self.assertFalse(focus.permits(1, 110, 20))

    def test_window_starts_after_playback_and_includes_reply_generation(self):
        focus = ConversationFocus()
        focus.accept(1, 100)
        self.assertTrue(focus.permits(1, 140, 20, playback_active=True))
        self.assertTrue(focus.permits(1, 159, 20, playback_finished_at=140))
        self.assertFalse(focus.permits(1, 161, 20, playback_finished_at=140))


class ConversationTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_mode_background_never_interrupts_and_named_controls_work(self):
        from bot import main
        player = NS(is_session_active=True, playback_finished_at=0, stop_playback=Mock())
        session = {"player": player}
        main.guild_sessions[783] = session
        async def generating(*args):
            await asyncio.Event().wait()
        async def route(user, text, turn, final=True):
            await main.route_transcript(783, user, str(user), text, turn_id=turn, is_final=final)
            await asyncio.sleep(0)
        with patch.object(config, "conversation_followup_seconds", 0), patch.object(main, "generate_reply", side_effect=generating) as reply:
            try:
                await route(1, "Murbot, explica a build", 1)
                task = session["current_turn_task"]
                stops = player.stop_playback.call_count
                for turn, user, text in ((2, 1, "Galera, olha esse jogo"), (3, 2, "Que legal"), (4, 1, "para"), (5, 1, "continua")):
                    await route(user, text, turn, final=False)
                    await route(user, text, turn)
                    self.assertIs(session["current_turn_task"], task)
                    self.assertFalse(task.done())
                self.assertEqual(player.stop_playback.call_count, stops)
                self.assertEqual(reply.await_count, 1)
                await route(1, "para, murbot", 6, final=False)
                self.assertTrue(task.cancelled())
                await route(1, "para, murbot", 6)
                self.assertIsNone(session["current_turn_task"])
                self.assertEqual(reply.await_count, 1)
                await route(1, "continua, murbot", 7)
                self.assertEqual(reply.await_count, 2)
                self.assertEqual(reply.call_args.args[-1], "continua, murbot")
            finally:
                await main.cancel_reply(session)
                main.guild_sessions.pop(783)

    async def test_natural_followups_other_speakers_timeout_and_quiet(self):
        from bot import main
        session = {"player": NS(is_session_active=False, playback_finished_at=0)}
        main.guild_sessions[781] = session
        clock = [100.0]
        async def route(user, text, turn):
            await main.route_transcript(781, user, str(user), text, turn_id=turn)
            await asyncio.sleep(0)
        with patch.object(config, "conversation_semantic_followups", False), patch.object(main, "cancel_reply", new_callable=AsyncMock), patch.object(main, "generate_reply", new_callable=AsyncMock) as reply, patch.object(config, "conversation_followup_seconds", 20), patch("bot.main.monotonic", side_effect=lambda: clock[0]):
            try:
                await route(1, "Como vai?", 1)
                reply.assert_not_called()
                await route(1, "Murbot, como vai?", 2)
                self.assertEqual(reply.await_count, 1)
                clock[0] = 105
                await route(1, "E o que mais você sabe fazer?", 3)
                self.assertEqual(reply.await_count, 2)
                await route(2, "Pessoal, olhem isso", 4)
                self.assertEqual(reply.await_count, 2)
                clock[0] = 126
                await route(1, "Outra pergunta", 5)
                self.assertEqual(reply.await_count, 2)
                await route(2, "Murbot, olá", 6)
                self.assertEqual(reply.await_count, 3)
                await route(1, "Outra pergunta", 7)
                self.assertEqual(reply.await_count, 3)
                await route(2, "Fica quieto", 8)
                self.assertEqual(reply.await_count, 3)
                await route(2, "Pode falar", 9)
                self.assertEqual(reply.await_count, 3)
            finally:
                main.guild_sessions.pop(781)

    async def test_followup_capture_time_not_delayed_decode_time(self):
        from bot import main
        focus = ConversationFocus(1, 100)
        session = {"player": NS(is_session_active=False, playback_finished_at=0), "conversation_focus": focus}
        main.guild_sessions[782] = session
        with patch.object(config, "conversation_semantic_followups", False), patch.object(main, "cancel_reply", new_callable=AsyncMock), patch.object(main, "generate_reply", new_callable=AsyncMock) as reply, patch.object(config, "conversation_followup_seconds", 20), patch("bot.main.monotonic", return_value=130):
            try:
                await main.route_transcript(782, 1, "Alice", "E por quê?", turn_id=2, started_at=119)
                await asyncio.sleep(0)
                reply.assert_awaited_once()
            finally:
                main.guild_sessions.pop(782)

    async def test_old_player_completion_cannot_extend_new_reply_focus(self):
        callbacks = []
        vc = Mock()
        vc.is_playing.return_value = False
        vc.play.side_effect = lambda source, *, after: callbacks.append(after)
        player = VoicePlaybackController(vc)
        source, _ = player.start_new_session()
        source.write_pcm(b"x" * 3840)
        player.ensure_playing()
        current, _ = player.start_new_session()
        current.write_pcm(b"y" * 3840)
        player.ensure_playing()
        callbacks[0](None)
        await asyncio.sleep(0)
        self.assertEqual(player.playback_finished_at, 0)
        callbacks[1](None)
        await asyncio.sleep(0)
        self.assertGreater(player.playback_finished_at, 0)
