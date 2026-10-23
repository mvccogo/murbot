import asyncio
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bot.brain.tools import execute_tool, TOOL_DEFINITIONS
from bot.brain.discord_tools import execute_discord_tool
from bot.brain.web_tools import public_url, PublicResolver, extract_page, browse_webpage
from bot.brain.llm import LLMConversationEngine
from bot.config import config
from bot.brain.utility_tools import calculate, convert_units, execute_utility_tool


def fixture():
    def channel(cid, name, kind="voice", hidden=False):
        return NS(id=cid, name=name, type=kind, permissions_for=lambda member: NS(view_channel=not hidden))
    room = channel(10, "Sala principal")
    def member(uid, name, bot=False):
        return NS(id=uid, name=name.lower(), display_name=name, bot=bot,
                  voice=NS(channel=room, self_mute=False, mute=False, self_deaf=False,
                           deaf=False, self_stream=False, self_video=False))
    alice, bob, murbot = member(1, "Alice"), member(2, "Bob"), member(3, "Murbot", True)
    room.members = [alice, bob, murbot]
    room.voice_states = {m.id: m.voice for m in room.members}
    guild = NS(id=100, name="Servidor A", me=murbot, channels=[room, channel(20, "geral", "text"), channel(30, "privado", "text", hidden=True)])
    guild.get_member = lambda uid: next((m for m in room.members if m.id == uid), None)
    vc = NS(is_connected=lambda: True, channel=room, guild=guild)
    sink = NS(speaking_snapshot=lambda: {2, 999})
    return {"voice_client": vc, "audio_sink": sink, "user_id": 1, "username": "Alice"}, room, guild


class DiscordTests(unittest.TestCase):
    def test_live_room_members_and_counts(self):
        context, room, _ = fixture()
        data = json.loads(execute_discord_tool("discord_list_room_members", {}, context))["data"]
        self.assertEqual([m["display_name"] for m in data["members"]], ["Alice", "Bob", "Murbot"])
        counts = json.loads(execute_discord_tool("discord_room_info", {}, context))["data"]
        self.assertEqual(counts["known_human_count"], 2)
        self.assertEqual(counts["known_bot_count"], 1)
        room.members.pop(1)
        room.voice_states.pop(2)
        later = json.loads(execute_discord_tool("discord_list_room_members", {}, context))["data"]
        self.assertEqual(later["total"], 2)

    def test_unknown_voice_identity_is_not_invented(self):
        context, room, _ = fixture()
        room.voice_states[4] = NS()
        data = json.loads(execute_discord_tool("discord_room_info", {}, context))["data"]
        self.assertEqual(data["participant_count"], 4)
        self.assertEqual(data["unknown_count"], 1)

    def test_member_details_and_recent_speakers(self):
        context, room, _ = fixture()
        room.members[1].voice.self_stream = True
        data = json.loads(execute_discord_tool("discord_get_room_member", {"member": "Bob"}, context))["data"]
        self.assertTrue(data["streaming"])
        speakers = json.loads(execute_discord_tool("discord_who_is_speaking", {}, context))["data"]["speakers"]
        self.assertEqual([m["id"] for m in speakers], ["2"])
        self.assertEqual(json.loads(execute_discord_tool("discord_get_room_member", {}, context))["data"]["id"], "1")

    def test_channels_respect_both_permissions_and_room_membership(self):
        context, room, guild = fixture()
        data = json.loads(execute_discord_tool("discord_list_channels", {"kind": "text"}, context))["data"]
        self.assertEqual([c["name"] for c in data["channels"]], ["geral"])
        guild.channels[1].permissions_for = lambda member: NS(view_channel=member.id != 1)
        self.assertEqual(json.loads(execute_discord_tool("discord_list_channels", {}, context))["data"]["channels"], [{"id": "10", "name": "Sala principal", "type": "voice"}])
        room.members[0].voice.channel = NS(id=999)
        with self.assertRaises(ValueError):
            execute_discord_tool("discord_room_info", {}, context)


class UtilityTests(unittest.TestCase):
    def test_arithmetic_and_units(self):
        self.assertEqual(calculate("(120 * 0.15) + 5"), 23)
        self.assertEqual(calculate("2 ** 3 + -1"), 7)
        self.assertAlmostEqual(convert_units(1, "mi", "km"), 1.609344)
        self.assertEqual(convert_units(32, "F", "C"), 0)
        self.assertAlmostEqual(convert_units(36, "km/h", "m/s"), 10)
        with self.assertRaises(ValueError):
            convert_units(1, "kg", "km")

    def test_calculator_rejects_code_and_extreme_exponents(self):
        for expression in ("__import__('os')", "True", "10 ** 100000000", "[1]", "open('file')"):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                calculate(expression)

    def test_current_time_uses_requested_offset(self):
        result = execute_utility_tool("get_current_time", {"utc_offset_minutes": -180})
        self.assertTrue(result["time"].endswith("-03:00"))
        with self.assertRaises(ValueError):
            execute_utility_tool("get_current_time", {"utc_offset_minutes": 2000})


class WebTests(unittest.IsolatedAsyncioTestCase):
    async def test_private_urls_credentials_and_non_web_schemes_are_rejected(self):
        for value in ("http://127.0.0.1", "http://[::1]", "http://10.0.0.1", "http://169.254.169.254", "http://localhost", "http://host.local", "file:///etc/passwd", "https://user:password@example.com", "http://example.com:8000"):
            with self.subTest(url=value), self.assertRaises(ValueError):
                public_url(value)
        self.assertEqual(str(public_url("https://example.com/page#fragment")), "https://example.com/page")

    async def test_connector_resolver_rejects_private_and_mixed_answers(self):
        records = [{"host": "93.184.216.34"}]
        delegate = NS(resolve=AsyncMock(return_value=records), close=AsyncMock())
        with patch("bot.brain.web_tools.aiohttp.resolver.ThreadedResolver", return_value=delegate):
            resolver = PublicResolver()
            self.assertIs(await resolver.resolve("example.com"), records)
            records.append({"host": "127.0.0.1"})
            with self.assertRaises(OSError):
                await resolver.resolve("example.com")
            await resolver.close()
            delegate.close.assert_awaited_once()

    async def test_page_extraction_keeps_sources_and_bounds_text(self):
        body = b'<html><title>Title</title><nav>menu</nav><main><script>bad script</script><p>Hello</p><a href="/next">Next</a><a href="http://127.0.0.1">Private</a>' + b'x' * 7000 + b'</main></html>'
        result = extract_page(body, "https://example.com")
        self.assertEqual(result["title"], "Title")
        self.assertNotIn("bad script", result["text"])
        self.assertNotIn("menu", result["text"])
        self.assertEqual(result["links"], [{"text": "Next", "url": "https://example.com/next"}])
        self.assertEqual(len(result["text"]), 6000)
        self.assertTrue(result["truncated"])
        self.assertTrue(result["untrusted_content"])

    async def test_redirect_to_private_target_is_not_fetched(self):
        response = NS(status=302, headers={"Location": "http://127.0.0.1/private"})
        class Response:
            async def __aenter__(self):
                return response
            async def __aexit__(self, *args):
                pass
        client = Mock()
        client.get.return_value = Response()
        class Client:
            async def __aenter__(self):
                return client
            async def __aexit__(self, *args):
                pass
        connector = NS(close=AsyncMock())
        resolver = NS(close=AsyncMock())
        with patch("bot.brain.web_tools.PublicResolver", return_value=resolver), patch("bot.brain.web_tools.aiohttp.TCPConnector", return_value=connector), patch("bot.brain.web_tools.aiohttp.ClientSession", return_value=Client()):
            with self.assertRaises(ValueError):
                await browse_webpage("https://example.com")
        client.get.assert_called_once()


class ToolPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_bad_arguments_cannot_override_session_or_fail_unhandled(self):
        context, _, _ = fixture()
        self.assertIn("desconhecidos", await execute_tool("discord_room_info", {"guild_id": 999}, context))
        self.assertIn("JSON", await execute_tool("web_search", [], context))
        self.assertIn("obrigatórios", await execute_tool("web_search", {}, context))
        with patch("bot.brain.tools.web_search", new_callable=AsyncMock, return_value={"results": []}) as search:
            result = await execute_tool("web_search", {"query": "Python documentation", "max_results": 2}, {})
            self.assertEqual(json.loads(result)["results"], [])
            search.assert_awaited_once_with("Python documentation", 2)
        names = [d["function"]["name"] for d in TOOL_DEFINITIONS]
        self.assertEqual(len(names), len(set(names)))

    async def test_fragmented_tool_call_uses_live_context_then_streams_answer(self):
        context, _, _ = fixture()
        class Stream:
            def __init__(self, deltas):
                self.deltas = iter(deltas)
                self.closed = False
            def __aiter__(self):
                return self
            async def __anext__(self):
                try:
                    return NS(choices=[NS(delta=next(self.deltas))])
                except StopIteration:
                    raise StopAsyncIteration
            async def close(self):
                self.closed = True
        first = Stream([
            NS(content=None, tool_calls=[NS(index=0, id="call_1", function=NS(name="discord_list_", arguments="{"))]),
            NS(content=None, tool_calls=[NS(index=0, id=None, function=NS(name="room_members", arguments="}"))]),
        ])
        second = Stream([NS(content="Alice e Bob estão na sala.", tool_calls=None)])
        create = AsyncMock(side_effect=[first, second])
        engine = LLMConversationEngine.__new__(LLMConversationEngine)
        engine.history = []
        engine.tool_context = {"voice_client": context["voice_client"], "audio_sink": context["audio_sink"]}
        engine._build_system_prompt = Mock(return_value="test")
        engine.client = NS(chat=NS(completions=NS(create=create)))
        with patch.object(config, "llm_tools_enabled", True):
            sentences = [s async for s in engine.stream_reply_sentences("Alice", 1, "Murbot, quem está na sala?", asyncio.Event())]
        self.assertEqual(sentences, ["Alice e Bob estão na sala."])
        messages = create.call_args_list[1].kwargs["messages"]
        tool_result = next(m for m in messages if m["role"] == "tool")
        self.assertEqual(tool_result["tool_call_id"], "call_1")
        self.assertEqual(json.loads(tool_result["content"])["data"]["total"], 3)
        self.assertTrue(first.closed and second.closed)

    async def test_tts_preserves_conversational_commas(self):
        from bot.providers.tts import TextToSpeechProvider
        provider = TextToSpeechProvider.__new__(TextToSpeechProvider)
        self.assertEqual(provider._prepare_text_prosody("Bom, vamos conversar."), "Bom, vamos conversar.")

    async def test_streamed_decimal_is_not_spoken_as_two_sentences(self):
        class Stream:
            def __init__(self):
                self.parts = iter(["O valor é 3.", "14 reais. ", "Certo?"])
            def __aiter__(self):
                return self
            async def __anext__(self):
                try:
                    return NS(choices=[NS(delta=NS(content=next(self.parts), tool_calls=None))])
                except StopIteration:
                    raise StopAsyncIteration
            async def close(self):
                pass
        engine = LLMConversationEngine.__new__(LLMConversationEngine)
        engine.history = []
        engine._build_system_prompt = Mock(return_value="test")
        engine.client = NS(chat=NS(completions=NS(create=AsyncMock(return_value=Stream()))))
        sentences = [s async for s in engine.stream_reply_sentences("Alice", 1, "Murbot, qual é o valor?", asyncio.Event())]
        self.assertEqual(sentences, ["O valor é 3.14 reais.", "Certo?"])


if __name__ == "__main__":
    unittest.main()
