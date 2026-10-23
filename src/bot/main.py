import logging
import asyncio
from time import monotonic
from collections import deque
import discord
from discord.ext import commands, voice_recv

from bot.config import config
from bot.logging_setup import configure_logging
from bot.voice.player import VoicePlaybackController
from bot.voice.audio_sink import MultiUserAudioSink
from bot.providers.stt import SpeechToTextProvider
from bot.providers.streaming_stt import StreamingSpeechRoom
from bot.providers.tts import TextToSpeechProvider
from bot.brain.llm import LLMConversationEngine
from bot.voice.addressing import classify_address
from bot.voice.conversation import ConversationFocus

logger = logging.getLogger("VoiceBot.Main")

# Bot Setup
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.guilds = True

bot = commands.Bot(command_prefix=config.command_prefix, intents=intents)

# Core AI Services
stt_provider: SpeechToTextProvider | None = None
tts_provider: TextToSpeechProvider | None = None

# Active Guild Sessions: guild_id -> dict
guild_sessions: dict[int, dict] = {}

def get_or_init_services():
    global stt_provider, tts_provider
    if stt_provider is None:
        stt_provider = SpeechToTextProvider()
    if tts_provider is None:
        tts_provider = TextToSpeechProvider()

@bot.event
async def on_ready():
    logger.info("==================================================")
    logger.info(f"Bot Logged in as: {bot.user} (ID: {bot.user.id})")
    logger.info(f"Active Mode: {config.mode.upper()}")
    logger.info(f"LLM Provider: {config.llm_provider} ({config.active_llm_model})")
    logger.info(f"STT Provider: {config.active_stt_provider}")
    active_voice = config.elevenlabs_voice_id if config.tts_provider == "elevenlabs" else config.kokoro_voice if config.tts_provider == "kokoro" else config.tts_voice
    logger.info(f"TTS Provider: {config.tts_provider} ({active_voice})")
    logger.info(f"Barge-In Interruption: {config.interruption_enabled}")
    logger.info("==================================================")
    get_or_init_services()

async def handle_speech_turn(guild_id: int, user_id: int, username: str, audio_16k_mono):
    session = guild_sessions.get(guild_id)
    if not session:
        return
    # Serialize transcription in capture order within a room. Never cancel
    # another speaker just because someone has finished an unrelated sentence.
    async with session["stt_lock"]:
        if guild_sessions.get(guild_id) is not session:
            return
        try:
            transcript = await stt_provider.transcribe(audio_16k_mono)
        except Exception:
            logger.exception("Transcription failed for %s", user_id)
            return
        if guild_sessions.get(guild_id) is not session:
            return
        await route_transcript(guild_id, user_id, username, transcript)


async def route_transcript(guild_id, user_id, username, transcript, *, turn_id=None, is_final=True, started_at=0.0):
    session = guild_sessions.get(guild_id)
    if not session:
        return
    # Semantic checks run outside the routing lock: a slow API must never delay
    # another speaker's explicit "para, Murbot". Unnamed partials never cancel.
    semantic_action = "ignore"
    semantic_snapshot = None
    focus = session.get("conversation_focus")
    if (config.conversation_semantic_followups and is_final and transcript.strip()
            and classify_address(transcript, config.bot_name, config.bot_aliases) == "ignore"
            and turn_id not in session.get("addressed_turns", ()) and focus):
        player = session["player"]
        finished_at = getattr(player, "playback_finished_at", 0.0)
        if focus.permits(user_id, started_at or monotonic(), config.conversation_followup_seconds,
                         playback_active=getattr(player, "is_session_active", False) is True,
                         playback_finished_at=finished_at if isinstance(finished_at, (int, float)) else 0.0):
            semantic_snapshot = (focus.user_id, focus.last_addressed_at, session.get("latest_addressed_turn"))
            engine = session.get("engine")
            if engine and callable(getattr(engine, "classify_followup", None)):
                semantic_action = await engine.classify_followup(username, user_id, transcript)
    async with session.setdefault("routing_lock", asyncio.Lock()):
        if guild_sessions.get(guild_id) is not session:
            return
        quiet_turns = session.setdefault("quiet_turns", deque(maxlen=256))
        interrupted = session.setdefault("interrupted_turns", deque(maxlen=256))
        addressed = session.setdefault("addressed_turns", deque(maxlen=256))
        focus = session.setdefault("conversation_focus", ConversationFocus())
        now = monotonic()
        action = classify_address(transcript, config.bot_name, config.bot_aliases)
        reason = "name_detected" if action != "ignore" else "name_not_detected" if transcript.strip() else "empty_transcript"
        if turn_id is not None and turn_id < session.get("latest_addressed_turn", -1):
            logger.info("Voice turn user=%s turn=%s routing=ignore reason=stale_result final=%s", user_id, turn_id, is_final)
            return  # A delayed decoder result must not cancel a newer request.
        if turn_id is not None and action != "ignore" and turn_id not in addressed:
            addressed.append(turn_id)
        if is_final and transcript.strip() and action == "ignore" and turn_id in addressed:
            # Full-utterance decoding can revise/drop an unfamiliar wake name
            # that was already recognized in a partial of this SAME utterance.
            action = classify_address(f"{config.bot_name}, {transcript}", config.bot_name, config.bot_aliases)
            reason = "name_detected_in_partial"
        player = session["player"]
        finished_at = getattr(player, "playback_finished_at", 0.0)
        if (action == "ignore" and semantic_snapshot is not None
                and semantic_snapshot == (focus.user_id, focus.last_addressed_at, session.get("latest_addressed_turn"))):
            action = semantic_action
            reason = "semantic_followup" if action != "ignore" else "background_or_uncertain"
        if not config.conversation_semantic_followups and action == "ignore" and transcript.strip() and focus.permits(
            user_id, started_at or now, config.conversation_followup_seconds,
            playback_active=getattr(player, "is_session_active", False) is True,
            playback_finished_at=finished_at if isinstance(finished_at, (int, float)) else 0.0,
        ):
            action = classify_address(f"{config.bot_name}, {transcript}", config.bot_name, config.bot_aliases)
            reason = "conversation_followup"
            if turn_id is not None and turn_id not in addressed:
                addressed.append(turn_id)
        if is_final and turn_id in quiet_turns:
            action = "stop"
            reason = "quiet_detected_in_partial"
        logger.info("Voice turn user=%s turn=%s routing=%s reason=%s final=%s", user_id, turn_id, action, reason, is_final)
        if config.stt_log_transcripts:
            logger.info("STT user=%s turn=%s final=%s text=%r", user_id, turn_id, is_final, transcript[:500])
        if action == "ignore":
            return
        if action == "stop":
            focus.clear()
        else:
            focus.accept(user_id, now)
        if not is_final:
            if action == "stop" and turn_id not in quiet_turns:
                quiet_turns.append(turn_id)
            if action != "stop" and not config.interruption_enabled:
                return
            if turn_id not in interrupted:
                session["latest_addressed_turn"] = turn_id
                interrupted.append(turn_id)
                await cancel_reply(session)
            return  # Never send an incomplete question to the LLM.
        if turn_id is not None:
            session["latest_addressed_turn"] = turn_id
        if action == "stop" or config.interruption_enabled:
            await cancel_reply(session)
        elif session.get("current_turn_task") and not session["current_turn_task"].done():
            return
        if action == "stop":
            return  # Quiet means quiet: no acknowledgment through TTS.
        session["current_turn_task"] = asyncio.create_task(
            generate_reply(session, user_id, username, transcript)
        )


async def cancel_reply(session):
    session["player"].stop_playback(is_barge_in=False)
    task = session.get("current_turn_task")
    if task and not task.done():
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    session["current_turn_task"] = None


async def generate_reply(session, user_id, username, transcript):
    player = session["player"]
    source, cancel_token = player.start_new_session()
    replies = session["engine"].stream_reply_sentences(
        username, user_id, transcript, cancel_token
    )
    try:
        async for sentence in replies:
            if cancel_token.is_set():
                break
            chunks = tts_provider.stream_pcm(sentence)
            try:
                async for pcm in chunks:
                    if cancel_token.is_set():
                        break
                    while source.buffered_seconds > 2.0 and not cancel_token.is_set():
                        await asyncio.sleep(0.02)
                    if cancel_token.is_set():
                        break
                    if pcm:
                        source.write_pcm(pcm)
                        player.ensure_playing()
            finally:
                await chunks.aclose()
    except asyncio.CancelledError:
        source.interrupt()
        raise
    except Exception:
        logger.exception("Reply generation failed")
    finally:
        await replies.aclose()
        source.mark_finished()


@bot.command(name="join", aliases=["entrar"])
async def join_command(ctx: commands.Context):
    """Faz o bot entrar no canal de voz do usuário e iniciar a escuta em tempo real."""
    if not ctx.author.voice or not ctx.author.voice.channel:
        return await ctx.send("❌ Você precisa estar conectado a um canal de voz!")

    channel = ctx.author.voice.channel
    guild_id = ctx.guild.id

    # If already connected in this guild, disconnect first
    old_session = guild_sessions.pop(guild_id, None)
    if old_session:
        old_session["sink"].cleanup()
        if old_session.get("streaming_stt"):
            await old_session["streaming_stt"].aclose()
        await cancel_reply(old_session)
        await old_session["engine"].client.close()
    if ctx.voice_client:
        await ctx.voice_client.disconnect()

    try:
        # Connect using voice_recv.VoiceRecvClient to receive audio streams
        vc: voice_recv.VoiceRecvClient = await channel.connect(cls=voice_recv.VoiceRecvClient)
        player = VoicePlaybackController(vc)

        async def callback(user_id, username, audio_data):
            await handle_speech_turn(guild_id, user_id, username, audio_data)

        async def on_transcript(event):
            if guild_sessions.get(guild_id) is session:
                await route_transcript(
                    guild_id, event.user_id, event.username, event.text,
                    turn_id=event.turn_id, is_final=event.is_final,
                    started_at=event.started_at,
                )

        streaming = None
        if config.stt_streaming_enabled or config.stt_provider == "deepgram":
            streaming = StreamingSpeechRoom(stt_provider, on_transcript)

        sink = MultiUserAudioSink(
            bot_user=bot.user,
            player_controller=player,
            on_speech_turn_callback=callback,
            on_audio_stream_frame=streaming.feed if streaming else None,
            on_stream_cleanup=streaming.close if streaming else None,
        )

        session = {
            "vc": vc,
            "player": player,
            "sink": sink,
            "engine": LLMConversationEngine(tool_context={"voice_client": vc, "audio_sink": sink}),
            "stt_lock": asyncio.Lock(),
            "streaming_stt": streaming,
        }
        guild_sessions[guild_id] = session

        logger.info(f"Iniciando escuta no canal de voz '{channel.name}' via vc.listen(sink)...")
        vc.listen(sink)
        logger.info(f"vc.listen(sink) ativado com sucesso! Bot está ouvindo o canal '{channel.name}'.")

        await ctx.send(f"🎙️ **Conectado ao canal {channel.name}!** Diga {config.bot_name} para começar. A mesma pessoa pode continuar sem repetir o nome por {config.conversation_followup_seconds:g}s após a resposta. Para interromper: '{config.bot_name}, fica quieto'.")

    except Exception as e:
        failed_session = guild_sessions.pop(guild_id, None)
        if failed_session:
            failed_session["sink"].cleanup()
            if failed_session.get("streaming_stt"):
                await failed_session["streaming_stt"].aclose()
            await cancel_reply(failed_session)
            await failed_session["engine"].client.close()
        if ctx.voice_client:
            await ctx.voice_client.disconnect()
        logger.error(f"Failed to join voice channel: {e}")
        await ctx.send(f"❌ Erro ao conectar ao canal de voz: {e}")

@bot.command(name="leave", aliases=["sair"])
async def leave_command(ctx: commands.Context):
    """Desconecta o bot do canal de voz."""
    guild_id = ctx.guild.id
    session = guild_sessions.pop(guild_id, None)

    if session:
        session["sink"].cleanup()
        if session.get("streaming_stt"):
            await session["streaming_stt"].aclose()
        await cancel_reply(session)
        await session["engine"].client.close()

    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send("👋 Desconectado do canal de voz.")
    else:
        await ctx.send("Não estou em nenhum canal de voz.")


@bot.event
async def on_voice_state_update(member, before, after):
    if bot.user and member.id == bot.user.id and after.channel is None:
        session = guild_sessions.pop(member.guild.id, None)
        if session:
            session["sink"].cleanup()
            if session.get("streaming_stt"):
                await session["streaming_stt"].aclose()
            await cancel_reply(session)
            await session["engine"].client.close()

@bot.command(name="mode", aliases=["modo"])
async def mode_command(ctx: commands.Context, new_mode: str = ""):
    """Alterna entre modo 'cloud' (APIs) e 'local' (Docker/Ollama offline)."""
    new_mode = new_mode.lower().strip()
    if new_mode not in ("cloud", "local"):
        return await ctx.send(
            f"ℹ️ Modo atual: **{config.mode.upper()}**.\n"
            f"Para alterar, use `{config.command_prefix}mode cloud` ou `{config.command_prefix}mode local`."
        )

    config.mode = new_mode
    # MODE selects local execution without losing the configured cloud provider.
    if new_mode == "cloud" and config.llm_provider in ("local", "ollama"):
        config.llm_provider = "cloud"

    # Re-init clients with new settings
    for session in guild_sessions.values():
        await cancel_reply(session)
        await session["engine"].client.close()
        session["engine"] = LLMConversationEngine(tool_context={"voice_client": session["vc"], "audio_sink": session["sink"]})

    await ctx.send(
        f"✅ Modo alterado com sucesso para **{new_mode.upper()}**!\n"
        f"• LLM: `{config.active_llm_model}` via `{config.active_llm_base_url}`\n"
        f"• STT: `{config.active_stt_provider}`"
    )

@bot.command(name="tts")
async def tts_command(ctx: commands.Context, provider_name: str = ""):
    """Alterna o provedor de TTS entre 'kokoro', 'edge' e 'openai'."""
    p = provider_name.lower().strip()
    if p not in ("kokoro", "edge", "openai", "elevenlabs"):
        return await ctx.send(
            f"ℹ️ Provedor TTS atual: **{config.tts_provider.upper()}**.\n"
            f"Para alterar, use `{config.command_prefix}tts kokoro` (SOTA Local), "
            f"`{config.command_prefix}tts edge`, `{config.command_prefix}tts openai` ou `{config.command_prefix}tts elevenlabs`."
        )

    config.tts_provider = p
    tts_provider.provider = p
    await ctx.send(f"🗣️ Provedor TTS alterado para **{p.upper()}**!")

@bot.command(name="voice", aliases=["voz"])
async def voice_command(ctx: commands.Context, voice_name: str = ""):
    """Altera a voz do Text-to-Speech (Kokoro: dora, alex, santa | Edge: francisca, antonio)."""
    voice_key = voice_name.lower().strip()
    
    # Kokoro pt-BR voices
    kokoro_map = {
        "dora": "pf_dora",
        "pf_dora": "pf_dora",
        "alex": "pm_alex",
        "pm_alex": "pm_alex",
        "santa": "pm_santa",
        "pm_santa": "pm_santa"
    }

    # Edge-TTS pt-BR voices
    edge_map = {
        "francisca": "pt-BR-FranciscaNeural",
        "antonio": "pt-BR-AntonioNeural",
        "thalita": "pt-BR-ThalitaNeural"
    }

    if not voice_key:
        return await ctx.send(
            f"ℹ️ **Configuração de Voz Atual**:\n"
            f"• Provedor: `{config.tts_provider}`\n"
            f"• Voz Kokoro: `{config.kokoro_voice}` (Opções: `dora`, `alex`, `santa`)\n"
            f"• Voz Edge: `{config.tts_voice}` (Opções: `francisca`, `antonio`, `thalita`)"
            f"\n• Voz ElevenLabs: `{config.elevenlabs_voice_id}` (use o ID da voz com `{config.command_prefix}voice ID`)"
        )

    if config.tts_provider == "elevenlabs":
        import re
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", voice_name.strip()):
            return await ctx.send("Use um ID de voz ElevenLabs válido.")
        config.elevenlabs_voice_id = voice_name.strip()
        return await ctx.send(f"🗣️ Voz ElevenLabs alterada para `{config.elevenlabs_voice_id}`.")

    if voice_key in kokoro_map:
        config.kokoro_voice = kokoro_map[voice_key]
        tts_provider.kokoro_voice = config.kokoro_voice
        await ctx.send(f"🗣️ Voz Kokoro alterada para **{config.kokoro_voice}**!")
    elif voice_key in edge_map:
        config.tts_voice = edge_map[voice_key]
        tts_provider.edge_voice = config.tts_voice
        await ctx.send(f"🗣️ Voz Edge alterada para **{config.tts_voice}**!")
    else:
        # Allow passing arbitrary voice name or mixture (e.g. pf_dora(2)+pm_alex(1))
        config.kokoro_voice = voice_key
        tts_provider.kokoro_voice = voice_key
        await ctx.send(f"🗣️ Voz personalizada definida para **{voice_key}**!")

@bot.command(name="status")
async def status_command(ctx: commands.Context):
    """Exibe o status completo e configuração do bot."""
    active_voice = config.elevenlabs_voice_id if config.tts_provider == "elevenlabs" else config.kokoro_voice if config.tts_provider == "kokoro" else config.tts_voice
    msg = (
        f"🤖 **Status da {config.bot_name}**\n"
        f"• **Modo Operacional**: `{config.mode.upper()}`\n"
        f"• **LLM Model**: `{config.active_llm_model}` ({config.active_llm_base_url})\n"
        f"• **STT Engine**: `{config.active_stt_provider}` (device: {config.local_whisper_device})\n"
        f"• **Streaming STT**: `{'WebSocket' if config.stt_provider == 'deepgram' else 'Incremental local' if config.stt_streaming_enabled and config.stt_provider == 'local' else 'Final only'}`\n"
        f"• **TTS Engine**: `{config.tts_provider.upper()}` (voz ativa: `{active_voice}`)\n"
        f"• **Barge-in / Interrupção**: `{'Ativo' if config.interruption_enabled else 'Desativado'}`\n"
        f"• **Sensibilidade VAD**: `{config.vad_threshold}`\n"
        f"• **Timeout Silêncio**: `{config.silence_duration_ms}ms`"
        f"\n• **Janela de conversa**: `{config.conversation_followup_seconds:g}s` (mesmo locutor)"
    )
    await ctx.send(msg)

@bot.command(name="clear", aliases=["limpar"])
async def clear_command(ctx: commands.Context):
    """Limpa o histórico da conversa atual."""
    session = guild_sessions.get(ctx.guild.id)
    if session:
        await cancel_reply(session)
        session["engine"].history.clear()
        session["conversation_focus"] = ConversationFocus()
    await ctx.send("🧹 Histórico da conversa limpo!")

def main():
    runtime_logging = configure_logging(config)
    if not config.discord_token or config.discord_token == "your_discord_bot_token_here":
        logger.error("DISCORD_TOKEN não configurado. Preencha o token no arquivo .env.")
        runtime_logging.close()
        return
    try:
        bot.run(config.discord_token, log_handler=None)
    except Exception:
        logger.exception("Bot stopped with an unhandled error")
        raise
    finally:
        runtime_logging.close()

if __name__ == "__main__":
    main()

