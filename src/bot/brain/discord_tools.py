"""Live, read-only Discord tools scoped to the calling voice session."""
import json
import time


def tool(name, description, properties=None, required=None):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {},
                       "required": required or [], "additionalProperties": False},
    }}


DISCORD_TOOL_DEFINITIONS = [
    tool("discord_room_info", "Consulta em tempo real o servidor e canal de voz onde você está, a quantidade de participantes e quem fez o pedido."),
    tool("discord_list_room_members", "Lista quem está AGORA na sua sala de voz: nomes, IDs, bots, microfone, fone e transmissão. Use para 'quem está na sala?' e contagens de pessoas."),
    tool("discord_who_is_speaking", "Consulta quais participantes tiveram fala detectada recentemente pelo VAD na sua sala. É uma estimativa recente, não uma transcrição."),
    tool("discord_get_room_member", "Consulta uma pessoa da sala atual pelo nome ou ID; sem argumento consulta o autor do pedido. Não pesquisa pessoas de outros canais.",
         {"member": {"type": "string", "description": "Nome, apelido ou ID da pessoa na sala. Opcional: omita para consultar quem está falando com você."}}),
    tool("discord_list_channels", "Lista canais deste servidor que tanto o autor do pedido quanto o bot podem ver. Não acessa outros servidores nem canais ocultos.",
         {"kind": {"type": "string", "enum": ["all", "voice", "text"], "description": "Tipo de canal; padrão all."}}),
]


def _session(context):
    vc = context.get("voice_client")
    if vc is None or not vc.is_connected() or vc.channel is None:
        raise ValueError("Não há uma sessão de voz conectada para este pedido.")
    guild, channel = vc.guild, vc.channel
    requester = guild.get_member(context["user_id"])
    # Recheck at execution: a user who left/moved cannot query the former room.
    if requester is None or requester.voice is None or requester.voice.channel is None or requester.voice.channel.id != channel.id:
        raise ValueError("O autor do pedido não está mais na sala de voz do bot.")
    if guild.me is None:
        raise ValueError("Não foi possível verificar as permissões do bot.")
    if not channel.permissions_for(requester).view_channel or not channel.permissions_for(guild.me).view_channel:
        raise ValueError("O canal não está visível para o autor e o bot.")
    return vc, guild, channel, requester


def _member(member, voice):
    return {
        "id": str(member.id), "display_name": member.display_name,
        "username": member.name, "bot": member.bot,
        "self_muted": bool(getattr(voice, "self_mute", False)),
        "server_muted": bool(getattr(voice, "mute", False)),
        "self_deafened": bool(getattr(voice, "self_deaf", False)),
        "server_deafened": bool(getattr(voice, "deaf", False)),
        "streaming": bool(getattr(voice, "self_stream", False)),
        "camera_on": bool(getattr(voice, "self_video", False)),
    }


def execute_discord_tool(name, args, context):
    vc, guild, channel, requester = _session(context)
    # voice_states includes uncached participants without requiring Member intent.
    states = channel.voice_states
    members = {m.id: m for m in channel.members}
    participants = []
    for user_id, voice in states.items():
        member = members.get(user_id) or guild.get_member(user_id)
        participants.append(_member(member, voice) if member else {
            "id": str(user_id), "display_name": None, "bot": None,
            "note": "Participante presente, mas identidade não está no cache.",
        })
    if name == "discord_room_info":
        result = {
            "guild": {"id": str(guild.id), "name": guild.name},
            "channel": {"id": str(channel.id), "name": channel.name,
                        "bitrate": getattr(channel, "bitrate", None),
                        "user_limit": getattr(channel, "user_limit", None)},
            "requester": {"id": str(requester.id), "display_name": requester.display_name},
            "participant_count": len(participants),
            "known_human_count": sum(p["bot"] is False for p in participants),
            "known_bot_count": sum(p["bot"] is True for p in participants),
            "unknown_count": sum(p["bot"] is None for p in participants),
        }
    elif name == "discord_list_room_members":
        result = {"channel": channel.name, "total": len(participants),
                  "members": participants[:100], "truncated": len(participants) > 100}
    elif name == "discord_get_room_member":
        query = str(args.get("member", requester.id)).strip().casefold()
        matches = [p for p in participants if query == p["id"] or
                   query == (p.get("display_name") or "").casefold() or
                   query == p.get("username", "").casefold()]
        if len(matches) == 1:
            result = matches[0]
        elif matches:
            result = {"error": "Há mais de uma pessoa com esse nome. Peça o ID.", "matches": matches}
        else:
            result = {"error": "Pessoa não encontrada na sala atual."}
    elif name == "discord_who_is_speaking":
        sink = context.get("audio_sink")
        detected = sink.speaking_snapshot() if sink is not None else None
        result = {"speakers": [p for p in participants if detected is not None and int(p["id"]) in detected],
                  "available": detected is not None,
                  "note": "Atividade de voz recente do VAD; não indica necessariamente que a pessoa ainda está falando ao ouvir a resposta."}
    elif name == "discord_list_channels":
        kind = args.get("kind", "all")
        if kind not in ("all", "voice", "text"):
            raise ValueError("kind deve ser all, voice ou text.")
        allowed_types = {"voice", "stage_voice"} if kind == "voice" else {"text", "news"} if kind == "text" else {"voice", "stage_voice", "text", "news", "forum"}
        visible = []
        for item in guild.channels:
            channel_type = str(item.type)
            if channel_type not in allowed_types:
                continue
            if item.permissions_for(requester).view_channel and item.permissions_for(guild.me).view_channel:
                visible.append({"id": str(item.id), "name": item.name, "type": channel_type})
        result = {"guild": guild.name, "channels": visible[:100], "truncated": len(visible) > 100}
    else:
        raise ValueError("Ferramenta Discord desconhecida.")
    return json.dumps({"observed_at_unix": time.time(), "data": result}, ensure_ascii=False)
