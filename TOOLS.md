# Murbot tools

Restart the bot and use `!join` after updating. `LLM_TOOLS_ENABLED=true` is enabled in the active environment. The configured model must support OpenAI-style streamed function calls.

| Tool | Example spoken request | Data used |
| --- | --- | --- |
| `discord_list_room_members` | “Murbot, quem está na sala?” | Current room members, names, IDs, bots, mute/deafen, screen sharing, camera. |
| `discord_room_info` | “Em qual canal estamos? Quantas pessoas tem aqui?” | Current server/channel, participant counts, requester's identity. Human/bot/unknown counts are distinct. |
| `discord_get_room_member` | “O Bob está com o microfone mutado?” | A person currently in this room; exact name, nickname, or ID. Omit the argument for the requester. |
| `discord_who_is_speaking` | “Quem está falando agora?” | Recent VAD activity among current participants. This is an estimate and can change before the spoken reply is delivered. |
| `discord_list_channels` | “Quais canais de voz existem neste servidor?” | Voice/text/forum channels visible to both the requester and bot, in this server only. |
| `web_search` | “Pesquisa as notícias sobre astronomia.” | Up to five search results, snippets, and source URLs. |
| `browse_webpage` | “Abra essa página e resuma o conteúdo.” | Public HTML/text content, page title, URL, and links for further browsing. |

Discord tools use the live voice client from the room's session; the model cannot supply a different server or channel ID. Each call rechecks that the requester is still in the room and can see it. Membership is read at execution time rather than retained in conversation history. Uncached voice identities are marked unknown. These tools read Discord state; they do not send messages, moderate members, or move people.

Search and webpage requests now trigger retrieval before the model answers, independently of its automatic tool choice. Explicit URLs (including spoken text containing domains such as x.com), search commands, and common current-information requests are detected in Portuguese and English. If the model starts claiming it cannot access information on another request, that sentence is withheld while a real search is attempted and the answer is regenerated. Ordinary conversation still streams immediately. Explicit requests not to search and `LLM_TOOLS_ENABLED=false` disable these automatic lookups.

A failed, empty, or recognized challenge page triggers a search for its public host/path. Search results are supplied to the model and up to two alternate public results are opened, stopping at the first usable page. Each retrieval sequence makes at most four calls; each call has its own timeout. Already attempted pages are skipped and page results/errors are cached for that request to avoid repeatedly fetching blocked sites. HTTP failures such as 403 return structured status information and a warning; they do not mean the whole bot failed. This fallback also applies to web tools selected by the model. The bot must distinguish a page it actually read from indexed snippets and report specific failures when retrieval does not succeed.

The web reader handles public HTTP/HTTPS pages and redirects. It validates the actual DNS answers used by its connector and rejects local/private destinations and URL credentials. Rejected URLs are not forwarded to search, and page fallback queries omit URL query strings. It limits response size, redirects, output text, and execution time. It does not run JavaScript, authenticate to websites, or parse PDFs. Search results and page text are marked untrusted, and the system prompt instructs the model to use them as evidence rather than instructions. The model is asked to name sources naturally during speech, without reading long URLs.

Existing memory and skill tools remain available. Tool arguments are validated, unknown arguments are rejected, calls time out after 15 seconds, and results are bounded. Voice interruption still cancels an in-flight tool await. Synchronous search work may finish in its background thread after cancellation; its network requests have their own timeout.

Validation:

```powershell
uv run python -m unittest discover -s tests -v
uv run python scripts/diagnose_tools.py --web
uv run python scripts/diagnose_tools.py --retrieval
```

The diagnostic uses the real local LLM with a simulated three-member room, then optionally performs a public search and Python.org fetch. It does not connect to Discord. Both checks passed with the configured local qwen2.5:3b model. Live Discord queries still require a restarted bot in a room. A small model can choose the wrong tool or summarize imperfectly even when execution succeeds; inspect logged tool names when debugging.

## Additional callable tools

The full registry now has 14 callable tools. In addition to the seven web/Discord tools above:

| Tool | What it does now | Example |
| --- | --- | --- |
| `calculate` | Bounded arithmetic without evaluating code. | “Quanto é 15% de 120 mais cinco?” |
| `convert_units` | Length, mass, speed, Celsius/Fahrenheit/Kelvin. | “Converte cinco milhas para quilômetros.” |
| `get_current_time` | Current date/time at a specified UTC offset; defaults to UTC-3. | “Que horas são?” |
| `save_memory` | Store a requested fact for the calling user. | “Lembre que eu prefiro respostas curtas.” |
| `recall_memory` | Look up that user's stored facts. | “Qual é minha preferência de resposta?” |
| `create_skill` | Save a requested Markdown instruction skill. | “Crie uma habilidade para narrar histórias curtas.” |
| `list_skills` | List existing instruction skills. | “Quais habilidades você tem?” |

Current web search uses DDGS search backends; it is web search, not a direct Google Search API integration. The LLM can search on request, read relevant public results, and summarize sources. Actual Google-backed results would need a backend such as Serper or an appropriate Google grounding/search integration and its credentials. An LLM provider's built-in search is a separate capability from Murbot's own callable tools.

## Suggested additions — not implemented yet

| Suggested tool | Useful behavior | Setup / scope |
| --- | --- | --- |
| `google_search` | Google-backed search results on request. | Search provider such as Serper; separate key and charges. |
| `search_news` | Recent articles with publication dates. | News/search backend; dates and source links. |
| `compare_sources` | Compare claims across several pages. | Compose existing search/reader tools and preserve citations. |
| `get_weather` | Local forecast, rain, temperature. | Weather/geocoding API; location and timezone. |
| `get_currency_rates` | Convert BRL/USD/EUR using current rates. | Exchange-rate API and observation timestamp. |
| `wikipedia_lookup` | Quick article summaries and links. | Public encyclopedia API. |
| `read_pdf` | Read a linked document and extract relevant passages. | Sandboxed PDF parser, public URL validation, size/page limits. |
| `extract_web_table` | Read a page's table for comparisons. | Structured HTML extraction; explicit source URL. |
| `convert_timezone` | Convert named cities/timezones with DST rules. | IANA timezone database; current clock tool only applies fixed offsets. |
| `set_timer` / `cancel_timer` | Track a cooking/game timer. | Persisted scheduler and speaker identity. |
| `create_reminder` / `list_reminders` | Remind a user later, surviving restarts. | Scheduler, timezone, and explicit delivery destination. |
| `save_note` / `find_notes` | Store and retrieve room/project notes. | Guild-scoped storage and access rules. |
| `create_todo` / `list_todos` | A group's short task list. | Guild/user ownership, completion and deletion actions. |
| `forget_memory` | Remove an outdated user preference. | Current user scope and exact-record selection. |
| `discord_read_recent_messages` | Read/summarize an explicitly requested visible text channel. | Requester and bot View Channel + Read Message History checks. |
| `discord_search_accessible_messages` | Search messages indexed by the bot. | Permission-aware local index; do not assume the normal user search API is available to bots. |
| `discord_post_summary` | Post a requested summary or sources to a chosen channel. | Explicit request, destination, Send Messages permission, mention suppression. |
| `discord_create_poll` | Run a room poll. | Explicit question/options and permitted text channel. |
| `discord_list_events` / `discord_create_event` | Inspect or schedule server events. | Timezone, event details, and Discord permissions. |
| `discord_get_roles` | Explain roles in the current server. | Visible guild roles; avoid pretending roles imply authority without effective permission checks. |
| `discord_move_member` | Move a consenting user between voice rooms. | Restricted actor/target/channel rules and Move Members permission. |
| `discord_timeout_member` | Moderator-requested timeout. | Moderator authorization, target/reason/duration validation and hierarchy checks. |
| `get_bot_diagnostics` | Explain provider failures or queue pressure. | Redacted metrics; never expose tokens or private raw audio. |
| `search_room_knowledge` | Find project/game rules from approved documents. | Curated guild-specific retrieval index. |

For this bot I would prioritize Google-backed search (if actual Google results matter), weather, reminders, permission-scoped message summaries, then polls/events. Keep deterministic utilities and read-only room tools easy to use. Add posting, scheduling, and moderation as explicit actions with clearly defined actor and destination rules, rather than letting arbitrary web content request them.

Fourteen tool definitions are a manageable starting point. If the registry grows to dozens, select a relevant subset per request or use a tool-discovery step so a small model is not distracted by every possible action. Test tool choice with natural Brazilian Portuguese requests and multi-step follow-ups before exposing new actions.
