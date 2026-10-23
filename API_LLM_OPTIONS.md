# Budget LLM APIs for Murbot

Checked October 23, 2026. Prices below are USD per one million text tokens, input/output separately. These are standard on-demand rates, not batch discounts. They exclude TTS/STT, search APIs, tax, and credit-purchase fees. Tool schemas, history, tool results, and some reasoning tokens contribute to usage. Provider price and model availability can change.

| Provider | Candidate model | Input | Output | Why evaluate it for Murbot |
| --- | --- | ---: | ---: | --- |
| [Groq](https://console.groq.com/docs/models) | `openai/gpt-oss-20b` | $0.075 | $0.30 | Cheapest starting point on this shortlist with fast inference and a familiar API. Test Portuguese and tool selection with real examples. |
| [Groq](https://console.groq.com/docs/models) | `openai/gpt-oss-120b` | $0.15 | $0.60 | My first value candidate: larger model, still inexpensive, with a documented fast serving path. |
| [Google Gemini](https://ai.google.dev/gemini-api/docs/pricing) | `gemini-2.5-flash-lite` | $0.10 | $0.40 | Very low text cost; evaluate current model lifecycle and tool compatibility before selecting it. |
| [Google Gemini](https://ai.google.dev/gemini-api/docs/pricing) | `gemini-3.1-flash-lite` | $0.25 | $1.50 | A newer inexpensive candidate aimed at high-volume agentic tasks. |
| [OpenRouter](https://openrouter.ai/qwen/qwen3-30b-a3b-instruct-2507) | `qwen/qwen3-30b-a3b-instruct-2507` | From $0.09 | From $0.30 | Non-thinking model positioned for multilingual instruction following and tool use. Aggregator pricing and performance depend on the selected backend. |
| [DeepSeek](https://api-docs.deepseek.com/quick_start/pricing/) | `deepseek-flash` | $0.15 off-peak / $0.30 peak | $0.60 off-peak / $1.20 peak | Budget alternative with documented tool calls; cache-hit input rates are lower. Thinking is enabled by default, so latency and tool-history handling need attention. |
| [Cerebras](https://www.cerebras.ai/pricing) | `gpt-oss-120b` | $0.35 | $0.75 | Speed-focused alternative. Published self-serve Developer terms describe development/evaluation use; check the production tier before deploying a public bot. |

These recommendations are candidates to benchmark, not measured Portuguese-quality rankings. For a voice assistant, measure time to the first useful sentence and tool-call accuracy rather than only tokens per second. A model that takes a long reasoning pause can feel slow despite a high output rate.

DeepSeek's published peak periods are 01:00–04:00 and 06:00–10:00 UTC on weekdays, excluding Chinese public holidays. The input prices in the table are cache misses. OpenRouter advertises model pricing from its available routes; credit-purchase fees and routing options need separate inspection.

My suggested order: Groq 120B for a quality/price baseline, Groq 20B for lower cost, then Gemini 3.1 Flash-Lite and the OpenRouter Qwen candidate as comparisons. This is a proposed evaluation order, not a provider switch already applied to the bot.

For example, one million input tokens plus 200,000 output tokens would be approximately $0.27 on Groq 120B at these listed rates, before other services. That is a token-workload example, not an estimate per hour of Discord conversation. ElevenLabs speech is billed separately.

## Configuration

Murbot currently uses an OpenAI-compatible Chat Completions client. A Groq experiment would use:

```dotenv
MODE=cloud
LLM_PROVIDER=cloud
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_API_KEY=your_groq_key
LLM_MODEL=openai/gpt-oss-120b
LLM_TOOLS_ENABLED=true
```

These legacy `OPENAI_*` names configure the text provider, not ElevenLabs. STT and TTS remain independently configured.

The active configuration now selects DeepSeek Flash:

```dotenv
MODE=cloud
LLM_PROVIDER=deepseek
DEEPSEEK_MODEL=deepseek-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
# Prefer DEEPSEEK_API_KEY; DEEP_SEEK_API_KEY and OPENAI_API_KEY are accepted fallbacks.
```

The supplied key is currently read from the existing `OPENAI_API_KEY`. The dedicated DeepSeek provider uses its own endpoint and model without changing speech-provider settings. Every DeepSeek request explicitly sends `extra_body={"thinking": {"type": "disabled"}}` for first-sentence latency and compatibility with the standard tool history. This includes subsequent tool rounds. Switching `!mode local` and back to `!mode cloud` preserves the configured cloud provider. Restart the bot after changing `.env`.

The tool-loop implementation handles standard streamed function-call fragments. Provider-specific reasoning fields or thought signatures are not retained today, so do not assume every reasoning model is a drop-in replacement. Before changing providers, test direct replies, multi-step tool calls, cancellations, and follow-up questions. Gemini and DeepSeek deserve particular compatibility checks; prefer a supported non-thinking mode for low-latency voice when available.

The current local diagnostic checks a simulated room with the configured local LLM, not these paid API candidates:

```powershell
uv run python scripts/diagnose_tools.py
```

Use `uv run python scripts/diagnose_tools.py --api` to test the active API with a short Portuguese greeting, a natural follow-up, and live tool execution against a simulated room. This uses API tokens but does not connect to Discord.
