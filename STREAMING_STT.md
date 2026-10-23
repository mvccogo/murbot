# Streaming speech recognition

Streaming is enabled in the current local configuration. Restart the bot with `uv run discord-voice-bot` and rejoin with `!join` to use the changes.

The receiver sends roughly 96 ms batches of 16 kHz mono audio as each person speaks. Each utterance has a unique room-local ID and its own transcription state. A partial transcript containing a configured bot name cancels current speech and generation. Unnamed partials never interrupt in the active semantic follow-up mode. A completed utterance from the focused speaker is checked for intent while playback continues; only a clear request directed at the bot can interrupt/start a response. Background conversation and uncertainty are ignored. Say “para, Murbot” to stop silently and “continua, Murbot” to resume an interrupted task. Partial transcripts never start an LLM reply. The final transcript starts at most one response; a quiet command recognized in a partial keeps that utterance quiet even if the final hypothesis changes.

Newer addressed turns take precedence. Late transcripts from an older turn cannot interrupt a newer request. Streams are cancelled on disconnect/rejoin. Audio, stream count, send queues, and stream lifetimes are bounded; overflow drops the affected utterance with a log message. The existing 20-second utterance cap remains.

## Existing local Whisper

```dotenv
STT_PROVIDER=local
STT_STREAMING_ENABLED=true
STT_STREAM_MIN_AUDIO_MS=640
STT_STREAM_INTERVAL_MS=320
STT_STREAM_WINDOW_SECONDS=6
STT_MAX_STREAMS=8
```

No additional Docker service or key is needed. Whisper itself is not a native streaming decoder: the adapter decodes the latest rolling window periodically, coalesces audio arriving during inference, and produces revisable partial hypotheses. Inference stays on a single dedicated worker to avoid concurrent GPU calls. At utterance completion, it performs one full final decode with the existing VAD/confidence filters. Partial decodes use greedy decoding and the Discord VAD boundary, without running a second endpoint detector. Debug WAV capture applies to final input only.

The minimum audio setting controls when the first decode may start, not when its result will arrive. The interval sets a minimum gap after each partial decode; inference time and other speakers' work add latency. Increasing these values reduces GPU work; decreasing them increases load and may worsen hypotheses. CPU fallback and simultaneous speakers can be substantially slower. The rolling text is intended for name/quiet recognition, not a stable transcript display. Final questions still wait for the configured silence timeout.

## Native WebSocket streaming

To select the Deepgram streaming API:

```dotenv
STT_PROVIDER=deepgram
STT_STREAMING_ENABLED=true
DEEPGRAM_API_KEY=your_key
DEEPGRAM_MODEL=nova-3
DEEPGRAM_LANGUAGE=pt-BR
DEEPGRAM_WS_URL=wss://api.deepgram.com/v1/listen
```

This provider streams PCM continuously over a WebSocket and receives interim/final segments concurrently. Each utterance opens one connection. Provider segment finals are accumulated, while local Discord VAD determines the end of the whole utterance. `CloseStream` flushes trailing recognition before exactly one application-level final event. Provider failures do not fall back to a partial transcript as a complete question. This option sends voice audio to the configured service and uses its account. A Docker service implementing the same Deepgram wire protocol can use a different WebSocket URL.

Existing `STT_PROVIDER=cloud` upload endpoints remain final-only. Enabling the streaming flag does not turn a `/audio/transcriptions` WAV-upload API into a streaming protocol, and the bot does not repeatedly upload partial clips to those providers.

The native adapter follows [Deepgram's live-audio protocol](https://developers.deepgram.com/reference/speech-to-text/listen-streaming) and [CloseStream behavior](https://developers.deepgram.com/docs/close-stream). Live account integration has not been tested; protocol tests use a simulated socket.

## Validation

Run:

```powershell
uv run python -m unittest discover -s tests -v
uv run python scripts/diagnose_streaming.py
```

The streaming diagnostic generates a neutral phrase with the local Kokoro service and feeds it to configured STT in paced 96 ms chunks. It asserts that an addressed partial arrives before the input ends and exactly one final transcript is emitted. It does not connect to Discord.

Measured with the existing cached local Whisper model:

- First partial: approximately 1.48 seconds after input started.
- First partial matching `Murbot`: approximately 2.06 seconds. The first hypothesis spelled it `Mourbot`, so the exact configured name gate correctly waited for the next hypothesis. Alternate spellings can be configured with `BOT_ALIASES`.
- Input ended at approximately 5.84 seconds; one final result arrived at 6.38 seconds.

These are one synthetic replay's timings, not an expected latency guarantee for live Discord. The real test used local GPU STT and Kokoro; native API tests are mocked. Verify with two live speakers: ordinary conversation should not interrupt, saying the name during an answer should interrupt before the question ends, and `Murbot, fica quieto` should produce silence with no acknowledgment.

## Natural follow-up conversation

The default/active settings are `CONVERSATION_FOLLOWUP_SECONDS=20` and `CONVERSATION_SEMANTIC_FOLLOWUPS=true`. During a reply and the post-playback window, the last addressed speaker's finished utterance is evaluated against recent conversation and the active request/response. This uses a small non-streamed, tool-free call to the configured LLM; DeepSeek thinking remains disabled. Only valid reply/stop decisions with confidence at least 0.9 are accepted. Failures, timeouts (three seconds), ambiguous intent, and background conversation leave playback unchanged. Model confidence is a heuristic, not a calibrated guarantee. API usage adds a short classification request per eligible final utterance.

The semantic call does not hold the routing lock. Explicit named requests/stop commands proceed while it is pending, and a result is discarded if a newer addressed turn or cleared focus supersedes it. The window uses utterance capture time and actual playback completion. Other speakers must first say the bot name. Set the duration to zero for strict name-only routing. Setting semantic follow-ups false with a positive duration restores the older timing-only behavior, which can mistake side conversations for follow-ups.

`uv run python scripts/diagnose_intent.py` checks five Portuguese examples against the configured model without Discord, audio, or tool execution. Live DeepSeek accepted two follow-up questions and ignored three side-conversation/comment examples; unit tests also cover timeout handling, uninterrupted playback, and explicit-stop priority.

A name recognized in a partial is remembered for that utterance even if Whisper drops it from the final text. It does not authorize another speaker or an empty final transcript. Diagnostic routing logs include a reason; STT_LOG_TRANSCRIPTS=true adds the recognized text when needed. The active configuration leaves transcript logging disabled.

Follow-up regression suite: 26 tests pass, including consecutive streamed requests, name revisions, focus expiry, quiet reset, other-speaker isolation, and stale playback-completion callbacks.
# Wake-name recognition

Local Whisper now receives the configured bot name and aliases as vocabulary hints (`hotwords`) for both partial and final decoding. Override with `STT_HOTWORDS` if needed. This helps unfamiliar names such as Murbot without injecting a canned sentence as an initial prompt. It remains a hint, not a dedicated wake-word detector; exact recognized names/aliases and the speaker's follow-up window determine whether a turn is accepted. Restart the bot after changing these settings. Do not add unrelated people's names as aliases to compensate for a missed wake name.

Interrupted LLM turns retain the accepted user request and any response sentences already emitted, marked as interrupted and possibly not fully heard. A follow-up such as “continua” therefore keeps the original task in context even if generation was cancelled during search or TTS. Pending search results are not fabricated or recorded as completed; the model may need to consult tools again.

