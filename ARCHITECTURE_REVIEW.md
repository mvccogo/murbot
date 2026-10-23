# Architecture review — 2026-10-23

The intended architecture is viable: Discord already identifies individual speakers, so per-user decoding and VAD can feed a shared room conversation without speaker diarization. Keep capture, routing, generation, and playback as separate responsibilities. Each speech provider should have its own endpoint and credentials.

## Findings and changes

| Priority | Finding | Change / status |
| --- | --- | --- |
| Critical | Installed voice-recv 0.5.2a179 removes transport encryption but does not decrypt DAVE. The raw payload was passed directly to Opus. A successful Opus decode is not proof that the input contained valid speech. | Added a DAVE adapter before decoding, using discord.py's negotiated DaveSession and the actual Discord user ID. Unknown identities, unready keys, and decryption failures are dropped. Pinned this receiver version because a future upstream DAVE implementation would require removing this adapter. Live encrypted Discord receive remains to be verified. |
| High | One stateful Silero model processed interleaved speakers. Recurrent context leaked between unrelated audio streams. | Each track now owns its VAD instance and decoder. Idle state resets between talk spurts. |
| High | PacketRouter wrote audio while an asyncio monitor concatenated and cleared the same buffers. | Added a common lock and dispatches callback creation through the event loop. Playback state is no longer mutated from the receive thread. |
| High | Every utterance cancelled the previous turn before STT. Every transcript was sent to the LLM. | Name routing happens after STT. Background speech is ignored. Quiet commands bypass the LLM and cancel both generation and playback. New addressed requests replace the active response. |
| High | All guilds shared one LLM history, and clear affected everyone. | Each active room session owns its engine/history. Clear and disconnect cancel the room response. |
| High | VAD threshold was 0.18 in the active environment. Peak normalization amplified accepted noise; Whisper VAD was disabled, and an initial prompt supplied invented transcription context. | Active threshold raised to 0.5. Removed normalization and prompt, enabled Whisper VAD, and filter low-confidence/no-speech/high-compression segments. Reject malformed, short, silent, and non-finite input. No arbitrary phrase blacklist was added. |
| High | STT used the LLM URL/key; TTS also used that URL, even when it pointed at a text-only provider. | Independent STT and TTS API configuration; lazy API TTS initialization also fixes switching from Kokoro/Edge to API TTS. MODE now selects the LLM without overriding STT. |
| Medium | Local inference ran on an unbounded default executor; cancellation could leave GPU inference running while another call began. | Dedicated single-worker executor with lock held inside inference. Per-room STT ordering and a limit of eight pending capture callbacks bound accepted work per room. Continuous utterances are capped at 20 seconds; idle tracks expire after 60 seconds. |
| Medium | Direct LLM replies waited for a complete non-streaming response. Only post-tool responses streamed. | Stream the initial response and assemble fragmented tool calls. Bound tool rounds and room history; close streams on cancellation. TTS remains sentence based. |
| Medium | stop_playback scheduled token cancellation even when already on the event loop, making cancellation timing harder to reason about. Late TTS writes could race an interruption. | Cancel immediately on the loop; capture the old token for thread dispatch. Recheck source state under the buffer lock. |
| Medium | Synchronous web search blocked the event loop. Skill filenames permitted traversal. | Offload search to a worker and restrict skill names. |
| Medium | .env was untracked and not ignored; memory/audio could be accidentally committed. | Added ignore rules and Docker context exclusions. Existing .env credentials were preserved. |
| Medium | uv_build inferred discord_voice_bot while the command imports bot. Docker installed the project before copying sources. | Set the build module to bot and split dependency/project installation into correct Docker layers. Built wheel and source distribution successfully. |

The repeated Portuguese phrase cannot be proven to have one cause without the original raw or decoded clip. Missing DAVE decryption is the strongest concrete receive-path defect; permissive VAD and noise amplification made hallucinated output more likely. Portuguese itself was explicitly forced by STT settings and the LLM prompt. STT_LANGUAGE can now be set to auto; the conversational system prompt still asks for Brazilian Portuguese.

## Evidence

- Eleven regression tests pass: DAVE decryption ordering/drop behavior, name routing, ignored background speech, quiet cancellation, cancellation-token lifetime, conversion frequency/amplitude, PCM output length, stale audio rejection, STT input and segment filtering, independent speaker buffers/preroll, and streaming cancellation.
- Real Kokoro Docker -> cached GPU Whisper test: 2.69-second neutral phrase, RMS 0.04787. Input "Murbot, me diga qual é a capital do Brasil." was transcribed correctly and routed as a request.
- Real local Ollama qwen2.5:3b request streamed a short Portuguese greeting successfully with the configured tool-capable request format.
- uv lock refreshed offline; uv build produced a wheel and source distribution containing the bot package.
- Docker inspection showed Ollama and Kokoro running. The optional bot container was not running.
- No bot was connected to Discord, no live channel audio was captured, and no deployment/restart was performed during this review. Docker GPU Whisper and hosted API providers have not been integration-tested.

## Remaining architecture work

1. **Name-aware interruption (implemented in follow-up).** Local Whisper now emits incremental rolling-window partials, and Deepgram supports native duplex WebSocket transcription. A recognized partial name cancels playback/generation before utterance completion; final text alone dispatches a reply. Turn IDs reject stale results; partial quiet intents stay quiet. Local recognition was measured at about 2.1 seconds from the start of paced synthetic input, so this is not a zero-latency wake-word detector. See STREAMING_STT.md. Native Deepgram has protocol tests but has not been tested against a live account.
2. **Conversation policy (updated after live feedback).** An addressed request now opens a configurable 20-second follow-up window for that speaker, extended through playback and measured from its actual completion. Other speakers must address the bot to take over; quiet requests clear focus. A partial name is remembered within its utterance. Literal matching and the timing window still cannot infer whether a focused speaker has turned to someone else, so semantic intent routing remains a possible improvement.
3. **Audio flow control.** The receive callback still performs resampling and VAD, and first use loads one model per speaker on the router thread. For larger rooms, move these stages behind bounded per-speaker queues on a dedicated worker; keep timestamps and packet-loss metrics. The present lock can briefly delay the event loop during processing. Short packet resampling filters also restart at each boundary; a stateful resampler would improve continuity.
4. **TTS delivery and history.** TTS downloads and decodes a whole sentence before queuing it; there is no bounded playback-ahead budget. Use streamed PCM with a small buffer for lower latency and memory use. Generated history may include text still buffered when interrupted. Track delivered speech and interruption markers instead. Physical playback stop is constrained by Discord/network buffering, not zero milliseconds.
5. **Provider lifecycle and configuration.** Clients/models load at startup. Runtime mode/TTS/voice commands remain global and are not permission-restricted; mode switches rebuild all room histories. Add a guild administrator policy, per-guild provider settings, graceful bot-wide shutdown, and close shared STT/TTS clients/executors. API providers need contract tests for supported tool and streaming formats. Kokoro's Edge fallback is internet access, so local mode is not a guaranteed offline policy.
6. **Tools and storage.** The web reader now restricts destinations to public addresses using the actual connector resolver, validates redirects, and bounds response size. Discord read tools are session-scoped and check requester membership and channel visibility. Memory is keyed by user ID and shared files, while learned skills are global instructions. Add guild scope, atomic file updates/locks, and an explicit policy for persistent skill creation. Web content/tool results should be treated as untrusted by the model. The initial filename fix does not address the whole trust boundary.
7. **Operations.** Add service health/readiness checks and validate the actual CUDA libraries in the bot image. Pin tested container image digests and monitor combined VRAM use for Ollama, Kokoro, and Whisper. Optional WAV capture is off by default and has no retention cap; enable only temporarily. Add metrics for decrypt/decode failures, VAD decisions, speech queue depth, dropped turns, STT/LLM/TTS latency, and wake-name recognition rate.

## Live acceptance checks

Restart the bot with `uv run discord-voice-bot`, then use `!join` in a test voice room. Verify these with headphones to reduce acoustic echo:

1. Two people converse without the name: no response and no interruption.
2. Each person independently says "Murbot, que horas são?": correctly recognized speaker and one response.
3. During a long answer, another speaker says "Murbot, fica quieto": audio and generation stop after recognition, without an acknowledgment or stale continuation.
4. During an answer, unrelated conversation continues: the answer should continue.
5. Another addressed question replaces the answer; leaving/rejoining does not resurrect a prior response.
6. Repeat across two guilds and verify no history crosses between rooms.
7. If transcription repeats unrelated text, enable temporary capture and listen to the exact 16 kHz WAV before changing model prompts. Check decrypt/decode errors first.

## Primary references checked

- [Upstream DAVE receive issue #64](https://github.com/imayhaveborkedit/discord-ext-voice-recv/issues/64)
- [Upstream DAVE receive proposal #58](https://github.com/imayhaveborkedit/discord-ext-voice-recv/pull/58/files)
- [Faster-whisper implementation and decoding options](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py)

The adapter was also checked against the installed davey type declarations and discord.py voice-state implementation. The upstream proposals are open changes, not evidence that a released receiver already includes the fix.

## Streaming STT follow-up

Local streaming is enabled in the active environment. Audio is forwarded in approximately 96 ms batches with one stream per utterance. Local partial inference uses a bounded rolling window, and exactly one final decode covers the utterance. Limits and cleanup prevent a slow/failed provider from creating unbounded audio queues. Existing REST-only STT providers do not poll WAV uploads. The detailed behavior and follow-up validation are recorded in STREAMING_STT.md.

## Follow-up conversation fix

Live terminal evidence showed continued audio/STT processing after the first answer, with later turns routed as ignore. The user confirmed those follow-ups omitted the bot name. The strict per-utterance name requirement was replaced by speaker-scoped conversation focus. The 26-test suite covers follow-up acceptance, expiry, quiet reset, speaker changes, playback completion, and repeated streamed requests. These changes require restarting the running Python process; they are not hot-loaded.

## Discord and web tools follow-up

Added live room information, participant/member state, recent speaking detection, and visible server channels to the model tool registry. Bound context is supplied by each guild session and preserved when changing the LLM mode. Search and page reading return source URLs; public page fetching restricts destinations, redirects, and sizes. Argument validation and call timeouts apply to the registry. The local LLM successfully invoked the room-member tool against a simulated room; live public search and a Python.org fetch passed. TOOLS.md documents behavior and examples.

## ElevenLabs API and tool inventory update

ElevenLabs HTTP streaming TTS is now enabled with Roberta (Brazilian Portuguese) and eleven_v4_turbo. PCM chunks are resampled incrementally, playback is paced with a small queue, and barge-in closes the response. A live short test returned audio successfully, with first PCM at 4.98 seconds; see ELEVENLABS_TTS.md for that latency limitation. The registry now includes calculation, unit conversion, and current-clock tools, for 14 tools total. TOOLS.md lists implemented tools and a separate proposed roadmap. API_LLM_OPTIONS.md records checked API price candidates without switching the existing LLM provider. Forty-seven tests pass; the current local LLM still selects the room-member tool with the expanded registry.
