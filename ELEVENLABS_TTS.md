# ElevenLabs streaming TTS

The active configuration now selects ElevenLabs HTTP streaming TTS with `eleven_v4_turbo`, Portuguese language hint `pt`, speed `1.0`, and the public Brazilian voice **Roberta — Smooth and Confident** (`RGymW84CSmfVugnA5tvA`). The voice was selected from the public Brazilian Portuguese conversational catalog; its listing allows free users. The API key remains in `.env` and is not written here.

```dotenv
TTS_PROVIDER=elevenlabs
# Keep your existing ELEVEN_LABS_API_KEY, or use ELEVENLABS_API_KEY.
ELEVENLABS_VOICE_ID=RGymW84CSmfVugnA5tvA
ELEVENLABS_MODEL=eleven_v4_turbo
ELEVENLABS_LANGUAGE=pt
ELEVENLABS_SPEED=1.0
```

The bot uses [ElevenLabs' streaming speech endpoint](https://elevenlabs.io/docs/api-reference/text-to-speech/stream):

```text
POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream
```

Each completed LLM sentence starts a request. The response is raw `pcm_24000`: signed 16-bit, 24 kHz mono audio. A stateful resampler converts arriving chunks to 48 kHz stereo for Discord, preserving sample boundaries and filter state across arbitrary HTTP chunks. Playback starts after the first converted chunk; it does not collect an entire MP3 or wait for the full response. The playback loop applies backpressure after about two seconds of queued audio, plus a single arriving chunk.

Barge-in cancels generation, closes the HTTP response, and flushes queued playback. It does not wait for synthesis to finish. Failed ElevenLabs requests are logged with HTTP status and do not silently switch to another provider or generate a duplicate sentence. The existing `synthesize()` method can still collect audio for diagnostic/legacy callers; Discord response playback uses `stream_pcm()`.

Run the bot again, then `!join` to apply the new provider. `!tts elevenlabs` selects it at runtime; `!voice <voice_id>` changes the voice. Runtime voice changes do not rewrite `.env`.

Verification performed:

- Account model/voice reads succeeded; the model supports Portuguese.
- One short live Portuguese phrase returned 163 PCM chunks, representing 3.92 seconds of speech. First PCM arrived at 4.98 seconds and the request completed at 5.77 seconds. That first test is noticeably slower than an ideal voice interaction; it is a measured request, not a latency guarantee. Changing model/voice can alter this, and it should be compared using the same phrase.
- Regression checks cover fragmented raw PCM, resampling equivalence, HTTP errors, early audio arrival, nested-generator cleanup, and cancelling in-flight synthesis during playback.
- No live Discord session was restarted during these checks.

The diagnostic below generates one short phrase using API credits; it does not connect to Discord:

```powershell
uv run python scripts/diagnose_elevenlabs.py --speak
```

Catalog-only checks do not synthesize speech:

```powershell
uv run python scripts/diagnose_elevenlabs.py
uv run python scripts/diagnose_elevenlabs.py --library
```
