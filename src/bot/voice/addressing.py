"""Deterministic voice addressing; background conversations never reach the LLM."""
import re
import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.findall(r"\w+", text))


def classify_address(text: str, name: str, aliases: str = "") -> str:
    """Return ignore, stop, or reply. Each turn must contain a configured name."""
    normalized = f" {normalize(text)} "
    names = [normalize(n) for n in [name, *aliases.split(",")] if normalize(n)]
    if not any(f" {n} " in normalized for n in names):
        return "ignore"
    stop_commands = (
        "fica quieto", "fique quieto", "fique em silencio", "fica em silencio",
        "cale a boca", "cala a boca", "pare de falar", "para de falar",
        "silencio", "quieto", "calado", "stop talking", "be quiet", "shut up", "stop",
        "pare", "parar", "cala", "chega",
    )
    remainder = normalize(text)
    for n in names:
        remainder = re.sub(rf"(?<!\w){re.escape(n)}(?!\w)", "", remainder).strip()
    if remainder in ("para", "para por favor", "quiet", "silence"):
        return "stop"
    if any(f" {command} " in normalized for command in stop_commands):
        return "stop"
    return "reply"
