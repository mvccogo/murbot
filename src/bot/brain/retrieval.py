"""Bounded web retrieval independent of the model's willingness to call tools."""
import json
import re
import unicodedata

from bot.brain.web_tools import public_url


def normalized(text):
    return "".join(c for c in unicodedata.normalize("NFD", text.casefold()) if not unicodedata.combining(c))


URL_PATTERN = re.compile(r"https?://[^\s<>]+|\b(?:www\.)?[a-z0-9][a-z0-9.-]*\.(?:com|org|net|io|dev|gov|edu|br)(?:/[^\s<>]*)?", re.I)


def search_allowed(text):
    return not re.search(
        r"\b(?:nao (?:pesquise|busque|procure|acesse|use (?:a )?internet)|sem (?:pesquisar|buscar|internet)|"
        r"(?:do not|don't|dont) (?:search|browse)|without (?:search|browsing))\b", normalized(text))


def lookup_plan(text):
    if not search_allowed(text):
        return None
    match = URL_PATTERN.search(text)
    if match:
        url = match.group().rstrip(".,;!?)]}\"'")
        return "browse_webpage", {"url": url if "://" in url else "https://" + url}
    if re.search(
        r"\b(?:pesquis\w*|busqu\w*|procure|procurar|google|googl\w*|search|look up|browse|"
        r"internet|webpage|website|site|pagina|noticias|news|clima|previsao do tempo|cotacao|"
        r"preco|precos|atualmente|hoje|latest|current|documentacao)\b", normalized(text)):
        return "web_search", {"query": text.strip()[:500]}
    return None


def access_refusal(text):
    text = normalized(text)
    return bool(re.search(
        r"(?:nao (?:tenho|possuo|consigo|posso)|sem acesso|(?:don.t|do not) have|cannot|can.t|unable to)"
        r".{0,100}(?:acess|access|internet|naveg|pesquis|buscar|consult|abrir|obter.{0,30}informa|"
        r"browse|search|retrieve|web|page|site|link|real.time)", text))


def decoded(result):
    try:
        value = json.loads(result)
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def readable(result):
    text = decoded(result).get("text", "")
    return len(text.strip()) >= 80 and not any(marker in normalized(text[:600]) for marker in (
        "just a moment", "verify you are human", "enable javascript", "access denied", "captcha",
    ))


async def retrieve(name, args, execute, context, cancel, *, initial_result=None):
    """Return actual tool attempts, with one search and up to two alternative pages.

    Each call keeps its own timeout in execute_tool. Never search a rejected URL
    or forward its query string/credentials to a search engine.
    """
    attempts = []
    visited = context.setdefault("_web_attempted_urls", set())

    async def call(tool, arguments):
        if cancel.is_set():
            return ""
        result = await execute(tool, arguments, context)
        attempts.append((tool, arguments, result))
        return result

    result = initial_result if initial_result is not None else await call(name, args)
    original = None
    if cancel.is_set():
        return attempts
    if name == "browse_webpage":
        if readable(result):
            return attempts
        try:
            original = public_url(args["url"])
        except (KeyError, ValueError, TypeError):
            return attempts
        visited.add(str(original.with_query(None)))
        # Host/path locate indexed copies without leaking URL tokens.
        query = f"{original.host} {original.path.replace('/', ' ')}".strip()[:500]
        result = await call("web_search", {"query": query})
    if cancel.is_set():
        return attempts
    opened = 0
    for item in decoded(result).get("results", []):
        try:
            target = public_url(item["url"])
        except (KeyError, ValueError, TypeError):
            continue
        identity = str(target.with_query(None))
        if identity in visited:
            continue
        visited.add(identity)
        page = await call("browse_webpage", {"url": str(target)})
        opened += 1
        if readable(page) or opened >= 2 or cancel.is_set():
            break
    return attempts


def append_attempts(messages, attempts):
    for name, args, result in attempts:
        call_id = f"retrieval_{len(messages)}"
        messages.append({"role": "assistant", "content": None, "tool_calls": [{
            "id": call_id, "type": "function", "function": {
                "name": name, "arguments": json.dumps(args, ensure_ascii=False),
            },
        }]})
        messages.append({"role": "tool", "tool_call_id": call_id, "content": result})
