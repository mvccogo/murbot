"""Public web search and bounded HTML reading for model tool calls."""
import asyncio
import ipaddress
import socket

import aiohttp
from bs4 import BeautifulSoup
from ddgs import DDGS
from yarl import URL


def public_url(value):
    url = URL(value)
    if url.scheme not in ("http", "https") or not url.host:
        raise ValueError("Use uma URL pública http ou https.")
    if url.user is not None or url.password is not None or url.port not in (80, 443):
        raise ValueError("URLs com credenciais ou portas alternativas não são aceitas.")
    host = url.host.casefold().rstrip(".")
    if "%" in host or host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("A navegação acessa apenas sites públicos.")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        pass  # Hostnames are checked by the actual connection's resolver.
    else:
        if not address.is_global:
            raise ValueError("A navegação acessa apenas endereços públicos.")
    return url.with_fragment(None)


class PublicResolver(aiohttp.abc.AbstractResolver):
    """Validate and return the exact DNS answers the connector will use."""
    def __init__(self):
        self.resolver = aiohttp.resolver.ThreadedResolver()

    async def resolve(self, host, port=0, family=socket.AF_INET):
        records = await self.resolver.resolve(host, port, family)
        if not records or any(not ipaddress.ip_address(r["host"]).is_global for r in records):
            raise OSError("O host não aponta exclusivamente para endereços públicos.")
        return records

    async def close(self):
        await self.resolver.close()


async def web_search(query, max_results=3):
    if not isinstance(query, str) or not query.strip() or len(query) > 500:
        raise ValueError("A busca precisa de uma consulta entre 1 e 500 caracteres.")
    if not isinstance(max_results, int) or isinstance(max_results, bool) or not 1 <= max_results <= 5:
        raise ValueError("max_results deve ser um inteiro de 1 a 5.")
    def search():
        with DDGS(timeout=8) as client:
            return list(client.text(query.strip(), max_results=max_results))
    results = await asyncio.to_thread(search)
    return {"query": query, "untrusted_content": True, "results": [
        {"title": str(r.get("title", ""))[:300], "snippet": str(r.get("body", ""))[:1500], "url": str(r.get("href", ""))[:2000]}
        for r in results[:max_results]
    ]}


def extract_page(body, url, charset="utf-8"):
    soup = BeautifulSoup(body.decode(charset or "utf-8", errors="replace"), "html.parser")
    title = soup.title.get_text(" ", strip=True)[:300] if soup.title else ""
    for item in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        item.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    text = " ".join(root.stripped_strings)
    links = []
    seen = set()
    for link in root.find_all("a", href=True):
        try:
            target = str(public_url(str(URL(url).join(URL(link["href"])))))
        except (ValueError, TypeError):
            continue
        if target not in seen:
            seen.add(target)
            links.append({"text": link.get_text(" ", strip=True)[:150], "url": target[:2000]})
        if len(links) == 8:
            break
    return {"url": str(url), "title": title, "text": text[:6000],
            "truncated": len(text) > 6000, "links": links, "untrusted_content": True}


async def browse_webpage(url):
    target = public_url(url)
    resolver = PublicResolver()
    connector = aiohttp.TCPConnector(resolver=resolver, use_dns_cache=False)
    try:
        async with aiohttp.ClientSession(connector=connector, trust_env=False,
                                        timeout=aiohttp.ClientTimeout(total=10),
                                        headers={"User-Agent": "Murbot/0.1 (public page reader)"}) as client:
            for _ in range(6):
                target = public_url(str(target))
                async with client.get(target, allow_redirects=False) as response:
                    if response.status in (301, 302, 303, 307, 308):
                        location = response.headers.get("Location")
                        if not location:
                            raise ValueError("Redirecionamento sem destino.")
                        target = public_url(str(target.join(URL(location))))
                        continue
                    if response.status != 200:
                        raise ValueError(f"O site retornou HTTP {response.status}.")
                    if response.content_type not in ("text/html", "application/xhtml+xml", "text/plain"):
                        raise ValueError("A ferramenta lê páginas HTML e texto; este formato não é suportado.")
                    body = bytearray()
                    async for chunk in response.content.iter_chunked(16384):
                        body.extend(chunk)
                        if len(body) > 1_000_000:
                            raise ValueError("Página excede o limite de 1 MB.")
                    return await asyncio.to_thread(extract_page, bytes(body), target, response.charset)
            raise ValueError("O site excedeu o limite de redirecionamentos.")
    finally:
        await connector.close()
        await resolver.close()
