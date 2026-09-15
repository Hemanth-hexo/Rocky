"""Web search + page-reading via Tavily (tavily.com) — the one deliberate
exception to Rocky's otherwise fully-local design. Everything else runs
offline; these two tools make an outbound HTTPS call because the entire
point is looking up things the local model's frozen training data can't
possibly know — anything after its training cutoff, current events,
today's specifics (see the "iOS 27" conversation that prompted this)."""

import os

import requests

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
_SEARCH_URL = "https://api.tavily.com/search"
_EXTRACT_URL = "https://api.tavily.com/extract"
_TIMEOUT = 15
_MAX_EXTRACT_CHARS = 8000


def web_search(query: str, max_results: int = 5) -> str:
    if not TAVILY_API_KEY:
        return "web search isn't configured — TAVILY_API_KEY is missing from .env"
    max_results = max(1, min(10, int(max_results)))
    try:
        resp = requests.post(
            _SEARCH_URL, json={"api_key": TAVILY_API_KEY, "query": query, "max_results": max_results}, timeout=_TIMEOUT
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"web search failed: {e}"
    results = resp.json().get("results", [])
    if not results:
        return f"no results found for '{query}'"
    blocks = []
    for r in results:
        snippet = (r.get("content") or "").strip()
        if len(snippet) > 300:
            snippet = snippet[:300] + "…"
        blocks.append(f"{r.get('title', 'untitled')}\n{r.get('url', '')}\n{snippet}")
    return "\n\n".join(blocks)


def fetch_url(url: str) -> str:
    if not TAVILY_API_KEY:
        return "web search isn't configured — TAVILY_API_KEY is missing from .env"
    try:
        resp = requests.post(_EXTRACT_URL, json={"api_key": TAVILY_API_KEY, "urls": [url]}, timeout=_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"couldn't fetch '{url}': {e}"
    data = resp.json()
    if not data.get("results"):
        failed = data.get("failed_results") or []
        reason = failed[0].get("error", "unknown error") if failed else "no content returned"
        return f"couldn't fetch '{url}': {reason}"
    content = data["results"][0].get("raw_content", "")
    if len(content) > _MAX_EXTRACT_CHARS:
        content = content[:_MAX_EXTRACT_CHARS] + f"\n\n[truncated — page is longer than {_MAX_EXTRACT_CHARS:,} characters]"
    return content


WEB_FUNCTIONS = {
    "web_search": web_search,
    "fetch_url": fetch_url,
}

WEB_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the live web for current information — anything that might have happened or "
                "changed after your training data, or that you're not confident about. Returns titles, "
                "URLs, and short snippets for each result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "max_results": {"type": "integer", "description": "How many results to return (default 5, max 10)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Read the actual content of one specific web page, given its URL — use after web_search when a snippet isn't enough detail.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "The URL to fetch"}},
                "required": ["url"],
            },
        },
    },
]
