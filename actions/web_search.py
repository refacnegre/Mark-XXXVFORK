# actions/web_search.py
# MARK XXV — Web Search (DDG only)


def _ddg_search(query: str, max_results: int = 6) -> list:
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", ""),
                }
            )
    return results


def _format_ddg(query: str, results: list) -> str:
    if not results:
        return f"No results found for: {query}"

    lines = [f"Search results for: {query}\n"]
    for i, r in enumerate(results, 1):
        if r.get("title"):
            lines.append(f"{i}. {r['title']}")
        if r.get("snippet"):
            lines.append(f"   {r['snippet']}")
        if r.get("url"):
            lines.append(f"   {r['url']}")
        lines.append("")
    return "\n".join(lines).strip()


def _compare(items: list, aspect: str) -> str:
    lines = [f"Comparison — {aspect.upper()}\n{'─' * 40}"]

    for item in items:
        try:
            results = _ddg_search(f"{item} {aspect}", max_results=3)
        except Exception as e:
            print(f"[WebSearch] Compare lookup failed for {item!r}: {e}")
            results = []

        lines.append(f"\n▸ {item}")
        if not results:
            lines.append("  • No data found.")
            continue

        for r in results[:2]:
            if r.get("snippet"):
                lines.append(f"  • {r['snippet']}")
            elif r.get("title"):
                lines.append(f"  • {r['title']}")

    return "\n".join(lines).strip()


def web_search(
    parameters: dict,
    response=None,
    player=None,
    session_memory=None,
) -> str:
    params = parameters or {}
    query = params.get("query", "").strip()
    mode = params.get("mode", "search").lower()
    items = params.get("items", [])
    aspect = params.get("aspect", "general")

    if not query and not items:
        return "Please provide a search query."

    if items and mode != "compare":
        mode = "compare"

    if player:
        player.write_log(f"[Search] {query or ', '.join(items)}")

    print(f"[WebSearch] Query={query!r} Mode={mode}")

    try:
        if mode == "compare" and items:
            result = _compare(items, aspect)
            print("[WebSearch] Compare done.")
            return result

        results = _ddg_search(query)
        result = _format_ddg(query, results)
        print(f"[WebSearch] DDG results={len(results)}")
        return result
    except Exception as e:
        print(f"[WebSearch] Failed: {e}")
        return "Search is temporarily unavailable. Please try again shortly."
