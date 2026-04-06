"""
Tool: fetch_arxiv
Searches arXiv for academic papers using the public arXiv API.
No API key required — uses urllib only.
"""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET


def fetch_arxiv(query: str, max_results: int = 5) -> list:
    """
    Searches arXiv for research papers matching the query using the arXiv public API.
    Use this tool when the user asks to find, search, or look up academic papers,
    research articles, or scientific publications on arXiv.

    Args:
        query (str): The search query or topic (e.g. "LLM agents", "transformer architecture").
        max_results (int): Maximum number of results to return. Defaults to 5.

    Returns:
        list[dict]: A list of papers, each with keys:
            - "title"   (str): Title of the paper.
            - "authors" (list[str]): List of author names.
            - "summary" (str): Short abstract of the paper.
            - "url"     (str): Link to the arXiv paper page.
            - "published" (str): Publication date (YYYY-MM-DD).
    """
    base_url = "http://export.arxiv.org/api/query"
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
    })
    url = f"{base_url}?{params}"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            xml_data = response.read().decode("utf-8")
    except Exception as e:
        print(f"Error fetching from arXiv API: {e}")
        return []

    # Parse the Atom XML response
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as e:
        print(f"Error parsing arXiv response XML: {e}")
        return []

    results = []
    for entry in root.findall("atom:entry", ns):
        title_el   = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        published_el = entry.find("atom:published", ns)
        id_el      = entry.find("atom:id", ns)
        authors    = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)
                      if a.find("atom:name", ns) is not None]

        results.append({
            "title":     title_el.text.strip()     if title_el     is not None else "N/A",
            "authors":   authors,
            "summary":   summary_el.text.strip()   if summary_el   is not None else "N/A",
            "url":       id_el.text.strip()        if id_el        is not None else "N/A",
            "published": (published_el.text[:10]   if published_el is not None else "N/A"),
        })

    return results


# ── Tool spec dict for the ReAct agent ──────────────────────────────────────
FETCH_ARXIV_SPEC = {
    "name": "fetch_arxiv",
    "description": (
        "Searches arXiv for academic research papers on a given topic. "
        "Use this when the user wants to find papers, articles, or research on arXiv. "
        "Args: query (str) — the topic or keywords to search for; "
        "max_results (int, optional) — number of results to return (default 5)."
    ),
    "function": fetch_arxiv,
}
