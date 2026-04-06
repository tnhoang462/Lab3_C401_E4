"""
Tool registry — wraps all tool implementations into the standard dict format
expected by ReActAgent: {"name": str, "description": str, "function": callable}

Every function signature: def fn(args: str) -> str
"""

from datetime import datetime
from src.tools.search_arxiv import search_arxiv, search_arxiv_multi
from src.tools.tavily_search import tavily_search
from src.tools.tavily_extract import TavilyExtractTool


# --- Wrappers ---

def _get_current_date(args: str) -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _search_tavily(query: str) -> str:
    urls = tavily_search(query)
    if not urls:
        return "No results found."
    return "\n".join(f"- {url}" for url in urls)


_extractor = None

def _fetch_tavily(url: str) -> str:
    global _extractor
    if _extractor is None:
        _extractor = TavilyExtractTool()
    result = _extractor.extract(url.strip(), summarize=True)
    if not result.results:
        return "No content extracted."
    return result.results[0].preview(max_chars=1000)


# --- Tool Definitions ---

TOOLS = [
    {
        "name": "get_current_date",
        "description": "Get the current date. Args: none",
        "function": _get_current_date,
    },
    {
        "name": "search_tavily",
        "description": (
            "Search the web for a broad overview of a topic. "
            "Returns a list of relevant URLs. "
            "Args: query (str). Example: 'latest research on LLM agents'"
        ),
        "function": _search_tavily,
    },
    {
        "name": "fetch_tavily",
        "description": (
            "Fetch and summarize the content of a URL. "
            "Use after search_tavily to read a specific page. "
            "Args: url (str). Example: 'https://example.com/article'"
        ),
        "function": _fetch_tavily,
    },
    {
        "name": "search_arxiv",
        "description": (
            "Search for scientific papers on ArXiv. "
            "Simple input: 'RAG language model 2024'. "
            "Advanced: 'ti:attention AND au:vaswani'. "
            "(ti=title, au=author, abs=abstract, all=all fields) "
            "Use for CS/AI/ML academic papers."
        ),
        "function": search_arxiv,
    },
    {
        "name": "search_arxiv_multi",
        "description": (
            "Search multiple topics on ArXiv in one call. "
            "Input: pipe-separated queries, max 5. "
            "Example: 'RAG 2024 | LoRA fine-tuning | ti:BERT AND abs:NLP'"
        ),
        "function": search_arxiv_multi,
    },
]
