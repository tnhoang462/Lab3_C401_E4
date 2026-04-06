import requests
import xml.etree.ElementTree as ET
import time
import threading
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ATOM_NS  = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"

# ArXiv recommends >= 3s between calls
RATE_LIMIT_SECONDS = 3
MAX_RETRIES        = 3
TIMEOUT_SECONDS    = 15

# Thread-safe rate limiter — tracks last call time globally
_lock          = threading.Lock()
_last_call_time: float = 0.0

# Rate Limiter
def _wait_for_rate_limit() -> None:
    global _last_call_time
    with _lock:
        now     = time.monotonic()
        elapsed = now - _last_call_time
        wait    = RATE_LIMIT_SECONDS - elapsed

        if wait > 0:
            print(f"   ⏳ Rate limit: waiting {wait:.1f}s...")
            time.sleep(wait)

        _last_call_time = time.monotonic()


# URL Builder
def build_search_url(
    query: str,
    max_results: int = 5,
    sort_by: str    = "submittedDate",
    sort_order: str = "descending",
) -> str:
    has_prefix = any(
        p in query for p in ["ti:", "au:", "abs:", "all:", "AND", "OR", "ANDNOT"]
    )

    if not has_prefix:
        words           = query.strip().split()
        processed_query = "+AND+".join(f"all:{w}" for w in words)
    else:
        processed_query = query

    return (
        f"{ARXIV_API_URL}"
        f"?search_query={processed_query}"
        f"&start=0"
        f"&max_results={max_results}"
        f"&sortBy={sort_by}"
        f"&sortOrder={sort_order}"
    )


def fetch_arxiv(url: str) -> str:

    headers = {"User-Agent": "ResearchAgent/1.0 (Student Lab; educational use)"}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _wait_for_rate_limit()

            print(f" Calling ArXiv API (attempt {attempt}/{MAX_RETRIES})...")
            print(f" {url[:110]}...")

            response = requests.get(url, headers=headers, timeout=TIMEOUT_SECONDS)

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                print(f"  429 Too Many Requests — waiting {retry_after}s...")
                time.sleep(retry_after)
                continue

            response.raise_for_status()

            xml_content = response.text
            print(f" Received {len(xml_content):,} chars")
            return xml_content

        except requests.exceptions.Timeout:
            wait = 5 * attempt
            print(f" Timeout on attempt {attempt}. Retrying in {wait}s...")
            if attempt < MAX_RETRIES:
                time.sleep(wait)
            else:
                raise requests.exceptions.Timeout(
                    f"ArXiv API timed out after {MAX_RETRIES} attempts."
                )

        except requests.exceptions.ConnectionError:
            wait = 5 * attempt
            print(f" Connection error on attempt {attempt}. Retrying in {wait}s...")
            if attempt < MAX_RETRIES:
                time.sleep(wait)
            else:
                raise requests.exceptions.ConnectionError(
                    "Cannot reach ArXiv API after multiple attempts."
                )

        except requests.exceptions.HTTPError as e:
            raise e

    raise RuntimeError("fetch_arxiv: exhausted all retries.")

def parse_arxiv_xml(xml_string: str) -> List[Dict]:
    root    = ET.fromstring(xml_string)
    entries = root.findall(f"{{{ATOM_NS}}}entry")

    if not entries:
        return []

    def get_text(element, tag: str, ns: str = ATOM_NS) -> str:
        found = element.find(f"{{{ns}}}{tag}")
        return found.text.strip() if found is not None and found.text else ""

    papers = []

    for entry in entries:
        title  = get_text(entry, "title")
        id_url = get_text(entry, "id")

        published      = get_text(entry, "published")
        published_date = published[:10] if published else "N/A"

        abstract = " ".join(get_text(entry, "summary").split())
        if len(abstract) > 500:
            abstract = abstract[:500] + "..."

        authors = [
            get_text(a, "name")
            for a in entry.findall(f"{{{ATOM_NS}}}author")
            if get_text(a, "name")
        ]
        authors_display = authors[:4]
        if len(authors) > 4:
            authors_display.append("et al.")

        pdf_url = next(
            (lnk.get("href", "") for lnk in entry.findall(f"{{{ATOM_NS}}}link")
             if lnk.get("title") == "pdf"),
            ""
        )

        cat_elem = entry.find(f"{{{ARXIV_NS}}}primary_category")
        category = cat_elem.get("term", "") if cat_elem is not None else ""

        papers.append({
            "title":        title,
            "arxiv_id":     id_url.replace("http://arxiv.org/abs/", "").strip(),
            "published":    published_date,
            "authors":      authors_display,
            "abstract":     abstract,
            "pdf_url":      pdf_url,
            "abstract_url": id_url,
            "category":     category,
        })

    return papers


def format_results(papers: List[Dict]) -> str:
    if not papers:
        return "No papers found.\nTip: try different keywords in English."

    lines = [f" Found {len(papers)} papers:\n", "=" * 60]

    for i, p in enumerate(papers, 1):
        lines += [
            f"\n[{i}] {p['title']}",
            f"     Authors  : {', '.join(p['authors'])}",
            f"     Published: {p['published']}  |  Category: {p['category']}",
            f"     Abstract : {p['abstract']}",
            f"     PDF      : {p['pdf_url']}",
            f"     URL      : {p['abstract_url']}",
            "-" * 60,
        ]

    return "\n".join(lines)

def search_arxiv(query: str) -> str:
    print(f"\n search_arxiv | query: '{query}'")
    try:
        url    = build_search_url(query=query, max_results=4)
        xml    = fetch_arxiv(url)
        papers = parse_arxiv_xml(xml)
        result = format_results(papers)
        print(f"   Done — {len(papers)} papers returned")
        return result

    except requests.exceptions.Timeout:
        return " Timeout: ArXiv did not respond. Try again later."
    except requests.exceptions.ConnectionError:
        return " Connection error: check your internet."
    except requests.exceptions.HTTPError as e:
        return f" HTTP {e.response.status_code}: {str(e)}"
    except ET.ParseError as e:
        return f" XML parse error: {str(e)}"
    except Exception as e:
        return f" Unexpected error: {str(e)}"


def search_arxiv_batch(queries: List[str], max_results_per_query: int = 3) -> Dict[str, str]:
    if not queries:
        return {}

    seen      = set()
    unique_qs = [q for q in queries if not (q in seen or seen.add(q))]

    print(f"\n   Batch search: {len(unique_qs)} queries (sequential)")

    results = {}
    for i, query in enumerate(unique_qs, 1):
        print(f"\n   [{i}/{len(unique_qs)}] Query: '{query}'")
        try:
            url    = build_search_url(query=query, max_results=max_results_per_query)
            xml    = fetch_arxiv(url) 
            papers = parse_arxiv_xml(xml)
            results[query] = format_results(papers)
        except Exception as e:
            results[query] = f"Error for '{query}': {str(e)}"

    print(f"\n Batch done — {len(results)} queries completed")
    return results


def format_batch_results(batch: Dict[str, str]) -> str:
    """
    Merge all batch results into one readable string for the agent.
    Each query section is clearly separated.
    """
    if not batch:
        return "No results."

    sections = []
    for i, (query, result) in enumerate(batch.items(), 1):
        sections.append(
            f"{'#' * 60}\n"
            f"QUERY {i}: {query}\n"
            f"{'#' * 60}\n"
            f"{result}"
        )
    return "\n\n".join(sections)


def search_arxiv_multi(queries_str: str) -> str:

    queries = [q.strip() for q in queries_str.split("|") if q.strip()]

    if not queries:
        return "No valid queries. Separate multiple queries with |"

    if len(queries) > 5:
        return "Max 5 queries per batch call to avoid rate limiting."

    print(f"\nsearch_arxiv_multi | {len(queries)} queries: {queries}")

    batch  = search_arxiv_batch(queries, max_results_per_query=3)
    return format_batch_results(batch)



# Tool Definition
ARXIV_TOOL = {
    "name": "search_arxiv",
    "description": (
        "Search for scientific papers on ArXiv. "
        "Simple input: 'RAG language model 2024'. "
        "Advanced: 'ti:attention AND au:vaswani'. "
        "(ti=title, au=author, abs=abstract, all=all fields) "
        "Use when you need academic papers or experimental results."
    ),
    "function": search_arxiv,
}

ARXIV_BATCH_TOOL = {
    "name": "search_arxiv_multi",
    "description": (
        "Search multiple topics on ArXiv in one call. "
        "Input: pipe-separated queries, max 5 queries. "
        "Example: 'RAG 2024 | LoRA fine-tuning | ti:BERT AND abs:NLP' "
        "Use when you need to compare multiple topics or cover multiple angles at once."
    ),
    "function": search_arxiv_multi,
}

ALL_ARXIV_TOOLS = [ARXIV_TOOL, ARXIV_BATCH_TOOL]