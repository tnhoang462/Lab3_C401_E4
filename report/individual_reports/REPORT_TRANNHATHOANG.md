# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Trần Nhật Hoàng
- **Student ID**: 2A202600431
- **Date**: 06-04-2026

---

## I. Technical Contribution (15 Points)

My main contribution in this lab was building two tools for web search and content extraction via the Tavily API, directly serving the ReAct loop of the agent.

### Modules Implemented

- **`src/tools/tavily_search.py`** — Web search tool using Tavily API
- **`src/tools/tavily_extract.py`** — Content extraction and summarization tool from URLs

---

### `tavily_search.py` — Web Search Tool

The `tavily_search` function provides web search capability in a "broad overview" style, returning a list of relevant URLs that the agent will use for further exploration.

```python
def tavily_search(query: str, max_results: int = 4, search_depth: str = "advanced") -> list:
    load_dotenv()
    # ...
    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        search_depth=search_depth,
        max_results=max_results,
    )
    results = response.get("results", [])
    return [res.get("url") for res in results if res.get("url")]
```

**Key Design Decisions:**
- Uses `search_depth="advanced"` by default to ensure result quality suitable for academic research.
- Two-tier safe error handling: checks for the library (`TavilyClient is None`) and verifies the API key before making calls.
- Returns a pure `list[str]` (URLs only), separating concerns — reading URL content is delegated to `tavily_extract.py`.

---

### `tavily_extract.py` — Content Extraction & Summarization Tool

This module is more complex, consisting of two dataclasses and a main class capable of calling the Tavily Extract API and then summarizing content via an LLM (OpenAI).

**Architecture:**

```
TavilyExtractTool
    ├── __init__()          # Initialize Tavily client + OpenAI client (optional)
    ├── _summarize()        # Call LLM to summarize raw content
    └── extract()           # Full pipeline: extract → (optional) summarize → return ExtractionResult

ExtractedContent            # Dataclass for each URL
ExtractionResult            # Aggregated dataclass for multiple URLs
```

**Key Snippet — `extract()` pipeline:**

```python
def extract(self, urls: list[str] | str, summarize: bool = True) -> ExtractionResult:
    if isinstance(urls, str):
        urls = [urls]

    response = self.client.extract(urls=urls)
    extraction = ExtractionResult()

    for item in response.get("results", []):
        url = item.get("url", "")
        raw = item.get("raw_content", "")
        summary = self._summarize(raw, url) if summarize and raw else None
        extraction.results.append(
            ExtractedContent(url=url, raw_content=raw, summary=summary, success=True)
        )
    # ... handle failed_results and URLs not present in the response
    return extraction
```

**Key Snippet — `_summarize()` using LLM:**

```python
def _summarize(self, text: str, url: str) -> str | None:
    truncated = truncate_to_tokens(text, self.max_tokens)  # ~4 chars/token
    response = self.llm_client.chat.completions.create(
        model=self.llm_model,
        max_tokens=self.max_tokens,
        messages=[
            {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Summarize the following content extracted from {url}:\n\n{truncated}"},
        ],
    )
    return response.choices[0].message.content.strip()
```

**Key Design Decisions:**
- `truncate_to_tokens()` limits input to ~1500 tokens (≈ 6000 characters) to avoid exceeding the LLM's context window.
- `summarize=False` allows the tool to operate in extract-only mode when no OpenAI key is available.
- Smart fallback: if `OPENAI_API_KEY` is not set, the tool still works and returns raw content.
- `ExtractedContent.preview()` provides safe output (max 500/1000 chars) for agent consumption.

---

### Integration into the ReAct Loop (via `tool_registry.py`)

The two tools are registered in the `TOOLS` list with wrapper functions conforming to the `fn(args: str) -> str` interface required by `ReActAgent`:

```python
def _search_tavily(query: str) -> str:
    urls = tavily_search(query)
    return "\n".join(f"- {url}" for url in urls)

def _fetch_tavily(url: str) -> str:
    global _extractor
    if _extractor is None:
        _extractor = TavilyExtractTool()   # lazy initialization, reuse instance
    result = _extractor.extract(url.strip(), summarize=True)
    return result.results[0].preview(max_chars=1000)
```

The agent follows this workflow: `search_tavily` → get URLs → `fetch_tavily` → read content → continue with `search_arxiv`/`search_pubmed`.

---

## II. Debugging Case Study (10 Points)

### Problem Description

During testing, the agent occasionally encountered a **tool not found (hallucination) error** when the LLM generated an incorrect tool name. For example, the LLM produced:

```
Action: tavily_search("LLM agents 2024")
```

instead of the correct registry name `search_tavily`. This caused `_execute_tool()` in `agent.py` to log a `HALLUCINATION_ERROR` event and return an error, breaking the agent's flow.

Additionally, another error occurred in `_fetch_tavily` when the agent called this tool with a URL that hadn't been validated from `search_tavily`, causing the Tavily Extract API to return `failed_results` for that URL. As a result, `result.results[0]` had `success=False`, and `preview()` returned the string `[Extraction failed: ...]` — this was **correct behavior**, but the agent couldn't properly handle this error message in the next step.

### Diagnosis

- **Root Cause 1 (Hallucination):** The initial system prompt listed tool names inconsistently. One part of the prompt said `search_tavily`, while another part said `tavily_search`. The LLM got confused and generated the wrong name.
- **Root Cause 2 (Extract failure):** The LLM sometimes selected PDF URLs or pages requiring JavaScript rendering — URLs that Tavily Extract does not support, returning errors.

### Solution

1. **Fixing hallucination:** Ensured that tool names in `tool_registry.py` (`search_tavily`, `fetch_tavily`) are fully consistent with the `Available tools:` section in the system prompt of `agent.py`. Using `tool['name']` directly from the `TOOLS` list to build the prompt (instead of hardcoding) eliminates this risk:

```python
tool_descriptions = "\n".join(
    [f"- {t['name']}: {t['description']}" for t in self.tools]
)
```

2. **Fixing extract failure:** Added instructions in the system prompt for the agent to prefer general HTML URLs (Wikipedia, blogs, news articles) over PDFs or JavaScript-heavy pages. Additionally, `ExtractedContent.preview()` returns clear error messages so the agent can decide to try a different URL.

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

### 1. Reasoning — The Role of the `Thought` Block

The `Thought` block in the ReAct agent functions as an **intermediate reasoning step** before taking action. While a Chatbot answers directly from parametric knowledge (which may be outdated), the ReAct agent "thinks" first in a format like:

> *"The user is asking about the latest research on LLM agents. I need to search the web first to get up-to-date information."*

Then it calls `search_tavily`. This helps the agent **avoid hallucination** with factual information (paper names, publication dates, experimental results) — a major weakness of pure Chatbots. The `Thought` block also helps the agent self-coordinate multi-step workflows: search → extract → academic search → synthesize.

### 2. Reliability — When is the Agent Worse than the Chatbot?

The agent is actually less effective than the Chatbot in the following cases:

| Scenario | Issue |
|---|---|
| Basic conceptual questions | Agent wastes 3-5 extra tool call steps for information the LLM already knows |
| Questions requiring quick answers | Latency increases significantly due to multiple sequential API calls |
| Tool errors or API rate limits | Agent gets stuck, loops, or exits with a `timeout` |
| Vague queries | Agent performs many unnecessary steps before realizing it can't find anything |

The Chatbot responds instantly with lower latency, making it suitable for general conversation or questions that don't require real-time information.

### 3. Observation — The Impact of Environment Feedback

`Observation` is the "environment feedback" after each tool call, appended directly to the prompt:

```
current_prompt += f"\n{content}\nObservation: {observation}\n"
```

This creates an **incremental learning loop**: each Observation provides additional context for the next generation. For example:

- Observation from `search_tavily` returns 4 URLs → agent selects the most relevant URL for `fetch_tavily`
- Observation from `fetch_tavily` returns a content summary → agent identifies the topic and decides keywords for `search_arxiv`

If the Observation is an error (inaccessible URL), the agent can adjust its strategy in the next step — something the Chatbot has no equivalent mechanism for.

---

## IV. Future Improvements (5 Points)

### Scalability — Asynchronous Processing

Currently, tool calls in the agent are performed **sequentially**. When searching multiple sources, each step must wait for the previous one to complete. To scale to production:

- **Async tool execution**: Use `asyncio` + `aiohttp` to call `search_tavily` and `search_arxiv` in parallel within the same step.
- **Tool queue**: Use a message queue (Redis/RabbitMQ) to dispatch tool calls, enabling horizontal scaling of tool workers independently from the agent.
- **Batch extraction**: `TavilyExtractTool.extract()` already supports multiple URLs, which can be leveraged to reduce the number of API round-trips.

### Safety — Controlling Agent Behavior

- **Supervisor LLM**: Add a "supervisor" LLM that inspects the output of each Thought-Action pair before execution, detecting abnormal behaviors (e.g., agent attempting to access URLs outside a whitelist).
- **Tool call budget**: Limit the number of times each tool can be called within a session (currently the system prompt has the rule "Each tool may only be called ONCE", but this is not enforced at the code level). A counter should be added in `_execute_tool()`.
- **Input sanitization**: Validate and sanitize arguments before calling tools to prevent injection via query strings.

### Performance — Optimizing Retrieval and Cost

- **Vector DB for tool retrieval**: With a large number of tools (>20), instead of listing all tools in the prompt, use embedding search (FAISS, Pinecone) to inject only the N most relevant tools for the user's query — significantly reducing token consumption.
- **Caching layer**: Cache `tavily_search` results by query hash in Redis (TTL ~1 hour) to avoid repeated API calls for the same query.
- **Streaming responses**: Implement streaming LLM output so the Thought block displays in real-time instead of waiting for the entire response, noticeably improving UX.

---

