# Group Report: Lab 3 - Production-Grade Agentic System

- **Team Name**: E4
- **Team Members**: Truong Dang Gia Huy, Nguyen Xuan Mong, Pham Do Ngoc Minh, Tran Nhat Hoang, Nguyen Ngoc Thang, Nguyen Xuan Hai, Tran Minh Toan.
- **Deployment Date**: 2026-04-06

---

## 1. Executive Summary

Our team built a ReAct (Reason + Act) research agent that autonomously searches the web and academic databases to answer complex research questions. Compared to a baseline chatbot that relies solely on parametric knowledge, our agent retrieves real-time information with verifiable sources.

- **Success Rate**: 6/8 test sessions reached Final Answer (75%)
- **Key Outcome**: The agent solved multi-step research queries by combining web search (Tavily), academic paper retrieval (ArXiv, PubMed), and structured reasoning — producing answers with real 2025-2026 paper citations that the baseline chatbot could not provide.

---

## 2. System Architecture & Tooling

### 2.1 ReAct Loop Implementation

```
User Input
    |
    v
[Get current date for context]
    |
    v
+---> LLM generates Thought + Action
|         |
|         v
|     Execute tool (search/fetch)
|         |
|         v
|     Observation (tool result)
|         |
|         v
|     Append to prompt context
|         |
+----< Has Final Answer? -- No --> loop (max 10 steps)
          |
          Yes
          |
          v
      Return Final Answer
```

Key design decisions:
- **String-based parsing** with regex (`Action:\s*(\w+)\(([^)]*)\)` and `Final Answer:\s*(.*)`)
- **Prompt accumulation**: Each step appends the full LLM response + Observation to `current_prompt`, giving the LLM full history
- **Date context injection**: `get_current_date` is called once at the start of `run()` and injected into the system prompt, avoiding wasting a tool call step

### 2.2 Tool Definitions (Inventory)

| Tool Name | Input Format | Use Case |
| :--- | :--- | :--- |
| `get_current_date` | none | Provides current date context to the LLM |
| `search_tavily` | `query (str)` | Web search for broad overview URLs on a topic |
| `fetch_tavily` | `url (str)` | Fetch and summarize content from a specific URL |
| `search_arxiv` | `query (str)` | Search ArXiv for CS/AI/ML academic papers |
| `search_pubmed` | `query (str)` | Search PubMed for biomedical/life-science papers |
| `fetch_pubmed` | `PMID (str)` | Fetch full details of a specific PubMed article |

Research workflow enforced in system prompt:
1. `search_tavily` -> get broad overview URLs
2. `fetch_tavily` -> read the most relevant URL to understand key concepts
3. `search_arxiv` or `search_pubmed` -> find academic papers
4. `fetch_pubmed` -> get full details of a PubMed article (if needed)

### 2.3 LLM Providers Used

- **Primary**: GPT-4o (OpenAI)
- **Attempted**: Qwen 3.6 Plus (OpenRouter free tier) — rate-limited (429), switched to GPT-4o

---

## 3. Telemetry & Performance Dashboard

Metrics collected from `logs/2026-04-06.log` across 8 agent sessions.

**GPT-4o Pricing**: $2.50/1M input tokens, $1.25/1M cached input, $10.00/1M output tokens.

- **Average Latency (P50)**: ~1,900ms per LLM call
- **Max Latency (P99)**: ~6,200ms (long Final Answer generation with 600+ tokens)
- **Average Tokens per Task**: ~800-2,500 tokens per full agent run
- **Total Cost of Test Runs**: ~$0.086 across all GPT-4o sessions

### Detailed Session Metrics

| Session | Query | Steps | Input Tokens | Output Tokens | Latency (total) | Cost | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | "What is today's date?" (Qwen) | 1 | 188 | 82 | 2.9s | N/A | LLM Error (429) |
| 2 | "What is today's date?" (GPT-4o) | 2 | 371 | 44 | 3.7s | $0.0014 | Final Answer |
| 3 | "Find recent research papers about ReAct agents" | 2 | 1,556 | 536 | 7.5s | $0.0093 | Final Answer |
| 4 | "Latest advances in ReAct agents for AI research?" | 5 | 5,730 | 412 | 10.0s | $0.0184 | Final Answer |
| 5 | "AI Agents in bioinformatics" | 5 | 4,272 | 571 | 13.4s | $0.0164 | Final Answer |
| 6 | "try again" | 1 | 529 | 53 | 3.1s | $0.0019 | Parse Error |
| 7 | "search about AI in bioinformatics" | 5 | 5,047 | 849 | 16.1s | $0.0211 | Final Answer |
| 8 | "ai in bioinformatics" | 4 | 3,656 | 840 | 10.5s | $0.0175 | Final Answer |

---

## 4. Root Cause Analysis (RCA) - Failure Traces

### Case Study 1: Rate Limit Error (Provider Failure)

- **Input**: "What is today's date?"
- **Model**: qwen/qwen3.6-plus:free (OpenRouter)
- **Log Evidence**:
  ```json
  {"event": "LLM_ERROR", "data": {"step": 2, "error": "Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429}}"}}
  {"event": "AGENT_END", "data": {"steps": 1, "status": "llm_error"}}
  ```
- **Root Cause**: Free-tier model on OpenRouter hit upstream rate limit from Alibaba after just 1 successful call.
- **Fix**: Switched to OpenAI GPT-4o with a paid API key. The `LLMProvider` abstract class made this a one-line change in `OpenAIProvider.__init__`.

### Case Study 2: Parse Error (Ambiguous Input)

- **Input**: "try again"
- **Log Evidence**:
  ```json
  {"event": "AGENT_STEP", "data": {"step": 1, "response": "Thought: It seems there might have been a misunderstanding..."}}
  {"event": "PARSE_ERROR", "data": {"step": 1}}
  {"event": "AGENT_END", "data": {"steps": 0, "status": "parse_error"}}
  ```
- **Root Cause**: The input "try again" is ambiguous — the LLM generated a Thought asking for clarification but produced neither an Action nor a Final Answer. The regex parser could not find a valid action.
- **Fix**: The agent gracefully handles this by returning `last_response` to the user instead of crashing. This is by design — the `_parse_response` returns `{"type": "error"}` which triggers a clean exit.

### Case Study 3: Tool Repetition (Prompt Gap)

- **Input**: "AI Agents in bioinformatics"
- **Log Evidence**:
  ```json
  {"event": "TOOL_SUCCESS", "data": {"tool": "search_arxiv", "args": "AI agents in bioinformatics 2026", "result_preview": "No papers found."}}
  {"event": "TOOL_SUCCESS", "data": {"tool": "search_arxiv", "args": "AI in bioinformatics", "result_preview": "No papers found."}}
  ```
- **Root Cause**: Two issues:
  1. The LLM included "2026" in the ArXiv search query, which is too restrictive for academic search
  2. The system prompt lacked a fallback rule, so the agent retried `search_arxiv` instead of switching to `search_pubmed`
- **Fix**: Added two explicit rules to the system prompt:
  - "Do NOT include the current year or date in search queries for search_arxiv or search_pubmed"
  - "If search_arxiv returns no results, try search_pubmed instead of repeating the same tool"

---

## 5. Ablation Studies & Experiments

### Experiment 1: Prompt v1 vs Prompt v2

**Diff**: Added two rules to the system prompt:
1. "Do NOT include the current year or date in search queries for search_arxiv or search_pubmed. Use only topic keywords."
2. "If search_arxiv returns no results, try search_pubmed (or vice versa) instead of repeating the same tool."

**Results**:

| Metric | Prompt v1 (Session 5) | Prompt v2 (Session 8) |
| :--- | :--- | :--- |
| Query | "AI Agents in bioinformatics" | "ai in bioinformatics" |
| search_arxiv calls | 2 (violated one-call rule) | 0 |
| search_pubmed calls | 0 (never tried) | 1 (correctly used) |
| Academic papers found | 0 | 4 PubMed articles |
| Steps | 5 | 4 (more efficient) |
| Total tokens | 4,843 ($0.0164) | 4,496 ($0.0175) |

**Conclusion**: Prompt v2 reduced invalid tool calls by 100% and improved academic paper retrieval from 0 to 4 papers for biomedical topics. The agent also completed in fewer steps and tokens.

### Experiment 2: Chatbot vs Agent

Same prompt: "What are the latest advances in AI agents for bioinformatics?"

| Aspect | Chatbot Baseline | ReAct Agent | Winner |
| :--- | :--- | :--- | :--- |
| Sources | None (parametric memory only) | 4 ArXiv papers + 1 web article with URLs | **Agent** |
| Recency | Generic, outdated (mentions AlphaFold generally) | Real 2025-2026 papers (Latent-Y, BioAgent Bench) | **Agent** |
| Verifiability | No references, cannot fact-check | Every claim has a clickable link | **Agent** |
| Latency | ~1.5s (single LLM call) | ~10-16s (multi-step with tool calls) | **Chatbot** |
| Token cost | 509 tokens ($0.0048) | 4,496 tokens ($0.0175) | **Chatbot** |
| Simple Q&A | Correct | Correct (but slower) | **Draw** |

**Conclusion**: The agent decisively wins on multi-step research tasks requiring real-time data and verifiable sources. The chatbot wins on latency and cost for simple questions. In a production system, a routing layer could direct simple queries to the chatbot and complex research queries to the agent.

---

## 6. Production Readiness Review

- **Security**:
  - API keys managed via `.env` file (not hardcoded)
  - `.gitignore` excludes `venv/` and sensitive files
  - Tool arguments are string-only, reducing injection surface

- **Guardrails**:
  - `max_steps=10` prevents infinite loops and runaway billing
  - Each tool limited to one call per session
  - Try/except around every LLM call and tool execution
  - Graceful degradation on parse errors (returns last response)

- **Observability**:
  - Every event logged as structured JSON with timestamps
  - LLM metrics tracked: tokens, latency, cost per request
  - LangSmith integration via `@traceable` decorators for distributed tracing

- **Scaling Considerations**:
  - Transition to LangGraph for more complex agent workflows with branching
  - Async tool calls for parallel execution (e.g., search ArXiv and PubMed simultaneously)
  - Response caching for repeated queries
  - Token budget management to control costs at scale

---
