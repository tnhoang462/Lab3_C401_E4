"""
Tavily Extract Tool - Extracts content from URLs and summarizes it
using an LLM for a student research agent.
"""

import os
import json
import logging
from dataclasses import dataclass, field
from tavily import TavilyClient
from dotenv import load_dotenv
from openai import OpenAI


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_TOKENS = 1500

SUMMARIZE_SYSTEM_PROMPT = (
    "You are a research assistant helping students. "
    "Summarize the following web page content clearly and concisely. "
    "Focus on the key ideas, findings, and facts. "
    "Use simple, academic-friendly language. "
    "Keep your summary under 300 words."
)


@dataclass
class ExtractedContent:
    """Represents content extracted (and optionally summarized) from a single URL."""
    url: str
    raw_content: str
    summary: str | None
    success: bool
    error: str | None = None

    def preview(self, max_chars: int = 500) -> str:
        if not self.success:
            return f"[Extraction failed: {self.error}]"
        text = self.summary or self.raw_content
        return text[:max_chars] + ("..." if len(text) > max_chars else "")


@dataclass
class ExtractionResult:
    """Aggregated result from extracting multiple URLs."""
    results: list[ExtractedContent] = field(default_factory=list)
    failed_urls: list[str] = field(default_factory=list)

    @property
    def successful_count(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def total_count(self) -> int:
        return len(self.results)

    def to_dict(self) -> dict:
        return {
            "total": self.total_count,
            "successful": self.successful_count,
            "failed_urls": self.failed_urls,
            "results": [
                {
                    "url": r.url,
                    "raw_content": r.raw_content,
                    "summary": r.summary,
                    "success": r.success,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


def truncate_to_tokens(text: str, max_tokens: int = MAX_TOKENS) -> str:
    """
    Approximate token truncation (~4 chars per token).
    Truncates text so it stays within the token budget.
    """
    approx_chars = max_tokens * 4
    if len(text) <= approx_chars:
        return text
    return text[:approx_chars] + f"... [truncated to ~{max_tokens} tokens]"


class TavilyExtractTool:
    """
    Wraps Tavily's extract API to pull content from URLs and
    optionally summarize it via an LLM for student research workflows.

    Usage:
        tool = TavilyExtractTool()

        # Extract + summarize (default)
        result = tool.extract(["https://en.wikipedia.org/wiki/Machine_learning"])

        # Extract only, no summarization
        result = tool.extract(["https://example.com"], summarize=False)

        for item in result.results:
            print(item.summary or item.raw_content)
    """

    def __init__(
        self,
        api_key: str | None = None,
        openai_api_key: str | None = None,
        max_tokens: int = MAX_TOKENS,
        llm_model: str = "gpt-4o-mini",
    ):
        load_dotenv()

        # Tavily setup
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Tavily API key is required. Pass it directly or set TAVILY_API_KEY env var."
            )
        self.client = TavilyClient(api_key=self.api_key)
        self.max_tokens = max_tokens

        # OpenAI setup for summarization
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.llm_model = llm_model
        if self.openai_api_key:
            self.llm_client = OpenAI(api_key=self.openai_api_key)
        else:
            self.llm_client = None
            logger.warning(
                "No OpenAI API key provided. Summarization will be disabled. "
                "Set OPENAI_API_KEY env var or pass openai_api_key to enable it."
            )

    def _summarize(self, text: str, url: str) -> str | None:
        """
        Summarize extracted text using the LLM.
        Returns the summary string, or None if summarization is unavailable.
        """
        if not self.llm_client:
            return None

        truncated = truncate_to_tokens(text, self.max_tokens)

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Summarize the following content extracted from {url}:\n\n"
                            f"{truncated}"
                        ),
                    },
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Summarization failed for {url}: {e}")
            return None

    def extract(self, urls: list[str] | str, summarize: bool = True) -> ExtractionResult:
        """
        Extract content from one or more URLs, with optional LLM summarization.

        Args:
            urls: A single URL string or a list of URLs to extract from.
            summarize: If True (default), each result is summarized via the LLM.
                       If False or no LLM client is configured, raw content is returned.

        Returns:
            ExtractionResult with extracted content, summaries, and metadata.
        """
        if isinstance(urls, str):
            urls = [urls]

        if not urls:
            logger.warning("No URLs provided for extraction.")
            return ExtractionResult()

        logger.info(f"Extracting content from {len(urls)} URL(s)...")

        try:
            response = self.client.extract(urls=urls)
        except Exception as e:
            logger.error(f"Tavily extract API call failed: {e}")
            return ExtractionResult(
                results=[
                    ExtractedContent(url=u, raw_content="", summary=None, success=False, error=str(e))
                    for u in urls
                ],
                failed_urls=urls,
            )

        extraction = ExtractionResult()
        successful_urls = set()

        for item in response.get("results", []):
            url = item.get("url", "")
            raw = item.get("raw_content", "")
            successful_urls.add(url)

            summary = None
            if summarize and raw:
                summary = self._summarize(raw, url)

            extraction.results.append(
                ExtractedContent(url=url, raw_content=raw, summary=summary, success=True)
            )

        for fail in response.get("failed_results", []):
            url = fail.get("url", "")
            error = fail.get("error", "Unknown error")
            extraction.failed_urls.append(url)
            extraction.results.append(
                ExtractedContent(url=url, raw_content="", summary=None, success=False, error=error)
            )

        # Catch any URLs not accounted for in the response
        for url in urls:
            if url not in successful_urls and url not in extraction.failed_urls:
                extraction.failed_urls.append(url)
                extraction.results.append(
                    ExtractedContent(
                        url=url, raw_content="", summary=None, success=False, error="No response from API"
                    )
                )

        logger.info(
            f"Extraction complete: {extraction.successful_count}/{extraction.total_count} succeeded."
        )
        return extraction
