import os
import re
from typing import List, Dict, Any, Optional
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker
from langsmith import traceable

class ReActAgent:
    """
    SKELETON: A ReAct-style Agent that follows the Thought-Action-Observation loop.
    Students should implement the core loop logic and tool execution.
    """

    def __init__(self, llm: LLMProvider, tools: List[Dict[str, Any]], max_steps: int = 10):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.history = []

    def get_system_prompt(self, date_context: str = "") -> str:
        """
        Build the system prompt with tool descriptions and ReAct format instructions.
        """
        tool_descriptions = "\n".join(
            [f"- {t['name']}: {t['description']}" for t in self.tools]
        )
        date_line = f"\nFor your context, today is: {date_context}" if date_context else ""
        return f"""You are a general research assistant.{date_line}

Available tools:
{tool_descriptions}

Research workflow — follow this order:
1. search_tavily → get broad overview URLs on the topic
2. fetch_tavily → read the most relevant URL to understand key concepts
3. search_arxiv (for CS/AI/ML) or search_pubmed (for biomedical) → find academic papers
4. fetch_arxiv or fetch_pubmed → get full details of a specific paper

Format — follow this exactly:

Thought: <reasoning>
Action: tool_name("argument")

After receiving an Observation, continue or finish:

Thought: <final reasoning>
Final Answer: <your response>

Rules:
- Only use tools listed above. Only ONE Action per step.
- Each tool may only be called ONCE. Do not repeat a tool you already used.
- Always Thought before Action or Final Answer.
- If a tool returns an error, try a different approach.
- If you can answer without tools, go directly to Final Answer.

Output format for Final Answer:
- Start with a clear, concise summary of findings
- Use bullet points or numbered lists for key points
- Include a "References" section at the end with:
  - Paper titles and ArXiv IDs or PubMed PMIDs found
  - URLs of web sources consulted
  - Format each reference as: [Title](URL) or [Title] (ArXiv: ID / PMID: ID)"""

    @traceable(name="ReActAgent.run")
    def run(self, user_input: str) -> str:
        """
        Execute the ReAct loop: Thought -> Action -> Observation until Final Answer or max_steps.
        """
        logger.log_event("AGENT_START", {"input": user_input, "model": self.llm.model_name})

        # Get current date for context if get_current_date tool is available
        date_context = ""
        for tool in self.tools:
            if tool['name'] == 'get_current_date':
                try:
                    date_context = tool['function']("")
                except Exception:
                    pass
                break

        current_prompt = user_input
        steps = 0
        last_response = ""
        exit_reason = "timeout"

        while steps < self.max_steps:
            # Generate LLM response
            try:
                result = self.llm.generate(current_prompt, system_prompt=self.get_system_prompt(date_context=date_context))
                content = result["content"]
            except Exception as e:
                logger.log_event("LLM_ERROR", {"step": steps + 1, "error": str(e)})
                last_response = f"Error communicating with LLM: {str(e)}"
                exit_reason = "llm_error"
                break
            last_response = content

            # Track metrics
            tracker.track_request(
                provider=result.get("provider", "unknown"),
                model=self.llm.model_name,
                usage=result.get("usage", {}),
                latency_ms=result.get("latency_ms", 0)
            )

            logger.log_event("AGENT_STEP", {
                "step": steps + 1,
                "response": content[:500],
                "usage": result.get("usage", {})
            })

            # Parse response
            parsed = self._parse_response(content)

            # Final Answer found
            if parsed["type"] == "final_answer":
                logger.log_event("AGENT_END", {"steps": steps + 1, "status": "final_answer"})
                return parsed["content"]

            # Action found — execute tool
            if parsed["type"] == "action":
                observation = self._execute_tool(parsed["tool_name"], parsed["args"])

                # Append the full exchange to the prompt
                current_prompt += f"\n{content}\nObservation: {observation}\n"
                steps += 1
                continue

            # Parse error — no Action or Final Answer found
            logger.log_event("PARSE_ERROR", {
                "step": steps + 1,
                "response": content[:500]
            })
            exit_reason = "parse_error"
            break

        # Max steps exceeded or parse error
        logger.log_event("AGENT_END", {"steps": steps, "status": exit_reason})
        return last_response if last_response else "Agent could not produce an answer."

    @traceable(name="ReActAgent._execute_tool")
    def _execute_tool(self, tool_name: str, args: str) -> str:
        """
        Execute a tool by name with the given arguments string.
        Returns the tool result as a string.
        """
        for tool in self.tools:
            if tool['name'] == tool_name:
                try:
                    result = tool['function'](args)
                    logger.log_event("TOOL_SUCCESS", {
                        "tool": tool_name,
                        "args": args,
                        "result_preview": str(result)[:200]
                    })
                    return str(result)
                except Exception as e:
                    logger.log_event("TOOL_ERROR", {
                        "tool": tool_name,
                        "args": args,
                        "error": str(e)
                    })
                    return f"Error calling {tool_name}: {str(e)}"

        logger.log_event("HALLUCINATION_ERROR", {
            "tool": tool_name,
            "available_tools": [t['name'] for t in self.tools]
        })
        return f"Error: Tool '{tool_name}' not found. Available tools: {', '.join(t['name'] for t in self.tools)}"

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM output for Final Answer or Action.
        Returns dict with 'type' key: 'final_answer', 'action', or 'error'.
        """
        # Check for Final Answer first
        final_match = re.search(r'Final Answer:\s*(.*)', response, re.DOTALL)
        if final_match:
            return {
                "type": "final_answer",
                "content": final_match.group(1).strip()
            }

        # Check for Action: tool_name("args") or tool_name(args)
        action_match = re.search(r'Action:\s*(\w+)\(([^)]*)\)', response)
        if action_match:
            tool_name = action_match.group(1).strip()
            raw_args = action_match.group(2).strip()
            # Strip surrounding quotes if present
            if (raw_args.startswith('"') and raw_args.endswith('"')) or \
               (raw_args.startswith("'") and raw_args.endswith("'")):
                raw_args = raw_args[1:-1]
            return {
                "type": "action",
                "tool_name": tool_name,
                "args": raw_args
            }

        return {"type": "error", "content": response}
