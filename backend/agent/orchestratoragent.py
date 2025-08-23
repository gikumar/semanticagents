from __future__ import annotations

import os
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()

try:
    import semantic_kernel as sk
    from semantic_kernel.connectors.ai.open_ai import AzureOpenAI
except Exception:  # pragma: no cover - fallback import hint
    sk = None  # type: ignore
    AzureOpenAI = None  # type: ignore


@dataclass
class AgentConfig:
    endpoint: str
    api_key: str
    deployment: str
    region: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1024

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
            deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            region=os.environ.get("AZURE_OPENAI_REGION"),
            temperature=float(os.environ.get("AZURE_OPENAI_TEMPERATURE", "0.2")),
            max_tokens=int(os.environ.get("AZURE_OPENAI_MAX_TOKENS", "1024")),
        )


class AzureFoundryAgent:
    """A compact agent that uses Semantic Kernel to plan, expand, and summarize.

    The implementation aims to be resilient to small SDK changes by providing
    a fallback REST client if the official connector is not importable.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.kernel = None
        self._ai_service = None
        self._ensure_kernel()
        self._register_skills()

    def _ensure_kernel(self) -> None:
        # Initialize kernel and AI service connector. Keep code tolerant if SDK missing.
        if sk is None or AzureOpenAI is None:
            # SDK not available: kernel will remain None; we'll use REST fallback when invoking.
            self.kernel = None
            self._ai_service = None
            return

        self.kernel = sk.Kernel()
        # Configure Azure OpenAI connector
        client = AzureOpenAI(
            endpoint=self.config.endpoint,
            api_key=self.config.api_key,
            deployment=self.config.deployment,
            default_temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        self._ai_service = client
        self.kernel.register_ai_service("azure_openai", client)

    # -----------------
    # Prompt templates
    # -----------------
    # Keep prompts concise but adjustable.
    PLANNER_PROMPT = (
        "You are an expert systems designer.\n"
        "Given the goal below, produce a numbered plan of 3-6 steps.\n\n"
        "Goal:\n{goal}\n\nPlan:"  # returns a numbered list
    )

    EXPAND_PROMPT = (
        "You are a software engineer. Expand the plan step into detailed actions and commands.\n"
        "Step:\n{step}\n\nContext:\n{context}\n\nDetailed expansion:"
    )

    SUMMARIZE_PROMPT = (
        "You are an assistant that summarizes results. Produce a 3-sentence summary.\n\n"
        "Results:\n{results}\n\nSummary:"
    )

    def _register_skills(self) -> None:
        # When semantic-kernel is available, register the prompt templates as semantic skills.
        if not self.kernel:
            return

        # Register planner
        planner = self.kernel.create_semantic_function(
            self.PLANNER_PROMPT,
            "planner",
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        self.kernel.register_semantic_function("foundry", "planner", planner)

        # Register expander
        expander = self.kernel.create_semantic_function(
            self.EXPAND_PROMPT,
            "expander",
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        self.kernel.register_semantic_function("foundry", "expander", expander)

        # Register summarizer
        summarizer = self.kernel.create_semantic_function(
            self.SUMMARIZE_PROMPT,
            "summarizer",
            max_tokens=300,
            temperature=0.1,
        )
        self.kernel.register_semantic_function("foundry", "summarizer", summarizer)

    # -----------------
    # Fallback REST call (very small wrapper)
    # -----------------
    async def _rest_call(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        """Simple REST fallback to Azure OpenAI completions/chat endpoint.

        Only used if the Semantic Kernel SDK isn't available.
        """
        import aiohttp
        import json

        # Determine endpoint url for chat completions
        base = self.config.endpoint.rstrip("/")
        # This URL pattern works for Azure OpenAI chat completions
        url = f"{base}/openai/deployments/{self.config.deployment}/chat/completions?api-version=2023-05-15"

        headers = {
            "Content-Type": "application/json",
            "api-key": self.config.api_key,
        }

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=json.dumps(payload)) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"Azure OpenAI REST call failed ({resp.status}): {body}")
                data = await resp.json()
                # Extract content: supports `choices` for chat
                choices = data.get("choices") or []
                if not choices:
                    return ""
                return choices[0].get("message", {}).get("content", "")

    # -----------------
    # Pipeline methods
    # -----------------
    async def plan(self, goal: str) -> List[str]:
        """Produce a numbered plan from the goal."""
        prompt = self.PLANNER_PROMPT.format(goal=goal)
        if self.kernel:
            result = await self.kernel.run_async("foundry.planner", input=prompt)
            text = result.get_output() if hasattr(result, "get_output") else str(result)
        else:
            text = await self._rest_call(prompt, max_tokens=300, temperature=self.config.temperature)

        # Parse numbered list into steps
        steps = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Accept lines like '1. Do X' or '1) Do X' or '1 - Do X'
            import re

            m = re.match(r"^\d+\s*[\.)\-]\s*(.+)$", line)
            if m:
                steps.append(m.group(1).strip())
            else:
                # If the line doesn't start with a number, but we currently have no steps,
                # treat entire block as a single step
                if not steps:
                    steps = [text.strip()]
                    break
        return steps

    async def expand_step(self, step: str, context: str = "") -> str:
        prompt = self.EXPAND_PROMPT.format(step=step, context=context)
        if self.kernel:
            result = await self.kernel.run_async("foundry.expander", input=prompt)
            text = result.get_output() if hasattr(result, "get_output") else str(result)
        else:
            text = await self._rest_call(prompt, max_tokens=600, temperature=self.config.temperature)
        return text.strip()

    async def summarize(self, results: str) -> str:
        prompt = self.SUMMARIZE_PROMPT.format(results=results)
        if self.kernel:
            result = await self.kernel.run_async("foundry.summarizer", input=prompt)
            text = result.get_output() if hasattr(result, "get_output") else str(result)
        else:
            text = await self._rest_call(prompt, max_tokens=200, temperature=0.1)
        return text.strip()

    async def run(self, goal: str, context: str = "") -> Dict[str, Any]:
        """Full pipeline: plan -> expand each step -> summarize."""
        plan_steps = await self.plan(goal)
        expansions: List[str] = []

        # Expand each step sequentially; for Foundry you may parallelize later.
        for step in plan_steps:
            expanded = await self.expand_step(step, context=context)
            expansions.append(f"Step: {step}\nExpansion:\n{expanded}")

        full_results = "\n\n".join(expansions)
        summary = await self.summarize(full_results)

        return {"goal": goal, "plan": plan_steps, "expansions": expansions, "summary": summary}


# -----------------
# Demo: run from CLI
# -----------------
async def _async_main():
    cfg = AgentConfig.from_env()
    if not cfg.endpoint or not cfg.api_key:
        raise RuntimeError(
            "Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY in environment."
        )

    agent = AzureFoundryAgent(cfg)

    # Example goal: adapt for your Foundry-specific task
    goal = (
        "Design a secure, scalable pipeline to ingest CSV telemetry, validate records, and store "
        "structured events in an Azure Data Lake. Include testing and monitoring steps."
    )

    out = await agent.run(goal, context="production; use secure credentials; compliance: GDPR")

    import json

    print(json.dumps(out, indent=2))


def main():
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
