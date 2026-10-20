"""Scripted stand-in for an LLM so CrewAI agents can be tested without any API calls."""

import json
from dataclasses import dataclass, field
from typing import Any

from crewai.llms.base_llm import BaseLLM
from pydantic import PrivateAttr

from crew import fact_lines
from tools import rule_findings

ROLES = ["Mission Analyst", "Security Analyst", "Risk Assessor", "Commander"]
TOOL_FOR_ROLE = {
    "Mission Analyst": "check_mission_status",
    "Security Analyst": "detect_security_threats",
    "Risk Assessor": "calculate_risk_score",
    "Commander": "get_command_policy",
}
FACT_FOR_ROLE = {"Mission Analyst": "mission", "Security Analyst": "security", "Risk Assessor": "risk", "Commander": "commander"}


@dataclass
class ToolCall:
    name: str
    arguments: dict = field(default_factory=dict)


@dataclass
class RecordedCall:
    role: str
    messages: list[dict]
    tools: list[dict] | None


class ScriptedLLM(BaseLLM):
    """Replays a per-agent-role script of tool calls and final answers.

    native=True mimics a function-calling model (the path CrewAI uses for OpenAI);
    native=False emits ReAct-style text that CrewAI parses itself.
    """

    native: bool = True
    _scripts: dict[str, list[ToolCall | str]] = PrivateAttr(default_factory=dict)
    _calls: list[RecordedCall] = PrivateAttr(default_factory=list)

    def __init__(self, scripts: dict[str, list[ToolCall | str]], **kwargs: Any) -> None:
        kwargs.setdefault("model", "scripted-test-model")
        super().__init__(**kwargs)
        self._scripts = {role: list(steps) for role, steps in scripts.items()}

    @property
    def calls(self) -> list[RecordedCall]:
        return self._calls

    def remaining(self, role: str) -> int:
        return len(self._scripts.get(role, []))

    def supports_function_calling(self) -> bool:
        return self.native

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None) -> Any:
        role = from_agent.role if from_agent is not None else "<no agent>"
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        self._calls.append(RecordedCall(role=role, messages=[dict(m) for m in messages], tools=tools))

        steps = self._scripts.get(role)
        if not steps:
            raise AssertionError(f"Unexpected LLM call for agent role {role!r}: script exhausted.")
        step = steps.pop(0)

        if isinstance(step, ToolCall):
            if self.native:
                return [{
                    "id": f"call_{len(self._calls)}",
                    "type": "function",
                    "function": {"name": step.name, "arguments": json.dumps(step.arguments)},
                }]
            return (
                f"Thought: I need the {step.name} result.\n"
                f"Action: {step.name}\n"
                f"Action Input: {json.dumps(step.arguments)}"
            )
        if self.native:
            return step
        return f"Thought: I now know the final answer.\nFinal Answer: {step}"


def honest_scripts(telemetry, overrides=None) -> dict[str, list[ToolCall | str]]:
    """Each agent calls its tool, then answers with the exact fact line plus commentary."""
    facts = fact_lines(rule_findings(telemetry))
    scripts = {
        role: [ToolCall(TOOL_FOR_ROLE[role]), f"{facts[FACT_FOR_ROLE[role]]}\nAnalysis from the {role}."]
        for role in ROLES
    }
    scripts.update(overrides or {})
    return scripts
