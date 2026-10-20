"""CrewAI crew of four drone security agents backed by the deterministic rule tools.

Each agent calls its rule tool and reports the result. A guardrail on every task
rejects any answer that does not repeat the tool's facts verbatim, and the
authoritative findings are always taken from the rules, never parsed from LLM text.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Callable

# Opt out of CrewAI's anonymous telemetry and trace upload unless explicitly enabled.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from crewai import LLM, Agent, Crew, Process, Task  # noqa: E402
from crewai.llms.base_llm import BaseLLM  # noqa: E402
from crewai.tasks.task_output import TaskOutput  # noqa: E402

from telemetry import DroneTelemetry  # noqa: E402
from tools import build_tools, rule_findings  # noqa: E402

DEFAULT_MODEL = "gpt-5"
GUARDRAIL_MAX_RETRIES = 2


class MissingAPIKeyError(RuntimeError):
    """Raised when the crew needs a real LLM but no OpenAI API key is configured."""


def make_llm() -> LLM:
    """Create the OpenAI-backed LLM for live runs, configured from environment variables."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise MissingAPIKeyError(
            "OPENAI_API_KEY is not set. Export it (or put it in an untracked .env file) "
            "to run the CrewAI agents, or use the rule-based mode, which needs no key."
        )
    return LLM(model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL), api_key=api_key)


def fact_lines(findings: dict) -> dict[str, str]:
    """The exact line each agent must report, derived from the deterministic rules."""
    threats = "; ".join(findings["security_risks"]) or "None"
    return {
        "mission": f"BATTERY: {findings['mission_analysis']}",
        "security": f"THREATS: {threats}",
        "risk": f"RISK: {findings['risk_level']} ({findings['risk_score']}/100)",
        "commander": f"DECISION: {findings['rule_based_decision']}",
    }


def require_line(expected: str, tool_name: str) -> Callable[[TaskOutput], tuple[bool, Any]]:
    """Build a guardrail that only accepts answers containing the tool's exact fact line."""

    def guardrail(output: TaskOutput) -> tuple[bool, Any]:
        lines = [line.strip() for line in output.raw.splitlines()]
        if expected in lines:
            return True, output.raw
        return False, (
            f"Your answer must contain this exact line, as returned by the {tool_name} tool: "
            f"'{expected}'. Do not change, estimate or soften tool results."
        )

    return guardrail


@dataclass
class CrewRun:
    """Outcome of a crew run: rule facts are authoritative, agent text is commentary."""

    findings: dict
    agent_outputs: dict[str, str] = field(default_factory=dict)


@dataclass
class DroneSecurityCrew:
    """Builds the four agents and their sequential tasks for one telemetry snapshot."""

    telemetry: DroneTelemetry
    llm: BaseLLM | None = None
    verbose: bool = False

    def __post_init__(self) -> None:
        self.llm = self.llm or make_llm()
        self.findings = rule_findings(self.telemetry)
        self.tools = build_tools(self.telemetry)
        self.facts = fact_lines(self.findings)
        self.agents = self._build_agents()
        self.tasks = self._build_tasks()
        self.crew = Crew(
            agents=list(self.agents.values()),
            tasks=list(self.tasks.values()),
            process=Process.sequential,
            memory=False,
            tracing=False,
            verbose=self.verbose,
        )

    def _agent(self, role: str, goal: str, backstory: str, tool_name: str) -> Agent:
        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            tools=[self.tools[tool_name]],
            llm=self.llm,
            allow_delegation=False,
            max_iter=5,
            verbose=self.verbose,
        )

    def _build_agents(self) -> dict[str, Agent]:
        return {
            "mission": self._agent(
                role="Mission Analyst",
                goal="Assess the drone's operational readiness from verified telemetry.",
                backstory=(
                    "A UAV flight operations specialist who judges whether a drone can safely "
                    "continue its mission, relying only on measured telemetry."
                ),
                tool_name="check_mission_status",
            ),
            "security": self._agent(
                role="Security Analyst",
                goal="Identify GPS jamming, navigation and communication threats to the drone.",
                backstory=(
                    "An electronic warfare analyst experienced in GNSS interference and link "
                    "degradation, who reports only threats confirmed by detection rules."
                ),
                tool_name="detect_security_threats",
            ),
            "risk": self._agent(
                role="Risk Assessor",
                goal="Report the authoritative mission risk score and explain what drives it.",
                backstory=(
                    "A safety assessor who never estimates risk by intuition: the score comes "
                    "from the risk scoring rules, and the job is to explain it clearly."
                ),
                tool_name="calculate_risk_score",
            ),
            "commander": self._agent(
                role="Commander",
                goal="Issue a command decision that is at least as cautious as safety policy requires.",
                backstory=(
                    "A drone mission commander accountable for aircraft safety, who follows "
                    "the mandatory command policy and adds concrete immediate actions."
                ),
                tool_name="get_command_policy",
            ),
        }

    def _task(self, key: str, tool_name: str, description: str, context: list[Task]) -> Task:
        fact = self.facts[key]
        return Task(
            description=(
                f"Drone {self.telemetry.drone_id}: {description} "
                f"Call the {tool_name} tool and base your answer only on what it returns."
            ),
            expected_output=(
                f"The first line copied exactly from the tool result in the form '{fact.split(':')[0]}: ...', "
                "followed by a short analysis of at most five sentences."
            ),
            agent=self.agents[key],
            context=context,
            guardrail=require_line(fact, tool_name),
            guardrail_max_retries=GUARDRAIL_MAX_RETRIES,
        )

    def _build_tasks(self) -> dict[str, Task]:
        mission = self._task(
            "mission",
            "check_mission_status",
            "Assess battery level and mission readiness.",
            context=[],
        )
        security = self._task(
            "security",
            "detect_security_threats",
            "Identify GPS jamming, GPS instability and communication threats.",
            context=[mission],
        )
        risk = self._task(
            "risk",
            "calculate_risk_score",
            "Report the mission risk level and score and explain the contributing threats.",
            context=[mission, security],
        )
        commander = self._task(
            "commander",
            "get_command_policy",
            "Decide the mission command and list immediate actions for the operator.",
            context=[mission, security, risk],
        )
        return {"mission": mission, "security": security, "risk": risk, "commander": commander}

    def run(self) -> CrewRun:
        result = self.crew.kickoff()
        outputs = {key: output.raw for key, output in zip(self.tasks, result.tasks_output)}
        return CrewRun(findings=self.findings, agent_outputs=outputs)
