"""CrewAI tools exposing the deterministic detection rules in agents.py to LLM agents.

Each tool is bound to one validated DroneTelemetry snapshot and accepts no arguments
from the LLM, so an agent can only ask the rules to run on the real telemetry; it
cannot supply (or invent) the values the rules evaluate.
"""

import json

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict

from agents import commander_agent, mission_agent, risk_agent, security_agent
from telemetry import DroneTelemetry


# Any arguments an LLM passes are discarded before the rule runs.
# The docstring is included in the tool description shown to the LLM.
class NoArguments(BaseModel):
    """This tool takes no arguments; call it with an empty input."""

    model_config = ConfigDict(extra="ignore")


def rule_findings(telemetry: DroneTelemetry) -> dict:
    """Run every rule once and return the ground-truth findings for a telemetry snapshot."""
    data = telemetry.to_rule_input()
    security_risks = security_agent(data)
    risk_level, risk_score = risk_agent(security_risks, data)
    return {
        "drone_id": telemetry.drone_id,
        "mission_analysis": mission_agent(data),
        "security_risks": security_risks,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "rule_based_decision": commander_agent(risk_level),
    }


class TelemetryRuleTool(BaseTool):
    """Base class for tools that run a deterministic rule on bound telemetry."""

    args_schema: type[BaseModel] = NoArguments
    telemetry: DroneTelemetry

    def _run(self, **_ignored) -> str:
        return json.dumps(self.evaluate())

    def evaluate(self) -> dict:
        raise NotImplementedError


class MissionStatusTool(TelemetryRuleTool):
    name: str = "check_mission_status"
    description: str = (
        "Checks the drone's operational readiness from its telemetry: battery level "
        "(critically low below 20%), mission status and altitude. Takes no input."
    )

    def evaluate(self) -> dict:
        t = self.telemetry
        return {
            "drone_id": t.drone_id,
            "battery_level": t.battery_level,
            "battery_assessment": mission_agent(t.to_rule_input()),
            "mission_status": t.mission_status,
            "altitude": t.altitude,
        }


class ThreatDetectionTool(TelemetryRuleTool):
    name: str = "detect_security_threats"
    description: str = (
        "Runs rule-based threat detection on the drone's telemetry: possible GPS jamming, "
        "GPS signal instability and weak communication signal (below 30%). Takes no input."
    )

    def evaluate(self) -> dict:
        t = self.telemetry
        threats = security_agent(t.to_rule_input())
        return {
            "drone_id": t.drone_id,
            "threats": threats,
            "threat_count": len(threats),
            "possible_jamming": t.possible_jamming,
            "gps_signal": t.gps_signal,
            "signal_strength": t.signal_strength,
        }


class RiskScoringTool(TelemetryRuleTool):
    name: str = "calculate_risk_score"
    description: str = (
        "Calculates the authoritative mission risk score (0-100) and level (Low below 40, "
        "Medium 40-69, High 70+) from the drone's telemetry. Takes no input."
    )

    def evaluate(self) -> dict:
        findings = rule_findings(self.telemetry)
        return {key: findings[key] for key in ("drone_id", "risk_level", "risk_score", "security_risks")}


class CommandPolicyTool(TelemetryRuleTool):
    name: str = "get_command_policy"
    description: str = (
        "Returns the mandatory rule-based command decision for the drone's current risk "
        "level, which any recommendation must not be less cautious than. Takes no input."
    )

    def evaluate(self) -> dict:
        findings = rule_findings(self.telemetry)
        return {key: findings[key] for key in ("drone_id", "risk_level", "rule_based_decision")}


def build_tools(telemetry: DroneTelemetry) -> dict[str, TelemetryRuleTool]:
    """Create one instance of every rule tool bound to the given telemetry, keyed by name."""
    tools = [
        MissionStatusTool(telemetry=telemetry),
        ThreatDetectionTool(telemetry=telemetry),
        RiskScoringTool(telemetry=telemetry),
        CommandPolicyTool(telemetry=telemetry),
    ]
    return {tool.name: tool for tool in tools}
