import itertools
import json
from pathlib import Path

import pytest
from crewai.tools import BaseTool

from agents import commander_agent, mission_agent, risk_agent, security_agent
from telemetry import DroneTelemetry, load_telemetry
from tools import build_tools, rule_findings

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "drone_telemetry.json"
TOOL_NAMES = {"check_mission_status", "detect_security_threats", "calculate_risk_score", "get_command_policy"}


@pytest.fixture
def sample():
    return load_telemetry(SAMPLE_PATH)


def run_tool(tools, name, **kwargs):
    return json.loads(tools[name].run(**kwargs))


def all_flag_combinations(base):
    """Yield telemetry for every combination of the four risk conditions."""
    for low_battery, jamming, weak_signal, gps_unstable in itertools.product([False, True], repeat=4):
        yield DroneTelemetry.model_validate(
            {
                **base,
                "battery_level": 18 if low_battery else 80,
                "possible_jamming": jamming,
                "signal_strength": 22 if weak_signal else 90,
                "gps_signal": "unstable" if gps_unstable else "stable",
            }
        )


def test_build_tools_returns_crewai_tools_with_unique_names(sample):
    tools = build_tools(sample)
    assert set(tools) == TOOL_NAMES
    for tool in tools.values():
        assert isinstance(tool, BaseTool)
        assert tool.description


def test_sample_telemetry_tool_outputs(sample):
    tools = build_tools(sample)

    assert run_tool(tools, "check_mission_status") == {
        "drone_id": "DRONE-001",
        "battery_level": 18,
        "battery_assessment": "Battery level is critically low.",
        "mission_status": "active",
        "altitude": 120,
    }
    assert run_tool(tools, "detect_security_threats") == {
        "drone_id": "DRONE-001",
        "threats": [
            "Possible GPS jamming detected.",
            "GPS signal instability detected.",
            "Communication signal is weak.",
        ],
        "threat_count": 3,
        "possible_jamming": True,
        "gps_signal": "unstable",
        "signal_strength": 22,
    }
    assert run_tool(tools, "calculate_risk_score")["risk_level"] == "High"
    assert run_tool(tools, "calculate_risk_score")["risk_score"] == 100
    assert run_tool(tools, "get_command_policy")["rule_based_decision"] == "Return to base immediately."


def test_tools_agree_with_rule_functions_for_every_risk_combination(nominal_telemetry):
    for telemetry in all_flag_combinations(nominal_telemetry):
        data = telemetry.to_rule_input()
        risks = security_agent(data)
        level, score = risk_agent(risks, data)
        tools = build_tools(telemetry)

        assert run_tool(tools, "check_mission_status")["battery_assessment"] == mission_agent(data)
        assert run_tool(tools, "detect_security_threats")["threats"] == risks
        risk = run_tool(tools, "calculate_risk_score")
        assert (risk["risk_level"], risk["risk_score"], risk["security_risks"]) == (level, score, risks)
        assert run_tool(tools, "get_command_policy")["rule_based_decision"] == commander_agent(level)


def test_rule_findings_matches_original_pipeline(sample):
    assert rule_findings(sample) == {
        "drone_id": "DRONE-001",
        "mission_analysis": "Battery level is critically low.",
        "security_risks": [
            "Possible GPS jamming detected.",
            "GPS signal instability detected.",
            "Communication signal is weak.",
        ],
        "risk_level": "High",
        "risk_score": 100,
        "rule_based_decision": "Return to base immediately.",
    }


def test_llm_supplied_arguments_cannot_change_tool_results(sample):
    tools = build_tools(sample)
    honest = run_tool(tools, "calculate_risk_score")
    tampered = run_tool(tools, "calculate_risk_score", battery_level=100, possible_jamming=False, risk_score=0)
    assert tampered == honest


@pytest.mark.parametrize("name", sorted(TOOL_NAMES))
def test_structured_tool_used_by_crewai_agents_matches_direct_run(sample, name):
    tool = build_tools(sample)[name]
    structured = tool.to_structured_tool()

    assert structured.name == name
    assert json.loads(structured.invoke(input={})) == json.loads(tool.run())
    assert json.loads(structured.invoke(input={"risk_score": 0})) == json.loads(tool.run())


def test_tools_are_deterministic(sample):
    first = {name: tool.run() for name, tool in build_tools(sample).items()}
    second = {name: tool.run() for name, tool in build_tools(sample).items()}
    assert first == second
