from pathlib import Path

import pytest
from crewai import Process
from crewai.llms.base_llm import BaseLLM
from pydantic import ValidationError

from crew import DroneSecurityCrew, MissingAPIKeyError, make_llm
from fake_llm import ROLES, TOOL_FOR_ROLE, ScriptedLLM, ToolCall, honest_scripts
from telemetry import DroneTelemetry, load_telemetry
from tools import build_tools, rule_findings

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "drone_telemetry.json"
@pytest.fixture
def sample():
    return load_telemetry(SAMPLE_PATH)


def calls_for(llm, role):
    return [call for call in llm.calls if call.role == role]


def transcript(call):
    return "\n".join(str(message.get("content")) for message in call.messages)


def test_crew_has_four_agents_with_one_rule_tool_each(sample):
    crew = DroneSecurityCrew(sample, llm=ScriptedLLM({}))

    assert [agent.role for agent in crew.crew.agents] == ROLES
    for agent in crew.crew.agents:
        assert agent.goal and agent.backstory
        assert [tool.name for tool in agent.tools] == [TOOL_FOR_ROLE[agent.role]]
        assert agent.allow_delegation is False


def test_tasks_run_sequentially_and_chain_context(sample):
    crew = DroneSecurityCrew(sample, llm=ScriptedLLM({}))
    mission, security, risk, commander = crew.crew.tasks

    assert crew.crew.process == Process.sequential
    assert [task.agent.role for task in crew.crew.tasks] == ROLES
    assert security.context == [mission]
    assert risk.context == [mission, security]
    assert commander.context == [mission, security, risk]
    assert all(task.guardrail is not None for task in crew.crew.tasks)


@pytest.mark.parametrize("native", [True, False], ids=["native-tool-calling", "react-text"])
def test_full_crew_run_executes_real_rule_tools(sample, native):
    llm = ScriptedLLM(honest_scripts(sample), native=native)

    run = DroneSecurityCrew(sample, llm=llm).run()

    assert [call.role for call in llm.calls] == [role for role in ROLES for _ in range(2)]
    for role in ROLES:
        assert llm.remaining(role) == 0
        # The second LLM call of each agent must contain the real tool output.
        assert build_tools(sample)[TOOL_FOR_ROLE[role]].run() in transcript(calls_for(llm, role)[1])

    assert run.findings == rule_findings(sample)
    assert run.agent_outputs["risk"].splitlines()[0] == "RISK: High (100/100)"
    assert run.agent_outputs["commander"].splitlines()[0] == "DECISION: Return to base immediately."


def test_native_mode_offers_only_the_agents_own_tool(sample):
    llm = ScriptedLLM(honest_scripts(sample))
    DroneSecurityCrew(sample, llm=llm).run()

    for role in ROLES:
        offered = [tool["function"]["name"] for tool in calls_for(llm, role)[0].tools]
        assert offered == [TOOL_FOR_ROLE[role]]


def test_later_agents_receive_earlier_task_outputs_as_context(sample):
    llm = ScriptedLLM(honest_scripts(sample))
    DroneSecurityCrew(sample, llm=llm).run()

    commander_prompt = transcript(calls_for(llm, "Commander")[0])
    for role in ["Mission Analyst", "Security Analyst", "Risk Assessor"]:
        assert f"Analysis from the {role}." in commander_prompt


def test_guardrail_rejects_altered_risk_score_and_agent_corrects_it(sample):
    llm = ScriptedLLM(
        honest_scripts(
            sample,
            {
                "Risk Assessor": [
                    ToolCall("calculate_risk_score"),
                    "RISK: Low (10/100)\nThe drone looks fine.",
                    "RISK: High (100/100)\nCorrected to match the risk scoring tool.",
                ]
            },
        )
    )

    run = DroneSecurityCrew(sample, llm=llm).run()

    retry_prompt = transcript(calls_for(llm, "Risk Assessor")[-1])
    assert "RISK: High (100/100)" in retry_prompt
    assert "Do not change, estimate or soften tool results." in retry_prompt
    assert run.agent_outputs["risk"].startswith("RISK: High (100/100)")


def test_guardrail_blocks_less_cautious_commander_decision(sample):
    unsafe = "DECISION: Continue mission normally.\nRisk is acceptable."
    llm = ScriptedLLM(honest_scripts(sample, {"Commander": [ToolCall("get_command_policy"), unsafe, unsafe, unsafe]}))

    with pytest.raises(Exception, match="Task failed guardrail validation"):
        DroneSecurityCrew(sample, llm=llm).run()


def test_findings_never_come_from_llm_text(sample):
    llm = ScriptedLLM(
        honest_scripts(
            sample,
            {
                "Commander": [
                    ToolCall("get_command_policy"),
                    "DECISION: Return to base immediately.\nRisk score is 0 and the mission can continue normally.",
                ]
            },
        )
    )

    run = DroneSecurityCrew(sample, llm=llm).run()

    assert run.findings["risk_score"] == 100
    assert run.findings["rule_based_decision"] == "Return to base immediately."


def test_medium_risk_scenario_uses_its_own_facts(nominal_telemetry):
    telemetry = DroneTelemetry.model_validate({**nominal_telemetry, "possible_jamming": True})
    llm = ScriptedLLM(honest_scripts(telemetry))

    run = DroneSecurityCrew(telemetry, llm=llm).run()

    assert run.agent_outputs["risk"].splitlines()[0] == "RISK: Medium (40/100)"
    assert run.agent_outputs["commander"].splitlines()[0] == (
        "DECISION: Continue mission with caution and increase monitoring."
    )


def test_validated_telemetry_is_immutable(sample):
    with pytest.raises(ValidationError):
        sample.battery_level = 100


def test_crew_without_llm_requires_api_key(sample):
    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY is not set"):
        DroneSecurityCrew(sample)


def test_make_llm_uses_configured_openai_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test-model")

    llm = make_llm()

    assert isinstance(llm, BaseLLM)
    assert llm.model == "gpt-test-model"
