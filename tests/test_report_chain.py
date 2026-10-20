from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from fake_chat import fake_model, narrative_message
from llm_config import MissingAPIKeyError
from report_chain import (
    CommanderNarrative,
    SecurityReport,
    build_narrative_chain,
    check_against_policy,
    generate_report,
    make_chat_model,
)
from telemetry import DroneTelemetry, load_telemetry
from tools import rule_findings

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "drone_telemetry.json"
HIGH_RISK_FACTS = {
    "drone_id": "DRONE-001",
    "mission_analysis": "Battery level is critically low.",
    "security_risks": [
        "Possible GPS jamming detected.",
        "GPS signal instability detected.",
        "Communication signal is weak.",
    ],
    "risk_level": "High",
    "risk_score": 100,
    "commander_decision": "Return to base immediately.",
}
SAFE_NARRATIVE = {
    "recommended_posture": "return_to_base",
    "decision": "Return to base immediately using inertial navigation.",
    "reasoning": "GPS jamming, an unstable GPS fix and a weak link at 18% battery make continued flight unsafe.",
    "recommended_actions": ["Switch to inertial navigation", "Initiate return-to-home", "Alert the ground station"],
}


@pytest.fixture
def sample():
    return load_telemetry(SAMPLE_PATH)


def facts_of(report: SecurityReport) -> dict:
    return report.model_dump(include=set(HIGH_RISK_FACTS))


def narrative(**overrides) -> CommanderNarrative:
    return CommanderNarrative(**{**SAFE_NARRATIVE, **overrides})


def blocking(report: SecurityReport) -> list[str]:
    return [issue.message for issue in report.policy_issues if issue.severity == "blocking"]


# --- Rules-only and consistent narratives ---------------------------------------------


def test_rules_only_report_needs_no_llm(sample):
    report = generate_report(sample)

    assert facts_of(report) == HIGH_RISK_FACTS
    assert report.narrative_source == "rules_only"
    assert report.recommended_actions == ["Return to base immediately."]
    assert report.reasoning.startswith("Rule-based assessment: High risk (100/100).")
    assert report.ai_recommendation is None and report.model is None


def test_consistent_llm_narrative_is_used_with_rule_facts(sample):
    report = generate_report(sample, fake_model(narrative_message(**SAFE_NARRATIVE)))

    assert facts_of(report) == HIGH_RISK_FACTS
    assert report.narrative_source == "llm"
    assert report.policy_issues == []
    assert report.reasoning == SAFE_NARRATIVE["reasoning"]
    assert report.recommended_actions == SAFE_NARRATIVE["recommended_actions"]
    assert report.ai_recommendation == SAFE_NARRATIVE["decision"]


def test_prompt_contains_verified_findings_policy_and_analyst_notes(sample):
    model = fake_model(narrative_message(**SAFE_NARRATIVE))

    generate_report(sample, model, agent_outputs={"risk": "RISK: High (100/100)\nAll factors present."})

    system, human = model.prompts[0]
    assert "at least as cautious as the required posture 'return_to_base'" in system.content
    assert '"risk_score": 100' in human.content
    assert "Possible GPS jamming detected." in human.content
    assert "Mandatory policy decision: Return to base immediately." in human.content
    assert "[risk] RISK: High (100/100)" in human.content
    assert model.bound_tools == [CommanderNarrative]


def test_llm_never_supplies_factual_fields(sample):
    # Extra keys an LLM might add are not part of the schema and cannot reach the report.
    report = generate_report(
        sample, fake_model(narrative_message(**SAFE_NARRATIVE, risk_score=5, risk_level="Low", drone_id="FAKE"))
    )

    assert facts_of(report) == HIGH_RISK_FACTS
    assert report.narrative_source == "llm"


def test_more_cautious_narrative_is_kept_but_flagged(sample):
    report = generate_report(
        sample, fake_model(narrative_message(**{**SAFE_NARRATIVE, "recommended_posture": "abort_mission"}))
    )

    assert report.narrative_source == "llm"
    assert report.commander_decision == "Return to base immediately."
    assert [issue.severity for issue in report.policy_issues] == ["warning"]


# --- Contradictory narratives -----------------------------------------------------------


def test_less_cautious_posture_is_blocked_and_cannot_override_policy(sample):
    unsafe = {
        "recommended_posture": "continue_normally",
        "decision": "Keep flying, the anomalies are minor.",
        "reasoning": "Signal issues are temporary.",
        "recommended_actions": ["Proceed with the mission"],
    }

    report = generate_report(sample, fake_model(narrative_message(**unsafe)))

    assert report.narrative_source == "llm_rejected"
    assert report.commander_decision == "Return to base immediately."
    assert report.recommended_actions == ["Return to base immediately."]
    assert "Signal issues are temporary." not in report.reasoning
    assert report.ai_recommendation == "Keep flying, the anomalies are minor."
    assert any("policy requires at least 'return_to_base'" in msg for msg in blocking(report))


def test_unsafe_decision_text_is_blocked_even_with_safe_posture(sample):
    report = generate_report(
        sample, fake_model(narrative_message(**{**SAFE_NARRATIVE, "decision": "Continue mission normally."}))
    )

    assert report.narrative_source == "llm_rejected"
    assert any("less cautious than the policy decision" in msg for msg in blocking(report))


def test_fabricated_risk_score_and_level_are_blocked(sample):
    report = generate_report(
        sample,
        fake_model(narrative_message(**{**SAFE_NARRATIVE, "reasoning": "This is a low risk situation with a risk score of 20."})),
    )

    assert report.narrative_source == "llm_rejected"
    assert report.risk_score == 100 and report.risk_level == "High"
    assert any("cited risk score 20" in msg for msg in blocking(report))
    assert any("cited Low risk" in msg for msg in blocking(report))


@pytest.mark.parametrize(
    "overrides, expected_blocking",
    [
        ({}, 0),
        ({"reasoning": "Verified risk is 100/100 and high risk."}, 0),
        ({"recommended_posture": "continue_with_caution"}, 1),
        ({"recommended_actions": ["Continue mission with caution and increase monitoring"]}, 1),
        ({"reasoning": "Medium risk overall."}, 1),
        ({"reasoning": "Score 70/100."}, 1),
    ],
)
def test_check_against_policy_for_high_risk(sample, overrides, expected_blocking):
    issues = check_against_policy(narrative(**overrides), rule_findings(sample))
    assert sum(issue.severity == "blocking" for issue in issues) == expected_blocking


def test_low_risk_allows_continue_normally(nominal_telemetry):
    telemetry = DroneTelemetry.model_validate(nominal_telemetry)
    low = {
        "recommended_posture": "continue_normally",
        "decision": "Continue mission normally.",
        "reasoning": "No threats detected and risk score is 0.",
        "recommended_actions": ["Maintain routine monitoring"],
    }

    report = generate_report(telemetry, fake_model(narrative_message(**low)))

    assert report.narrative_source == "llm"
    assert (report.risk_level, report.risk_score) == ("Low", 0)
    assert report.policy_issues == []


# --- Malformed responses and failures -------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        AIMessage(content="Return to base immediately. Risk is high."),
        narrative_message(decision="Return to base.", reasoning="Jamming."),
        narrative_message(**{**SAFE_NARRATIVE, "recommended_posture": "hover_and_wait"}),
        narrative_message(**{**SAFE_NARRATIVE, "recommended_actions": []}),
        narrative_message(**{**SAFE_NARRATIVE, "recommended_actions": "Return home"}),
        narrative_message(**{**SAFE_NARRATIVE, "decision": ""}),
    ],
    ids=["plain-text", "missing-fields", "invalid-posture", "no-actions", "actions-not-a-list", "empty-decision"],
)
def test_malformed_response_falls_back_to_rules(sample, message):
    report = generate_report(sample, fake_model(message))

    assert report.narrative_source == "llm_rejected"
    assert facts_of(report) == HIGH_RISK_FACTS
    assert report.recommended_actions == ["Return to base immediately."]
    assert len(blocking(report)) == 1


def test_model_error_falls_back_to_rules(sample):
    report = generate_report(sample, fake_model())  # no scripted messages: the model raises

    assert report.narrative_source == "llm_rejected"
    assert facts_of(report) == HIGH_RISK_FACTS
    assert blocking(report)[0].startswith("AI narrative unavailable:")


# --- Real ChatOpenAI wiring (constructed offline, never invoked) ----------------------


def test_make_chat_model_requires_api_key():
    with pytest.raises(MissingAPIKeyError):
        make_chat_model()


def test_chat_openai_chain_builds_offline(monkeypatch):
    from langchain_openai import ChatOpenAI

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test-model")

    model = make_chat_model()
    chain = build_narrative_chain(model)

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-test-model"
    assert chain.first.input_variables == sorted(["findings", "policy_decision", "required_posture", "analyst_notes"])
