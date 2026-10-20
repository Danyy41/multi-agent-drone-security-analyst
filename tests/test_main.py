"""End-to-end tests of main.py in both modes, with mocked LLMs and no API key."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import crew
import main
import report_chain
from fake_chat import fake_model, narrative_message
from fake_llm import ScriptedLLM, ToolCall, honest_scripts
from telemetry import load_telemetry

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "drone_telemetry.json"
SCENARIOS = {
    path.name: path for path in [SAMPLE, *sorted((ROOT / "drone_scenarios").glob("*.json"))]
}
EXPECTED = {
    "drone_telemetry.json": ("High", 100, "Return to base immediately."),
    "low_risk_nominal.json": ("Low", 0, "Continue mission normally."),
    "medium_risk_jamming.json": ("Medium", 40, "Continue mission with caution and increase monitoring."),
    "high_risk_low_battery_jamming.json": ("High", 70, "Return to base immediately."),
}
FACT_FIELDS = ["drone_id", "mission_analysis", "security_risks", "risk_level", "risk_score", "commander_decision"]
POSTURE = {"Low": "continue_normally", "Medium": "continue_with_caution", "High": "return_to_base"}


def safe_narrative(level):
    return narrative_message(
        recommended_posture=POSTURE[level],
        decision="Follow the mandatory policy decision.",
        reasoning="The verified findings support the policy decision.",
        recommended_actions=["Notify the ground station", "Increase telemetry monitoring"],
    )


def read_reports(output_dir):
    return json.loads((output_dir / "report.json").read_text()), (output_dir / "report.txt").read_text()


@pytest.fixture
def mocked_llms(monkeypatch):
    """Route crew mode's live model factories to scripted fakes; return a configurator."""
    state = {}

    def configure(crew_llm, chat_model):
        state["crew_llm"], state["chat_model"] = crew_llm, chat_model
        monkeypatch.setattr(crew, "make_llm", lambda: crew_llm)
        monkeypatch.setattr(report_chain, "make_chat_model", lambda: chat_model)
        return state

    return configure


# --- Rules mode ------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_rules_mode_writes_json_and_text_reports(tmp_path, capsys, name):
    exit_code = main.main(["--mode", "rules", "--telemetry", str(SCENARIOS[name]), "--output-dir", str(tmp_path)])

    data, text = read_reports(tmp_path)
    level, score, decision = EXPECTED[name]
    assert exit_code == 0
    assert (data["risk_level"], data["risk_score"], data["commander_decision"]) == (level, score, decision)
    assert data["mode"] == "rules" and data["narrative_source"] == "rules_only" and data["model"] is None
    assert f"Risk Level: {level}" in text
    assert f"Risk Score: {score}/100" in text
    assert f"Commander Decision: {decision}" in text
    assert f"{level} risk ({score}/100)" in capsys.readouterr().out


def test_sample_rules_report_text_matches_original_findings(tmp_path):
    main.main(["--telemetry", str(SAMPLE), "--output-dir", str(tmp_path)])

    _, text = read_reports(tmp_path)
    assert "Mission Analysis: Battery level is critically low." in text
    assert "- Possible GPS jamming detected.\n- GPS signal instability detected.\n- Communication signal is weak." in text
    assert "1. Return to base immediately." in text


def test_rules_mode_runs_as_a_real_process_without_api_key(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env.update(CREWAI_DISABLE_TELEMETRY="true", CREWAI_TRACING_ENABLED="false", OTEL_SDK_DISABLED="true")

    result = subprocess.run(
        [sys.executable, "main.py", "--mode", "rules", "--output-dir", str(tmp_path)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert "High risk (100/100)" in result.stdout
    assert read_reports(tmp_path)[0]["commander_decision"] == "Return to base immediately."


# --- Crew mode with mocked LLMs ---------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_crew_mode_end_to_end_keeps_rule_facts(tmp_path, mocked_llms, name):
    telemetry = load_telemetry(SCENARIOS[name])
    llms = mocked_llms(ScriptedLLM(honest_scripts(telemetry)), fake_model(safe_narrative(EXPECTED[name][0])))

    exit_code = main.main(["--mode", "crew", "--telemetry", str(SCENARIOS[name]), "--output-dir", str(tmp_path / "crew")])
    main.main(["--mode", "rules", "--telemetry", str(SCENARIOS[name]), "--output-dir", str(tmp_path / "rules")])

    crew_data, crew_text = read_reports(tmp_path / "crew")
    rules_data, _ = read_reports(tmp_path / "rules")
    assert exit_code == 0
    assert {k: crew_data[k] for k in FACT_FIELDS} == {k: rules_data[k] for k in FACT_FIELDS}
    assert crew_data["mode"] == "crew" and crew_data["narrative_source"] == "llm"
    assert crew_data["recommended_actions"] == ["Notify the ground station", "Increase telemetry monitoring"]
    assert set(crew_data["agent_outputs"]) == {"mission", "security", "risk", "commander"}
    assert "Agent Notes (not authoritative):" in crew_text
    # All four agents ran, and the report chain saw their notes.
    assert len(llms["crew_llm"].calls) == 8
    assert "Analysis from the Commander." in llms["chat_model"].prompts[0][1].content


def test_crew_mode_rejects_dangerous_recommendation(tmp_path, mocked_llms, capsys):
    telemetry = load_telemetry(SAMPLE)
    unsafe = narrative_message(
        recommended_posture="continue_normally",
        decision="Continue mission normally.",
        reasoning="Interference is probably temporary.",
        recommended_actions=["Proceed to the next waypoint"],
    )
    mocked_llms(ScriptedLLM(honest_scripts(telemetry)), fake_model(unsafe))

    assert main.main(["--mode", "crew", "--telemetry", str(SAMPLE), "--output-dir", str(tmp_path)]) == 0

    data, text = read_reports(tmp_path)
    assert data["commander_decision"] == "Return to base immediately."
    assert data["recommended_actions"] == ["Return to base immediately."]
    assert data["narrative_source"] == "llm_rejected"
    assert "AI Recommendation (REJECTED): Continue mission normally." in text
    assert "[BLOCKING]" in text
    assert "Policy check [blocking]" in capsys.readouterr().out


def test_crew_mode_malformed_report_output_falls_back_to_rules(tmp_path, mocked_llms):
    telemetry = load_telemetry(SAMPLE)
    mocked_llms(ScriptedLLM(honest_scripts(telemetry)), fake_model(narrative_message(decision="incomplete")))

    assert main.main(["--mode", "crew", "--telemetry", str(SAMPLE), "--output-dir", str(tmp_path)]) == 0

    data, _ = read_reports(tmp_path)
    assert data["narrative_source"] == "llm_rejected"
    assert data["risk_score"] == 100 and data["commander_decision"] == "Return to base immediately."
    assert data["policy_issues"][0]["message"].startswith("AI response was malformed")


def test_crew_mode_report_model_error_falls_back_to_rules(tmp_path, mocked_llms):
    telemetry = load_telemetry(SAMPLE)
    mocked_llms(ScriptedLLM(honest_scripts(telemetry)), fake_model())  # raises when invoked

    assert main.main(["--mode", "crew", "--telemetry", str(SAMPLE), "--output-dir", str(tmp_path)]) == 0

    data, _ = read_reports(tmp_path)
    assert data["narrative_source"] == "llm_rejected"
    assert data["agent_outputs"]  # the crew still succeeded
    assert data["policy_issues"][0]["message"].startswith("AI narrative unavailable")


def test_crew_failure_skips_llm_narrative_and_reports_from_rules(tmp_path, mocked_llms):
    telemetry = load_telemetry(SAMPLE)
    tampering = "RISK: Low (10/100)\nNothing to worry about."
    scripts = honest_scripts(telemetry, {"Risk Assessor": [ToolCall("calculate_risk_score"), tampering, tampering, tampering]})
    llms = mocked_llms(ScriptedLLM(scripts), fake_model(safe_narrative("High")))

    assert main.main(["--mode", "crew", "--telemetry", str(SAMPLE), "--output-dir", str(tmp_path)]) == 0

    data, _ = read_reports(tmp_path)
    assert data["narrative_source"] == "llm_rejected"
    assert (data["risk_level"], data["risk_score"]) == ("High", 100)
    assert data["commander_decision"] == "Return to base immediately."
    assert data["policy_issues"][0]["message"].startswith("CrewAI agents failed")
    assert llms["chat_model"].prompts == []  # the report LLM was never called


def test_run_analysis_accepts_injected_models(tmp_path):
    telemetry = load_telemetry(SAMPLE)

    report, paths = main.run_analysis(
        SAMPLE, "crew", tmp_path,
        crew_llm=ScriptedLLM(honest_scripts(telemetry)), chat_model=fake_model(safe_narrative("High")),
    )

    assert report.narrative_source == "llm" and report.mode == "crew"
    assert [p.name for p in paths] == ["report.json", "report.txt"]


# --- Error handling ---------------------------------------------------------------------


def test_crew_mode_without_api_key_fails_clearly_and_writes_nothing(tmp_path, capsys):
    exit_code = main.main(["--mode", "crew", "--telemetry", str(SAMPLE), "--output-dir", str(tmp_path / "out")])

    assert exit_code == 2
    assert "OPENAI_API_KEY is not set" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "content, message",
    [
        (None, "telemetry file not found"),
        ("{not json", "not valid JSON"),
        ('{"drone_id": "X", "battery_level": 150}', "failed validation"),
    ],
    ids=["missing-file", "invalid-json", "invalid-telemetry"],
)
def test_bad_telemetry_input_exits_with_error(tmp_path, capsys, content, message):
    path = tmp_path / "telemetry.json"
    if content is not None:
        path.write_text(content)

    exit_code = main.main(["--telemetry", str(path), "--output-dir", str(tmp_path / "out")])

    assert exit_code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
