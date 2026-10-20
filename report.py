"""Write a SecurityReport as structured JSON and as a readable text report."""

from pathlib import Path

from report_chain import SecurityReport

NARRATIVE_SOURCE_LABELS = {
    "llm": "AI commander (passed policy checks)",
    "rules_only": "Deterministic rules (no LLM used)",
    "llm_rejected": "Deterministic rules (AI narrative rejected)",
}


def render_text(report: SecurityReport) -> str:
    lines = [
        "MULTI-AGENT DRONE SECURITY ANALYSIS REPORT",
        "",
        f"Drone ID: {report.drone_id}",
        f"Mode: {report.mode}",
        f"Risk Level: {report.risk_level}",
        f"Risk Score: {report.risk_score}/100",
        f"Commander Decision: {report.commander_decision}",
        "",
        f"Mission Analysis: {report.mission_analysis}",
        "",
        "Security Risks:",
        *([f"- {risk}" for risk in report.security_risks] or ["- None detected"]),
        "",
        f"Reasoning [source: {NARRATIVE_SOURCE_LABELS[report.narrative_source]}]:",
        report.reasoning,
        "",
        "Recommended Actions:",
        *[f"{number}. {action}" for number, action in enumerate(report.recommended_actions, start=1)],
    ]

    if report.ai_recommendation is not None:
        status = "REJECTED" if report.narrative_source == "llm_rejected" else "accepted"
        lines += ["", f"AI Recommendation ({status}): {report.ai_recommendation}"]

    if report.policy_issues:
        lines += ["", "Policy Checks:"]
        lines += [f"- [{issue.severity.upper()}] {issue.message}" for issue in report.policy_issues]

    if report.model:
        lines += ["", f"Model: {report.model}"]

    if report.agent_outputs:
        lines += ["", "Agent Notes (not authoritative):"]
        for name, text in report.agent_outputs.items():
            lines += [f"[{name}]", text, ""]

    lines += [
        "",
        "Risk score, risk level, threats and commander decision are produced by deterministic rules.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_reports(report: SecurityReport, output_dir: str | Path) -> tuple[Path, Path]:
    """Write report.json and report.txt into output_dir and return their paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    text_path = output_dir / "report.txt"
    json_path.write_text(report.model_dump_json(indent=2) + "\n")
    text_path.write_text(render_text(report))
    return json_path, text_path
