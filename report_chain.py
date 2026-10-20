"""LangChain pipeline that turns deterministic rule findings into a structured SecurityReport.

The LLM only writes the commander narrative (posture, decision wording, reasoning and
actions). Every factual field is copied from rule_findings(), the narrative is checked
against the deterministic command policy, and a narrative that is malformed, cites
wrong facts or is less cautious than policy is discarded rather than reported.
"""

import json
import re
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from agents import commander_agent
from llm_config import openai_settings
from telemetry import DroneTelemetry
from tools import rule_findings

Posture = Literal["continue_normally", "continue_with_caution", "return_to_base", "abort_mission"]

# Higher is more cautious.
POSTURE_SEVERITY: dict[str, int] = {
    "continue_normally": 0,
    "continue_with_caution": 1,
    "return_to_base": 2,
    "abort_mission": 3,
}
POLICY_POSTURE: dict[str, str] = {
    "Low": "continue_normally",
    "Medium": "continue_with_caution",
    "High": "return_to_base",
}


class CommanderNarrative(BaseModel):
    """Commander recommendation for a drone security report, based only on the verified findings."""

    recommended_posture: Posture = Field(
        description="Mission posture to adopt. Must be at least as cautious as the required posture."
    )
    decision: str = Field(min_length=1, description="One-sentence command decision for the operator.")
    reasoning: str = Field(min_length=1, description="Why this decision follows from the verified findings.")
    recommended_actions: list[str] = Field(
        min_length=1, max_length=8, description="Concrete immediate actions for the operator, most urgent first."
    )


class PolicyIssue(BaseModel):
    severity: Literal["blocking", "warning"]
    message: str


class SecurityReport(BaseModel):
    """Final report: facts from the deterministic rules, narrative from the LLM only if it passed checks."""

    drone_id: str
    mission_analysis: str
    security_risks: list[str]
    risk_level: str
    risk_score: int
    commander_decision: str = Field(description="Authoritative rule-based decision; never replaced by the LLM.")
    reasoning: str
    recommended_actions: list[str]
    ai_recommendation: str | None = None
    ai_posture: Posture | None = None
    narrative_source: Literal["llm", "rules_only", "llm_rejected"]
    policy_issues: list[PolicyIssue] = Field(default_factory=list)
    model: str | None = None
    mode: Literal["rules", "crew"] = "rules"
    agent_outputs: dict[str, str] = Field(default_factory=dict, description="CrewAI agent commentary (not authoritative).")


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the Commander of a drone security team. You write the command narrative for a "
            "security report. The verified findings below were produced by deterministic detection "
            "rules and are authoritative: do not change, re-estimate or contradict them, and do not "
            "introduce any risk score, risk level or threat that is not listed. Your recommended "
            "posture must be at least as cautious as the required posture '{required_posture}'.",
        ),
        (
            "human",
            "Verified findings:\n{findings}\n\n"
            "Mandatory policy decision: {policy_decision}\n\n"
            "Analyst notes from the agent crew (context only, not authoritative):\n{analyst_notes}\n\n"
            "Write the commander narrative.",
        ),
    ]
)


def make_chat_model() -> BaseChatModel:
    """Create the OpenAI chat model for live runs, configured from environment variables."""
    from langchain_openai import ChatOpenAI

    api_key, model = openai_settings()
    return ChatOpenAI(model=model, api_key=api_key)


def build_narrative_chain(chat_model: BaseChatModel) -> Runnable:
    """prompt | model with structured output; returns {'raw', 'parsed', 'parsing_error'}."""
    structured = chat_model.with_structured_output(
        CommanderNarrative, method="function_calling", include_raw=True
    )
    return PROMPT | structured


def _cited_scores(text: str) -> set[int]:
    patterns = [r"\b(\d{1,3})\s*/\s*100\b", r"risk score(?:\s+(?:of|is|was))?[\s:]*(\d{1,3})\b"]
    return {int(match) for pattern in patterns for match in re.findall(pattern, text, re.IGNORECASE)}


def _cited_levels(text: str) -> set[str]:
    patterns = [r"\b(low|medium|high)[\s-]+risk\b", r"risk level(?:\s+(?:of|is|was))?[\s:]*(low|medium|high)\b"]
    return {match.capitalize() for pattern in patterns for match in re.findall(pattern, text, re.IGNORECASE)}


def check_against_policy(narrative: CommanderNarrative, findings: dict) -> list[PolicyIssue]:
    """Compare an LLM narrative with the deterministic findings and command policy."""
    issues: list[PolicyIssue] = []
    level = findings["risk_level"]
    required = POLICY_POSTURE[level]
    proposed = narrative.recommended_posture

    if POSTURE_SEVERITY[proposed] < POSTURE_SEVERITY[required]:
        issues.append(PolicyIssue(
            severity="blocking",
            message=f"AI recommended '{proposed}' but policy requires at least '{required}' for {level} risk.",
        ))
    elif POSTURE_SEVERITY[proposed] > POSTURE_SEVERITY[required]:
        issues.append(PolicyIssue(
            severity="warning",
            message=f"AI recommended a more cautious posture '{proposed}' than policy '{required}'; "
                    "the policy decision is kept and the escalation is flagged for operator review.",
        ))

    commands = " ".join([narrative.decision, *narrative.recommended_actions]).lower()
    for other_level, posture in POLICY_POSTURE.items():
        weaker_decision = commander_agent(other_level).rstrip(".").lower()
        if POSTURE_SEVERITY[posture] < POSTURE_SEVERITY[required] and weaker_decision in commands:
            issues.append(PolicyIssue(
                severity="blocking",
                message=f"AI decision or actions say '{commander_agent(other_level)}', which is less "
                        f"cautious than the policy decision '{findings['rule_based_decision']}'.",
            ))

    text = " ".join([narrative.decision, narrative.reasoning, *narrative.recommended_actions])
    for score in sorted(_cited_scores(text) - {findings["risk_score"]}):
        issues.append(PolicyIssue(
            severity="blocking",
            message=f"AI cited risk score {score}, but the verified risk score is {findings['risk_score']}.",
        ))
    for cited in sorted(_cited_levels(text) - {level}):
        issues.append(PolicyIssue(
            severity="blocking",
            message=f"AI cited {cited} risk, but the verified risk level is {level}.",
        ))
    return issues


def _rule_based_reasoning(findings: dict) -> str:
    risks = " ".join(findings["security_risks"]) or "None detected."
    return (
        f"Rule-based assessment: {findings['risk_level']} risk ({findings['risk_score']}/100). "
        f"{findings['mission_analysis']} Threats: {risks}"
    )


def generate_report(
    telemetry: DroneTelemetry,
    chat_model: BaseChatModel | None = None,
    agent_outputs: dict[str, str] | None = None,
) -> SecurityReport:
    """Build the SecurityReport; without a chat model the report is purely rule-based."""
    findings = rule_findings(telemetry)
    facts = {
        "drone_id": findings["drone_id"],
        "mission_analysis": findings["mission_analysis"],
        "security_risks": findings["security_risks"],
        "risk_level": findings["risk_level"],
        "risk_score": findings["risk_score"],
        "commander_decision": findings["rule_based_decision"],
    }
    fallback = {
        "reasoning": _rule_based_reasoning(findings),
        "recommended_actions": [findings["rule_based_decision"]],
    }
    if agent_outputs:
        facts["agent_outputs"] = dict(agent_outputs)

    if chat_model is None:
        return SecurityReport(**facts, **fallback, narrative_source="rules_only")

    model_name = getattr(chat_model, "model_name", None) or getattr(chat_model, "model", None)
    inputs = {
        "findings": json.dumps(
            {k: v for k, v in facts.items() if k not in ("commander_decision", "agent_outputs")}, indent=2
        ),
        "policy_decision": findings["rule_based_decision"],
        "required_posture": POLICY_POSTURE[findings["risk_level"]],
        "analyst_notes": "\n\n".join(f"[{k}] {v}" for k, v in (agent_outputs or {}).items()) or "None.",
    }

    try:
        result = build_narrative_chain(chat_model).invoke(inputs)
    except Exception as error:  # the report must still be produced from the rules
        issue = PolicyIssue(severity="blocking", message=f"AI narrative unavailable: {type(error).__name__}: {error}")
        return SecurityReport(**facts, **fallback, narrative_source="llm_rejected", policy_issues=[issue], model=model_name)

    narrative = result.get("parsed")
    if narrative is None:
        error = result.get("parsing_error") or "no structured output was returned"
        issue = PolicyIssue(severity="blocking", message=f"AI response was malformed: {error}")
        return SecurityReport(**facts, **fallback, narrative_source="llm_rejected", policy_issues=[issue], model=model_name)

    issues = check_against_policy(narrative, findings)
    audit = {"ai_recommendation": narrative.decision, "ai_posture": narrative.recommended_posture}
    if any(issue.severity == "blocking" for issue in issues):
        return SecurityReport(
            **facts, **fallback, **audit, narrative_source="llm_rejected", policy_issues=issues, model=model_name
        )
    return SecurityReport(
        **facts,
        **audit,
        reasoning=narrative.reasoning,
        recommended_actions=narrative.recommended_actions,
        narrative_source="llm",
        policy_issues=issues,
        model=model_name,
    )
