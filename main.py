"""Command-line entry point for the Multi-Agent Drone Security Analyst.

--mode rules  Deterministic detection rules only. Offline, no API key, no cost.
--mode crew   CrewAI agents plus the LangChain report chain. Calls OpenAI and needs OPENAI_API_KEY.

Both modes take risk score, risk level, threats and the commander decision from the rules.
"""

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from llm_config import MissingAPIKeyError
from report import write_reports
from report_chain import PolicyIssue, SecurityReport, generate_report
from telemetry import load_telemetry

EXIT_OK = 0
EXIT_INPUT_ERROR = 2


def run_crew_mode(telemetry, crew_llm=None, chat_model=None) -> SecurityReport:
    """Run the CrewAI agents, then the LangChain report chain on their notes.

    crew_llm and chat_model default to the live OpenAI models; tests inject fakes.
    """
    from crew import DroneSecurityCrew, make_llm
    from report_chain import make_chat_model

    crew_llm = crew_llm or make_llm()
    chat_model = chat_model or make_chat_model()

    try:
        crew_run = DroneSecurityCrew(telemetry, llm=crew_llm).run()
    except Exception as error:
        # Without trustworthy agent output, skip the LLM narrative and report from the rules.
        issue = PolicyIssue(severity="blocking", message=f"CrewAI agents failed: {type(error).__name__}: {error}")
        report = generate_report(telemetry)
        return report.model_copy(update={"narrative_source": "llm_rejected", "policy_issues": [issue]})

    return generate_report(telemetry, chat_model, agent_outputs=crew_run.agent_outputs)


def run_analysis(telemetry_path, mode="rules", output_dir="outputs", crew_llm=None, chat_model=None):
    """Validate telemetry, build the report for the chosen mode and write it to output_dir."""
    telemetry = load_telemetry(telemetry_path)
    if mode == "crew":
        report = run_crew_mode(telemetry, crew_llm=crew_llm, chat_model=chat_model)
    else:
        report = generate_report(telemetry)
    report = report.model_copy(update={"mode": mode})
    return report, write_reports(report, output_dir)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["rules", "crew"], default="rules")
    parser.add_argument("--telemetry", default="drone_telemetry.json", help="Path to a telemetry JSON file.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for report.json and report.txt.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        report, (json_path, text_path) = run_analysis(args.telemetry, args.mode, args.output_dir)
    except FileNotFoundError as error:
        print(f"Error: telemetry file not found: {error.filename}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    except json.JSONDecodeError as error:
        print(f"Error: telemetry file is not valid JSON: {error}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    except ValidationError as error:
        print(f"Error: telemetry failed validation:\n{error}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    except MissingAPIKeyError as error:
        print(f"Error: {error}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    print(f"Drone {report.drone_id}: {report.risk_level} risk ({report.risk_score}/100)")
    print(f"Commander decision: {report.commander_decision}")
    print(f"Narrative source: {report.narrative_source}")
    for issue in report.policy_issues:
        print(f"Policy check [{issue.severity}]: {issue.message}")
    print(f"Reports written to {json_path} and {text_path}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
