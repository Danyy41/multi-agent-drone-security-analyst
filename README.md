# Multi-Agent Drone Security Analyst

A multi-agent AI workflow that analyzes drone telemetry, detects security and operational risks, calculates mission risk levels, and uses an LLM Commander Agent to recommend commander-level actions through automated GitHub Actions execution.

## Overview

This project demonstrates Agentic AI concepts by simulating a drone security monitoring system.

Multiple specialized agents collaborate to analyze telemetry data, assess mission risk, and generate a final mission decision.

## Agent Architecture

```text
Drone Telemetry
      ↓
Mission Agent
      ↓
Security Agent
      ↓
Risk Agent
      ↓
LLM Commander Agent
      ↓
Final Report
```

## Agents

### Mission Agent

Evaluates mission status and operational conditions.

Examples:

- Low battery detection
- Mission readiness assessment

### Security Agent

Identifies security threats.

Examples:

- GPS jamming detection
- Signal instability detection
- Weak communication detection

### Risk Agent

Calculates overall mission risk.

Outputs:

- Low Risk
- Medium Risk
- High Risk

### LLM Commander Agent

Uses an OpenAI language model to generate the final commander-level decision, reasoning, immediate actions, and risk explanation.

Examples:

- Continue mission
- Continue with caution
- Return to base immediately
- Abort mission and execute recovery procedure

## Project Structure

```text
multi-agent-drone-security-analyst
│
├── agents.py
├── llm_commander_agent.py
├── main.py
├── drone_telemetry.json
├── memory.json
├── requirements.txt
├── README.md
│
├── outputs/
│   └── report.txt
│
└── .github/
    └── workflows/
        └── drone-security.yml
```

## Example Input

```json
{
  "drone_id": "DRONE-001",
  "battery_level": 18,
  "gps_signal": "unstable",
  "signal_strength": 22,
  "altitude": 120,
  "mission_status": "active",
  "possible_jamming": true
}
```

## Example Output

```text
Mission Analysis:
Battery level is critically low.

Security Risks:
- Possible GPS jamming detected
- GPS signal instability detected
- Communication signal is weak

Risk Level:
High

Risk Score:
100

LLM Commander Decision:
Abort mission immediately and execute Return-to-Home using non-GNSS navigation if available.

Reasoning:
The drone is operating with low battery, unstable GPS, weak communications, and possible GPS jamming. These combined conditions increase the risk of loss of control or unsafe landing.
```

## GitHub Actions Automation

The system automatically executes whenever code is pushed to the repository.

Workflow:

```text
Push Event
      ↓
GitHub Actions
      ↓
Install Dependencies
      ↓
Run Multi-Agent Analysis
      ↓
Generate Report
```

## Agentic AI Concepts Demonstrated

- Multi-Agent Orchestration
- Agent Specialization
- LLM-Powered Decision Making
- Prompt Engineering
- Workflow Automation
- State Management
- Persistent Memory
- Risk Assessment
- Tool Usage
- Event-Driven Execution
- GitHub Actions Integration
- Secret Management with GitHub Actions

## Skills Demonstrated

- Python Development
- OpenAI API Integration
- Agent Workflow Design
- Autonomous Decision Pipelines
- Security Monitoring Concepts
- Drone Telemetry Analysis
- GitHub Actions
- CI/CD Automation
- System Architecture Design
- GenAI Application Development

## Future Enhancements

- Multiple telemetry scenarios
- Real drone telemetry streams
- Threat intelligence integration
- Drone fleet monitoring
- RAG-based mission knowledge base
- Multi-drone coordination
- Dashboard visualization
