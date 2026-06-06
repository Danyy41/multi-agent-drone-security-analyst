from openai import OpenAI

client = OpenAI()

def llm_commander_agent(telemetry, risks, risk_level, risk_score):
    prompt = f"""
You are a drone security commander agent.

Analyze this drone telemetry and return:
1. Commander decision
2. Reasoning
3. Immediate action
4. Risk explanation

Telemetry:
{telemetry}

Detected risks:
{risks}

Risk level: {risk_level}
Risk score: {risk_score}
"""

    response = client.responses.create(
        model="gpt-5",
        input=prompt
    )

    return response.output_text
