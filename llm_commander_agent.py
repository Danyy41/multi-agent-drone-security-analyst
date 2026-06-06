def llm_commander_agent(telemetry, risks, risk_level, risk_score):
    prompt = f"""
You are a drone security commander agent.

Analyze this drone telemetry:

Drone ID: {telemetry["drone_id"]}
Battery Level: {telemetry["battery_level"]}
GPS Signal: {telemetry["gps_signal"]}
Signal Strength: {telemetry["signal_strength"]}
Possible Jamming: {telemetry["possible_jamming"]}

Detected Risks:
{risks}

Risk Level: {risk_level}
Risk Score: {risk_score}

Give a commander-level decision and explain why.
"""

    return {
        "commander_prompt": prompt,
        "commander_decision": "Return to base immediately due to GPS instability, possible jamming, weak signal, and critical battery level."
    }
