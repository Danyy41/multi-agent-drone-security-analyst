def mission_agent(telemetry):
    if telemetry["battery_level"] < 20:
        return "Battery level is critically low."
    return "Battery level is acceptable."


def security_agent(telemetry):
    risks = []

    if telemetry["possible_jamming"]:
        risks.append("Possible GPS jamming detected.")

    if telemetry["gps_signal"] == "unstable":
        risks.append("GPS signal instability detected.")

    if telemetry["signal_strength"] < 30:
        risks.append("Communication signal is weak.")

    return risks


def risk_agent(risks, telemetry):
    risk_score = 0

    if telemetry["battery_level"] < 20:
        risk_score += 30

    if telemetry["possible_jamming"]:
        risk_score += 40

    if telemetry["signal_strength"] < 30:
        risk_score += 20

    if telemetry["gps_signal"] == "unstable":
        risk_score += 10

    if risk_score >= 70:
        return "High", risk_score
    elif risk_score >= 40:
        return "Medium", risk_score
    else:
        return "Low", risk_score


def commander_agent(risk_level):
    if risk_level == "High":
        return "Return to base immediately."
    elif risk_level == "Medium":
        return "Continue mission with caution and increase monitoring."
    else:
        return "Continue mission normally."
