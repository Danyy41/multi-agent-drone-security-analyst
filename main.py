import json
from agents import mission_agent, security_agent, risk_agent, commander_agent

with open("drone_telemetry.json", "r") as file:
    telemetry = json.load(file)

mission_result = mission_agent(telemetry)
security_risks = security_agent(telemetry)
risk_level, risk_score = risk_agent(security_risks, telemetry)
command_decision = commander_agent(risk_level)

result = {
    "drone_id": telemetry["drone_id"],
    "mission_analysis": mission_result,
    "security_risks": security_risks,
    "risk_level": risk_level,
    "risk_score": risk_score,
    "commander_decision": command_decision
}

with open("memory.json", "w") as memory_file:
    json.dump(result, memory_file, indent=4)

with open("outputs/report.txt", "w") as report:
    report.write("MULTI-AGENT DRONE SECURITY ANALYSIS REPORT\n\n")
    report.write(f"Drone ID: {result['drone_id']}\n")
    report.write(f"Mission Analysis: {result['mission_analysis']}\n")
    report.write(f"Risk Level: {result['risk_level']}\n")
    report.write(f"Risk Score: {result['risk_score']}\n")
    report.write(f"Commander Decision: {result['commander_decision']}\n\n")
    report.write("Security Risks:\n")

    for risk in result["security_risks"]:
        report.write(f"- {risk}\n")

print(result)
