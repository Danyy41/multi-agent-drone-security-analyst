from pathlib import Path

import pytest
from pydantic import ValidationError

from agents import mission_agent, risk_agent, security_agent
from telemetry import DroneTelemetry, load_telemetry

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "drone_telemetry.json"


def test_sample_telemetry_file_is_valid_and_scores_high():
    telemetry = load_telemetry(SAMPLE_PATH).to_rule_input()

    assert telemetry["drone_id"] == "DRONE-001"
    assert mission_agent(telemetry) == "Battery level is critically low."
    assert risk_agent(security_agent(telemetry), telemetry) == ("High", 100)


def test_valid_telemetry_round_trips_to_rule_input(nominal_telemetry):
    assert DroneTelemetry.model_validate(nominal_telemetry).to_rule_input() == nominal_telemetry


def test_extra_fields_are_preserved(nominal_telemetry):
    telemetry = DroneTelemetry.model_validate({**nominal_telemetry, "heading": 270})
    assert telemetry.to_rule_input()["heading"] == 270


@pytest.mark.parametrize(
    "field", ["drone_id", "battery_level", "gps_signal", "signal_strength", "altitude", "mission_status", "possible_jamming"]
)
def test_missing_required_field_is_rejected(nominal_telemetry, field):
    data = {k: v for k, v in nominal_telemetry.items() if k != field}
    with pytest.raises(ValidationError):
        DroneTelemetry.model_validate(data)


@pytest.mark.parametrize(
    "field, value",
    [
        ("battery_level", -1),
        ("battery_level", 101),
        ("signal_strength", -5),
        ("signal_strength", 150),
        ("gps_signal", "lost"),
        ("drone_id", ""),
        ("possible_jamming", "false"),
        ("possible_jamming", 1),
        ("battery_level", "low"),
    ],
)
def test_invalid_values_are_rejected(nominal_telemetry, field, value):
    with pytest.raises(ValidationError):
        DroneTelemetry.model_validate({**nominal_telemetry, field: value})


def test_load_telemetry_rejects_invalid_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"drone_id": "X", "battery_level": 50}')
    with pytest.raises(ValidationError):
        load_telemetry(bad)


def test_network_is_blocked_during_tests():
    import socket

    with pytest.raises(RuntimeError, match="Network access is disabled"):
        socket.create_connection(("api.openai.com", 443))
