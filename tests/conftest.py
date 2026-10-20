import socket

import pytest

LLM_KEY_ENV_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LANGCHAIN_API_KEY", "LANGSMITH_API_KEY")


@pytest.fixture(autouse=True)
def no_api_keys_or_network(monkeypatch):
    """Guarantee tests never use real credentials or reach a paid API."""
    for name in LLM_KEY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    def blocked(*args, **kwargs):
        raise RuntimeError("Network access is disabled during tests.")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def nominal_telemetry():
    return {
        "drone_id": "DRONE-TEST",
        "battery_level": 80,
        "gps_signal": "stable",
        "signal_strength": 90,
        "altitude": 100,
        "mission_status": "active",
        "possible_jamming": False,
    }
