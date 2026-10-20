"""Characterization tests locking in the original rule-based behavior of agents.py."""

import pytest

from agents import commander_agent, mission_agent, risk_agent, security_agent

JAMMING = "Possible GPS jamming detected."
GPS_UNSTABLE = "GPS signal instability detected."
WEAK_SIGNAL = "Communication signal is weak."


def make_telemetry(base, low_battery=False, jamming=False, weak_signal=False, gps_unstable=False):
    return {
        **base,
        "battery_level": 18 if low_battery else 80,
        "possible_jamming": jamming,
        "signal_strength": 22 if weak_signal else 90,
        "gps_signal": "unstable" if gps_unstable else "stable",
    }


# (low_battery, jamming, weak_signal, gps_unstable) -> (score, level); weights 30/40/20/10.
RISK_TABLE = [
    ((False, False, False, False), 0, "Low"),
    ((False, False, False, True), 10, "Low"),
    ((False, False, True, False), 20, "Low"),
    ((False, False, True, True), 30, "Low"),
    ((False, True, False, False), 40, "Medium"),
    ((False, True, False, True), 50, "Medium"),
    ((False, True, True, False), 60, "Medium"),
    ((False, True, True, True), 70, "High"),
    ((True, False, False, False), 30, "Low"),
    ((True, False, False, True), 40, "Medium"),
    ((True, False, True, False), 50, "Medium"),
    ((True, False, True, True), 60, "Medium"),
    ((True, True, False, False), 70, "High"),
    ((True, True, False, True), 80, "High"),
    ((True, True, True, False), 90, "High"),
    ((True, True, True, True), 100, "High"),
]


@pytest.mark.parametrize("flags, expected_score, expected_level", RISK_TABLE)
def test_risk_agent_full_truth_table(nominal_telemetry, flags, expected_score, expected_level):
    telemetry = make_telemetry(nominal_telemetry, *flags)
    level, score = risk_agent(security_agent(telemetry), telemetry)
    assert (level, score) == (expected_level, expected_score)


@pytest.mark.parametrize(
    "battery, expected",
    [
        (0, "Battery level is critically low."),
        (19.9, "Battery level is critically low."),
        (20, "Battery level is acceptable."),
        (100, "Battery level is acceptable."),
    ],
)
def test_mission_agent_battery_threshold(nominal_telemetry, battery, expected):
    assert mission_agent({**nominal_telemetry, "battery_level": battery}) == expected


def test_security_agent_reports_no_risks_for_nominal_telemetry(nominal_telemetry):
    assert security_agent(nominal_telemetry) == []


def test_security_agent_detects_gps_jamming(nominal_telemetry):
    assert security_agent({**nominal_telemetry, "possible_jamming": True}) == [JAMMING]


def test_security_agent_reports_all_risks_in_stable_order(nominal_telemetry):
    telemetry = make_telemetry(nominal_telemetry, jamming=True, weak_signal=True, gps_unstable=True)
    assert security_agent(telemetry) == [JAMMING, GPS_UNSTABLE, WEAK_SIGNAL]


@pytest.mark.parametrize("strength, weak", [(29, True), (29.9, True), (30, False), (31, False)])
def test_security_agent_signal_strength_threshold(nominal_telemetry, strength, weak):
    risks = security_agent({**nominal_telemetry, "signal_strength": strength})
    assert (WEAK_SIGNAL in risks) is weak


@pytest.mark.parametrize(
    "level, expected",
    [
        ("High", "Return to base immediately."),
        ("Medium", "Continue mission with caution and increase monitoring."),
        ("Low", "Continue mission normally."),
    ],
)
def test_commander_agent_policy(level, expected):
    assert commander_agent(level) == expected
