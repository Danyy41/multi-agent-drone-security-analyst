import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool


class DroneTelemetry(BaseModel):
    """Validated drone telemetry snapshot consumed by the detection rules in agents.py."""

    # Unknown fields are kept so richer telemetry feeds are not rejected.
    # Frozen so nothing (including agent tooling) can alter telemetry after validation.
    model_config = ConfigDict(extra="allow", frozen=True)

    drone_id: str = Field(min_length=1)
    battery_level: float = Field(ge=0, le=100, description="Battery charge in percent.")
    gps_signal: Literal["stable", "unstable"]
    signal_strength: float = Field(ge=0, le=100, description="Link quality in percent.")
    altitude: float = Field(description="Altitude in metres.")
    mission_status: str = Field(min_length=1)
    # Strict so a string such as "false" cannot be read as truthy by the jamming rule.
    possible_jamming: StrictBool

    def to_rule_input(self) -> dict:
        """Return the plain dict shape the rule functions in agents.py expect."""
        return self.model_dump()


def load_telemetry(path: str | Path) -> DroneTelemetry:
    """Load and validate a telemetry JSON file, raising pydantic.ValidationError if invalid."""
    with open(path, "r") as file:
        return DroneTelemetry.model_validate(json.load(file))
