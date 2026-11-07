"""Parse Stanford Cars class names into make / model / year / body type.

Class names look like "Aston Martin V8 Vantage Coupe 2012": make is the first
token (or a known multi-word make), year is the last token, model is the rest.
"""
from __future__ import annotations

from dataclasses import dataclass

MULTI_WORD_MAKES = ("AM General", "Aston Martin", "Land Rover")

# Coarse body type derived from the model string; used for geometry analysis
# and as an extra grouping in the confusion / failure analysis.
_PICKUP_SUFFIXES = ("Crew Cab", "Extended Cab", "Regular Cab", "Club Cab", "Quad Cab", "SuperCab")
_BODY_TYPES = ("Sedan", "SUV", "Coupe", "Convertible", "Hatchback", "Wagon", "Minivan", "Van")


@dataclass(frozen=True)
class CarLabel:
    class_id: int  # 0-based
    name: str
    make: str
    model: str
    year: int
    body_type: str

    @property
    def make_model(self) -> str:
        return f"{self.make} {self.model}"


def parse_class_name(name: str, class_id: int = -1) -> CarLabel:
    name = " ".join(name.strip().split())
    tokens = name.split(" ")
    if not tokens[-1].isdigit() or len(tokens[-1]) != 4:
        raise ValueError(f"Class name has no trailing 4-digit year: {name!r}")
    year = int(tokens[-1])
    rest = " ".join(tokens[:-1])

    make = next((m for m in MULTI_WORD_MAKES if rest == m or rest.startswith(m + " ")), tokens[0])
    model = rest[len(make):].strip()
    if not model:
        raise ValueError(f"Could not separate make and model in {name!r}")

    return CarLabel(class_id, name, make, model, year, body_type_of(model))


def body_type_of(model: str) -> str:
    if model.endswith(_PICKUP_SUFFIXES):
        return "Pickup"
    last = model.split(" ")[-1]
    if last in _BODY_TYPES:
        return last
    if model.endswith("Cargo Van") or model.endswith("Passenger Van"):
        return "Van"
    return "Other"


def folder_name(class_name: str) -> str:
    """Folder name used by the Kaggle mirror ("Ram C/V ..." -> "Ram C-V ...")."""
    return class_name.replace("/", "-")
