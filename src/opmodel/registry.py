from __future__ import annotations

from opmodel.models.roofline import RooflineModel


def create_model(name: str):
    if name == "roofline":
        return RooflineModel()
    raise ValueError(f"Unknown model: {name}")
