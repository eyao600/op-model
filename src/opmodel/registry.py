from __future__ import annotations

from opmodel.models.base import BaseModel
from opmodel.models.extended_roofline import ExtendedRooflineModel
from opmodel.models.roofline import RooflineModel


def create_model(name: str):
    if name == "base":
        return BaseModel()
    if name == "roofline":
        return RooflineModel()
    if name == "extended_roofline":
        return ExtendedRooflineModel()
    raise ValueError(f"Unknown model: {name}")
