from __future__ import annotations

from typing import Any, Mapping

from opmodel.api import LocalOp, OpKind, OpProfile
from opmodel.hardware import HardwareSpec


class DispatchingOpModel:
    def __init__(self, estimators: Mapping[OpKind, Any]):
        self._estimators = dict(estimators)

    def predict(self, op: LocalOp, hardware: HardwareSpec) -> OpProfile:
        try:
            estimator = self._estimators[op.kind]
        except KeyError as exc:
            raise NotImplementedError(f"Unsupported op kind: {op.kind}") from exc
        return estimator.estimate(op, hardware)
