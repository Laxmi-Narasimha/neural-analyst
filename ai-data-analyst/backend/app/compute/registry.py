from __future__ import annotations

from typing import Any

from app.compute.operators.base import Operator
from app.compute.operators.eda import P0_EDA_OPERATORS


class OperatorRegistry:
    def __init__(self) -> None:
        self._ops: dict[str, Operator] = {}

    def register(self, op: Operator) -> None:
        self._ops[op.name] = op

    def get(self, name: str) -> Operator:
        if name not in self._ops:
            raise KeyError(name)
        return self._ops[name]

    def list(self) -> list[str]:
        return sorted(self._ops.keys())


def default_registry() -> OperatorRegistry:
    reg = OperatorRegistry()
    for op in P0_EDA_OPERATORS:
        reg.register(op)
    return reg

