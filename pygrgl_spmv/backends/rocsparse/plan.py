"""Concrete rocSPARSE plan types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping

import pygrgl

from pygrgl_spmv.backends.types import SparseFormat, StoredMatrix, parse_sparse_format, parse_store, transpose_compatible_format


class Operation(IntEnum):
    N = 111
    T = 112


class DenseOrder(IntEnum):
    ROW = 0
    COL = 1


class SpMMAlgorithm(IntEnum):
    DEFAULT = 0
    CSR = 1
    COO_SEGMENTED = 2
    COO_ATOMIC = 3
    CSR_ROW_SPLIT = 4
    CSR_NNZ_SPLIT = 5
    COO_SEGMENTED_ATOMIC = 6
    CSR_MERGE_PATH = 9


class SpMVAlgorithm(IntEnum):
    DEFAULT = 0
    COO = 1
    CSR_ADAPTIVE = 2
    CSR_ROWSPLIT = 3
    COO_ATOMIC = 5
    CSR_LRB = 7
    CSR_NNZSPLIT = 8


def _parse_operation(value) -> Operation:
    if isinstance(value, Operation):
        return value
    key = str(value).strip().upper()
    if key == "N":
        return Operation.N
    if key == "T":
        return Operation.T
    return Operation(int(key))


def _parse_dense_order(value) -> DenseOrder:
    if isinstance(value, DenseOrder):
        return value
    key = str(value).strip().upper()
    if key == "ROW":
        return DenseOrder.ROW
    if key == "COL":
        return DenseOrder.COL
    return DenseOrder(int(key))


def _parse_algo(value) -> SpMMAlgorithm:
    if isinstance(value, SpMMAlgorithm):
        return value
    key = str(value).strip().upper()
    return SpMMAlgorithm[key]


def _normalize_scratch(value: object) -> str:
    if value is None or str(value).strip().lower() == "none":
        return "none"
    raise ValueError("rocSPARSE supports only scratch='none'")


@dataclass(frozen=True)
class RocsparsePlan:
    store: StoredMatrix
    fmt: SparseFormat
    op_a: Operation
    op_b: Operation
    order_b: DenseOrder
    order_c: DenseOrder
    algo: SpMMAlgorithm
    scratch: str = "none"
    spmv: SpMVAlgorithm | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "store", parse_store(self.store))
        object.__setattr__(self, "fmt", parse_sparse_format(self.fmt))
        object.__setattr__(self, "op_a", _parse_operation(self.op_a))
        object.__setattr__(self, "op_b", _parse_operation(self.op_b))
        object.__setattr__(self, "order_b", _parse_dense_order(self.order_b))
        object.__setattr__(self, "order_c", _parse_dense_order(self.order_c))
        object.__setattr__(self, "algo", _parse_algo(self.algo))
        object.__setattr__(self, "scratch", _normalize_scratch(self.scratch))
        if self.spmv is not None:
            value = self.spmv
            object.__setattr__(self, "spmv", value if isinstance(value, SpMVAlgorithm)
                               else SpMVAlgorithm[str(value).upper()])
        if not self.supported:
            raise ValueError(f"unsupported rocSPARSE plan: {self}")

    @property
    def direction(self):
        return (
            pygrgl.TraversalDirection.UP
            if (self.store == StoredMatrix.N) == (self.op_a == Operation.N)
            else pygrgl.TraversalDirection.DOWN
        )

    @property
    def supported(self) -> bool:
        if self.fmt == SparseFormat.COO:
            mm = self.algo in {SpMMAlgorithm.DEFAULT, SpMMAlgorithm.COO_SEGMENTED,
                              SpMMAlgorithm.COO_ATOMIC, SpMMAlgorithm.COO_SEGMENTED_ATOMIC}
            mv = self.spmv in {None, SpMVAlgorithm.DEFAULT, SpMVAlgorithm.COO, SpMVAlgorithm.COO_ATOMIC}
        else:
            mm = self.algo in {SpMMAlgorithm.DEFAULT, SpMMAlgorithm.CSR,
                              SpMMAlgorithm.CSR_ROW_SPLIT, SpMMAlgorithm.CSR_NNZ_SPLIT,
                              SpMMAlgorithm.CSR_MERGE_PATH}
            mv = self.spmv not in {SpMVAlgorithm.COO, SpMVAlgorithm.COO_ATOMIC}
        # ROCm 10.1 returns not_implemented for adaptive/LRB with an
        # effective CSR transpose (including normal CSC). No silent dispatch.
        effective_transpose = (self.fmt == SparseFormat.CSC) != (self.op_a == Operation.T)
        if self.fmt != SparseFormat.COO and effective_transpose and self.spmv in {
            SpMVAlgorithm.CSR_ADAPTIVE, SpMVAlgorithm.CSR_LRB,
        }:
            mv = False
        return mm and mv

    @classmethod
    def from_dict(cls, mapping: Mapping[str, object]) -> "RocsparsePlan":
        allowed = {"store", "fmt", "opA", "opB", "orderB", "orderC", "algo", "scratch", "spmv"}
        extra = sorted(set(mapping) - allowed)
        if extra:
            raise ValueError(f"unknown RocsparsePlan field(s): {extra}")
        return cls(
            store=parse_store(mapping["store"]),
            fmt=parse_sparse_format(mapping["fmt"]),
            op_a=_parse_operation(mapping["opA"]),
            op_b=_parse_operation(mapping["opB"]),
            order_b=_parse_dense_order(mapping["orderB"]),
            order_c=_parse_dense_order(mapping["orderC"]),
            algo=_parse_algo(mapping["algo"]),
            scratch=mapping.get("scratch", "none"),
            spmv=mapping.get("spmv"),
        )

    def can_share_storage_with(self, other: "RocsparsePlan") -> bool:
        if self.store == other.store:
            return self.fmt == other.fmt
        if self.fmt == SparseFormat.COO and other.fmt == SparseFormat.COO:
            return False
        return transpose_compatible_format(self.fmt) == other.fmt


@dataclass(frozen=True)
class RocsparsePlanPair:
    plan_up: RocsparsePlan | None
    plan_down: RocsparsePlan | None

    def __post_init__(self) -> None:
        if self.plan_up is None and self.plan_down is None:
            raise ValueError("at least one of plan_up/plan_down must be provided")
        if self.plan_up is not None and str(getattr(self.plan_up.direction, "name", self.plan_up.direction)).lower() != "up":
            raise ValueError(f"rocSPARSE plan_up expects a UP plan, got {self.plan_up}")
        if self.plan_down is not None and str(getattr(self.plan_down.direction, "name", self.plan_down.direction)).lower() != "down":
            raise ValueError(f"rocSPARSE plan_down expects a DOWN plan, got {self.plan_down}")

    @classmethod
    def from_dicts(
        cls,
        plan_up: Mapping[str, object] | None,
        plan_down: Mapping[str, object] | None,
    ) -> "RocsparsePlanPair":
        return cls(
            plan_up=None if plan_up is None else RocsparsePlan.from_dict(plan_up),
            plan_down=None if plan_down is None else RocsparsePlan.from_dict(plan_down),
        )


__all__ = [
    "RocsparsePlan",
    "RocsparsePlanPair",
    "DenseOrder",
    "Operation",
    "SpMMAlgorithm",
    "SpMVAlgorithm",
]
