"""Resident, single-GPU ROCm backend (CuPy, rocSPARSE, HIP VMM)."""

from .plan import RocsparsePlan, RocsparsePlanPair, DenseOrder, Operation, SpMMAlgorithm, SpMVAlgorithm
from .backend import RocsparseRuntime, RocsparseLayout, plan_rocsparse_layout

__all__ = [
    "RocsparsePlan", "RocsparsePlanPair", "RocsparseRuntime", "RocsparseLayout",
    "plan_rocsparse_layout", "DenseOrder", "Operation", "SpMMAlgorithm", "SpMVAlgorithm",
]
