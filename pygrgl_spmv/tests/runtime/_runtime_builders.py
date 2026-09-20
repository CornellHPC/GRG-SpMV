"""Small concrete helpers for building runtime layouts in tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pygrgl_spmv import RuntimeRequirements
from pygrgl_spmv.backends.mkl import MklPlan, MklPlanPair, MklRuntime, plan_mkl_layout
from pygrgl_spmv.backends.reference import ReferencePlan, ReferencePlanPair, ReferenceRuntime, plan_reference_layout
# The local copy was field-for-field identical. Aliased rather than renamed, because
# three call sites bind a local named `requirements` that would shadow it.
from pygrgl_spmv.testing import requirements as full_requirements

DEFAULT_GPU_BUDGET = 1_000_000_000


def reference_pair(
    *,
    plan_up: ReferencePlan | None = None,
    plan_down: ReferencePlan | None = None,
) -> ReferencePlanPair:
    return ReferencePlanPair(
        plan_up=ReferencePlan(store="N", fmt="CSR") if plan_up is None else plan_up,
        plan_down=ReferencePlan(store="T", fmt="CSC") if plan_down is None else plan_down,
    )


def mkl_pair(
    *,
    plan_up: MklPlan | None = None,
    plan_down: MklPlan | None = None,
) -> MklPlanPair:
    return MklPlanPair(
        plan_up=MklPlan(store="N", fmt="CSR", n_threads=1) if plan_up is None else plan_up,
        plan_down=MklPlan(store="T", fmt="CSC", n_threads=1) if plan_down is None else plan_down,
    )


def cusparse_pair(
    *,
    plan_up=None,
    plan_down=None,
):
    from pygrgl_spmv.backends.cusparse import CusparsePlan, CusparsePlanPair

    return CusparsePlanPair(
        plan_up=(
            CusparsePlan(
                store="N",
                fmt="CSR",
                op_a="N",
                op_b="N",
                order_b="ROW",
                order_c="ROW",
                algo="DEFAULT",
                scratch="none",
            )
            if plan_up is None
            else plan_up
        ),
        plan_down=(
            CusparsePlan(
                store="T",
                fmt="CSC",
                op_a="N",
                op_b="N",
                order_b="ROW",
                order_c="ROW",
                algo="DEFAULT",
                scratch="none",
            )
            if plan_down is None
            else plan_down
        ),
    )


def build_reference_layout(
    artifacts,
    *,
    dtype=np.float64,
    requirements: RuntimeRequirements | None = None,
    pair: ReferencePlanPair | None = None,
):
    return plan_reference_layout(
        artifacts=[Path(path) for path in artifacts],
        pair=reference_pair() if pair is None else pair,
        dtype=np.dtype(dtype),
        requirements=full_requirements() if requirements is None else requirements,
    )


def build_mkl_layout(
    artifacts,
    *,
    dtype=np.float64,
    requirements: RuntimeRequirements | None = None,
    pair: MklPlanPair | None = None,
):
    return plan_mkl_layout(
        artifacts=[Path(path) for path in artifacts],
        pair=mkl_pair() if pair is None else pair,
        dtype=np.dtype(dtype),
        requirements=full_requirements() if requirements is None else requirements,
    )


def build_cusparse_layout(
    artifacts,
    *,
    dtype=np.float64,
    requirements: RuntimeRequirements | None = None,
    pair=None,
    vram_budget_bytes: int = DEFAULT_GPU_BUDGET,
    ring_buffer_size: int = 0,
    allow_residency: bool = True,
    device: int = 0,
    stream=0,
):
    from pygrgl_spmv.backends.cusparse import plan_cusparse_layout

    return plan_cusparse_layout(
        artifacts=[Path(path) for path in artifacts],
        pair=cusparse_pair() if pair is None else pair,
        dtype=np.dtype(dtype),
        requirements=full_requirements() if requirements is None else requirements,
        vram_budget_bytes=int(vram_budget_bytes),
        ring_buffer_size=int(ring_buffer_size),
        allow_residency=bool(allow_residency),
        device=device,
        stream=stream,
    )


def nonreference_backend_cases():
    """Both backends unconditionally, selected by marks.

    Probing here instead dropped the cusparse param outright on a GPU-less host, so
    `--backend=cusparse` reported success having run no GPU case at all -- and the
    probe ran at import, before the collection hook, importing torch and cupy even
    under `--backend=mkl`.
    """
    return (
        pytest.param("mkl", id="mkl", marks=pytest.mark.mkl),
        pytest.param("cusparse", id="cusparse", marks=[pytest.mark.gpu, pytest.mark.cusparse]),
    )


def build_layout_for_backend(backend_name: str, artifacts, **kwargs):
    match str(backend_name):
        case "reference":
            return build_reference_layout(artifacts, **kwargs)
        case "mkl":
            return build_mkl_layout(artifacts, **kwargs)
        case "cusparse":
            return build_cusparse_layout(artifacts, **kwargs)
        case _:
            raise ValueError(f"unknown backend {backend_name!r}")


def runtime_cls_for_backend(backend_name: str):
    match str(backend_name):
        case "reference":
            return ReferenceRuntime
        case "mkl":
            return MklRuntime
        case "cusparse":
            from pygrgl_spmv.backends.cusparse import CusparseRuntime

            return CusparseRuntime
        case _:
            raise ValueError(f"unknown backend {backend_name!r}")
