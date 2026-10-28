"""Small adapters for vendor plans, buffers and prepared entry points."""

import importlib

import pytest

GPU_BACKENDS = tuple(pytest.param(name, id=name, marks=getattr(pytest.mark, name))
                     for name in ("cusparse", "rocsparse"))


def plan_types(backend):
    module = importlib.import_module(f"pygrgl_spmv.backends.{backend}")
    prefix = "Cusparse" if backend == "cusparse" else "Rocsparse"
    return getattr(module, prefix + "Plan"), getattr(module, prefix + "PlanPair")
