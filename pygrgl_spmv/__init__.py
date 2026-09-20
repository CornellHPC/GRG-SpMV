"""Runtime-owned GRG sparse matmul package."""

from pygrgl_spmv.backends.mkl import MklPlan, MklPlanPair, MklRuntime, plan_mkl_layout
from pygrgl_spmv.backends.reference import ReferencePlan, ReferencePlanPair, ReferenceRuntime, plan_reference_layout
from pygrgl_spmv.grg import RuntimeRequirements, simple_convert
from pygrgl_spmv.adaptor import (
    CapturedBoundGRG,
    CaptureSpec,
    RunConfigs,
    MklBackendConfig,
    CusparseBackendConfig,
    make_backend_mkl,
    make_backend_cusparse,
    make_runconfig_kernel,
    make_runconfig_pca,
    make_runconfig_bolt,
    make_runconfig_gwas,
    load_grg_spmv_single,
    load_grg_spmv_multi,
)

__all__ = [
    "CaptureSpec",
    "CapturedBoundGRG",
    "CusparseBackendConfig",
    "MklBackendConfig",
    "MklPlan",
    "MklPlanPair",
    "MklRuntime",
    "ReferencePlan",
    "ReferencePlanPair",
    "ReferenceRuntime",
    "RunConfigs",
    "RuntimeRequirements",
    "load_grg_spmv_multi",
    "load_grg_spmv_single",
    "make_backend_cusparse",
    "make_backend_mkl",
    "make_runconfig_bolt",
    "make_runconfig_gwas",
    "make_runconfig_kernel",
    "make_runconfig_pca",
    "plan_mkl_layout",
    "plan_reference_layout",
    "simple_convert",
]
