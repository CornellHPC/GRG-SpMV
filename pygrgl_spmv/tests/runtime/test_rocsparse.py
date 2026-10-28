"""rocSPARSE explicit algorithm dispatch regressions."""


import numpy as np
import pytest

from pygrgl_spmv import testing
from pygrgl_spmv.backends.rocsparse import RocsparsePlan, RocsparsePlanPair, RocsparseRuntime, plan_rocsparse_layout
from pygrgl_spmv.tests.conftest import tol

pytestmark = pytest.mark.rocsparse


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("algo", ["DEFAULT", "COO_SEGMENTED", "COO_ATOMIC", "COO_SEGMENTED_ATOMIC"])
@pytest.mark.parametrize("spmv", [None, "DEFAULT", "COO", "COO_ATOMIC"])
def test_coo_algorithms_and_explicit_width_dispatch(primary_artifact, dtype, algo, spmv):
    pair = RocsparsePlanPair(
        RocsparsePlan("N", "COO", "N", "N", "ROW", "ROW", algo, spmv=spmv),
        RocsparsePlan("T", "COO", "N", "N", "ROW", "ROW", algo, spmv=spmv),
    )
    req = testing.requirements(max_k=4)
    layout = plan_rocsparse_layout(artifacts=[primary_artifact], pair=pair, requirements=req, dtype=dtype)
    with RocsparseRuntime(layout) as runtime, testing.load_reference(primary_artifact, req=req, dtype=dtype) as ref:
        for direction, n in (("up", ref.num_samples), ("down", ref.num_mutations)):
            for k in (1, 4):
                x = np.random.default_rng(17).normal(size=(k, n)).astype(dtype)
                np.testing.assert_allclose(runtime.grgs[0].matmul(x, direction), ref.matmul(x, direction),
                                           atol=tol(dtype)[0], rtol=tol(dtype)[1])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("spmv", [None, "DEFAULT", "CSR_ADAPTIVE", "CSR_ROWSPLIT", "CSR_LRB", "CSR_NNZSPLIT"])
def test_explicit_spmv_capture(primary_artifact, missing_artifact, dtype, spmv):
    from dataclasses import replace
    from pygrgl_spmv.backends.rocsparse.backend import standard_pair
    from pygrgl_spmv.tests.runtime.test_adaptor_capture import _check_native_capture_mixed_width_streams_and_teardown

    pair = standard_pair()
    # Adaptive/LRB require effective CSR; NNZ-split is supported for transpose only.
    down_fmt = "CSR" if spmv in {"CSR_ADAPTIVE", "CSR_LRB"} else "CSC"
    up_spmv = "CSR_ROWSPLIT" if spmv == "CSR_NNZSPLIT" else spmv
    pair = replace(pair, plan_up=replace(pair.plan_up, spmv=up_spmv),
                   plan_down=replace(pair.plan_down, fmt=down_fmt, spmv=spmv))
    _check_native_capture_mixed_width_streams_and_teardown(
        primary_artifact, missing_artifact, dtype, "rocsparse", pair)
