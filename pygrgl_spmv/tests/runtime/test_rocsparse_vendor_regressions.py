"""rocSPARSE ABI and SDK regressions; shared semantics live in shared families."""

import subprocess
import sys
from pathlib import Path

import pytest

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from pygrgl_spmv.backends.types import SparseFormat
from pygrgl_spmv.backends.rocsparse.backend import standard_pair
from pygrgl_spmv.backends.rocsparse.ffi import Handle
from pygrgl_spmv.backends.rocsparse.operations import SparseOperation
from pygrgl_spmv.backends.rocsparse.vmm import SharedOnes


from pygrgl_spmv.backends.rocsparse import ffi


@pytest.mark.rocsparse
def test_library_uses_pip_sdk_without_an_override(monkeypatch):
    monkeypatch.delenv("ROCM_PATH", raising=False)
    root = subprocess.check_output(
        [sys.executable, "-m", "rocm_sdk", "path", "--root"], text=True, timeout=30,
    ).strip()
    ffi.library.cache_clear()
    try:
        assert Path(ffi.library()._name).resolve() == (Path(root) / "lib/librocsparse.so").resolve()
    finally:
        ffi.library.cache_clear()


@pytest.mark.parametrize("root", ["", "missing-sdk"])
def test_invalid_sdk_override_does_not_fall_back(monkeypatch, tmp_path, root):
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / root) if root else "")
    ffi.library.cache_clear()
    try:
        with pytest.raises(ImportError, match="root is empty|library not found"):
            ffi.library()
    finally:
        ffi.library.cache_clear()


@pytest.mark.parametrize("failure", ["timeout", "exit", "missing"])
def test_sdk_discovery_failure_is_an_installation_error(monkeypatch, failure):
    def fail(command, **kwargs):
        assert command == [sys.executable, "-m", "rocm_sdk", "path", "--root"]
        assert kwargs["timeout"] == 30
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 30)
        if failure == "exit":
            raise subprocess.CalledProcessError(1, command)
        raise OSError("missing interpreter")

    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.setattr(subprocess, "check_output", fail)
    ffi.library.cache_clear()
    try:
        with pytest.raises(ImportError, match="Install the pip ROCm SDK"):
            ffi.library()
    finally:
        ffi.library.cache_clear()


@pytest.mark.rocsparse
def test_failed_capture_drops_graphs_before_stack_unwinds(primary_artifact):
    from contextlib import ExitStack
    from pygrgl_spmv import CaptureSpec, RunConfigs, load_grg_spmv_multi, make_backend_rocsparse, testing

    # The second duplicate spec fails after one successful graph capture.
    # Retaining the traceback must not retain a graph referencing freed state.
    spec = CaptureSpec("up", 1)
    with pytest.raises(ValueError, match="Duplicate CaptureSpec") as error:
        with ExitStack() as stack:
            load_grg_spmv_multi([primary_artifact], make_backend_rocsparse(capture=True),
                               RunConfigs(testing.requirements(), (spec, spec)), stack)
    frames = [entry.frame.f_locals for entry in error.traceback if entry.name == "capture_grg"]
    assert len(frames) == 1
    assert frames[0]["graphs"] == {}
    assert frames[0]["graph"] is None


@pytest.mark.rocsparse
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("k", [1, 2, 10])
@pytest.mark.parametrize("algo", ["DEFAULT", "CSR", "CSR_NNZ_SPLIT", "CSR_ROW_SPLIT", "CSR_MERGE_PATH"])
def test_sparse_first_row_accumulated_once(dtype, k, algo):
    blocked = k <= 2 and algo in {"DEFAULT", "CSR", "CSR_NNZ_SPLIT"}
    _check_sparse_first_row(dtype, k, algo, None, blocked)


@pytest.mark.rocsparse
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("spmv", ["DEFAULT", "CSR_ADAPTIVE", "CSR_ROWSPLIT", "CSR_LRB", "CSR_NNZSPLIT"])
def test_native_spmv_first_row_accumulated_once(dtype, spmv):
    _check_sparse_first_row(dtype, 1, "CSR_ROW_SPLIT", spmv, spmv == "CSR_NNZSPLIT")


def _check_sparse_first_row(dtype, k, algo, spmv, blocked):
    import cupy as cp

    # ROCm 10.1 NNZ-split adds row 0 twice at widths 1/2 on this six-edge
    # matrix. DEFAULT and CSR dispatch to the same affected implementation.
    a = sp.csr_matrix((np.ones(6), ([0, 0, 0, 0, 1, 39], [0, 1, 2, 3, 4, 5])), shape=(124, 10))
    block = SimpleNamespace(struct0=cp.asarray(a.indptr), struct1=cp.asarray(a.indices),
                            nrows=124, ncols=10, nnz=6, fmt=SparseFormat.CSR)
    stream = cp.cuda.get_current_stream()
    handle = Handle(stream)
    ones = SharedOnes(6, dtype, 0, "materialized")
    op = None
    try:
        source, destination = cp.ones((10, k), dtype=dtype), cp.ones((124, k), dtype=dtype)
        plan = replace(standard_pair().plan_up, algo=algo, spmv=spmv)
        if blocked:
            with pytest.raises(ValueError, match="incorrect"):
                SparseOperation(handle, stream, block, ones, source, destination, plan)
        else:
            op = SparseOperation(handle, stream, block, ones, source, destination, plan)
            assert op.is_vector is (spmv is not None and k == 1)
            op()
            stream.synchronize()
            np.testing.assert_array_equal(cp.asnumpy(destination), a @ np.ones((10, k)) + 1)
    finally:
        if op is not None:
            op.close()
        ones.close()
        handle.close()
