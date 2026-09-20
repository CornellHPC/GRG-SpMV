"""Adaptor surface contracts: factories, validation, threading, and missingness.

Complements test_adaptor_capture.py (which owns capture keys and replay parity).
Everything here except the marked GPU tests runs on CPU.
"""

from __future__ import annotations

import contextlib
import os

import numpy as np
import pygrgl
import pytest

from pygrgl_spmv import (
    CusparseBackendConfig,
    MklBackendConfig,
    load_grg_spmv_multi,
    load_grg_spmv_single,
    make_backend_cusparse,
    make_backend_mkl,
    make_runconfig_bolt,
    make_runconfig_gwas,
    make_runconfig_kernel,
    make_runconfig_pca,
    testing,
)
from pygrgl_spmv.adaptor import (
    _physical_cores,
    _resolve_capture_k,
    _resolve_device,
    _resolve_mkl_threads,
    _validate_capture_spec,
)
from pygrgl_spmv import adaptor as adaptor_mod
from pygrgl_spmv.backends.types import Direction
from pygrgl_spmv.tests.conftest import DATA_DTYPE, tol


# ---------------------------------------------------------------------------
# Backend factories
# ---------------------------------------------------------------------------


def test_make_backend_mkl_accepts_scalar_and_dict():
    assert make_backend_mkl() == MklBackendConfig(n_threads=0, optimize=False)
    assert make_backend_mkl(n_threads=4, optimize=True) == MklBackendConfig(n_threads=4, optimize=True)
    mapping = {"chr1": {"mkl_threads": [2, 1]}}
    assert make_backend_mkl(n_threads=mapping).n_threads is mapping


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        pytest.param({"n_threads": -1}, ValueError, id="negative"),
        pytest.param({"n_threads": "4"}, TypeError, id="string"),
        pytest.param({"nthreads": 4}, TypeError, id="typo"),
    ],
)
def test_make_backend_mkl_rejects_bad_arguments(kwargs, exc):
    with pytest.raises(exc):
        make_backend_mkl(**kwargs)


def test_make_backend_cusparse_defaults_and_dict():
    assert make_backend_cusparse() == CusparseBackendConfig(
        device=0, allow_residency=True, vram_budget_mb=0, capture=False, native=False
    )
    mapping = {"chr1": {"cuda_device": 1}}
    assert make_backend_cusparse(device=mapping).device is mapping


@pytest.mark.parametrize(
    ("kwargs", "exc", "match"),
    [
        pytest.param({"device": -1}, ValueError, "device must be >= 0", id="negative-device"),
        pytest.param({"device": "0"}, TypeError, "device must be int or dict", id="string-device"),
        pytest.param({"vram_budget_mb": -1}, ValueError, "must be >= 0", id="negative-budget"),
        pytest.param({"vram_budget_mb": 1.5}, TypeError, "must be int", id="float-budget"),
        pytest.param(
            {"allow_residency": False},
            ValueError,
            "must be > 0 in streaming mode",
            id="streaming-needs-budget",
        ),
        pytest.param({"devcie": 0}, TypeError, "unexpected keyword", id="typo"),
    ],
)
def test_make_backend_cusparse_rejects_bad_arguments(kwargs, exc, match):
    with pytest.raises(exc, match=match):
        make_backend_cusparse(**kwargs)


def test_native_without_capture_is_ignored_not_an_error():
    """Documented behaviour; the guard lives in the caller, not here."""
    cfg = make_backend_cusparse(native=True)
    assert cfg.native is True and cfg.capture is False


# ---------------------------------------------------------------------------
# Run-config factories
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [make_runconfig_pca, make_runconfig_bolt, make_runconfig_gwas],
    ids=["pca", "bolt", "gwas"],
)
def test_runconfig_factories_reject_unknown_kwargs(factory):
    with pytest.raises(TypeError, match="unexpected keyword arguments"):
        factory(force_spmmm=True)


def test_runconfig_bolt_typo_names_the_right_function():
    """The message used to name make_runconfig_boltlmm(), which does not exist."""
    with pytest.raises(TypeError, match=r"make_runconfig_bolt\(\)"):
        make_runconfig_bolt(nope=1)


@pytest.mark.parametrize("factory", [make_runconfig_pca, make_runconfig_gwas], ids=["pca", "gwas"])
@pytest.mark.parametrize("maxk", [2.5, 0, -1, "4"], ids=["float", "zero", "negative", "string"])
def test_maxk_validation_is_uniform(factory, maxk):
    """pca used to accept maxk=2.5 and pass it straight into max_k_up."""
    with pytest.raises(ValueError, match="maxk must be an int"):
        factory(maxk=maxk)


@pytest.mark.parametrize(
    ("maxk", "force_spmm", "expected"),
    [(1, False, 1), (1, True, 2), (4, False, 4), (4, True, 4), (8, False, 8)],
)
def test_resolve_capture_k(maxk, force_spmm, expected):
    assert _resolve_capture_k(maxk, force_spmm) == expected


def test_make_runconfig_kernel_requires_a_valid_direction():
    with pytest.raises(ValueError, match="direction must be"):
        make_runconfig_kernel("sideways", 1)


def test_validate_capture_spec_checks_direction_and_k():
    from pygrgl_spmv import CaptureSpec

    req = testing.requirements(max_k=2)
    with pytest.raises(ValueError, match="direction must be"):
        _validate_capture_spec(CaptureSpec("sideways", 1), req)
    with pytest.raises(ValueError, match="k must be an int"):
        _validate_capture_spec(CaptureSpec("up", 0), req)
    with pytest.raises(ValueError, match=r"exceeds the declared max_k_up"):
        _validate_capture_spec(CaptureSpec("up", 4), testing.requirements(max_k=1))
    with pytest.raises(ValueError, match=r"exceeds the declared max_k_down"):
        _validate_capture_spec(CaptureSpec("down", 4), testing.requirements(max_k=1))


# ---------------------------------------------------------------------------
# Threading and device resolution
# ---------------------------------------------------------------------------


def test_physical_cores_does_not_count_smt_siblings():
    """psutil is not a dependency, so this used to return os.cpu_count()."""
    physical = _physical_cores()
    logical = os.cpu_count() or 1
    assert 1 <= physical <= logical
    sysfs = os.path.join("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list")
    if os.path.isfile(sysfs):
        with open(sysfs) as handle:
            siblings = handle.read().strip()
        if "," in siblings or "-" in siblings:
            assert physical < logical, "SMT is enabled but physical == logical"


def test_physical_cores_counts_only_cpus_in_the_affinity_mask():
    """Reported whole-host cores inside a cpuset; then min(cores, affinity) reported
    the logical count, the double-count this function exists to prevent."""
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        pytest.skip("no CPU affinity control on this platform")
    sysfs = "/sys/devices/system/cpu/cpu0/topology/thread_siblings_list"
    if not os.path.isfile(sysfs):
        pytest.skip("no sysfs CPU topology")
    with open(sysfs) as handle:
        raw = handle.read().strip()
    if "," not in raw and "-" not in raw:
        pytest.skip("SMT is disabled, so one core has nothing to over-count")
    lo, _, hi = raw.replace("-", ",").partition(",")
    one_core = {int(lo), int(hi)}

    original = os.sched_getaffinity(0)
    assert _physical_cores() <= len(original)
    try:
        os.sched_setaffinity(0, one_core)
        _physical_cores.cache_clear()
        assert _physical_cores() == 1, "both siblings of one core must count as one core"
    finally:
        os.sched_setaffinity(0, original)
        _physical_cores.cache_clear()


@pytest.mark.parametrize(
    ("cpu_max", "expected"),
    [
        pytest.param("200000 100000", 2, id="2-cpus"),
        pytest.param("250000 100000", 2, id="2.5-cpus-floors"),
        pytest.param("110000 100000", 1, id="1.1-cpus-floors"),
        pytest.param("50000 100000", 1, id="half-a-cpu-floors-to-one"),
        pytest.param("max 100000", None, id="unlimited"),
    ],
)
def test_cgroup_cpu_quota_is_read_and_floored(tmp_path, monkeypatch, cpu_max, expected):
    """A CPU *limit* is a quota, not a cpuset, so sched_getaffinity cannot see it and
    n_threads=0 handed MKL every host core inside a container. Floored, not ceiled:
    ceiling would hand out 3 threads against --cpus=2.5 and reintroduce throttling."""
    (tmp_path / "cpu.max").write_text(cpu_max)
    monkeypatch.setattr(adaptor_mod, "_CGROUP_ROOT", tmp_path)
    assert adaptor_mod._cgroup_cpu_quota() == expected

    # _physical_cores is cached, so a quota change is invisible without clearing it.
    _physical_cores.cache_clear()
    try:
        assert _physical_cores() == (expected if expected is not None else _physical_cores())
    finally:
        monkeypatch.undo()
        _physical_cores.cache_clear()


def test_resolve_mkl_threads_scalar_and_auto(tmp_path):
    path = tmp_path / "chr1.grg_spmv"
    assert _resolve_mkl_threads(make_backend_mkl(n_threads=3), path, 2) == (3, 3)
    auto = _resolve_mkl_threads(make_backend_mkl(n_threads=0), path, 4)
    assert auto == (max(_physical_cores() // 4, 1),) * 2


def test_resolve_mkl_threads_dict_form_and_errors(tmp_path):
    path = tmp_path / "chr1.grg_spmv"
    cfg = make_backend_mkl(n_threads={"chr1": {"mkl_threads": [2, 1]}})
    assert _resolve_mkl_threads(cfg, path, 1) == (2, 1)
    with pytest.raises(KeyError, match="not found in n_threads config"):
        _resolve_mkl_threads(cfg, tmp_path / "chr9.grg_spmv", 1)
    with pytest.raises(KeyError, match="missing 'mkl_threads'"):
        _resolve_mkl_threads(make_backend_mkl(n_threads={"chr1": {}}), path, 1)


def test_auto_threads_warns_when_files_outnumber_cores(tmp_path):
    path = tmp_path / "chr1.grg_spmv"
    with pytest.warns(RuntimeWarning, match="using 1 thread per file"):
        assert _resolve_mkl_threads(make_backend_mkl(n_threads=0), path, _physical_cores() + 1) == (1, 1)


@pytest.mark.cuda13
@pytest.mark.cusparse
def test_resolve_device_dict_form_and_errors(tmp_path):
    path = tmp_path / "chr1.grg_spmv"
    cfg = make_backend_cusparse(device={"chr1": {"cuda_device": 0}})
    assert _resolve_device(cfg, path) == 0
    with pytest.raises(KeyError, match="not found in device config"):
        _resolve_device(cfg, tmp_path / "chr9.grg_spmv")
    with pytest.raises(KeyError, match="missing 'cuda_device'"):
        _resolve_device(make_backend_cusparse(device={"chr1": {}}), path)
    with pytest.raises(RuntimeError, match="is not available"):
        _resolve_device(make_backend_cusparse(device=9999), path)


# ---------------------------------------------------------------------------
# load_grg_spmv_* argument validation
# ---------------------------------------------------------------------------


def test_load_validates_its_arguments(primary_artifact, primary_grg_path):
    stack = contextlib.ExitStack()
    backend = make_backend_mkl()
    req = make_runconfig_kernel("up", 1)
    with pytest.raises(ValueError, match="must be non-empty"):
        load_grg_spmv_multi([], backend, req, stack)
    with pytest.raises(ValueError, match=r"expected a \.grg_spmv file"):
        load_grg_spmv_multi([primary_grg_path], backend, req, stack)
    with pytest.raises(FileNotFoundError):
        load_grg_spmv_multi(["/nonexistent/x.grg_spmv"], backend, req, stack)
    with pytest.raises(TypeError, match="backend must be"):
        load_grg_spmv_multi([primary_artifact], object(), req, stack)
    with pytest.raises(TypeError, match="req must be"):
        load_grg_spmv_multi([primary_artifact], backend, object(), stack)
    with pytest.raises(ValueError, match="dtype must be float32 or float64"):
        load_grg_spmv_multi([primary_artifact], backend, req, stack, dtype=np.float16)


# ---------------------------------------------------------------------------
# Direction handling
# ---------------------------------------------------------------------------


@pytest.mark.cuda13
@pytest.mark.cusparse
@pytest.mark.parametrize(
    "direction",
    ["up", "UP", Direction.UP, pygrgl.TraversalDirection.UP],
    ids=["lower", "upper", "Direction", "TraversalDirection"],
)
def test_captured_matmul_accepts_every_direction_form(primary_artifact, direction):
    """Only the lowercase literal and Direction (a StrEnum) used to work."""
    with testing.load(primary_artifact, backend="cusparse", capture=True,
                      req=make_runconfig_kernel("up", 1)) as grg:
        ones = np.ones((1, grg.num_samples), dtype=DATA_DTYPE)
        assert float(grg.matmul(ones, direction).sum()) > 0


@pytest.mark.cuda13
@pytest.mark.cusparse
def test_captured_matmul_rejects_an_unknown_direction(primary_artifact):
    with testing.load(primary_artifact, backend="cusparse", capture=True,
                      req=make_runconfig_kernel("up", 1)) as grg:
        with pytest.raises(ValueError, match="(?i)unknown direction"):
            grg.matmul(np.ones((1, grg.num_samples), dtype=DATA_DTYPE), "sideways")


# ---------------------------------------------------------------------------
# Missingness -- all paths must agree
# ---------------------------------------------------------------------------


def _expected_miss(grg_obj, k: int) -> np.ndarray:
    ones = np.ones((k, grg_obj.num_samples), dtype=DATA_DTYPE)
    miss = np.zeros((k, grg_obj.num_mutations), dtype=DATA_DTYPE)
    pygrgl.matmul(grg_obj, ones, pygrgl.TraversalDirection.UP, miss=miss)
    return miss


# Marks, not bare strings: the collection hook filters on item.keywords, so an
# unmarked "cusparse" param initialised CUDA under --backend mkl and -m "not cuda13".
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("reference", id="reference"),
        pytest.param("mkl", id="mkl", marks=pytest.mark.mkl),
        pytest.param("cusparse", id="cusparse", marks=[pytest.mark.cuda13, pytest.mark.cusparse]),
    ],
)
@pytest.mark.parametrize("k", [1, 2], ids=["k1", "k2"])
def test_up_miss_output_matches_pygrgl_on_every_backend(missing_artifact, missing_grg, backend, k):
    if not testing.is_available(backend):
        pytest.skip(f"{backend} unavailable")
    expected = _expected_miss(missing_grg, k)
    assert expected.sum() > 0, "fixture has no missingness; the test would be vacuous"
    ones = np.ones((k, missing_grg.num_samples), dtype=DATA_DTYPE)
    got = np.zeros_like(expected)
    with testing.load(missing_artifact, backend=backend, max_k=k) as grg:
        grg.matmul(ones, "up", miss=got)
    np.testing.assert_allclose(got, expected, atol=1e-9)


@pytest.mark.cuda13
@pytest.mark.cusparse
@pytest.mark.parametrize("k", [1, 2], ids=["k1", "k2"])
def test_captured_copy_mode_accumulates_miss_into_the_callers_array(missing_artifact, missing_grg, k):
    expected = _expected_miss(missing_grg, k)
    ones = np.ones((k, missing_grg.num_samples), dtype=DATA_DTYPE)
    got = np.zeros_like(expected)
    with testing.load(missing_artifact, backend="cusparse", capture=True,
                      req=make_runconfig_gwas(maxk=k)) as grg:
        grg.matmul(ones, "up", miss=got)
    np.testing.assert_allclose(got, expected, atol=1e-9)


@pytest.mark.cuda13
@pytest.mark.cusparse
def test_native_mode_accumulates_miss_into_the_callers_device_array(missing_artifact, missing_grg):
    """Native mode updates a caller-OWNED device array in place, correctly.

    The known failure mode is a caller that hands over ``cupy.asarray(host_miss)``
    -- a temporary -- and then reads the host array, losing every count. That is a
    caller bug and cannot be detected from here, which is why the rejection message
    for a host array explicitly warns against it (see the test below).
    """
    import cupy as cp

    expected = _expected_miss(missing_grg, 1)
    with testing.load(missing_artifact, backend="cusparse", capture=True,
                      native=True, req=make_runconfig_gwas(maxk=1)) as grg:
        device = cp.cuda.Device(grg._src_tensors[next(iter(grg._src_tensors))].device.index)
        with device:
            ones = cp.ones((1, missing_grg.num_samples), dtype=DATA_DTYPE)
            miss = cp.zeros((1, missing_grg.num_mutations), dtype=DATA_DTYPE)
            grg.matmul(ones, "up", miss=miss)
            got = cp.asnumpy(miss)
    np.testing.assert_allclose(got, expected, atol=1e-9)
    assert got.sum() == pytest.approx(expected.sum())


@pytest.mark.cuda13
@pytest.mark.cusparse
def test_native_mode_host_miss_message_warns_against_the_temporary(missing_artifact, missing_grg):
    """The old message read as an instruction to write the bug.

    It said only "expected cupy.ndarray ..., got ndarray", and the natural way to
    satisfy that is ``cupy.asarray(miss)`` -- which silently discards the result.
    """
    with testing.load(missing_artifact, backend="cusparse", capture=True,
                      native=True, req=make_runconfig_gwas(maxk=1)) as grg:
        import cupy as cp

        device_index = grg._src_tensors[next(iter(grg._src_tensors))].device.index
        with cp.cuda.Device(device_index):
            ones = cp.ones((1, missing_grg.num_samples), dtype=DATA_DTYPE)
            host_miss = np.zeros((1, missing_grg.num_mutations), dtype=DATA_DTYPE)
            with pytest.raises(TypeError) as excinfo:
                grg.matmul(ones, "up", miss=host_miss)
    message = str(excinfo.value)
    assert "in-place accumulator" in message
    assert "cupy.asarray()" in message


@pytest.mark.cuda13
@pytest.mark.cusparse
def test_down_miss_input_matches_between_eager_and_captured(missing_artifact, missing_grg):
    runconfig = make_runconfig_bolt()
    rng = np.random.default_rng(4242)
    x = rng.standard_normal((1, missing_grg.num_mutations), dtype=DATA_DTYPE)
    miss_in = rng.standard_normal((1, missing_grg.num_mutations), dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    with contextlib.ExitStack() as stack:
        eager = load_grg_spmv_single(missing_artifact, make_backend_cusparse(device=0), runconfig, stack)
        captured = load_grg_spmv_single(
            missing_artifact, make_backend_cusparse(device=0, capture=True), runconfig, stack
        )
        expected = eager.matmul(x, "down", miss=miss_in.copy())
        got = captured.matmul(x, "down", miss=miss_in.copy())
    np.testing.assert_allclose(got, expected, atol=atol, rtol=rtol)
