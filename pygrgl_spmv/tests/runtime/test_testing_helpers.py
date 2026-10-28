"""Contract for the shipped ``pygrgl_spmv.testing`` helpers.

These are public API for downstream suites, so the properties that make them
usable are pinned here: no ``test_``-prefixed public name, no pytest import, a
context-manager lifetime with no caller-supplied ExitStack, fork-safe
availability probes, and a strict keyword allowlist.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pygrgl
import pytest

from pygrgl_spmv import testing
from pygrgl_spmv.tests.conftest import DATA_DTYPE, tol


# ---------------------------------------------------------------------------
# Shape of the public surface
# ---------------------------------------------------------------------------


def test_no_public_name_would_be_collected_as_a_test():
    """A ``test_``-prefixed helper is collected by whichever suite imports it.

    Importing such a name into a test module yields
    ``fixture 'grg_path' not found`` instead of a usable helper, so the prefix is
    unshippable regardless of taste.
    """
    offenders = [name for name in testing.__all__ if name.startswith("test_")]
    assert not offenders
    exported = [name for name in vars(testing) if not name.startswith("_")]
    assert not [name for name in exported if name.startswith("test_")]


def test_importable_without_pytest_and_without_touching_gpu_modules():
    """pytest must not become a runtime dependency of the shipped package, and
    importing this module must not drag in torch or cupy -- it is the first thing a
    consumer's CPU-only collection phase imports. The torch/cupy half of the name
    used to go unchecked, so `import torch` at module scope would have passed."""
    script = textwrap.dedent(
        """
        import importlib.abc
        import json
        import sys

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in {"pytest", "_pytest"}:
                    raise ModuleNotFoundError(fullname)
                return None

        sys.meta_path.insert(0, Blocker())
        from pygrgl_spmv import testing
        print(json.dumps({
            "backends": list(testing.BACKENDS),
            "pytest_imported": "pytest" in sys.modules,
            "gpu_imported": sorted(m for m in ("torch", "cupy") if m in sys.modules),
        }))
        """
    )
    result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
    import json

    payload = json.loads(result.stdout)
    assert payload == {
        "backends": ["reference", "mkl", "cusparse", "rocsparse"],
        "pytest_imported": False,
        "gpu_imported": [],
    }


@pytest.mark.parametrize("backend", [
    pytest.param(name, id=name, marks=getattr(pytest.mark, name))
    for name in ("cusparse", "rocsparse")
])
def test_availability_probes_are_fork_safe(backend):
    # A fresh parent prevents earlier numerical tests from initializing its driver.
    script = textwrap.dedent(
        """
        import ctypes
        import multiprocessing as mp
        import sys
        from pygrgl_spmv import testing

        backend = sys.argv[1]
        assert testing.is_available(backend)
        if backend == "cusparse":
            ctx = ctypes.c_void_p()
            rc = ctypes.CDLL("libcuda.so.1").cuCtxGetCurrent(ctypes.byref(ctx))
            assert rc == 3, f"CUDA driver initialized by probe: {rc}"

        def allocate():
            import cupy as cp
            assert float(cp.ones(4).sum()) == 4.0

        child = mp.get_context("fork").Process(target=allocate)
        child.start()
        child.join(30)
        if child.is_alive():
            child.kill()
            child.join()
            raise AssertionError("first child allocation timed out")
        assert child.exitcode == 0, child.exitcode
        """
    )
    subprocess.run([sys.executable, "-c", script, backend], check=True, timeout=60)


@pytest.mark.parametrize("outcome", [0, 1, "timeout", "missing"])
def test_rocsparse_probe_failure_and_cache(monkeypatch, outcome):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command[0] == sys.executable
        assert kwargs["timeout"] == 30
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, 30)
        if outcome == "missing":
            raise OSError("interpreter unavailable")
        return subprocess.CompletedProcess(command, outcome)

    testing._probe_rocsparse.cache_clear()
    monkeypatch.setattr(subprocess, "run", run)
    try:
        assert testing._probe_rocsparse() is (outcome == 0)
        assert testing._probe_rocsparse() is (outcome == 0)
        assert len(calls) == 1
    finally:
        testing._probe_rocsparse.cache_clear()


@pytest.mark.parametrize("is_hip", [False, True])
@pytest.mark.parametrize("count", [0, 1])
def test_cusparse_probe_rejects_hip_cupy_before_device_count(monkeypatch, is_hip, count):
    calls = []

    def device_count():
        calls.append(True)
        return count

    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(is_hip=is_hip))))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(device_count=device_count)))
    testing._probe_cusparse.cache_clear()
    try:
        assert testing._probe_cusparse() is (not is_hip and count > 0)
        assert calls == ([] if is_hip else [True])
    finally:
        testing._probe_cusparse.cache_clear()


@pytest.mark.parametrize("backend", ["cusparse", "rocsparse"])
def test_unavailable_backend_skips_by_default_but_explicit_selection_fails(backend):
    node = f"{Path(__file__).resolve()}::test_load_yields_a_working_grg_on_every_backend[{backend}]"
    command = [sys.executable, "-m", "pytest", "-q", node]
    env = dict(os.environ, PYGRGL_SPMV_DISABLE_GPU="1")
    optional = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert optional.returncode == 0, optional.stdout + optional.stderr
    assert "1 skipped" in optional.stdout
    required = subprocess.run(command + [f"--backend={backend}"], env=env,
                              capture_output=True, text=True, timeout=60)
    assert required.returncode == 4, required.stdout + required.stderr
    assert f"--backend={backend} explicitly requested" in required.stderr


def test_is_available_dispatches_to_the_shipped_predicates(monkeypatch):
    """The identity form of this (`f() is f()` on bools) could not fail, and neither
    could comparing is_available("mkl") against the same function it dispatches to."""
    assert testing.is_available("reference") is True
    for backend in ("mkl", "cusparse", "rocsparse"):
        for answer in (True, False):
            monkeypatch.setitem(testing._PREDICATES, backend, lambda answer=answer: answer)
            assert testing.is_available(backend) is answer
            assert (backend in testing.available_backends()) is answer
    # Still a ValueError, not the raw KeyError, now that triton is gone.
    with pytest.raises(ValueError, match="unknown backend"):
        testing.is_available("triton")


@pytest.mark.parametrize("backend", ["cusparse", "rocsparse"])
def test_the_gpu_probe_is_cached_but_the_env_check_is_not(backend):
    probe = getattr(testing, f"_probe_{backend}")
    available = getattr(testing, f"is_{backend}_available")
    assert hasattr(probe, "cache_info")
    assert not hasattr(available, "cache_info")
    assert probe.cache_info().maxsize is None


def test_available_backends_is_ordered_and_filterable():
    backends = testing.available_backends()
    assert backends[0] == "reference"
    assert set(backends) <= set(testing.BACKENDS)
    assert list(backends) == [b for b in testing.BACKENDS if b in backends]
    assert testing.available_backends(include_reference=False) == tuple(
        b for b in backends if b != "reference"
    )


@pytest.mark.parametrize("backend", ["cusparse", "rocsparse"])
def test_gpu_probe_reads_the_disable_env_var_on_every_call(monkeypatch, backend):
    available = getattr(testing, f"is_{backend}_available")
    probe = getattr(testing, f"_probe_{backend}")
    for on in ("1", "true", "yes", "on", " TRUE "):
        monkeypatch.setenv("PYGRGL_SPMV_DISABLE_GPU", on)
        assert available() is False
        assert backend not in testing.available_backends()
    for off in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("PYGRGL_SPMV_DISABLE_GPU", off)
        assert available() is probe()
    monkeypatch.delenv("PYGRGL_SPMV_DISABLE_GPU")
    assert available() is probe()


# ---------------------------------------------------------------------------
# requirements() / artifact_for()
# ---------------------------------------------------------------------------


def test_requirements_mirrors_the_dataclass_fields():
    import dataclasses

    from pygrgl_spmv import RuntimeRequirements

    req = testing.requirements()
    assert isinstance(req, RuntimeRequirements)
    field_names = {f.name for f in dataclasses.fields(RuntimeRequirements)}
    import inspect

    kwargs = set(inspect.signature(testing.requirements).parameters) - {"max_k"}
    assert kwargs == field_names, "requirements() must mirror RuntimeRequirements field-for-field"


def test_requirements_max_k_shorthand_sets_both_directions():
    req = testing.requirements(max_k=5)
    assert (req.max_k_up, req.max_k_down) == (5, 5)
    req = testing.requirements(max_k_up=3, max_k_down=7)
    assert (req.max_k_up, req.max_k_down) == (3, 7)
    assert testing.requirements(need_init_xtx=False).need_init_xtx is False


def test_artifact_for_converts_and_is_not_a_cache(primary_grg_path, tmp_path):
    first = testing.artifact_for(primary_grg_path, tmp_path)
    assert first.is_file()
    assert first.suffix == ".grg_spmv"
    assert "float64" in first.name
    stamp = first.stat().st_mtime_ns
    second = testing.artifact_for(primary_grg_path, tmp_path)
    assert second == first
    assert second.stat().st_mtime_ns != stamp, "artifact_for() must reconvert, not serve a cached file"


def test_artifact_for_separates_dtypes(primary_grg_path, tmp_path):
    f64 = testing.artifact_for(primary_grg_path, tmp_path, dtype=np.float64)
    f32 = testing.artifact_for(primary_grg_path, tmp_path, dtype=np.float32)
    assert f64 != f32


# ---------------------------------------------------------------------------
# load() / load_many()
# ---------------------------------------------------------------------------


def _backend_params():
    return [
        pytest.param(
            name,
            id=name,
            marks=() if name == "reference" else getattr(pytest.mark, name),
        )
        for name in testing.BACKENDS
    ]


@pytest.mark.parametrize("backend", _backend_params())
def test_load_yields_a_working_grg_on_every_backend(primary_artifact, primary_grg, backend):
    """The downstream pattern: one parametrize over available_backends(), no skipif."""
    if not testing.is_available(backend):
        pytest.skip(f"{backend} unavailable")
    rng = np.random.default_rng(11)
    x = rng.standard_normal((2, primary_grg.num_samples), dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    with testing.load(primary_artifact, backend=backend, max_k=2) as grg:
        got = grg.matmul(x, "up")
    expected = np.asarray(pygrgl.matmul(primary_grg, x, pygrgl.TraversalDirection.UP))
    np.testing.assert_allclose(got, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("backend", _backend_params())
def test_per_backend_sugar_matches_the_primitive(primary_artifact, backend):
    if not testing.is_available(backend):
        pytest.skip(f"{backend} unavailable")
    sugar = {"reference": testing.load_reference, "mkl": testing.load_mkl, "cusparse": testing.load_cusparse, "rocsparse": testing.load_rocsparse}
    with sugar[backend](primary_artifact, max_k=2) as a, testing.load(
        primary_artifact, backend=backend, max_k=2
    ) as b:
        assert type(a) is type(b)
        assert a.num_samples == b.num_samples


def test_load_many_returns_one_grg_per_artifact_in_order(primary_artifact, missing_artifact):
    with testing.load_many([primary_artifact, missing_artifact], max_k=2) as grgs:
        assert len(grgs) == 2
        assert grgs[0].num_mutations != grgs[1].num_mutations
        assert [g.artifact_path for g in grgs] == [primary_artifact, missing_artifact]


def test_load_many_rejects_a_bare_path(primary_artifact):
    with pytest.raises(TypeError, match="use load\\(\\) for a single one"):
        with testing.load_many(primary_artifact):
            pass


def test_load_releases_resources_on_exit(primary_artifact):
    with testing.load(primary_artifact, backend="reference", max_k=2) as grg:
        runtime = grg._runtime
    with pytest.raises(RuntimeError, match="must be entered"):
        runtime._call_scope().__enter__()


@pytest.mark.cusparse
def test_captured_grg_refuses_to_replay_after_its_runtime_is_released(primary_artifact):
    """Post-close replay used to corrupt other allocations rather than fail."""
    with testing.load(primary_artifact, backend="cusparse", max_k=2, capture=True) as grg:
        ones = np.ones((1, grg.num_samples), dtype=DATA_DTYPE)
        assert float(grg.matmul(ones, "up").sum()) > 0
    with pytest.raises(RuntimeError, match="after its runtime was released"):
        grg.matmul(ones, "up")


def test_load_rejects_a_grg_input(primary_grg_path):
    """Conversion is a separate, explicit step -- load() never does it for you."""
    with pytest.raises(ValueError, match="expected a .grg_spmv artifact"):
        with testing.load(primary_grg_path):
            pass


def test_load_rejects_a_missing_artifact(tmp_path):
    with pytest.raises(FileNotFoundError):
        with testing.load(tmp_path / "absent.grg_spmv"):
            pass


def test_load_rejects_an_unknown_backend(primary_artifact):
    with pytest.raises(ValueError, match="unknown backend"):
        with testing.load(primary_artifact, backend="triton"):
            pass


@pytest.mark.parametrize(
    ("backend", "kwarg"),
    [
        pytest.param("cusparse", "allow_resideny", id="cusparse-typo"),
        pytest.param("cusparse", "ring_buffer_size", id="cusparse-unsupported"),
        pytest.param("cusparse", "stream", id="cusparse-hardcoded"),
        pytest.param("rocsparse", "allow_resideny", id="rocsparse-typo"),
        pytest.param("rocsparse", "ring_buffer_size", id="rocsparse-unsupported"),
        pytest.param("rocsparse", "stream", id="rocsparse-hardcoded"),
        pytest.param("rocsparse", "vram_budget_mb", id="rocsparse-streaming"),
        pytest.param("mkl", "vram_budget_mb", id="mkl-wrong-backend"),
        # Named parameters, so they bypassed the allowlist: the README-recommended
        # load(backend="reference", capture=True) returned an eager GRG with no
        # error, and a downstream "captured replay" test passed without a graph.
        pytest.param("reference", "capture", id="reference-capture"),
        pytest.param("cusparse", "n_threads", id="cusparse-n_threads"),
        pytest.param("rocsparse", "n_threads", id="rocsparse-n_threads"),
        pytest.param("mkl", "native", id="mkl-native"),
        pytest.param("mkl", "plans", id="mkl-plans"),
    ],
)
def test_unknown_backend_kwargs_are_a_loud_error(primary_artifact, backend, kwarg):
    """A permissive **kw silently turned a typo into a no-op."""
    with pytest.raises(TypeError, match="does not accept"):
        with testing.load(primary_artifact, backend=backend, **{kwarg: 1}):
            pass


def test_max_k_together_with_req_is_an_error(primary_artifact):
    """Passing both used to discard max_k silently, then fail on the input shape."""
    from pygrgl_spmv import make_runconfig_gwas

    with pytest.raises(TypeError, match="either req or max_k"):
        with testing.load(primary_artifact, req=make_runconfig_gwas(), max_k=64):
            pass


def test_requirements_can_be_supplied_directly(primary_artifact):
    req = testing.requirements(max_k=1, need_init_matrix=False)
    with testing.load(primary_artifact, backend="reference", req=req) as grg:
        assert grg._runtime.layout.requirements.max_k_up == 1
        assert grg._runtime.layout.requirements.need_init_matrix is False


def test_runconfig_is_accepted_as_req(primary_artifact):
    from pygrgl_spmv import make_runconfig_gwas

    with testing.load(primary_artifact, backend="reference", req=make_runconfig_gwas(maxk=3)) as grg:
        assert grg._runtime.layout.requirements.max_k_up == 3
