"""Contract for the shipped ``pygrgl_spmv.testing`` helpers.

These are public API for downstream suites, so the properties that make them
usable are pinned here: no ``test_``-prefixed public name, no pytest import, a
context-manager lifetime with no caller-supplied ExitStack, fork-safe
availability probes, and a strict keyword allowlist.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

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
        "backends": ["reference", "mkl", "cusparse"],
        "pytest_imported": False,
        "gpu_imported": [],
    }


def test_availability_probes_do_not_initialise_the_cuda_driver():
    """The probes must be fork-safe.

    ``torch.cuda.is_available()`` and ``cupy.cuda.runtime.getDeviceCount()`` both
    call ``cuInit``, after which a ``fork()``ed child cannot use CUDA at all --
    even though no CUDA *context* exists. Shipping that would break any consumer
    using multiprocessing's default start method.
    """
    script = textwrap.dedent(
        """
        import ctypes, json
        from pygrgl_spmv import testing
        available = testing.is_cusparse_available()
        ctx = ctypes.c_void_p()
        rc = ctypes.CDLL("libcuda.so.1").cuCtxGetCurrent(ctypes.byref(ctx))
        print(json.dumps({"available": bool(available), "rc": int(rc)}))
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"could not probe the CUDA driver here: {result.stderr.strip()[:200]}")
    import json

    payload = json.loads(result.stdout)
    # 3 == CUDA_ERROR_NOT_INITIALIZED. Anything else means the probe called cuInit.
    assert payload["rc"] == 3, f"is_cusparse_available() initialised the driver (rc={payload['rc']})"


def test_is_available_dispatches_to_the_shipped_predicates(monkeypatch):
    """The identity form of this (`f() is f()` on bools) could not fail, and neither
    could comparing is_available("mkl") against the same function it dispatches to."""
    assert testing.is_available("reference") is True
    for backend, predicate in (("mkl", "is_mkl_available"), ("cusparse", "is_cusparse_available")):
        for answer in (True, False):
            monkeypatch.setitem(testing._PREDICATES, backend, lambda answer=answer: answer)
            assert testing.is_available(backend) is answer
            assert (backend in testing.available_backends()) is answer
    # Still a ValueError, not the raw KeyError, now that triton is gone.
    with pytest.raises(ValueError, match="unknown backend"):
        testing.is_available("triton")


def test_the_gpu_probe_is_cached_but_the_env_check_is_not():
    """_probe_cusparse carries the @cache; is_cusparse_available deliberately does
    not, so the kill switch stays live. Asserting on __wrapped__ rather than on
    bool identity, which is true either way."""
    assert hasattr(testing._probe_cusparse, "cache_info")
    assert not hasattr(testing.is_cusparse_available, "cache_info")
    assert testing._probe_cusparse.cache_info().maxsize is None


def test_available_backends_is_ordered_and_filterable():
    backends = testing.available_backends()
    assert backends[0] == "reference"
    assert set(backends) <= set(testing.BACKENDS)
    assert list(backends) == [b for b in testing.BACKENDS if b in backends]
    assert testing.available_backends(include_reference=False) == tuple(
        b for b in backends if b != "reference"
    )


def test_gpu_probe_reads_the_disable_env_var_on_every_call(monkeypatch):
    """The env read used to sit inside the @cache, so it was inert once anything had
    probed, and needed cache_clear() here -- which only ever tested the cold path.
    And "0" is a non-empty string, so the conventional 0/1 pair disabled the GPU.
    """
    monkeypatch.setenv("PYGRGL_SPMV_DISABLE_GPU", "1")
    assert testing.is_cusparse_available() is False
    assert "cusparse" not in testing.available_backends()

    # Against the probe rather than True, so this also holds on a host with no GPU.
    for off in ("0", "false", "no", ""):
        monkeypatch.setenv("PYGRGL_SPMV_DISABLE_GPU", off)
        assert testing.is_cusparse_available() is testing._probe_cusparse()

    monkeypatch.delenv("PYGRGL_SPMV_DISABLE_GPU")
    assert testing.is_cusparse_available() is testing._probe_cusparse()


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
            marks=[pytest.mark.gpu, pytest.mark.cusparse] if name == "cusparse"
            else ([pytest.mark.mkl] if name == "mkl" else []),
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
    sugar = {"reference": testing.load_reference, "mkl": testing.load_mkl, "cusparse": testing.load_cusparse}
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


@pytest.mark.gpu
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
        pytest.param("mkl", "vram_budget_mb", id="mkl-wrong-backend"),
        # Named parameters, so they bypassed the allowlist: the README-recommended
        # load(backend="reference", capture=True) returned an eager GRG with no
        # error, and a downstream "captured replay" test passed without a graph.
        pytest.param("reference", "capture", id="reference-capture"),
        pytest.param("cusparse", "n_threads", id="cusparse-n_threads"),
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
