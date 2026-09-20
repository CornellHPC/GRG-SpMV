"""Test helpers for downstream consumers of ``pygrgl-spmv``.

This module ships in the wheel (``pygrgl_spmv/tests/**`` does not), so downstream
suites can parameterize their own tests over our backends without hand-rolling
planner and runtime boilerplate.

Two constraints shape the API, both of them measured rather than stylistic:

* **No public name starts with ``test_``.** pytest *collects* module-level
  callables whose name starts with ``test_``, so a consumer writing
  ``from pygrgl_spmv.testing import test_load_mkl`` in a test module gets
  ``ERROR ... fixture 'grg_path' not found`` rather than a helper.
* **pytest is never imported here.** It is not a runtime dependency of this
  package and must not become one. There is deliberately no ``pytest11`` plugin
  entry point either: probing backends at import time costs ~2 s in *every*
  pytest run of *every* project that installs us, to save a four-line fixture.

The loaders are context managers, so there is no ``exit_stack`` argument to pass
around. The lifetime being managed is real -- for cuSPARSE, one runtime and one
dense arena are shared by every artifact on a device, plus one prepared-op
context per captured graph -- but the test framework already owns a stack for
you: pytest ``yield`` fixtures and :meth:`unittest.TestCase.enterContext` both
are one.

Typical use, pytest::

    import pytest
    from pygrgl_spmv import testing as spmv_test

    @pytest.fixture(scope="session")
    def artifact(tmp_path_factory):
        return spmv_test.artifact_for("chr1.grg", tmp_path_factory.mktemp("art"))

    @pytest.fixture(params=spmv_test.available_backends(), ids=lambda b: b)
    def grg(request, artifact):
        with spmv_test.load(artifact, backend=request.param, max_k=4) as g:
            yield g

and ``unittest``::

    for backend in spmv_test.available_backends():
        grg = self.enterContext(spmv_test.load(artifact, backend=backend, max_k=4))
"""

from __future__ import annotations

import contextlib
import functools
import os
from numbers import Integral
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from pygrgl_spmv.grg import RuntimeRequirements, simple_convert
from pygrgl_spmv.grg.artifact import GRG_SPMV_SUFFIX

__all__ = [
    "BACKENDS",
    "artifact_for",
    "available_backends",
    "is_available",
    "is_cusparse_available",
    "is_mkl_available",
    "load",
    "load_cusparse",
    "load_many",
    "load_mkl",
    "load_reference",
    "requirements",
]

#: Every backend this package can execute, in preference order. ``reference`` is
#: a pure NumPy/SciPy implementation that is always available and is the parity
#: oracle the other backends are checked against.
BACKENDS = ("reference", "mkl", "cusparse")


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


@functools.cache
def is_mkl_available() -> bool:
    """Whether ``libmkl_rt.so`` can be loaded in this process.

    This is the same load the MKL planner performs, so it is an exact proxy
    rather than a guess. The result is cached; the underlying dlopen and ABI
    probe happen once.
    """
    try:
        from pygrgl_spmv.backends.mkl import ffi as mkl_ffi

        mkl_ffi._ensure_loaded()
        return True
    except Exception:
        return False


def is_cusparse_available() -> bool:
    """Whether CuPy, torch and at least one visible GPU are present.

    Set ``PYGRGL_SPMV_DISABLE_GPU`` to ``1``/``true``/``yes``/``on`` to force
    ``False``; every other value, including ``0`` and ``off``, leaves the GPU
    enabled. Read outside the cached probe, so a consumer setting it from
    ``pytest_configure`` or a fixture is still honoured after something has probed.
    """
    # Only the documented values disable. An allowlist rather than "anything truthy",
    # because a consumer exporting =off or =2 to mean "do not disable" would otherwise
    # lose cuSPARSE silently and their whole GPU matrix would skip, green.
    if os.environ.get("PYGRGL_SPMV_DISABLE_GPU", "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    return _probe_cusparse()


@functools.cache
def _probe_cusparse() -> bool:
    """The cached half of :func:`is_cusparse_available`.

    Deliberately fork-safe. ``torch.cuda.is_available()`` and
    ``cupy.cuda.runtime.getDeviceCount()`` both call ``cuInit``, which leaves the
    process unable to use CUDA in a ``fork()``ed child
    (``cudaErrorInitializationError``) even though no CUDA *context* is created.
    A library-level predicate must not do that to its callers, so this goes
    through NVML via ``torch.cuda.device_count()``, which leaves the driver
    uninitialised. The trade is that NVML can report a device that
    ``cudaSetDevice`` would later reject (MIG or cgroup restrictions); that
    surfaces as a clear CUDA error at load time, whereas a fork-poisoned process
    surfaces as a baffling failure in an unrelated worker.
    """
    try:
        import cupy  # noqa: F401
        import torch
    except Exception:
        return False
    try:
        return int(torch.cuda.device_count()) > 0
    except Exception:
        return False


_PREDICATES = {
    "reference": lambda: True,
    "mkl": is_mkl_available,
    "cusparse": is_cusparse_available,
}


def is_available(backend: str) -> bool:
    """Whether *backend* can be used in this process."""
    try:
        predicate = _PREDICATES[str(backend)]
    except KeyError:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}") from None
    return predicate()


def available_backends(*, include_reference: bool = True) -> tuple[str, ...]:
    """Backends usable in this process, in :data:`BACKENDS` order.

    This is the one call a downstream suite needs in order to parameterize over
    our backends; it replaces per-backend ``skipif`` boilerplate, and a backend
    added or removed here shows up automatically.
    """
    names = BACKENDS if include_reference else BACKENDS[1:]
    return tuple(name for name in names if is_available(name))


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def requirements(
    *,
    max_k: int | None = None,
    max_k_up: int | None = None,
    max_k_down: int | None = None,
    need_down_miss_input: bool = True,
    need_up_miss_output: bool = True,
    need_init_vector: bool = True,
    need_init_matrix: bool = True,
    need_init_xtx: bool = True,
) -> RuntimeRequirements:
    """An everything-enabled :class:`RuntimeRequirements`, with a ``max_k`` shorthand.

    Keyword names mirror ``RuntimeRequirements`` field for field, including the
    ``need_`` prefixes, so this is a drop-in replacement rather than a second
    vocabulary to learn. The only additions are the permissive defaults and
    ``max_k``, which sets both directions at once and cannot be combined with them.
    """
    if max_k is not None:
        if max_k_up is not None or max_k_down is not None:
            raise TypeError("pass either max_k or max_k_up/max_k_down, not both")
        max_k_up = max_k_down = max_k

    # Every width, not just max_k: int() silently accepted True and truncated 2.9, and
    # the >= 1 bound was left to RuntimeRequirements, which reports it under max_k_up
    # even when the caller passed max_k.
    def _width(value, name):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be an int >= 1, got {value!r}")
        return int(value)

    name = "max_k" if max_k is not None else "max_k_up"
    max_k_up = 8 if max_k_up is None else _width(max_k_up, name)
    max_k_down = 8 if max_k_down is None else _width(max_k_down, name if max_k is not None else "max_k_down")
    return RuntimeRequirements(
        max_k_up=max_k_up,
        max_k_down=max_k_down,
        need_down_miss_input=bool(need_down_miss_input),
        need_up_miss_output=bool(need_up_miss_output),
        need_init_vector=bool(need_init_vector),
        need_init_matrix=bool(need_init_matrix),
        need_init_xtx=bool(need_init_xtx),
    )


def artifact_for(grg_path, out_dir, *, dtype=np.float64) -> Path:
    """Convert *grg_path* into a ``.grg_spmv`` artifact under *out_dir*.

    A thin naming convenience over :func:`pygrgl_spmv.simple_convert`, and
    deliberately a separate call from :func:`load`: conversion stays a visible
    step whose cost is never hidden inside a load. It does not cache -- call it
    from a session-scoped fixture if you want to convert once per run.

    Named from the input's basename, so ``chr1/part.grg`` and ``chr2/part.grg``
    need different *out_dir*\\ s or the second overwrites the first.
    """
    src = Path(os.fspath(grg_path))
    dst_dir = Path(os.fspath(out_dir))
    return simple_convert(
        src, dst_dir / f"{src.stem}.{np.dtype(dtype).name}{GRG_SPMV_SUFFIX}", dtype=dtype
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

#: Backend-specific arguments accepted through ``**kw``, mirroring each
#: ``make_backend_*`` factory's keywords so the factories keep owning the defaults.
#: Anything else is a TypeError -- that is the whole point. A permissive ``**kw``
#: silently turned a typo'd ``allow_resideny=True`` into a no-op, and ``capture``,
#: ``native``, ``device``, ``n_threads`` and ``plans`` were named parameters that
#: bypassed the check entirely and were dropped by backends that cannot honour them.
_EXTRA_KW = {
    "reference": frozenset({"plans"}),
    "mkl": frozenset({"n_threads", "optimize"}),
    "cusparse": frozenset({"allow_residency", "capture", "device", "native", "vram_budget_mb"}),
}


def _as_paths(artifacts) -> list[Path]:
    if isinstance(artifacts, (str, os.PathLike)):
        raise TypeError("load_many() expects a sequence of artifacts; use load() for a single one")
    paths = [Path(os.fspath(a)) for a in artifacts]
    if not paths:
        raise ValueError("artifacts must be non-empty")
    for path in paths:
        if path.suffix != GRG_SPMV_SUFFIX:
            raise ValueError(
                f"expected a {GRG_SPMV_SUFFIX} artifact, got {path}; "
                "convert the .grg first with pygrgl_spmv.simple_convert() or artifact_for()"
            )
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def _bare_requirements(req):
    from pygrgl_spmv.adaptor import RunConfigs

    return req.req if isinstance(req, RunConfigs) else req


def _reference_runtime(paths, req, dtype, plans):
    from pygrgl_spmv.backends.reference import (
        ReferencePlan,
        ReferencePlanPair,
        ReferenceRuntime,
        plan_reference_layout,
    )

    pair = plans or ReferencePlanPair(
        plan_up=ReferencePlan(store="N", fmt="CSR"),
        plan_down=ReferencePlan(store="T", fmt="CSC"),
    )
    return ReferenceRuntime(
        plan_reference_layout(
            artifacts=paths, pair=pair, dtype=dtype, requirements=_bare_requirements(req)
        )
    )


@contextlib.contextmanager
def _adaptor_loaded(paths, backend_cfg, req, dtype):
    """Own the ExitStack the adaptor's load functions require."""
    from pygrgl_spmv.adaptor import load_grg_spmv_multi

    with contextlib.ExitStack() as stack:
        yield load_grg_spmv_multi([str(p) for p in paths], backend_cfg, req, stack, dtype)


@contextlib.contextmanager
def load_many(
    artifacts: Sequence,
    *,
    backend: str = "reference",
    req=None,
    max_k: int | None = None,
    dtype=np.float64,
    **kw,
) -> Iterator[list]:
    """Load ``.grg_spmv`` artifacts on *backend* and yield the bound GRGs.

    Every resource -- runtime arenas, cuSPARSE handles, captured CUDA graphs and
    their prepared-op contexts -- is released on exit, in dependency order. The
    returned GRGs must not be used afterwards.

    Pass ``req`` to supply your own ``RuntimeRequirements`` or a ``RunConfigs``
    from ``make_runconfig_*``; otherwise ``requirements(max_k=max_k)`` is used. Note
    that a bare ``RuntimeRequirements`` captures only the default up/down pair, so
    under ``capture=True`` a shape like ``by_individual=True`` needs a ``RunConfigs``.

    Everything else goes through ``**kw`` to the matching ``make_backend_*``
    factory; see :data:`_EXTRA_KW` for what each backend accepts.
    """
    backend = str(backend)
    if backend not in _EXTRA_KW:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    unknown = sorted(set(kw) - _EXTRA_KW[backend])
    if unknown:
        accepted = sorted(_EXTRA_KW[backend]) or "none"
        raise TypeError(
            f"backend={backend!r} does not accept {unknown}; "
            f"accepted for this backend: {accepted}"
        )

    paths = _as_paths(artifacts)
    if req is not None and max_k is not None:
        # Both used to be accepted, and max_k was silently discarded.
        raise TypeError("pass either req or max_k, not both; max_k only builds a default req")
    if req is None:
        req = requirements(max_k=8 if max_k is None else max_k)
    # The reference branch below bypasses load_grg_spmv_multi, which is where the
    # adaptor validates these -- so float16 used to load with inf init biases, and a
    # wrong-typed req died as an AttributeError inside the planner, on the very
    # backend the other two are compared against.
    from pygrgl_spmv.adaptor import _validate_dtype, _validate_req

    _validate_req(req)
    dtype = _validate_dtype(dtype)

    # kw is already restricted to each factory's own keywords.
    match backend:
        case "reference":
            with _reference_runtime(paths, req, dtype, kw.get("plans")) as runtime:
                yield list(runtime.grgs)
        case "mkl":
            from pygrgl_spmv.adaptor import make_backend_mkl

            with _adaptor_loaded(paths, make_backend_mkl(**kw), req, dtype) as grgs:
                yield grgs
        case "cusparse":
            from pygrgl_spmv.adaptor import make_backend_cusparse

            with _adaptor_loaded(paths, make_backend_cusparse(**kw), req, dtype) as grgs:
                yield grgs


@contextlib.contextmanager
def load(
    artifact,
    *,
    backend: str = "reference",
    req=None,
    max_k: int | None = None,
    dtype=np.float64,
    **kw,
) -> Iterator[object]:
    """Single-artifact form of :func:`load_many`."""
    with load_many(
        [artifact], backend=backend, req=req, max_k=max_k, dtype=dtype, **kw
    ) as grgs:
        yield grgs[0]


def load_reference(artifact, **kwargs):
    """:func:`load` pinned to the reference backend."""
    return load(artifact, backend="reference", **kwargs)


def load_mkl(artifact, **kwargs):
    """:func:`load` pinned to the MKL backend."""
    return load(artifact, backend="mkl", **kwargs)


def load_cusparse(artifact, **kwargs):
    """:func:`load` pinned to the cuSPARSE backend."""
    return load(artifact, backend="cusparse", **kwargs)
