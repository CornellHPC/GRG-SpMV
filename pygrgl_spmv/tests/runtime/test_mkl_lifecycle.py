"""MklRuntime.__enter__ failure-path ownership, driven by a stubbed MKL FFI.

These tests deliberately carry no ``mkl`` marker. They install ``_FakeMklLib`` in
place of the real library, so they exercise ``MklRuntime.__enter__``'s cleanup
control flow on hosts without ``libmkl_rt.so`` -- which is where a leak on the
failure path would otherwise go unnoticed.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

import pygrgl_spmv.backends.mkl.ffi as mkl_ffi
from pygrgl_spmv import MklRuntime
from pygrgl_spmv.tests.runtime._runtime_builders import build_mkl_layout, full_requirements
from pygrgl_spmv.tests.runtime.test_mkl import _FakeMklLib


@pytest.fixture
def fake_mkl(monkeypatch) -> _FakeMklLib:
    fake = _FakeMklLib()
    monkeypatch.setattr(mkl_ffi, "_ensure_loaded", lambda: (fake, np.int32, ctypes.c_int))
    return fake


def _layout(artifact):
    return build_mkl_layout([artifact], requirements=full_requirements(max_k_up=2, max_k_down=2))


def test_mkl_enter_failure_mid_block_loop_destroys_every_handle(missing_artifact, fake_mkl, monkeypatch):
    """A failure part way through one artifact's block loop must leak nothing.

    ``_MklArtifact`` is appended to ``artifacts`` only after its whole block loop
    finishes, so before the ``pending_grids`` fix ``_destroy_handles(artifacts)``
    could not see any handle built for the in-flight artifact. ``__enter__`` then
    called ``_shared_values.destroy()``, which munmaps the values buffer those
    still-live handles point into.

    The missingness fixture is used deliberately: 20 levels / 81 non-empty blocks,
    so there are many handles to lose. msprime has one block and cannot express it.
    """
    real_build = MklRuntime._build_handle
    calls = {"n": 0}
    fail_at = 40

    def _boom(self, matrix, plan):
        calls["n"] += 1
        if calls["n"] == fail_at:
            raise RuntimeError("injected failure building a sparse handle")
        return real_build(self, matrix, plan)

    monkeypatch.setattr(MklRuntime, "_build_handle", _boom)

    runtime = MklRuntime(_layout(missing_artifact))
    with pytest.raises(RuntimeError, match="injected failure building a sparse handle"):
        runtime.__enter__()

    created = list(fake_mkl.created_handles)
    destroyed = set(fake_mkl.destroyed_handles)
    assert created, "fixture built no handles; the test would be vacuous"
    leaked = [h for h in created if h not in destroyed]
    assert not leaked, f"{len(leaked)} of {len(created)} MKL handles leaked on the failure path"

    # The values buffer must also be released, and the runtime left unusable.
    assert runtime._shared_values is None
    assert runtime._up_workspace is None
    assert runtime._down_workspace is None
    # Left non-empty, a later __exit__ destroys these handles a second time.
    assert runtime._artifacts == ()
    runtime.__exit__(None, None, None)
    with pytest.raises(RuntimeError, match="must be entered"):
        runtime._call_scope().__enter__()


def test_mkl_successful_enter_then_exit_destroys_every_handle(missing_artifact, fake_mkl):
    """Baseline for the test above: the success path must be leak-free too."""
    with MklRuntime(_layout(missing_artifact)) as runtime:
        assert runtime.grgs
        assert fake_mkl.created_handles
        assert not fake_mkl.destroyed_handles

    created = list(fake_mkl.created_handles)
    destroyed = set(fake_mkl.destroyed_handles)
    leaked = [h for h in created if h not in destroyed]
    assert not leaked, f"{len(leaked)} of {len(created)} MKL handles leaked on the success path"
