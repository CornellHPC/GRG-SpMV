"""rocSPARSE ABI and SDK regressions; shared semantics live in shared families."""

import subprocess
import sys
from pathlib import Path

import pytest

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
