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
