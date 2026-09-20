"""``simple_convert()`` contract: the only way to produce a ``.grg_spmv`` artifact.

Conversion is explicit-in / explicit-out. There is no output-directory form, no
source-path-derived destination, and no in-memory-GRG form -- so there is nothing
that could be mistaken for an implicit cache, and conversion time can never be
counted as GRG-SpMV runtime.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePath

import numpy as np
import pytest

from pygrgl_spmv import simple_convert
from pygrgl_spmv.grg.artifact import load_grg_spmv


class _PathWrapper(os.PathLike[str]):
    def __init__(self, value: str) -> None:
        self._value = value

    def __fspath__(self) -> str:
        return self._value


def test_writes_artifact_to_the_exact_requested_path(primary_grg_path, tmp_path):
    dst = tmp_path / "chr1.grg_spmv"
    artifact = simple_convert(primary_grg_path, dst)
    assert artifact == dst.resolve()
    assert artifact.is_file()
    # Nothing else is created next to it -- no mirrored source path, no cache tree.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["chr1.grg_spmv"]


@pytest.mark.parametrize(
    ("out_name", "expected"),
    [
        pytest.param("a.grg_spmv", "a.grg_spmv", id="already-suffixed"),
        pytest.param("b", "b.grg_spmv", id="no-suffix"),
        pytest.param("c.v2", "c.v2.grg_spmv", id="other-suffix"),
        pytest.param("x.y.z.grg_spmv", "x.y.z.grg_spmv", id="dotted-stem"),
    ],
)
def test_appends_the_artifact_suffix_when_absent(primary_grg_path, tmp_path, out_name, expected):
    artifact = simple_convert(primary_grg_path, tmp_path / out_name)
    assert artifact.name == expected
    assert artifact.is_file()


def test_creates_missing_parent_directories(primary_grg_path, tmp_path):
    artifact = simple_convert(primary_grg_path, tmp_path / "nested" / "deep" / "chr1")
    assert artifact == (tmp_path / "nested" / "deep" / "chr1.grg_spmv").resolve()
    assert artifact.is_file()


def test_returns_a_resolved_path(primary_grg_path, tmp_path):
    """scan_grg_spmv() resolves but BoundGRG.artifact_path does not.

    Returning a resolved path keeps those two identities equal by construction,
    instead of relying on the caller having passed an already-absolute path.
    """
    link = tmp_path / "link"
    link.symlink_to(tmp_path)
    artifact = simple_convert(primary_grg_path, link / "via-symlink.grg_spmv")
    assert artifact.is_absolute()
    assert artifact == artifact.resolve()


def test_overwrites_an_existing_artifact_atomically(primary_grg_path, tmp_path):
    dst = tmp_path / "chr1.grg_spmv"
    dst.write_bytes(b"stale garbage that is not an npz")
    artifact = simple_convert(primary_grg_path, dst)
    state = load_grg_spmv(artifact, np.float64)
    assert state.num_samples > 0
    # The atomic write must not leave its temp file behind.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["chr1.grg_spmv"]


@pytest.mark.parametrize(
    "path_factory",
    [
        pytest.param(PurePath, id="purepath"),
        pytest.param(_PathWrapper, id="pathlike-wrapper"),
        pytest.param(str, id="str"),
    ],
)
def test_accepts_generic_pathlikes_for_both_arguments(primary_grg_path, tmp_path, path_factory):
    artifact = simple_convert(path_factory(primary_grg_path), path_factory(str(tmp_path / "out.grg_spmv")))
    assert artifact.is_file()


def test_rejects_a_non_grg_input(tmp_path):
    bogus = tmp_path / "file.txt"
    bogus.write_text("not a grg")
    with pytest.raises(ValueError, match=r"expected a \.grg input file"):
        simple_convert(bogus, tmp_path / "out.grg_spmv")


def test_rejects_a_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError, match="GRG input file not found"):
        simple_convert(tmp_path / "absent.grg", tmp_path / "out.grg_spmv")


def test_rejects_a_directory_as_output(primary_grg_path, tmp_path):
    """``simple_convert(src, "artifacts/")`` used to write a sibling *file*."""
    existing = tmp_path / "artifacts"
    existing.mkdir()
    with pytest.raises(IsADirectoryError, match="must name a .grg_spmv file"):
        simple_convert(primary_grg_path, existing)


@pytest.mark.parametrize("spelling", ["trailing-slash", "suffixed-is-a-dir"])
def test_rejects_a_directory_before_doing_any_work(primary_grg_path, tmp_path, monkeypatch, spelling):
    """Both spellings used to reach save_grg_spmv and fail after the full compile."""
    import pygrgl_spmv.grg as grg_module

    def _should_not_load(*_args, **_kwargs):
        raise AssertionError("simple_convert() must reject a directory before loading the GRG")

    monkeypatch.setattr(grg_module.pygrgl, "load_immutable_grg", _should_not_load)

    if spelling == "trailing-slash":
        target = f"{tmp_path / 'artifacts'}/"
    else:
        (tmp_path / "foo.grg_spmv").mkdir()
        target = tmp_path / "foo"

    with pytest.raises(IsADirectoryError, match="must name a .grg_spmv file"):
        simple_convert(primary_grg_path, target)
    assert not (tmp_path / "artifacts.grg_spmv").exists()


def test_reports_an_unloadable_grg_instead_of_crashing_in_the_compiler(
    primary_grg_path, tmp_path, monkeypatch
):
    """A corrupt .grg used to die as an AttributeError on NoneType in the compiler."""
    import pygrgl_spmv.grg as grg_module

    monkeypatch.setattr(grg_module.pygrgl, "load_immutable_grg", lambda *a, **k: None)
    with pytest.raises(ValueError, match="could not load GRG"):
        simple_convert(primary_grg_path, tmp_path / "out.grg_spmv")


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(np.float16, id="float16"),
        pytest.param(np.complex64, id="complex64"),
        pytest.param(np.int32, id="int32"),
        pytest.param(np.longdouble, id="longdouble"),
    ],
)
def test_rejects_dtypes_no_backend_can_consume(primary_grg_path, tmp_path, dtype):
    """float16 used to silently write +inf init biases; longdouble wrote float128."""
    with pytest.raises(ValueError, match="dtype must be float32 or float64"):
        simple_convert(primary_grg_path, tmp_path / "out.grg_spmv", dtype=dtype)


def test_validates_before_doing_any_work(primary_grg_path, tmp_path, monkeypatch):
    """A bad call must cost nothing, not fail after a full compile."""
    import pygrgl_spmv.grg as grg_module

    def _should_not_load(*_args, **_kwargs):
        raise AssertionError("simple_convert() must validate before loading the GRG")

    monkeypatch.setattr(grg_module.pygrgl, "load_immutable_grg", _should_not_load)
    with pytest.raises(ValueError, match="dtype must be float32 or float64"):
        simple_convert(primary_grg_path, tmp_path / "out.grg_spmv", dtype=np.float16)


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_init_biases_are_present_and_typed(primary_grg_path, tmp_path, dtype):
    artifact = simple_convert(primary_grg_path, tmp_path / "out.grg_spmv", dtype=dtype)
    state = load_grg_spmv(artifact, dtype)
    assert state.init_vector_up_bias is not None
    assert state.init_vector_down_bias is not None
    assert state.init_vector_up_bias.shape == (state.num_mutations,)
    assert state.init_vector_down_bias.shape == (state.num_samples,)
    assert state.init_vector_up_bias.dtype == np.dtype(dtype)


def test_is_byte_reproducible_for_a_fixed_input_and_dtype(primary_grg_path, tmp_path):
    """Nothing else protects this, and any compile refactor could break it."""
    first = simple_convert(primary_grg_path, tmp_path / "one.grg_spmv").read_bytes()
    second = simple_convert(primary_grg_path, tmp_path / "two.grg_spmv").read_bytes()
    assert first == second


def test_loads_grg_with_down_edges_only_and_does_not_compute_missing_coals(
    primary_grg_path,
    monkeypatch,
    tmp_path,
):
    import pygrgl_spmv.grg as grg_module

    real_loader = grg_module.pygrgl.load_immutable_grg
    calls = {"load_up_edges": [], "calculate_missing_coals": 0}

    class _Proxy:
        def __init__(self, inner):
            self._inner = inner

        def calculate_missing_coals(self):
            calls["calculate_missing_coals"] += 1
            return self._inner.calculate_missing_coals()

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def _wrapped_loader(path, *args, **kwargs):
        calls["load_up_edges"].append(bool(kwargs.get("load_up_edges", False)))
        return _Proxy(real_loader(path, *args, **kwargs))

    monkeypatch.setattr(grg_module.pygrgl, "load_immutable_grg", _wrapped_loader)

    artifact = simple_convert(primary_grg_path, tmp_path / "out.grg_spmv")
    assert artifact.is_file()
    assert calls["load_up_edges"] == [False]
    assert calls["calculate_missing_coals"] == 0


def test_no_implicit_output_directory_api_remains():
    """The old cache-shaped API must be gone, not merely discouraged."""
    import pygrgl_spmv
    import pygrgl_spmv.grg.artifact as artifact_mod

    assert not hasattr(pygrgl_spmv, "convert")
    assert not hasattr(artifact_mod, "artifact_path_for_grg")
    assert not hasattr(artifact_mod, "_scan_grg_spmv_cached")


def test_cli_reports_errors_instead_of_tracebacks(primary_grg_path, tmp_path, capsys):
    """The CLI had no coverage, and everything the write path raises reached the user
    as a traceback with exit 1 instead of one ``error:`` line."""
    from pygrgl_spmv.__main__ import main

    dst = tmp_path / "cli.grg_spmv"
    assert main(["convert", primary_grg_path, str(dst)]) == 0
    assert dst.is_file()
    assert str(dst) in capsys.readouterr().out

    blocked = tmp_path / "not-a-dir"
    blocked.write_text("")
    assert main(["convert", primary_grg_path, str(blocked / "x.grg_spmv")]) == 2
    assert "error:" in capsys.readouterr().err
