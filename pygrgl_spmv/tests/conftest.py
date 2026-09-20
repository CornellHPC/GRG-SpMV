"""Shared fixtures and pytest controls for the runtime-era test suite."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygrgl
import pytest
from pygrgl_spmv import simple_convert
from pygrgl_spmv.testing import is_cusparse_available, is_mkl_available

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIMARY_GRG = str(REPO_ROOT / "pygrgl_spmv" / "tests" / "data" / "msprime.example.igd.final.grg")
DEFAULT_MISSING_GRG = str(REPO_ROOT / "pygrgl_spmv" / "tests" / "data" / "test-200-samples.miss.final.grg")

DATA_DTYPE = np.float64


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        default="all",
        choices=["mkl", "cusparse", "all"],
        help="Which backend-specific tests to include; shared/reference tests always run.",
    )
    parser.addoption(
        "--stress",
        action="store_true",
        default=False,
        help="Run long streamed GPU stress tests.",
    )
    parser.addoption(
        "--grg",
        default=DEFAULT_PRIMARY_GRG,
        help="Primary GRG file used by tests.",
    )
    parser.addoption(
        "--missing-grg",
        default=DEFAULT_MISSING_GRG,
        help="Missingness GRG file used by tests.",
    )


def pytest_collection_modifyitems(config, items):
    backend = str(config.getoption("--backend"))
    stress = bool(config.getoption("--stress"))

    match backend:
        case "all":
            pass
        case "mkl":
            skip = pytest.mark.skip(reason="--backend=mkl")
            for item in items:
                if "cusparse" in item.keywords or "cuda13" in item.keywords:
                    item.add_marker(skip)
        case "cusparse":
            skip = pytest.mark.skip(reason="--backend=cusparse")
            for item in items:
                if "mkl" in item.keywords:
                    item.add_marker(skip)
        case _:
            raise ValueError(f"unexpected --backend value {backend!r}")

    # The shipped predicates rather than hand-rolled probes: the suite's old cuSPARSE
    # probe called cuInit, leaving every forked child unable to use CUDA. Called here
    # rather than at module scope because the cuSPARSE one imports torch and cupy
    # (~1.4 s), which --backend=mkl then never pays for.
    if not is_mkl_available():
        skip = pytest.mark.skip(reason="MKL runtime unavailable (libmkl_rt.so not found)")
        for item in items:
            if "mkl" in item.keywords:
                item.add_marker(skip)

    if backend != "mkl" and not is_cusparse_available():
        skip = pytest.mark.skip(reason="cuSPARSE runtime unavailable (CuPy + CUDA not found)")
        for item in items:
            if "cusparse" in item.keywords:
                item.add_marker(skip)

    if not stress:
        skip = pytest.mark.skip(reason="stress tests require --stress")
        for item in items:
            if "stress" in item.keywords:
                item.add_marker(skip)


@dataclass(frozen=True)
class GrgFixture:
    """One (name, grg path, loaded grg, converted artifact) bundle."""

    name: str
    path: str
    grg: object
    artifact: Path


@pytest.fixture(params=["msprime", "missing"])
def any_grg(request) -> GrgFixture:
    """Run a structural test on both fixtures, not just the trivial one.

    ``msprime.example`` is 2 levels with a single non-empty block, so every
    per-(level, block) invariant is trivially satisfied on it -- that is how the
    cuSPARSE DOWN ``ext_scratch`` index reversal survived for so long.
    ``test-200-samples.miss`` is 20 levels with 81 non-empty blocks and carries
    missingness, so it can actually express an ordering mismatch.

    Resolved lazily: declaring all six as parameters let a missing --missing-grg
    skip the *msprime* param too, silently zeroing both halves.
    """
    prefix = "primary" if request.param == "msprime" else "missing"
    return GrgFixture(
        request.param,
        *(request.getfixturevalue(f"{prefix}_{name}") for name in ("grg_path", "grg", "artifact")),
    )


def tol(dtype) -> tuple[float, float]:
    return (1e-3, 1e-3) if np.dtype(dtype) == np.float32 else (1e-5, 1e-5)


def binary_pm1(rng: np.random.Generator, shape: tuple[int, ...], dtype) -> np.ndarray:
    return rng.choice(np.array([-1.0, 1.0], dtype=np.dtype(dtype)), size=shape)


@pytest.fixture(scope="session")
def primary_grg_path(request) -> str:
    path = Path(str(request.config.getoption("--grg")))
    if not path.exists():
        pytest.skip(f"primary GRG file not found: {path}")
    return str(path)


@pytest.fixture(scope="session")
def missing_grg_path(request) -> str:
    path = Path(str(request.config.getoption("--missing-grg")))
    if not path.exists():
        pytest.skip(f"missingness GRG file not found: {path}")
    return str(path)


@pytest.fixture(scope="session")
def primary_grg(primary_grg_path):
    return pygrgl.load_immutable_grg(primary_grg_path, load_up_edges=True)


@pytest.fixture(scope="session")
def missing_grg(missing_grg_path):
    return pygrgl.load_immutable_grg(missing_grg_path, load_up_edges=True)


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory) -> Path:
    """Session-scoped directory for artifacts converted by the suite.

    A fresh temp directory per session, deliberately: artifacts are inputs now,
    not a cache, so every run converts exactly once and can never pick up a
    stale artifact left by an older format version.
    """
    return tmp_path_factory.mktemp("grg_spmv_artifacts")


@pytest.fixture(scope="session")
def primary_artifact(primary_grg_path, artifact_dir) -> Path:
    return simple_convert(primary_grg_path, artifact_dir / "primary.grg_spmv")


@pytest.fixture(scope="session")
def missing_artifact(missing_grg_path, artifact_dir) -> Path:
    return simple_convert(missing_grg_path, artifact_dir / "missing.grg_spmv")


