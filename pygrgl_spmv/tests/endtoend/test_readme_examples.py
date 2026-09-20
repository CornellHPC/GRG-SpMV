"""The README's grapp examples, executed.

The three grapp blocks of the project README's Basic Usage section are reproduced here,
differing only in the artifact paths, the phenotype and the choice of k. They are
hand-copied, not extracted, so this pins that the documented calls *work* -- not that
the README still says exactly this. The one exception is the import order, which is
checked against README.md itself below.

This exists because reading the examples was not enough. The PCA block shipped broken
twice -- once with an import order that hit a circular import inside grapp, once
pairing ``threads=4`` with ``capture=False``, which the eager path rejects as a
concurrent call on one runtime.
"""

from __future__ import annotations

import contextlib
import pathlib
import re
import subprocess
import sys

import numpy as np
import pytest

from pygrgl_spmv import (
    load_grg_spmv_multi,
    load_grg_spmv_single,
    make_backend_cusparse,
    make_runconfig_gwas,
    make_runconfig_pca,
)

pytestmark = [pytest.mark.gpu, pytest.mark.cusparse]

# "grapp", not "grapp.linalg": importing the latter here would pre-satisfy the circular
# import the order test below exists to catch.
pytest.importorskip("grapp", reason="grapp is only in the dev extra")

README = pathlib.Path(__file__).resolve().parents[3] / "README.md"


def test_readme_import_order_survives_a_fresh_interpreter():
    """grapp.grg_calculator imported before grapp.linalg hits a circular import inside
    grapp, so the README's order is load-bearing. Checked in a subprocess against the
    README's own text: in-process the order cannot be observed once anything has
    imported grapp.linalg."""
    checked = 0
    for block in re.findall(r"```python\n(.*?)```", README.read_text(), re.S):
        imports = [l for l in block.splitlines() if re.match(r"\s*(import|from)\s", l)]
        if not any("grapp" in l for l in imports):
            continue
        checked += 1
        done = subprocess.run([sys.executable, "-c", "\n".join(imports)], capture_output=True, text=True)
        assert done.returncode == 0, (
            f"a README block's imports fail in a fresh interpreter:\n{chr(10).join(imports)}\n{done.stderr}"
        )
    assert checked >= 1, "no README block imports grapp; has the example moved?"


@pytest.fixture
def two_artifacts(primary_artifact, tmp_path):
    """Two artifacts from one source, so they share a device group and a runtime.

    grapp asserts every GRG has the same samples, so this cannot be the primary plus
    the missingness fixture -- and one artifact would not reproduce the concurrency
    the README's ``threads=4`` needs.
    """
    import shutil

    second = tmp_path / "chr2.grg_spmv"
    shutil.copyfile(primary_artifact, second)
    return [str(primary_artifact), str(second)]


def test_pca_example_runs(two_artifacts):
    from grapp.linalg import PCs                      # linalg first, per the README
    from grapp.grg_calculator import GRGSpMVCalculator

    backend = make_backend_cusparse(device=0, capture=True)
    req = make_runconfig_pca()
    with contextlib.ExitStack() as stack:
        grgs = [
            GRGSpMVCalculator(g)
            for g in load_grg_spmv_multi(two_artifacts, backend, req, stack)
        ]
        pcs_df = PCs(grgs, k=10, threads=4)
    assert len(pcs_df) == grgs[0].num_individuals
    assert list(pcs_df.columns)[0] == "PC1"


def test_pca_example_with_include_eig_runs(two_artifacts):
    """include_eig drives a width-k product, so maxk must cover it."""
    from grapp.linalg import PCs                      # linalg first, per the README
    from grapp.grg_calculator import GRGSpMVCalculator

    k = 3
    backend = make_backend_cusparse(device=0, capture=True)
    with contextlib.ExitStack() as stack:
        grgs = [
            GRGSpMVCalculator(g)
            for g in load_grg_spmv_multi(two_artifacts, backend, make_runconfig_pca(maxk=k), stack)
        ]
        pcs_df, eig_vals, eig_vecs = PCs(grgs, k=k, include_eig=True, threads=4)
    assert eig_vals.shape == (k,)
    assert pcs_df.shape[1] == k


def test_summary_statistics_example_runs(missing_artifact):
    from grapp.grg_calculator import GRGSpMVCalculator
    from grapp.util import allele_counts, allele_frequencies

    backend = make_backend_cusparse(device=0, capture=True)
    with contextlib.ExitStack() as stack:
        grg = GRGSpMVCalculator(
            load_grg_spmv_single(str(missing_artifact), backend, make_runconfig_gwas(), stack)
        )
        freqs = allele_frequencies(grg, adjust_missing=True)
        counts, missing = allele_counts(grg, return_missing=True)
    assert freqs.shape == counts.shape == (grg.num_mutations,)
    assert missing.sum() > 0, "fixture has no missingness; the example would be vacuous"


def test_association_example_runs(missing_artifact):
    from grapp.assoc import linear_assoc_no_covar
    from grapp.grg_calculator import GRGSpMVCalculator

    backend = make_backend_cusparse(device=0, capture=True)
    with contextlib.ExitStack() as stack:
        grg = GRGSpMVCalculator(
            load_grg_spmv_single(str(missing_artifact), backend, make_runconfig_gwas(), stack)
        )
        Y = np.random.default_rng(0).standard_normal(grg.num_individuals)
        results = linear_assoc_no_covar(grg, Y)
    assert len(results) == grg.num_mutations
