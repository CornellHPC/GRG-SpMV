"""Capture-key correctness for the adaptor's CUDA-graph path.

This is the surface that used to return a plausible wrong number with no signal:
``CapturedBoundGRG._key`` degraded a cache miss to the plain graph, discarding the
caller's ``init`` and ``miss``. The tests here pin (a) that every key a
``make_runconfig_*`` factory declares is actually captured and self-consistent,
(b) that every captured key agrees with the eager path, and (c) that an
uncaptured key fails loudly.

The factory snapshots are CPU-only and cost nothing, which matters: the whole
class of defect these guard against is a run-configuration mistake, and until now
nothing checked a run configuration at all without a GPU.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from pygrgl_spmv import (
    CaptureSpec,
    RunConfigs,
    RuntimeRequirements,
    load_grg_spmv_single,
    make_backend_cusparse,
    make_runconfig_bolt,
    make_runconfig_gwas,
    make_runconfig_kernel,
    make_runconfig_pca,
    testing,
)
from pygrgl_spmv.adaptor import _validate_capture_spec
from pygrgl_spmv.tests.conftest import DATA_DTYPE, tol


def _keys(runconfig) -> set[tuple]:
    return {
        (s.direction, s.by_individual, s.init_mode, s.use_miss, s.emit_all_nodes)
        for s in runconfig.capture_ops
    }


_FACTORIES = [
    pytest.param(lambda: make_runconfig_kernel("up", 1), "kernel", id="kernel"),
    pytest.param(lambda: make_runconfig_kernel("down", 4), "kernel-down-k4", id="kernel-down-k4"),
    pytest.param(make_runconfig_pca, "pca", id="pca"),
    pytest.param(lambda: make_runconfig_pca(maxk=4), "pca-maxk4", id="pca-maxk4"),
    pytest.param(lambda: make_runconfig_pca(force_spmm=True), "pca-spmm", id="pca-spmm"),
    pytest.param(make_runconfig_bolt, "bolt", id="bolt"),
    pytest.param(lambda: make_runconfig_bolt(force_spmm=True), "bolt-spmm", id="bolt-spmm"),
    pytest.param(make_runconfig_gwas, "gwas", id="gwas"),
    pytest.param(lambda: make_runconfig_gwas(maxk=4), "gwas-maxk4", id="gwas-maxk4"),
    pytest.param(lambda: make_runconfig_gwas(sample_variance=False), "gwas-binom", id="gwas-binom"),
    pytest.param(lambda: make_runconfig_gwas(force_spmm=True), "gwas-spmm", id="gwas-spmm"),
]


# ---------------------------------------------------------------------------
# Factory self-consistency -- CPU only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("factory", "name"), _FACTORIES)
def test_every_capture_spec_is_legal_against_its_own_requirements(factory, name):
    """A spec that its own RuntimeRequirements forbids is a config bug.

    _validate_capture_spec previously ran only inside _capture_grg, i.e. only on
    a GPU during a real load, so a factory could ship an impossible spec and
    nothing would notice until someone ran it.
    """
    runconfig = factory()
    for spec in runconfig.capture_ops:
        _validate_capture_spec(spec, runconfig.req)


@pytest.mark.parametrize(("factory", "name"), _FACTORIES)
def test_capture_specs_are_unique(factory, name):
    specs = factory().capture_ops
    assert len(_keys(factory())) == len(specs), "duplicate CaptureSpec would raise at capture time"


@pytest.mark.parametrize(("factory", "name"), _FACTORIES)
def test_declared_need_flags_are_actually_used(factory, name):
    """Every need_* flag must be backed by a spec that uses it.

    An unused flag is not free: dropping the unused need_init_vector from
    make_runconfig_pca shrank the planned layout by num_nodes * itemsize, which
    scales with graph size.
    """
    runconfig = factory()
    keys = _keys(runconfig)
    init_modes = {k[2] for k in keys}
    if runconfig.req.need_init_vector:
        assert "vector" in init_modes, f"{name} declares need_init_vector but captures no vector init"
    if runconfig.req.need_init_matrix:
        assert "matrix" in init_modes, f"{name} declares need_init_matrix but captures no matrix init"
    if runconfig.req.need_init_xtx:
        assert "xtx" in init_modes, f"{name} declares need_init_xtx but captures no xtx init"
    if runconfig.req.need_up_miss_output:
        assert any(k[0] == "up" and k[3] for k in keys)
    if runconfig.req.need_down_miss_input:
        assert any(k[0] == "down" and k[3] for k in keys)


def test_runconfig_key_snapshot():
    """Frozen snapshot of each factory's capture set.

    This is the regression af892e3 ("Fix PCA captured ops") was, and the same
    class of mistake was live in two more places until strict _key exposed it.
    A deliberate change here should be a visible diff, not a silent one.
    """
    assert _keys(make_runconfig_pca()) == {
        ("up", False, "none", False, False),
        ("up", False, "none", True, False),
        ("up", True, "none", False, False),
        ("down", True, "none", False, False),
        # haploid=True gives by_individual=False, and eigsh drives both directions.
        ("down", False, "none", False, False),
    }
    assert _keys(make_runconfig_gwas()) == {
        ("up", False, "none", False, False),
        ("up", False, "none", True, False),
        ("up", False, "xtx", False, False),
        ("up", True, "none", False, False),
        ("up", True, "none", True, False),
    }
    assert _keys(make_runconfig_gwas(sample_variance=False)) == {
        ("up", False, "none", False, False),
        ("up", False, "none", True, False),
        ("up", True, "none", False, False),
        ("up", True, "none", True, False),
    }
    assert _keys(make_runconfig_bolt()) == {
        ("up", False, "none", False, False),
        ("up", False, "none", True, False),
        ("up", False, "xtx", False, False),
        ("up", True, "none", False, False),
        ("up", True, "none", True, False),
        ("up", True, "xtx", False, False),
        ("down", False, "none", False, False),
        ("down", False, "none", True, False),
        ("down", True, "none", False, False),
        ("down", True, "none", True, False),
    }
    assert _keys(make_runconfig_kernel("up", 1)) == {("up", False, "none", False, False)}


def test_grapp_diag_xtx_idiom_is_captured_by_both_factories():
    """``matmul(ones, UP, init="xtx")`` uses the by_individual=False default.

    grapp's two diag(X^T X) call sites (util/simple.py::variance and
    assoc::_computeDiagXTX) rely on that default. Neither bolt nor gwas captured
    it, so the old fallback silently returned plain allele counts instead -- and
    because g**2 == g for 0/1 genotypes, the leading values matched.
    """
    idiom = ("up", False, "xtx", False, False)
    assert idiom in _keys(make_runconfig_bolt())
    assert idiom in _keys(make_runconfig_gwas())


def test_plain_up_is_captured_wherever_the_fallback_used_to_land():
    """The old fallback substituted ('up', False, 'none', False).

    Under every gwas variant that key was itself uncaptured, so
    allele_counts(grg) and allele_frequencies(grg) died on a bare KeyError.
    """
    fallback = ("up", False, "none", False, False)
    for factory in (make_runconfig_gwas, lambda: make_runconfig_gwas(sample_variance=False),
                    make_runconfig_pca, make_runconfig_bolt):
        assert fallback in _keys(factory()), f"{factory} still cannot serve a plain UP matmul"


def test_force_spmm_widens_k_only_where_it_can():
    """force_spmm exists to escape k=1; at maxk>=2 the path is already SpMM.

    Not a defect -- the capture widths are byte-identical -- but it is documented
    as the CUDA precision workaround, so pin the actual semantics.
    """
    assert all(s.k == 1 for s in make_runconfig_pca().capture_ops)
    assert all(s.k == 2 for s in make_runconfig_pca(force_spmm=True).capture_ops)
    assert all(s.k == 4 for s in make_runconfig_pca(maxk=4).capture_ops)
    assert all(s.k == 4 for s in make_runconfig_pca(maxk=4, force_spmm=True).capture_ops)


# ---------------------------------------------------------------------------
# Replay behaviour -- needs a GPU
#
# Marked per test, not with a module-level `pytestmark`: that would also mark the
# CPU-only factory tests above, which exist to run without a GPU.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _pair(artifact, runconfig, device=0):
    """Yield (eager, captured) GRGs for the same artifact and run configuration."""
    with contextlib.ExitStack() as stack:
        eager = load_grg_spmv_single(
            artifact, make_backend_cusparse(device=device), runconfig, stack
        )
        captured = load_grg_spmv_single(
            artifact, make_backend_cusparse(device=device, capture=True), runconfig, stack
        )
        yield eager, captured


def _operand(grg, spec):
    if spec.direction == "down":
        cols = grg.num_mutations
    else:
        cols = grg.num_individuals if spec.by_individual else grg.num_samples
    return np.random.default_rng(9091).standard_normal((spec.k, cols), dtype=DATA_DTYPE)


@pytest.mark.cuda13
@pytest.mark.cusparse
@pytest.mark.parametrize(("factory", "name"), _FACTORIES)
def test_every_captured_key_matches_the_eager_path(missing_artifact, factory, name):
    """Parity for the whole capture surface, on the fixture that has missingness.

    Compared per element with an absolute floor, never as a scalar fingerprint:
    replay is not bit-deterministic (multi-stream accumulation gives ~1e-14
    run-to-run drift), and a summed comparison produces spurious mismatches.
    """
    runconfig = factory()
    atol, rtol = tol(DATA_DTYPE)
    with _pair(missing_artifact, runconfig) as (eager, captured):
        for spec in runconfig.capture_ops:
            x = _operand(captured, spec)
            kwargs = {"by_individual": spec.by_individual}
            if spec.init_mode == "xtx":
                kwargs["init"] = "xtx"
            miss_shape = (spec.k, captured.num_mutations)
            expected = eager.matmul(
                x, spec.direction,
                **kwargs,
                **({"miss": np.zeros(miss_shape, dtype=DATA_DTYPE)} if spec.use_miss else {}),
            )
            got = captured.matmul(
                x, spec.direction,
                **kwargs,
                **({"miss": np.zeros(miss_shape, dtype=DATA_DTYPE)} if spec.use_miss else {}),
            )
            np.testing.assert_allclose(
                got, expected, atol=atol, rtol=rtol,
                err_msg=f"{name}: captured != eager for {spec}",
            )


@pytest.mark.cuda13
@pytest.mark.cusparse
def test_uncaptured_key_raises_instead_of_returning_the_no_init_answer(missing_artifact):
    """The headline fix: this used to return a confidently wrong number."""
    runconfig = make_runconfig_pca()
    with contextlib.ExitStack() as stack:
        grg = load_grg_spmv_single(
            missing_artifact, make_backend_cusparse(device=0, capture=True), runconfig, stack
        )
        x = np.zeros((1, grg.num_individuals), dtype=DATA_DTYPE)
        with pytest.raises(ValueError, match="no CUDA graph was captured"):
            grg.matmul(x, "up", by_individual=True, init=np.array([3.0]))


@pytest.mark.cuda13
@pytest.mark.cusparse
def test_uncaptured_key_error_names_both_sides(missing_artifact):
    with contextlib.ExitStack() as stack:
        grg = load_grg_spmv_single(
            missing_artifact, make_backend_cusparse(device=0, capture=True),
            make_runconfig_kernel("up", 1), stack
        )
        with pytest.raises(ValueError) as excinfo:
            grg.matmul(np.ones((1, grg.num_mutations), dtype=DATA_DTYPE), "down")
    message = str(excinfo.value)
    assert "direction='down'" in message
    assert "Captured keys:" in message
    assert "('up', False, 'none', False, False)" in message


@pytest.mark.cuda13
@pytest.mark.cusparse
@pytest.mark.parametrize("payload", ["init", "miss"])
def test_short_init_or_miss_is_rejected_not_broadcast(missing_artifact, payload):
    """A k-1 init against a k=2 input used to be silently duplicated."""
    runconfig = RunConfigs(
        req=RuntimeRequirements(
            max_k_up=2, max_k_down=2, need_down_miss_input=True, need_up_miss_output=True,
            need_init_vector=True, need_init_matrix=False, need_init_xtx=False,
        ),
        capture_ops=(
            CaptureSpec("up", 2, init_mode="vector"),
            CaptureSpec("up", 2, use_miss=True),
        ),
    )
    with contextlib.ExitStack() as stack:
        grg = load_grg_spmv_single(
            missing_artifact, make_backend_cusparse(device=0, capture=True), runconfig, stack
        )
        x = np.ones((2, grg.num_samples), dtype=DATA_DTYPE)
        kwargs = (
            {"init": np.ones(1, dtype=DATA_DTYPE)}
            if payload == "init"
            else {"miss": np.zeros((1, grg.num_mutations), dtype=DATA_DTYPE)}
        )
        with pytest.raises(ValueError, match="shape mismatch"):
            grg.matmul(x, "up", **kwargs)


@pytest.mark.cuda13
@pytest.mark.cusparse
@pytest.mark.parametrize(
    "make_bad",
    [
        pytest.param(lambda g: np.asarray(1.0, dtype=DATA_DTYPE), id="rank-0"),
        pytest.param(lambda g: np.zeros((0, g.num_samples), dtype=DATA_DTYPE), id="k-zero"),
    ],
)
def test_degenerate_input_shapes_raise_like_the_eager_path(missing_artifact, make_bad):
    """Used to raise IndexError, and to return a silently empty answer, where the
    eager path raises ValueError for both."""
    with contextlib.ExitStack() as stack:
        grg = load_grg_spmv_single(
            missing_artifact, make_backend_cusparse(device=0, capture=True),
            make_runconfig_pca(maxk=2), stack,
        )
        with pytest.raises(ValueError):
            grg.matmul(make_bad(grg), "up")


@pytest.mark.cuda13
@pytest.mark.cusparse
def test_short_input_is_still_zero_padded_and_truncated(missing_artifact):
    """The pad/truncate path is load-bearing for any block solver that varies k."""
    runconfig = make_runconfig_pca(maxk=4)
    atol, rtol = tol(DATA_DTYPE)
    with _pair(missing_artifact, runconfig) as (eager, captured):
        for k in (1, 2, 3, 4):
            x = np.random.default_rng(31).standard_normal((k, captured.num_samples), dtype=DATA_DTYPE)
            got = captured.matmul(x, "up")
            assert got.shape == (k, captured.num_mutations)
            np.testing.assert_allclose(got, eager.matmul(x, "up"), atol=atol, rtol=rtol)


@pytest.mark.cuda13
@pytest.mark.cusparse
@pytest.mark.parametrize("direction", ["up", "down"])
def test_emit_all_nodes_is_supported_when_captured(missing_artifact, missing_grg, direction):
    """It used to be an ``assert``, which python -O strips.

    grapp forwards the flag, so under -O a caller asking for all-node output
    silently received endpoint output instead.
    """
    import pygrgl

    runconfig = RunConfigs(
        req=RuntimeRequirements(
            max_k_up=2, max_k_down=2, need_down_miss_input=False, need_up_miss_output=False,
            need_init_vector=False, need_init_matrix=False, need_init_xtx=False,
        ),
        capture_ops=(CaptureSpec(direction, 2, emit_all_nodes=True),),
    )
    cols = missing_grg.num_samples if direction == "up" else missing_grg.num_mutations
    x = np.random.default_rng(77).standard_normal((2, cols), dtype=DATA_DTYPE)
    atol, rtol = tol(DATA_DTYPE)
    with contextlib.ExitStack() as stack:
        grg = load_grg_spmv_single(
            missing_artifact, make_backend_cusparse(device=0, capture=True), runconfig, stack
        )
        got = grg.matmul(x, direction, emit_all_nodes=True)
    assert got.shape == (2, missing_grg.num_nodes)
    tdir = pygrgl.TraversalDirection.UP if direction == "up" else pygrgl.TraversalDirection.DOWN
    expected = np.asarray(pygrgl.matmul(missing_grg, x, tdir, emit_all_nodes=True))
    np.testing.assert_allclose(got, expected, atol=atol, rtol=rtol)


@pytest.mark.cuda13
@pytest.mark.cusparse
def test_emit_all_nodes_raises_when_not_captured(missing_artifact):
    with contextlib.ExitStack() as stack:
        grg = load_grg_spmv_single(
            missing_artifact, make_backend_cusparse(device=0, capture=True),
            make_runconfig_pca(), stack
        )
        with pytest.raises(ValueError, match="emit_all_nodes=True"):
            grg.matmul(np.ones((1, grg.num_samples), dtype=DATA_DTYPE), "up", emit_all_nodes=True)


@pytest.mark.cuda13
@pytest.mark.cusparse
def test_load_many_returns_grgs_in_input_order(missing_artifact, primary_artifact):
    """mikado zips chromosome labels against this list positionally."""
    from pygrgl_spmv import load_grg_spmv_multi

    paths = [missing_artifact, primary_artifact, missing_artifact]
    with contextlib.ExitStack() as stack:
        grgs = load_grg_spmv_multi(
            paths, make_backend_cusparse(device=0), make_runconfig_kernel("up", 1), stack
        )
    assert [g.artifact_path for g in grgs] == paths


@pytest.mark.cuda13
@pytest.mark.cusparse
def test_same_device_grgs_share_one_lock_and_runtime(missing_artifact, primary_artifact):
    from pygrgl_spmv import load_grg_spmv_multi

    with contextlib.ExitStack() as stack:
        grgs = load_grg_spmv_multi(
            [missing_artifact, primary_artifact],
            make_backend_cusparse(device=0, capture=True),
            make_runconfig_kernel("up", 1),
            stack,
        )
        assert grgs[0]._device_lock is grgs[1]._device_lock
        assert grgs[0]._grg._runtime is grgs[1]._grg._runtime


@pytest.mark.cuda13
@pytest.mark.cusparse
def test_release_drops_the_graphs_and_staging_buffers(missing_artifact):
    """The release hook used to set a bool and nothing else, so a retained handle
    pinned the graph pools and staging arenas: measured +46 MiB per load.

    Hand-rolled rather than a make_runconfig_* factory because no factory populates
    _init_tensors: only vector/matrix init is staged, and xtx is not. Under
    make_runconfig_pca two of the five assertions below held before release too.
    """
    runconfig = RunConfigs(
        req=RuntimeRequirements(
            max_k_up=2, max_k_down=2, need_down_miss_input=True, need_up_miss_output=False,
            need_init_vector=True, need_init_matrix=False, need_init_xtx=False,
        ),
        capture_ops=(
            CaptureSpec("up", 2, init_mode="vector"),
            CaptureSpec("down", 2, use_miss=True),
            CaptureSpec("up", 2),
        ),
    )
    owned_names = ("_graphs", "_src_tensors", "_init_tensors", "_miss_tensors", "_prepared_ops")
    with contextlib.ExitStack() as stack:
        grg = load_grg_spmv_single(
            missing_artifact, make_backend_cusparse(device=0, capture=True), runconfig, stack
        )
        for name in owned_names:
            assert getattr(grg, name), f"{name} empty before release; the check below is vacuous"
    assert grg._closed
    for name in owned_names:
        assert getattr(grg, name) == {}


@pytest.mark.cuda13
@pytest.mark.cusparse
def test_up_only_missingness_allocates_no_miss_staging_buffer(missing_artifact, monkeypatch):
    """Sizing shared_miss from UP reserved k * num_mutations that nothing aliased."""
    runconfig = make_runconfig_gwas(maxk=2)
    assert not any(s.direction == "down" for s in runconfig.capture_ops)
    assert runconfig.req.need_up_miss_output and not runconfig.req.need_down_miss_input

    # shared_miss is a local in _capture_grg and never reaches the wrapper when no DOWN
    # spec aliases it, so _miss_tensors == {} holds either way. Watch the allocation
    # itself: sizing it from UP would ask for exactly k * num_mutations elements.
    import torch

    sizes: list[int] = []
    real_zeros = torch.zeros

    def recording_zeros(*args, **kwargs):
        if args and isinstance(args[0], int):
            sizes.append(args[0])
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", recording_zeros)
    with contextlib.ExitStack() as stack:
        grg = load_grg_spmv_single(
            missing_artifact, make_backend_cusparse(device=0, capture=True), runconfig, stack
        )
        monkeypatch.undo()
        assert grg._miss_tensors == {}
        forbidden = 2 * grg.num_mutations
        assert sizes, "no flat torch.zeros seen; the watch is not wired up"
        assert forbidden not in sizes, (
            f"shared_miss was still sized from UP's miss_output ({forbidden} elements)"
        )
        ones = np.ones((1, grg.num_samples), dtype=DATA_DTYPE)
        miss = np.zeros((1, grg.num_mutations), dtype=DATA_DTYPE)
        grg.matmul(ones, "up", miss=miss)
        assert miss.sum() > 0, "fixture has no missingness; the test would be vacuous"


@pytest.mark.cuda13
@pytest.mark.cusparse
def test_captured_miss_with_emit_all_nodes_fails_like_the_eager_path(missing_artifact):
    """grapp forwards both flags, and strict _key would answer with a wrong exception
    type and a remedy that _prepare_cuda_spec refuses to build."""
    with contextlib.ExitStack() as stack:
        grg = load_grg_spmv_single(
            missing_artifact, make_backend_cusparse(device=0, capture=True),
            make_runconfig_gwas(maxk=1), stack,
        )
        x = np.ones((1, grg.num_samples), dtype=DATA_DTYPE)
        miss = np.zeros((1, grg.num_mutations), dtype=DATA_DTYPE)
        with pytest.raises(RuntimeError, match="cannot be mixed with"):
            grg.matmul(x, "up", miss=miss, emit_all_nodes=True)


_INIT_VECTOR_RUNCONFIG = RunConfigs(
    req=RuntimeRequirements(
        max_k_up=2, max_k_down=2, need_down_miss_input=False, need_up_miss_output=True,
        need_init_vector=True, need_init_matrix=False, need_init_xtx=False,
    ),
    capture_ops=(
        CaptureSpec("up", 2, init_mode="vector"),
        CaptureSpec("up", 2, use_miss=True),
        CaptureSpec("up", 2),
    ),
)


@pytest.mark.cuda13
@pytest.mark.cusparse
@pytest.mark.parametrize("payload", ["init", "miss"])
def test_init_and_miss_must_share_the_inputs_dtype(primary_artifact, missing_artifact, payload):
    """An int32 input with a float64 init used to be accepted and then truncated to the
    int32 output, losing the init's fractional part with no signal -- verified as
    [135 494 225 197] where float64 gives [135.5 494. 225.5 197.]. Both arrays are
    validated against the *capture* dtype, which allows int32 wherever float64 is
    expected, so neither check caught the mismatch between them. Eager has always
    raised; these now agree, message included."""
    artifact = missing_artifact if payload == "miss" else primary_artifact
    with contextlib.ExitStack() as stack:
        grg = load_grg_spmv_single(
            artifact, make_backend_cusparse(device=0, capture=True), _INIT_VECTOR_RUNCONFIG, stack
        )
        int_x = np.ones((1, grg.num_samples), dtype=np.int32)
        f64_x = np.ones((1, grg.num_samples), dtype=DATA_DTYPE)

        def call(x, dtype):
            arg = (
                {"init": np.array([0.5], dtype=dtype)}
                if payload == "init"
                else {"miss": np.zeros((1, grg.num_mutations), dtype=dtype)}
            )
            return grg.matmul(x, "up", **arg)

        for x, dtype in ((int_x, np.float64), (f64_x, np.int32)):
            with pytest.raises(TypeError, match="must match the dtype of the input matrix"):
                call(x, dtype)

        # Matching dtypes still work, and an int32 input with no payload at all still
        # returns int32 -- the one divergence from eager that is deliberate.
        call(f64_x, np.float64)
        call(int_x, np.int32)
        assert grg.matmul(int_x, "up").dtype == np.int32
