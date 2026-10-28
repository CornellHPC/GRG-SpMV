from __future__ import annotations

from contextlib import ExitStack
import functools
from types import SimpleNamespace

import numpy as np
import pygrgl
import pytest
import scipy.sparse as sp

from pygrgl_spmv.backends.base import materialize_sparse_block
from pygrgl_spmv.backends.cusparse import CusparsePlan, CusparsePlanPair, CusparseRuntime
from pygrgl_spmv.backends.cusparse.backend import _block_plan
from pygrgl_spmv.backends.cusparse.ffi import CudaVmmDriver
from pygrgl_spmv.backends.reference import ReferenceRuntime
from pygrgl_spmv.backends.types import Direction, SparseFormat
from pygrgl_spmv.grg.artifact import scan_grg_spmv
from pygrgl_spmv.tests.conftest import DATA_DTYPE, tol
from pygrgl_spmv.tests.runtime._runtime_builders import (
    build_cusparse_layout,
    build_reference_layout,
    full_requirements,
)
from pygrgl_spmv.tests.runtime._streaming_cases import (
    THREE_BLOCK_EXACTNESS_MODES,
    THREE_BLOCK_TRANSITION_MODES,
    ThreeBlockMode,
    _equal_block_budget_components,
    clear_cupy_state,
    cusparse_ring_thresholds,
    expected_down,
    expected_up,
    prepare_cusparse_stream_stress_case,
    three_block_mode_budget_bytes,
    write_overlap_band_artifact,
    write_three_level_band_artifact,
)

cp = pytest.importorskip("cupy")

pytestmark = [pytest.mark.cusparse]

_MODE_BY_NAME = {mode.name: mode for mode in THREE_BLOCK_TRANSITION_MODES}
_THREE_BLOCK_KEYS = ((1, 0), (2, 0), (2, 1))


def _requirements(runtime_k: int):
    return full_requirements(max_k_up=int(runtime_k), max_k_down=int(runtime_k))


def _cupy_device_nbytes(value, seen: set[int]) -> int:
    if value is None or not hasattr(value, "data"):
        return 0
    nbytes = int(value.nbytes)
    if nbytes == 0:
        return 0
    ptr = int(value.data.ptr)
    if ptr in seen:
        return 0
    seen.add(ptr)
    return nbytes


def _cupy_nested_nbytes(values, seen: set[int]) -> int:
    if values is None:
        return 0
    total = 0
    for value in values:
        if isinstance(value, (list, tuple)):
            total += _cupy_nested_nbytes(value, seen)
        else:
            total += _cupy_device_nbytes(value, seen)
    return total


def _shared_ones_nbytes(shared_ones, seen: set[int]) -> int:
    if shared_ones is None or int(shared_ones.physical_nbytes) == 0:
        return 0
    if shared_ones._materialized is not None:
        return _cupy_device_nbytes(shared_ones._materialized, seen)
    ptr = int(shared_ones.ptr)
    if ptr in seen:
        return 0
    seen.add(ptr)
    return int(shared_ones.physical_nbytes)


def _cusparse_owned_device_nbytes(runtime: CusparseRuntime) -> int:
    seen: set[int] = set()
    total = _shared_ones_nbytes(runtime._shared_ones, seen)
    for value in (runtime._alpha, runtime._beta_zero, runtime._beta_one, runtime._io0, runtime._io1, runtime._aux):
        total += _cupy_device_nbytes(value, seen)
    for values in (
        runtime._slot_struct0,
        runtime._slot_struct1,
        runtime._up_level_bufs,
        runtime._down_level_bufs,
        runtime._up_src_bufs,
        runtime._down_src_bufs,
        runtime._up_scratch,
        runtime._down_scratch,
        runtime._ext_main_up,
        runtime._ext_main_down,
        runtime._ext_scratch_up,
        runtime._ext_scratch_down,
    ):
        total += _cupy_nested_nbytes(values, seen)
    for artifact in runtime._artifacts:
        for selector in (artifact.mut_selector, artifact.miss_selector):
            total += _cupy_nested_nbytes(selector.rows_by_level, seen)
            total += _cupy_nested_nbytes(selector.cols_by_level, seen)
        for value in (
            artifact.node_perm,
            artifact.sample_to_individual,
            artifact.xtx_bias,
            artifact.init_vector_up_bias,
            artifact.init_vector_down_bias,
            artifact.init_xtx_up_bias,
            artifact.init_xtx_down_bias,
        ):
            total += _cupy_device_nbytes(value, seen)
        for ops_by_level in (artifact.up_ops, artifact.down_ops):
            for ops in ops_by_level:
                for op in ops:
                    total += _cupy_device_nbytes(op.block.struct0, seen)
                    total += _cupy_device_nbytes(op.block.struct1, seen)
    return total


def _owner_keys(layout, *, resident: bool) -> tuple[tuple[int, int], ...]:
    keys = [
        (int(block.dst_level), int(block.src_level))
        for artifact in layout.artifacts
        for block in (*artifact.blocks_up, *artifact.blocks_down)
        if bool(block.resident) == bool(resident)
    ]
    return tuple(sorted(keys))


def _nonzero_slot_count(layout) -> int:
    return sum(1 for slot in layout.slot_plans if slot.nbytes > 0)


def _build_layout(artifact, *, runtime_k: int, ring_buffer_size: int, budget_bytes: int, allow_residency: bool = True, stream=0):
    return build_cusparse_layout(
        [artifact],
        requirements=_requirements(runtime_k),
        ring_buffer_size=int(ring_buffer_size),
        vram_budget_bytes=int(budget_bytes),
        allow_residency=bool(allow_residency),
        stream=stream,
    )


def test_cusparse_layout_planning_uses_sparse_metadata_only(tmp_path, monkeypatch):
    artifact = write_three_level_band_artifact(tmp_path, "metadata-only-cusparse", n=8, bandwidth=2)

    def _fail_sparse_block_read(_path):
        raise AssertionError("cuSPARSE layout planning should not read sparse block arrays")

    def _fail_host_load(*_args, **_kwargs):
        raise AssertionError("cuSPARSE layout planning should not load artifact selectors")

    monkeypatch.setattr("pygrgl_spmv.backends.cusparse.backend.iter_artifact_blocks", _fail_sparse_block_read)
    monkeypatch.setattr("pygrgl_spmv.backends.cusparse.backend._load_grg_spmv_host", _fail_host_load)
    layout = build_cusparse_layout([artifact], requirements=_requirements(2))
    assert layout.max_levels == 3
    assert len(layout.ext_main_up) == layout.max_levels
    assert len(layout.ext_main_down) == layout.max_levels
    assert len(layout.ext_scratch_up) == layout.max_levels
    assert len(layout.ext_scratch_down) == layout.max_levels
    assert layout.bytes_by_category["ext"] >= 0
    assert layout.budget_items
    assert layout.bytes_total > 0


def _full_budget_components(artifact, *, runtime_k: int) -> tuple[int, int, int]:
    layout = build_cusparse_layout(
        [artifact],
        requirements=_requirements(runtime_k),
        ring_buffer_size=0,
        vram_budget_bytes=1_000_000_000_000,
    )
    return _equal_block_budget_components(layout)


def _source_cols(grg, direction: str) -> int:
    return int(grg.num_samples if direction == "up" else grg.num_mutations)


def _expected_prepared_chain(ref_grg, order, src: np.ndarray, init_first: np.ndarray, init_second: np.ndarray) -> np.ndarray:
    mid = ref_grg.matmul(src, order[0], init=init_first)
    return ref_grg.matmul(mid, order[1], init=init_second)


def _enter_prepared_chain(stack: ExitStack, grg, order, *, k: int):
    first = stack.enter_context(grg.prepare_matmul_cuda(direction=order[0], k=int(k), init_mode="vector"))
    second = stack.enter_context(grg.prepare_matmul_cuda(direction=order[1], k=int(k), init_mode="vector"))
    assert first.output.data_ptr() == second.input.data_ptr()
    return first, second


def _capture_prepared_chain(first, second, src, init_first, init_second, result=None) -> None:
    first.input.copy_(src)
    first.init_vector.copy_(init_first)
    first()
    second.init_vector.copy_(init_second)
    second()
    if result is not None:
        result.copy_(second.output)


def _build_multi_grg_graph_layout(artifacts, *, streamed: bool, stream):
    layout = build_cusparse_layout(
        artifacts,
        requirements=_requirements(1),
        ring_buffer_size=2 if streamed else 0,
        vram_budget_bytes=1_000_000_000,
        allow_residency=not bool(streamed),
        stream=stream,
    )
    if streamed:
        assert layout.allow_residency is False
        assert layout.requested_ring_buffer_size == 2
        assert layout.allocated_ring_buffer_size == 2
        assert all(
            not block.resident
            for artifact in layout.artifacts
            for block in (*artifact.blocks_up, *artifact.blocks_down)
        )
    else:
        assert layout.allocated_ring_buffer_size == 0
    return layout


def _assert_budget_accounting(layout, *, chosen_budget: int, fixed_bytes: int, block_bytes: int) -> None:
    assert sum(item.nbytes for item in layout.budget_items) == layout.bytes_total
    assert layout.required_budget_for_full_residency == int(fixed_bytes + 3 * block_bytes)
    assert layout.bytes_total <= int(chosen_budget) <= layout.required_budget_for_full_residency


def _assert_all_streamed_no_residency(layout, *, requested_ring_buffer_size: int, expected_slot_count: int, fixed_bytes: int, block_bytes: int) -> None:
    assert layout.allow_residency is False
    assert layout.requested_ring_buffer_size == int(requested_ring_buffer_size)
    assert layout.allocated_ring_buffer_size == int(expected_slot_count)
    assert _owner_keys(layout, resident=True) == ()
    assert _owner_keys(layout, resident=False) == _THREE_BLOCK_KEYS
    assert layout.bytes_by_category["resident_sparse"] == 0
    resident_items = [item for item in layout.budget_items if item.kind == "resident_sparse"]
    slot_items = [item for item in layout.budget_items if item.kind == "ring_slot"]
    assert resident_items == []
    assert len(slot_items) == int(expected_slot_count)
    assert layout.bytes_total == int(fixed_bytes + expected_slot_count * block_bytes)
    assert layout.required_budget_for_full_residency == int(fixed_bytes + 3 * block_bytes)


def _assert_three_block_mode(layout, mode: ThreeBlockMode, *, chosen_budget: int, fixed_bytes: int, block_bytes: int) -> None:
    assert layout.requested_ring_buffer_size == int(mode.requested_ring_buffer_size)
    assert layout.allocated_ring_buffer_size == int(mode.slot_count)
    assert _owner_keys(layout, resident=True) == tuple(sorted(mode.resident_blocks))
    assert _owner_keys(layout, resident=False) == tuple(sorted(mode.streamed_blocks))
    assert _nonzero_slot_count(layout) == int(mode.slot_count)
    resident_items = [item for item in layout.budget_items if item.kind == "resident_sparse"]
    slot_items = [item for item in layout.budget_items if item.kind == "ring_slot"]
    assert len(resident_items) == len(mode.resident_blocks)
    assert len(slot_items) == int(mode.slot_count)
    _assert_budget_accounting(layout, chosen_budget=chosen_budget, fixed_bytes=fixed_bytes, block_bytes=block_bytes)


def _below_transition(mode: ThreeBlockMode) -> ThreeBlockMode | None:
    mapping = {
        "ring0-resident": None,
        "ring1-stream": None,
        "ring1-hybrid": _MODE_BY_NAME["ring1-stream"],
        "ring1-resident": _MODE_BY_NAME["ring1-hybrid"],
        "ring2-stream": None,
        "ring2-resident": _MODE_BY_NAME["ring2-stream"],
        "ring3-resident": None,
    }
    return mapping[mode.name]


def _build_with_mode_warning(artifact, *, runtime_k: int, mode: ThreeBlockMode, fixed_bytes: int, block_bytes: int):
    budget_bytes = three_block_mode_budget_bytes(mode, fixed_bytes=fixed_bytes, block_bytes=block_bytes)
    if mode.expect_warning:
        with pytest.warns(RuntimeWarning, match="requested ring_buffer_size"):
            layout = _build_layout(
                artifact,
                runtime_k=runtime_k,
                ring_buffer_size=mode.requested_ring_buffer_size,
                budget_bytes=budget_bytes,
            )
    else:
        layout = _build_layout(
            artifact,
            runtime_k=runtime_k,
            ring_buffer_size=mode.requested_ring_buffer_size,
            budget_bytes=budget_bytes,
        )
    return layout, budget_bytes


def _assert_planner_transition(artifact, *, runtime_k: int, mode: ThreeBlockMode) -> None:
    fixed_bytes, block_bytes, owner_block_count = _full_budget_components(artifact, runtime_k=runtime_k)
    assert owner_block_count == 3
    layout, budget_bytes = _build_with_mode_warning(
        artifact,
        runtime_k=runtime_k,
        mode=mode,
        fixed_bytes=fixed_bytes,
        block_bytes=block_bytes,
    )
    _assert_three_block_mode(layout, mode, chosen_budget=budget_bytes, fixed_bytes=fixed_bytes, block_bytes=block_bytes)

    predecessor = _below_transition(mode)
    lower_budget = int(budget_bytes - 1)
    if predecessor is None:
        with pytest.raises(ValueError):
            _build_layout(
                artifact,
                runtime_k=runtime_k,
                ring_buffer_size=mode.requested_ring_buffer_size,
                budget_bytes=lower_budget,
            )
        return
    if predecessor.expect_warning:
        with pytest.warns(RuntimeWarning, match="requested ring_buffer_size"):
            lower_layout = _build_layout(
                artifact,
                runtime_k=runtime_k,
                ring_buffer_size=mode.requested_ring_buffer_size,
                budget_bytes=lower_budget,
            )
    else:
        lower_layout = _build_layout(
            artifact,
            runtime_k=runtime_k,
            ring_buffer_size=mode.requested_ring_buffer_size,
            budget_bytes=lower_budget,
        )
    _assert_three_block_mode(lower_layout, predecessor, chosen_budget=lower_budget, fixed_bytes=fixed_bytes, block_bytes=block_bytes)


def _run_exactness_case(artifact, case, *, runtime_k: int, mode: ThreeBlockMode, order, backend="cusparse") -> None:
    from pygrgl_spmv.tests.runtime._runtime_builders import build_layout_for_backend, runtime_cls_for_backend
    if backend == "rocsparse":
        assert mode.requested_ring_buffer_size == 0
        layout = build_layout_for_backend(backend, [artifact], requirements=_requirements(runtime_k))
    else:
        fixed_bytes, block_bytes, owner_block_count = _full_budget_components(artifact, runtime_k=runtime_k)
        assert owner_block_count == 3
        layout, budget_bytes = _build_with_mode_warning(
            artifact,
            runtime_k=runtime_k,
            mode=mode,
            fixed_bytes=fixed_bytes,
            block_bytes=block_bytes,
        )
        _assert_three_block_mode(layout, mode, chosen_budget=budget_bytes, fixed_bytes=fixed_bytes, block_bytes=block_bytes)
    with runtime_cls_for_backend(backend)(layout) as runtime:
        (grg,) = runtime.grgs
        for run_idx, direction in enumerate(order):
            seed = 41_000 + 1_000 * runtime_k + 100 * run_idx + sum(ord(ch) for ch in mode.name)
            rng = np.random.default_rng(seed)
            primary = rng.choice(np.array([-1.0, 1.0], dtype=DATA_DTYPE), size=(int(runtime_k), case.n))
            if direction == "up":
                expected = expected_up(primary.T, shifts=case.shifts, bandwidth=case.bandwidth).T
            else:
                expected = expected_down(primary.T, shifts=case.shifts, bandwidth=case.bandwidth).T
            actual = grg.matmul(primary, direction)
            np.testing.assert_array_equal(actual, expected)


@functools.cache
def _spin_kernel():
    module = cp.RawModule(
        code=r"""
        extern "C" __global__ void spin(unsigned long long iters) {
          unsigned long long start = clock64();
          while (clock64() - start < iters) {}
        }
        """,
    )
    return module.get_function("spin")


def _warm_spin_kernel() -> None:
    _spin_kernel()((1,), (1,), (1_000_000,))
    cp.cuda.runtime.deviceSynchronize()


def _run_overlap_case(tmp_path, order, requested_ring_buffer_size, monkeypatch, *, n: int, bandwidth: int, runtime_k: int) -> None:
    clear_cupy_state()
    _warm_spin_kernel()
    artifact = write_overlap_band_artifact(tmp_path, f"cusparse-overlap-{n}-{bandwidth}", n=n, bandwidth=bandwidth)
    budget_bytes = cusparse_ring_thresholds(
        artifact,
        requirements=_requirements(runtime_k),
        total_vram_bytes=prepare_cusparse_stream_stress_case().total_vram_bytes,
        max_ring_buffer_size=2,
    )[int(requested_ring_buffer_size) - 1]
    layout = _build_layout(
        artifact,
        runtime_k=runtime_k,
        ring_buffer_size=requested_ring_buffer_size,
        budget_bytes=budget_bytes,
    )
    ref_layout = build_reference_layout([artifact], requirements=_requirements(runtime_k))
    state = {"active": True, "compute": [], "copy": []}
    original_launch = CusparseRuntime._launch_spmm

    def _wrapped_launch(self, artifact, direction, op, dst_desc, beta_ptr, src_desc, ext) -> None:
        stream = cp.cuda.get_current_stream()
        if state["active"]:
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record(stream)
            original_launch(self, artifact, direction, op, dst_desc, beta_ptr, src_desc, ext)
            with stream:
                _spin_kernel()((1,), (1,), (10_000_000,))
            end.record(stream)
            state["compute"].append((start, end))
            return
        original_launch(self, artifact, direction, op, dst_desc, beta_ptr, src_desc, ext)

    def _wrapped_copy(self, copy_done, compute_done, dst_level: int, op_idx: int, op) -> None:
        if op.slot is None or op.host0 is None or op.host1 is None:
            return
        stream = self._slot_copy_streams[op.slot]
        with stream:
            if op.prev_in_slot is not None:
                prev_dst, prev_idx = op.prev_in_slot
                stream.wait_event(compute_done[prev_dst][prev_idx])
            start = None
            end = None
            if state["active"]:
                start = cp.cuda.Event()
                end = cp.cuda.Event()
                start.record(stream)
            cp.cuda.runtime.memcpyAsync(
                self._slot_struct0[op.slot].data.ptr,
                int(np.asarray(op.host0).ctypes.data),
                int(op.host0.nbytes),
                cp.cuda.runtime.memcpyHostToDevice,
                stream.ptr,
            )
            cp.cuda.runtime.memcpyAsync(
                self._slot_struct1[op.slot].data.ptr,
                int(np.asarray(op.host1).ctypes.data),
                int(op.host1.nbytes),
                cp.cuda.runtime.memcpyHostToDevice,
                stream.ptr,
            )
            if state["active"]:
                assert start is not None and end is not None
                end.record(stream)
                state["copy"].append((start, end))
            copy_done[dst_level][op_idx].record(stream)

    monkeypatch.setattr(CusparseRuntime, "_launch_spmm", _wrapped_launch)
    monkeypatch.setattr(CusparseRuntime, "_copy_to_slot", _wrapped_copy)

    def _has_overlap() -> bool:
        for compute_start, compute_end in state["compute"]:
            if cp.cuda.get_elapsed_time(compute_start, compute_end) <= 0.0:
                continue
            for copy_start, copy_end in state["copy"]:
                if cp.cuda.get_elapsed_time(copy_start, copy_end) <= 0.0:
                    continue
                if cp.cuda.get_elapsed_time(copy_start, compute_end) > 0.0 and cp.cuda.get_elapsed_time(compute_start, copy_end) > 0.0:
                    return True
        return False

    with ReferenceRuntime(ref_layout) as ref_runtime, CusparseRuntime(layout) as runtime:
        (ref_grg,) = ref_runtime.grgs
        (grg,) = runtime.grgs
        for run_idx, direction in enumerate(order):
            size = ref_grg.num_samples if direction == "up" else ref_grg.num_mutations
            rng = np.random.default_rng(51_000 + 1_000 * requested_ring_buffer_size + 10 * run_idx + (0 if direction == "up" else 1))
            primary = rng.choice(np.array([-1.0, 1.0], dtype=DATA_DTYPE), size=(int(runtime_k), size))
            expected = ref_grg.matmul(primary, direction)
            actual = grg.matmul(primary, direction)
            np.testing.assert_array_equal(actual, expected)
            cp.cuda.runtime.deviceSynchronize()
            if run_idx == 0:
                if int(requested_ring_buffer_size) == 1:
                    assert not _has_overlap()
                else:
                    assert _has_overlap()
                state["active"] = False
    clear_cupy_state()


@pytest.fixture(scope="module")
def cusparse_stream_stress_case():
    return prepare_cusparse_stream_stress_case()


@pytest.fixture(scope="module")
def gpu_small_stream_case():
    return SimpleNamespace(n=64, bandwidth=8, shifts=(0, 8, 16))


@pytest.fixture(scope="module")
def cusparse_small_stream_artifact(artifact_dir, gpu_small_stream_case):
    return write_three_level_band_artifact(
        artifact_dir,
        f"gpu-small-stream-n{gpu_small_stream_case.n}",
        n=gpu_small_stream_case.n,
        bandwidth=gpu_small_stream_case.bandwidth,
        shifts=gpu_small_stream_case.shifts,
    )


@pytest.fixture(scope="module")
def cusparse_stream_stress_artifact(artifact_dir, cusparse_stream_stress_case):
    return write_three_level_band_artifact(
        artifact_dir,
        f"gpu-stream-stress-n{cusparse_stream_stress_case.n}",
        n=cusparse_stream_stress_case.n,
        bandwidth=cusparse_stream_stress_case.bandwidth,
        shifts=cusparse_stream_stress_case.shifts,
    )


def test_cusparse_executes_on_foreign_torch_stream(primary_artifact, primary_grg):
    torch = pytest.importorskip("torch")
    with torch.cuda.device(0):
        requested = torch.cuda.Stream()
    layout = build_cusparse_layout([primary_artifact], device=0, stream=requested, requirements=_requirements(2))
    with CusparseRuntime(layout) as runtime:
        (grg,) = runtime.grgs
        rng = np.random.default_rng(7302)
        x_up = rng.standard_normal((2, primary_grg.num_samples), dtype=DATA_DTYPE)
        x_down = rng.standard_normal((2, primary_grg.num_mutations), dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(
            grg.matmul(x_up, "up"),
            np.asarray(pygrgl.matmul(primary_grg, x_up, pygrgl.TraversalDirection.UP)),
            atol=atol,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            grg.matmul(x_down, "down"),
            np.asarray(pygrgl.matmul(primary_grg, x_down, pygrgl.TraversalDirection.DOWN)),
            atol=atol,
            rtol=rtol,
        )


def test_cusparse_csc_block_plan_uses_separate_struct_dtypes_for_large_row_bound():
    huge = int(np.iinfo(np.int32).max) + 2
    fake = SimpleNamespace(dst_level=1, src_level=0, shape=(huge, 1), nnz=1)
    plan = CusparsePlan(store="N", fmt="CSC", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT")
    block = _block_plan(0, Direction.UP, False, fake, plan)
    assert block.stored_shape == (huge, 1)
    assert block.struct0_dtype == np.dtype(np.int32)
    assert block.struct1_dtype == np.dtype(np.int64)


def test_cusparse_owned_device_bytes_do_not_exceed_tight_budget(primary_artifact):
    clear_cupy_state()
    base = build_cusparse_layout([primary_artifact], requirements=_requirements(2), vram_budget_bytes=1_000_000_000_000)
    layout = build_cusparse_layout([primary_artifact], requirements=_requirements(2), vram_budget_bytes=base.bytes_total)
    assert layout.bytes_total == base.bytes_total
    with CusparseRuntime(layout) as runtime:
        assert _cusparse_owned_device_nbytes(runtime) <= layout.vram_budget_bytes
    clear_cupy_state()


@pytest.mark.parametrize(
    ("vram_budget_bytes", "ring_buffer_size", "case"),
    [
        pytest.param(0, 0, "full", id="zero-budget-ring0"),
        pytest.param(0, 1, "full_warn", id="zero-budget-ring1"),
        pytest.param("tight", 0, "ring_error", id="tight-ring0"),
        pytest.param("tight", 1, "streamed", id="tight-ring1"),
    ],
)
def test_cusparse_vram_budget_zero_sentinel_and_tight_ring(
    cusparse_small_stream_artifact,
    vram_budget_bytes,
    ring_buffer_size,
    case,
):
    full = _build_layout(
        cusparse_small_stream_artifact,
        runtime_k=2,
        ring_buffer_size=0,
        budget_bytes=0,
    )
    block_bytes = min(block.nbytes for artifact in full.artifacts for block in (*artifact.blocks_up, *artifact.blocks_down))
    budget = full.required_budget_for_full_residency - block_bytes if vram_budget_bytes == "tight" else vram_budget_bytes
    if case == "ring_error":
        with pytest.raises(ValueError, match="ring_buffer_size"):
            _build_layout(
                cusparse_small_stream_artifact,
                runtime_k=2,
                ring_buffer_size=ring_buffer_size,
                budget_bytes=budget,
            )
        return
    if case == "full_warn":
        with pytest.warns(RuntimeWarning, match="all sparse blocks fit resident"):
            layout = _build_layout(
                cusparse_small_stream_artifact,
                runtime_k=2,
                ring_buffer_size=ring_buffer_size,
                budget_bytes=budget,
            )
    else:
        layout = _build_layout(
            cusparse_small_stream_artifact,
            runtime_k=2,
            ring_buffer_size=ring_buffer_size,
            budget_bytes=budget,
        )
    if case in {"full", "full_warn"}:
        assert _owner_keys(layout, resident=False) == ()
        assert layout.allocated_ring_buffer_size == 0
        assert layout.vram_budget_bytes == layout.required_budget_for_full_residency
        assert layout.bytes_total == layout.required_budget_for_full_residency
    else:
        assert layout.allocated_ring_buffer_size == 1
        assert layout.bytes_total <= layout.vram_budget_bytes
        assert layout.bytes_total < layout.required_budget_for_full_residency


def test_cusparse_layout_rejects_negative_vram_budget(cusparse_small_stream_artifact):
    with pytest.raises(ValueError, match="vram_budget_bytes"):
        _build_layout(
            cusparse_small_stream_artifact,
            runtime_k=2,
            ring_buffer_size=0,
            budget_bytes=-1,
        )


@pytest.mark.parametrize("requested_ring_buffer_size", [1, 2, 3], ids=["ring1", "ring2", "ring3"])
def test_cusparse_allow_residency_false_forces_all_streamed_layout(cusparse_small_stream_artifact, requested_ring_buffer_size):
    fixed_bytes, block_bytes, owner_block_count = _full_budget_components(cusparse_small_stream_artifact, runtime_k=2)
    assert owner_block_count == 3
    budget_bytes = fixed_bytes + 3 * block_bytes
    layout = _build_layout(
        cusparse_small_stream_artifact,
        runtime_k=2,
        ring_buffer_size=requested_ring_buffer_size,
        budget_bytes=budget_bytes,
        allow_residency=False,
    )
    _assert_all_streamed_no_residency(
        layout,
        requested_ring_buffer_size=requested_ring_buffer_size,
        expected_slot_count=requested_ring_buffer_size,
        fixed_bytes=fixed_bytes,
        block_bytes=block_bytes,
    )
    if requested_ring_buffer_size < 3:
        assert layout.bytes_total < layout.required_budget_for_full_residency
    else:
        assert layout.bytes_total == layout.required_budget_for_full_residency


def test_cusparse_allow_residency_false_rejects_ring_zero(cusparse_small_stream_artifact):
    with pytest.raises(ValueError, match="ring_buffer_size"):
        _build_layout(
            cusparse_small_stream_artifact,
            runtime_k=1,
            ring_buffer_size=0,
            budget_bytes=1_000_000_000_000,
            allow_residency=False,
        )


def test_cusparse_allow_residency_false_warns_when_requested_ring_exceeds_streamed_blocks(cusparse_small_stream_artifact):
    fixed_bytes, block_bytes, owner_block_count = _full_budget_components(cusparse_small_stream_artifact, runtime_k=2)
    assert owner_block_count == 3
    budget_bytes = fixed_bytes + 3 * block_bytes
    with pytest.warns(RuntimeWarning, match="requested ring_buffer_size"):
        layout = _build_layout(
            cusparse_small_stream_artifact,
            runtime_k=2,
            ring_buffer_size=4,
            budget_bytes=budget_bytes,
            allow_residency=False,
        )
    _assert_all_streamed_no_residency(
        layout,
        requested_ring_buffer_size=4,
        expected_slot_count=3,
        fixed_bytes=fixed_bytes,
        block_bytes=block_bytes,
    )


@pytest.mark.parametrize(
    ("mode", "allocation_granularity", "expected_mode"),
    [
        pytest.param("unsupported", 0, "materialized", id="unsupported"),
        pytest.param("savings", 64, "vmm", id="vmm"),
        pytest.param("no-savings", 1 << 30, "materialized", id="no-savings"),
    ],
)
def test_cusparse_shared_ones_plan_modes(primary_artifact, monkeypatch, mode, allocation_granularity, expected_mode):
    monkeypatch.setattr(CudaVmmDriver, "current_context", lambda self: 1)
    monkeypatch.setattr(CudaVmmDriver, "vmm_supported", lambda self, device_id: mode != "unsupported")
    monkeypatch.setattr(CudaVmmDriver, "allocation_granularity", lambda self, device_id, *, recommended: allocation_granularity)
    layout = build_cusparse_layout([primary_artifact], requirements=_requirements(2))
    assert layout.shared_ones.mode == expected_mode
    if expected_mode == "vmm":
        assert layout.shared_ones.physical_bytes < layout.shared_ones.logical_bytes
    else:
        assert layout.shared_ones.physical_bytes == layout.shared_ones.logical_bytes


@pytest.mark.parametrize("runtime_k", [1, 2], ids=["k1", "k2"])
# A numeric token is the only way to get a *partial* per-level scratch mask, which is
# the case the per-level sizing walk was rewritten for and had no coverage of.
@pytest.mark.parametrize("scratch", ["all", "1"], ids=["scratch-all", "scratch-level1"])
def test_cusparse_scratch_enabled_up_matches_reference(any_grg, runtime_k, scratch):
    primary_artifact, primary_grg = any_grg.artifact, any_grg.grg
    pair = CusparsePlanPair(
        plan_up=CusparsePlan(store="N", fmt="CSR", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT", scratch=scratch),
        plan_down=None,
    )
    layout = build_cusparse_layout([primary_artifact], pair=pair, requirements=_requirements(runtime_k))
    with CusparseRuntime(layout) as runtime:
        (grg,) = runtime.grgs
        rng = np.random.default_rng(5011 + runtime_k)
        x = rng.standard_normal((runtime_k, primary_grg.num_samples), dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(grg.matmul(x, "up"), np.asarray(pygrgl.matmul(primary_grg, x, pygrgl.TraversalDirection.UP)), atol=atol, rtol=rtol)


@pytest.mark.parametrize("runtime_k", [1, 2], ids=["k1", "k2"])
# DOWN needs the partial mask more than UP does: the index reversal these tests were
# written for lived in this direction, and only a numeric token exercises a plan where
# some levels use ext_scratch and others ext_main.
@pytest.mark.parametrize("scratch", ["all", "0"], ids=["scratch-all", "scratch-level0"])
def test_cusparse_scratch_enabled_down_matches_reference(any_grg, runtime_k, scratch):
    primary_artifact, primary_grg = any_grg.artifact, any_grg.grg
    pair = CusparsePlanPair(
        plan_up=None,
        plan_down=CusparsePlan(store="T", fmt="CSC", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT", scratch=scratch),
    )
    layout = build_cusparse_layout([primary_artifact], pair=pair, requirements=_requirements(runtime_k))
    with CusparseRuntime(layout) as runtime:
        (grg,) = runtime.grgs
        rng = np.random.default_rng(6011 + runtime_k)
        x = rng.standard_normal((runtime_k, primary_grg.num_mutations), dtype=DATA_DTYPE)
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(grg.matmul(x, "down"), np.asarray(pygrgl.matmul(primary_grg, x, pygrgl.TraversalDirection.DOWN)), atol=atol, rtol=rtol)


@pytest.mark.parametrize("direction", ["up", "down"])
def test_cusparse_scratch_external_buffers_are_not_overrun(any_grg, direction):
    """Guard-region canary on every per-op cuSPARSE external workspace.

    Comparing results against a reference cannot catch an undersized external
    workspace: cuSPARSE writes past the end into whatever the pool put next and
    still returns the right numbers. _query_spmm_buffer_sizes walks query_blocks
    in artifact order (ascending parent level) while _build_ops assigns op_idx
    via iter_direction_level_pairs, which for DOWN is the exact reverse -- so a
    per-slot size would be handed to a different op than the one it was measured
    for. Only a canary sees that.
    """
    guard, sentinel = 4096, 0xAB
    plan = CusparsePlan(
        store="N" if direction == "up" else "T",
        fmt="CSR" if direction == "up" else "CSC",
        op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT", scratch="all",
    )
    pair = CusparsePlanPair(
        plan_up=plan if direction == "up" else None,
        plan_down=None if direction == "up" else plan,
    )
    layout = build_cusparse_layout([any_grg.artifact], pair=pair, requirements=_requirements(2))
    with CusparseRuntime(layout) as runtime:
        (grg,) = runtime.grgs
        table = runtime._ext_scratch_up if direction == "up" else runtime._ext_scratch_down
        watched = {}
        for level, row in enumerate(table):
            for op_idx, buf in enumerate(row):
                if buf is None:
                    continue
                nbytes = int(buf.nbytes)
                padded = cp.full((nbytes + guard,), sentinel, dtype=cp.uint8)
                watched[(level, op_idx)] = (padded, nbytes)
                row[op_idx] = padded[:nbytes]
        assert watched, "fixture produced no scratch buffers; the test would be vacuous"

        cols = any_grg.grg.num_samples if direction == "up" else any_grg.grg.num_mutations
        rng = np.random.default_rng(4242)
        x = rng.standard_normal((2, cols), dtype=DATA_DTYPE)
        got = grg.matmul(x, direction)
        cp.cuda.Device(layout.device).synchronize()

        overruns = {
            key: int(cp.count_nonzero(padded[nbytes:] != sentinel))
            for key, (padded, nbytes) in watched.items()
        }
        bad = {key: n for key, n in overruns.items() if n}
        assert not bad, (
            f"{len(bad)}/{len(watched)} cuSPARSE {direction} external scratch buffers "
            f"were written past their planned end: {sorted(bad.items())[:5]}"
        )

        expected = np.asarray(pygrgl.matmul(
            any_grg.grg, x,
            pygrgl.TraversalDirection.UP if direction == "up" else pygrgl.TraversalDirection.DOWN,
        ))
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(got, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("runtime_k", [1, 2], ids=["k1", "k2"])
@pytest.mark.parametrize("mode", THREE_BLOCK_TRANSITION_MODES, ids=[mode.name for mode in THREE_BLOCK_TRANSITION_MODES])
def test_cusparse_small_three_block_planner_transitions(cusparse_small_stream_artifact, runtime_k, mode):
    _assert_planner_transition(cusparse_small_stream_artifact, runtime_k=runtime_k, mode=mode)


@pytest.mark.stress
@pytest.mark.parametrize("runtime_k", [2], ids=["k2"])
@pytest.mark.parametrize("mode", THREE_BLOCK_TRANSITION_MODES, ids=[mode.name for mode in THREE_BLOCK_TRANSITION_MODES])
def test_cusparse_large_three_block_planner_transitions(cusparse_stream_stress_artifact, runtime_k, mode):
    _assert_planner_transition(cusparse_stream_stress_artifact, runtime_k=runtime_k, mode=mode)


@pytest.mark.parametrize("order", [("up", "down"), ("down", "up")], ids=["up-down", "down-up"])
@pytest.mark.parametrize("runtime_k", [1, 2], ids=["k1", "k2"])
@pytest.mark.parametrize("requested_ring_buffer_size", [1, 2, 3], ids=["ring1", "ring2", "ring3"])
def test_cusparse_small_three_block_exactness_allow_residency_false(order, runtime_k, requested_ring_buffer_size, gpu_small_stream_case, cusparse_small_stream_artifact):
    fixed_bytes, block_bytes, owner_block_count = _full_budget_components(cusparse_small_stream_artifact, runtime_k=runtime_k)
    assert owner_block_count == 3
    budget_bytes = fixed_bytes + requested_ring_buffer_size * block_bytes
    clear_cupy_state()
    layout = _build_layout(
        cusparse_small_stream_artifact,
        runtime_k=runtime_k,
        ring_buffer_size=requested_ring_buffer_size,
        budget_bytes=budget_bytes,
        allow_residency=False,
    )
    _assert_all_streamed_no_residency(
        layout,
        requested_ring_buffer_size=requested_ring_buffer_size,
        expected_slot_count=requested_ring_buffer_size,
        fixed_bytes=fixed_bytes,
        block_bytes=block_bytes,
    )
    with CusparseRuntime(layout) as runtime:
        (grg,) = runtime.grgs
        for run_idx, direction in enumerate(order):
            seed = 43_000 + 1_000 * runtime_k + 100 * run_idx + 10 * requested_ring_buffer_size
            rng = np.random.default_rng(seed)
            primary = rng.choice(np.array([-1.0, 1.0], dtype=DATA_DTYPE), size=(int(runtime_k), gpu_small_stream_case.n))
            if direction == "up":
                expected = expected_up(primary.T, shifts=gpu_small_stream_case.shifts, bandwidth=gpu_small_stream_case.bandwidth).T
            else:
                expected = expected_down(primary.T, shifts=gpu_small_stream_case.shifts, bandwidth=gpu_small_stream_case.bandwidth).T
            np.testing.assert_array_equal(grg.matmul(primary, direction), expected)
    clear_cupy_state()


@pytest.mark.parametrize("order", [("up", "down"), ("down", "up")], ids=["up-down", "down-up"])
@pytest.mark.parametrize("requested_ring_buffer_size", [1, 2], ids=["ring1", "ring2"])
def test_cusparse_small_stream_copy_overlaps_compute(tmp_path, order, requested_ring_buffer_size, monkeypatch):
    _run_overlap_case(tmp_path, order, requested_ring_buffer_size, monkeypatch, n=64, bandwidth=8, runtime_k=1)


@pytest.mark.stress
@pytest.mark.parametrize("order", [("up", "down"), ("down", "up")], ids=["up-down", "down-up"])
@pytest.mark.parametrize("requested_ring_buffer_size", [1, 2], ids=["ring1", "ring2"])
def test_cusparse_stream_copy_overlaps_compute(tmp_path, order, requested_ring_buffer_size, monkeypatch):
    _run_overlap_case(tmp_path, order, requested_ring_buffer_size, monkeypatch, n=4096, bandwidth=64, runtime_k=2)


def test_cusparse_prepare_cuda_graph_capture_streamed(cusparse_small_stream_artifact):
    torch = pytest.importorskip("torch")
    fixed_bytes, block_bytes, owner_block_count = _full_budget_components(cusparse_small_stream_artifact, runtime_k=1)
    assert owner_block_count == 3
    with torch.cuda.device(0):
        capture_stream = torch.cuda.Stream()
    layout = _build_layout(
        cusparse_small_stream_artifact,
        runtime_k=1,
        ring_buffer_size=2,
        budget_bytes=fixed_bytes + 2 * block_bytes,
        allow_residency=False,
        stream=capture_stream,
    )
    ref_layout = build_reference_layout([cusparse_small_stream_artifact], requirements=_requirements(1))
    with ReferenceRuntime(ref_layout) as ref_runtime:
        (ref_grg,) = ref_runtime.grgs
        order = ("down", "up")
        src_np = np.arange(_source_cols(ref_grg, order[0]), dtype=DATA_DTYPE).reshape(1, -1)
        init_first_np = np.asarray([0.5], dtype=DATA_DTYPE)
        init_second_np = np.asarray([1.5], dtype=DATA_DTYPE)
        expected = _expected_prepared_chain(ref_grg, order, src_np, init_first_np, init_second_np)
    with CusparseRuntime(layout) as runtime, ExitStack() as stack:
        (grg,) = runtime.grgs
        first, second = _enter_prepared_chain(stack, grg, order, k=1)
        src = torch.from_numpy(src_np).to(device=first.input.device)
        init_first = torch.from_numpy(init_first_np).to(device=first.init_vector.device)
        init_second = torch.from_numpy(init_second_np).to(device=second.init_vector.device)
        graph = torch.cuda.CUDAGraph()
        # All capture-sensitive setup must already be complete once the prepared ops are entered.
        with torch.cuda.graph(graph, stream=capture_stream):
            _capture_prepared_chain(first, second, src, init_first, init_second)
        graph.replay()
        cp.cuda.runtime.deviceSynchronize()
        actual = second.output.cpu().numpy().copy()
    np.testing.assert_allclose(actual, expected, atol=tol(DATA_DTYPE)[0], rtol=tol(DATA_DTYPE)[1])
