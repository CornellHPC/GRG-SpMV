"""Resident GPU contracts shared by CUDA and ROCm; vendor internals stay separate."""

from contextlib import ExitStack
from types import SimpleNamespace

import numpy as np
import pygrgl
import pytest

from pygrgl_spmv.backends.reference import ReferenceRuntime
from pygrgl_spmv.backends.types import SparseFormat
from pygrgl_spmv.grg.artifact import scan_grg_spmv
from pygrgl_spmv.tests.conftest import DATA_DTYPE, tol
from pygrgl_spmv.tests.runtime._gpu import (
    GPU_BACKENDS, Graph, assert_reset, copy_to, dense_entries, empty_like,
    new_stream, plan_types, prepare, to_device, to_host,
)
from pygrgl_spmv.tests.runtime._runtime_builders import (
    build_layout_for_backend, build_reference_layout, full_requirements, runtime_cls_for_backend,
)
from pygrgl_spmv.tests.runtime._streaming_cases import (
    THREE_BLOCK_EXACTNESS_MODES, clear_cupy_state,
    write_overlap_band_artifact, write_three_level_band_artifact,
)
from pygrgl_spmv.tests.runtime.test_cusparse import (
    _run_exactness_case, gpu_small_stream_case, cusparse_small_stream_artifact,
    cusparse_stream_stress_case, cusparse_stream_stress_artifact,
)

cp = pytest.importorskip("cupy")


@pytest.fixture(params=GPU_BACKENDS)
def gpu(request):
    name = request.param
    Plan, Pair = plan_types(name)

    def layout(artifacts, **kwargs):
        return build_layout_for_backend(name, artifacts, **kwargs)
    return name, layout, runtime_cls_for_backend(name), Plan, Pair


def _requirements(runtime_k: int):
    return full_requirements(max_k_up=int(runtime_k), max_k_down=int(runtime_k))


def _assert_runtime_matches_reference(grg, primary_grg, *, seed: int) -> None:
    rng = np.random.default_rng(seed)
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


def _prepared_direction_enum(direction: str):
    return pygrgl.TraversalDirection.UP if direction == "up" else pygrgl.TraversalDirection.DOWN


def _prepared_input_cols(grg, direction: str) -> int:
    return int(grg.num_samples if direction == "up" else grg.num_mutations)


@pytest.mark.parametrize("direction", ["up", "down"])
def test_gpu_prepared_same_direction_mixed_widths(primary_artifact, primary_grg, monkeypatch, direction, gpu):
    backend, build_layout, Runtime, Plan, Pair = gpu
    layout = build_layout([primary_artifact], requirements=_requirements(4))
    rng = np.random.default_rng(120_401 if direction == "up" else 120_402)
    x1_a = rng.standard_normal((1, _prepared_input_cols(primary_grg, direction)), dtype=DATA_DTYPE)
    x4 = rng.standard_normal((4, _prepared_input_cols(primary_grg, direction)), dtype=DATA_DTYPE)
    x1_b = rng.standard_normal((1, _prepared_input_cols(primary_grg, direction)), dtype=DATA_DTYPE)
    direction_enum = _prepared_direction_enum(direction)
    expected = [
        np.asarray(pygrgl.matmul(primary_grg, x1_a, direction_enum)),
        np.asarray(pygrgl.matmul(primary_grg, x4, direction_enum)),
        np.asarray(pygrgl.matmul(primary_grg, x1_b, direction_enum)),
    ]
    seen = []
    with Runtime(layout) as runtime, ExitStack() as stack:
        (grg,) = runtime.grgs
        op1 = stack.enter_context(prepare(grg, direction=direction, k=1))
        op4 = stack.enter_context(prepare(grg, direction=direction, k=4))
        dense_map = dense_entries(runtime, direction)
        assert set(dense_map) == {1, 4}
        assert dense_map[1][0] is op1._dense_state
        assert dense_map[4][0] is op4._dense_state

        original = runtime._enqueue_wavefront

        def record_dense(artifact, direction_arg, dense):
            seen.append(dense)
            return original(artifact, direction_arg, dense)

        monkeypatch.setattr(runtime, "_enqueue_wavefront", record_dense)
        actual = []
        for op, values in ((op1, x1_a), (op4, x4), (op1, x1_b)):
            copy_to(op.input, values)
            op()
            actual.append(to_host(op.output))
        assert seen == [op1._dense_state, op4._dense_state, op1._dense_state]
    assert dense_entries(runtime, direction) == {}
    for value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, expected_value, atol=tol(DATA_DTYPE)[0], rtol=tol(DATA_DTYPE)[1])


@pytest.mark.parametrize("direction", ["up", "down"])
def test_gpu_prepared_same_width_refcount(primary_artifact, primary_grg, direction, gpu):
    backend, build_layout, Runtime, Plan, Pair = gpu
    layout = build_layout([primary_artifact], requirements=_requirements(4))
    rng = np.random.default_rng(120_501 if direction == "up" else 120_502)
    x = rng.standard_normal((2, _prepared_input_cols(primary_grg, direction)), dtype=DATA_DTYPE)
    expected = np.asarray(pygrgl.matmul(primary_grg, x, _prepared_direction_enum(direction)))
    with Runtime(layout) as runtime:
        (grg,) = runtime.grgs
        with prepare(grg, direction=direction, k=2) as outer:
            with prepare(grg, direction=direction, k=2) as inner:
                dense_map = dense_entries(runtime, direction)
                assert set(dense_map) == {2}
                assert outer._dense_state is inner._dense_state
                assert dense_map[2][0] is outer._dense_state
                assert dense_map[2][1] == 2
            dense_map = dense_entries(runtime, direction)
            assert set(dense_map) == {2}
            assert dense_map[2][1] == 1
            copy_to(outer.input, x)
            outer()
            actual = to_host(outer.output)
    assert dense_entries(runtime, direction) == {}
    np.testing.assert_allclose(actual, expected, atol=tol(DATA_DTYPE)[0], rtol=tol(DATA_DTYPE)[1])


def test_gpu_runtime_exposes_requested_stream_and_stream_ptr(primary_artifact, gpu):
    backend, build_layout, Runtime, Plan, Pair = gpu
    with cp.cuda.Device(0):
        requested = cp.cuda.Stream(non_blocking=True)
    layout = build_layout([primary_artifact], stream=requested)
    with Runtime(layout) as runtime:
        assert int(runtime.device.id) == int(layout.device)
        assert runtime.stream is not None
        assert runtime.stream_ptr == layout.stream_ptr == int(requested.ptr)
        assert int(runtime.stream.ptr) == int(requested.ptr)
        assert len(runtime.grgs) == 1
        assert runtime.grgs[0].device == layout.device


def test_gpu_runtime_reusable_across_context_cycles(primary_artifact, primary_grg, gpu):
    backend, build_layout, Runtime, Plan, Pair = gpu
    layout = build_layout([primary_artifact], requirements=_requirements(2))
    runtime = Runtime(layout)
    with runtime:
        if backend == "cusparse":
            first_handle = runtime._cslib
            assert first_handle is not None
        else:
            first_ones = runtime._shared_ones
            assert first_ones is not None
        (grg,) = runtime.grgs
        _assert_runtime_matches_reference(grg, primary_grg, seed=9101)
    assert_reset(runtime)
    with runtime:
        if backend == "cusparse":
            assert runtime._cslib is not None
            assert runtime._cslib is not first_handle
        else:
            assert runtime._shared_ones is not None
            assert runtime._shared_ones is not first_ones
        (grg,) = runtime.grgs
        _assert_runtime_matches_reference(grg, primary_grg, seed=9102)
    assert_reset(runtime)


def test_gpu_runtime_exit_releases_owned_state(primary_artifact, gpu):
    backend, build_layout, Runtime, Plan, Pair = gpu
    layout = build_layout([primary_artifact], requirements=_requirements(2))
    runtime = Runtime(layout)
    with runtime:
        if backend == "cusparse":
            assert runtime._cslib is not None
        else:
            assert runtime._shared_ones is not None
        assert runtime._artifacts
        assert runtime._grgs
    assert_reset(runtime)


def test_gpu_runtime_enter_failure_releases_owned_state(primary_artifact, monkeypatch, gpu):
    backend, build_layout, Runtime, Plan, Pair = gpu
    layout = build_layout([primary_artifact, primary_artifact], requirements=_requirements(1))
    runtime = Runtime(layout)
    original = Runtime._build_artifact
    calls = {"count": 0}

    def _boom_after_first(self, *args):
        calls["count"] += 1
        if calls["count"] == 1:
            return original(self, *args)
        raise RuntimeError("boom")

    monkeypatch.setattr(Runtime, "_build_artifact", _boom_after_first)
    with pytest.raises(RuntimeError, match="boom"):
        runtime.__enter__()
    assert_reset(runtime)


def test_gpu_distinct_external_streams_use_distinct_runtime_handoffs(primary_artifact, primary_grg, gpu):
    backend, build_layout, Runtime, Plan, Pair = gpu
    ref_layout = build_reference_layout([primary_artifact], requirements=_requirements(2))
    with cp.cuda.Device(0):
        requested_a = cp.cuda.Stream(non_blocking=True)
        requested_b = cp.cuda.Stream(non_blocking=True)
    layout_a = build_layout([primary_artifact], stream=requested_a, requirements=_requirements(2))
    layout_b = build_layout([primary_artifact], stream=requested_b, requirements=_requirements(2))
    with ReferenceRuntime(ref_layout) as ref_runtime, Runtime(layout_a) as runtime_a, Runtime(layout_b) as runtime_b:
        assert runtime_a.stream_ptr == int(requested_a.ptr)
        assert runtime_b.stream_ptr == int(requested_b.ptr)
        assert runtime_a.stream_ptr != runtime_b.stream_ptr
        (ref_grg,) = ref_runtime.grgs
        (grg_a,) = runtime_a.grgs
        (grg_b,) = runtime_b.grgs
        rng = np.random.default_rng(9103)
        x_up = rng.standard_normal((2, primary_grg.num_samples), dtype=DATA_DTYPE)
        x_down = rng.standard_normal((2, primary_grg.num_mutations), dtype=DATA_DTYPE)
        expected_up = ref_grg.matmul(x_up, "up")
        expected_down = ref_grg.matmul(x_down, "down")
        np.testing.assert_allclose(grg_a.matmul(x_up, "up"), expected_up, atol=tol(DATA_DTYPE)[0], rtol=tol(DATA_DTYPE)[1])
        np.testing.assert_allclose(grg_b.matmul(x_up, "up"), expected_up, atol=tol(DATA_DTYPE)[0], rtol=tol(DATA_DTYPE)[1])
        np.testing.assert_allclose(grg_a.matmul(x_down, "down"), expected_down, atol=tol(DATA_DTYPE)[0], rtol=tol(DATA_DTYPE)[1])
        np.testing.assert_allclose(grg_b.matmul(x_down, "down"), expected_down, atol=tol(DATA_DTYPE)[0], rtol=tol(DATA_DTYPE)[1])


def test_gpu_f_order_input(primary_artifact, primary_grg, gpu):
    backend, build_layout, Runtime, Plan, Pair = gpu
    layout = build_layout([primary_artifact], requirements=_requirements(4))
    with Runtime(layout) as runtime:
        (grg,) = runtime.grgs
        rng = np.random.default_rng(9005)
        x = np.asfortranarray(rng.standard_normal((4, primary_grg.num_samples), dtype=DATA_DTYPE))
        atol, rtol = tol(DATA_DTYPE)
        np.testing.assert_allclose(grg.matmul(x, "up"), np.asarray(pygrgl.matmul(primary_grg, x, pygrgl.TraversalDirection.UP)), atol=atol, rtol=rtol)


@pytest.mark.parametrize("fmt", [SparseFormat.CSR, SparseFormat.CSC, SparseFormat.COO], ids=["csr", "csc", "coo"])
def test_gpu_primary_grg_exact_binary(primary_artifact, primary_grg, fmt, gpu):
    backend, build_layout, Runtime, Plan, Pair = gpu
    if fmt == SparseFormat.COO:
        up_plan = Plan(store="N", fmt="COO", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT")
        down_plan = Plan(store="T", fmt="COO", op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT")
    else:
        up_plan = Plan(store="N", fmt=fmt, op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT")
        down_plan = Plan(store="T", fmt=(SparseFormat.CSC if fmt == SparseFormat.CSR else SparseFormat.CSR), op_a="N", op_b="N", order_b="ROW", order_c="ROW", algo="DEFAULT")
    pair = Pair(plan_up=up_plan, plan_down=down_plan)
    layout = build_layout([primary_artifact], pair=pair, requirements=_requirements(4))
    with Runtime(layout) as runtime:
        (grg,) = runtime.grgs
        rng = np.random.default_rng(8001)
        x_up = rng.choice(np.array([-1.0, 1.0], dtype=DATA_DTYPE), size=(4, primary_grg.num_samples))
        x_down = rng.choice(np.array([-1.0, 1.0], dtype=DATA_DTYPE), size=(4, primary_grg.num_mutations))
        np.testing.assert_array_equal(grg.matmul(x_up, "up"), np.asarray(pygrgl.matmul(primary_grg, x_up, pygrgl.TraversalDirection.UP)))
        np.testing.assert_array_equal(grg.matmul(x_down, "down"), np.asarray(pygrgl.matmul(primary_grg, x_down, pygrgl.TraversalDirection.DOWN)))


_MIXED_HEIGHT_GRAPH_ORDERS = (("up", "down"), ("down", "up"))


def _build_multi_grg_graph_layout(gpu, artifacts, *, streamed, stream):
    backend, build_layout, *_ = gpu
    if backend == "rocsparse":
        if streamed:
            pytest.skip("rocSPARSE streaming is unsupported")
        return build_layout(artifacts, requirements=_requirements(1), stream=stream)
    from pygrgl_spmv.tests.runtime.test_cusparse import _build_multi_grg_graph_layout as cuda_layout
    return cuda_layout(artifacts, streamed=streamed, stream=stream)


def _enter_prepared_chain(stack, grg, order, *, k):
    first = stack.enter_context(prepare(grg, direction=order[0], k=k, init_mode="vector"))
    second = stack.enter_context(prepare(grg, direction=order[1], k=k, init_mode="vector"))
    if not isinstance(first.output, cp.ndarray):
        assert first.output.data_ptr() == second.input.data_ptr()
    return first, second


def _capture_prepared_chain(first, second, src, init_first, init_second, result=None):
    copy_to(first.input, src)
    copy_to(first.init_vector, init_first)
    if isinstance(first.output, cp.ndarray):
        runtime = first._runtime
        runtime._enqueue_prepared(first._artifact, first._spec, first)
        copy_to(second.input, first.output)
    else:
        first()
    # CUDA operations share the initialization arena; preserve the original
    # order so the second initializer cannot overwrite the first one early.
    copy_to(second.init_vector, init_second)
    if isinstance(first.output, cp.ndarray):
        runtime._enqueue_prepared(second._artifact, second._spec, second)
    else:
        second()
    if result is not None:
        copy_to(result, second.output)

def _source_cols(grg, direction: str) -> int:
    return int(grg.num_samples if direction == "up" else grg.num_mutations)


def _expected_prepared_chain(ref_grg, order, src: np.ndarray, init_first: np.ndarray, init_second: np.ndarray) -> np.ndarray:
    mid = ref_grg.matmul(src, order[0], init=init_first)
    return ref_grg.matmul(mid, order[1], init=init_second)


def _mixed_height_graph_artifacts(tmp_path) -> tuple[object, object]:
    tall = write_overlap_band_artifact(tmp_path, "tall-h4", n=16, bandwidth=4)
    short = write_three_level_band_artifact(tmp_path, "short-h3", n=16, bandwidth=4)
    assert scan_grg_spmv(tall).num_levels == 4
    assert scan_grg_spmv(short).num_levels == 3
    return tall, short


def _multi_grg_graph_cases(artifacts) -> list[SimpleNamespace]:
    out = []
    ref_layout = build_reference_layout(artifacts, requirements=_requirements(1))
    with ReferenceRuntime(ref_layout) as ref_runtime:
        for idx, (ref_grg, order) in enumerate(zip(ref_runtime.grgs, _MIXED_HEIGHT_GRAPH_ORDERS, strict=True)):
            cols = _source_cols(ref_grg, order[0])
            src = np.arange(cols, dtype=DATA_DTYPE).reshape(1, cols) + DATA_DTYPE(idx * 100)
            init_first = np.asarray([0.25 + idx], dtype=DATA_DTYPE)
            init_second = np.asarray([0.75 + idx], dtype=DATA_DTYPE)
            out.append(
                SimpleNamespace(
                    order=order,
                    src=src,
                    init_first=init_first,
                    init_second=init_second,
                    expected=_expected_prepared_chain(ref_grg, order, src, init_first, init_second),
                )
            )
    return out


def test_gpu_prepare_graph_capture_resident(primary_artifact, gpu):
    backend, build_layout, Runtime, Plan, Pair = gpu
    capture_stream = new_stream(backend)
    layout = build_layout([primary_artifact], requirements=_requirements(2), stream=capture_stream)
    ref_layout = build_reference_layout([primary_artifact], requirements=_requirements(2))
    with ReferenceRuntime(ref_layout) as ref_runtime:
        (ref_grg,) = ref_runtime.grgs
        rng = np.random.default_rng(98_201)
        order = ("down", "up")
        src_np = rng.standard_normal((2, _source_cols(ref_grg, order[0])), dtype=DATA_DTYPE)
        init_first_np = rng.standard_normal((2,), dtype=DATA_DTYPE)
        init_second_np = rng.standard_normal((2,), dtype=DATA_DTYPE)
        expected = _expected_prepared_chain(ref_grg, order, src_np, init_first_np, init_second_np)
    with Runtime(layout) as runtime, ExitStack() as stack:
        (grg,) = runtime.grgs
        first, second = _enter_prepared_chain(stack, grg, order, k=2)
        src = to_device(backend, src_np)
        init_first = to_device(backend, init_first_np)
        init_second = to_device(backend, init_second_np)
        graph = Graph(backend, runtime)
        # All capture-sensitive setup must already be complete once the prepared ops are entered.
        with graph.capture(capture_stream):
            _capture_prepared_chain(first, second, src, init_first, init_second)
        graph.replay()
        cp.cuda.runtime.deviceSynchronize()
        actual = to_host(second.output)
    np.testing.assert_allclose(actual, expected, atol=tol(DATA_DTYPE)[0], rtol=tol(DATA_DTYPE)[1])


@pytest.mark.parametrize("streamed", [False, True], ids=["resident", "streamed"])
def test_gpu_graph_capture_multi_grg_single_graph_mixed_heights(tmp_path, streamed, gpu):
    backend, build_layout, Runtime, Plan, Pair = gpu
    artifacts = _mixed_height_graph_artifacts(tmp_path)
    cases = _multi_grg_graph_cases(artifacts)
    capture_stream = new_stream(backend)
    layout = _build_multi_grg_graph_layout(gpu, artifacts, streamed=streamed, stream=capture_stream)
    with Runtime(layout) as runtime, ExitStack() as stack:
        chains = [
            _enter_prepared_chain(stack, grg, case.order, k=1)
            for grg, case in zip(runtime.grgs, cases, strict=True)
        ]
        srcs = [
            to_device(backend, case.src)
            for chain, case in zip(chains, cases, strict=True)
        ]
        init_first = [
            to_device(backend, case.init_first)
            for chain, case in zip(chains, cases, strict=True)
        ]
        init_second = [
            to_device(backend, case.init_second)
            for chain, case in zip(chains, cases, strict=True)
        ]
        results = [empty_like(chain[1].output) for chain in chains]
        graph = Graph(backend, runtime)
        with graph.capture(capture_stream):
            for chain, src, first_init, second_init, result in zip(chains, srcs, init_first, init_second, results, strict=True):
                _capture_prepared_chain(chain[0], chain[1], src, first_init, second_init, result)
        graph.replay()
        cp.cuda.runtime.deviceSynchronize()
        actual = [to_host(result) for result in results]
    for case, value in zip(cases, actual, strict=True):
        np.testing.assert_allclose(value, case.expected, atol=tol(DATA_DTYPE)[0], rtol=tol(DATA_DTYPE)[1])


@pytest.mark.parametrize("streamed", [False, True], ids=["resident", "streamed"])
def test_gpu_graph_capture_multi_grg_sequential_mixed_heights(tmp_path, streamed, gpu):
    backend, build_layout, Runtime, Plan, Pair = gpu
    artifacts = _mixed_height_graph_artifacts(tmp_path)
    cases = _multi_grg_graph_cases(artifacts)
    capture_stream = new_stream(backend)
    layout = _build_multi_grg_graph_layout(gpu, artifacts, streamed=streamed, stream=capture_stream)
    with Runtime(layout) as runtime, ExitStack() as stack:
        chains = [
            _enter_prepared_chain(stack, grg, case.order, k=1)
            for grg, case in zip(runtime.grgs, cases, strict=True)
        ]
        srcs = [
            to_device(backend, case.src)
            for chain, case in zip(chains, cases, strict=True)
        ]
        init_first = [
            to_device(backend, case.init_first)
            for chain, case in zip(chains, cases, strict=True)
        ]
        init_second = [
            to_device(backend, case.init_second)
            for chain, case in zip(chains, cases, strict=True)
        ]
        captures = []
        for chain, src, first_init, second_init in zip(chains, srcs, init_first, init_second, strict=True):
            graph = Graph(backend, runtime)
            with graph.capture(capture_stream):
                _capture_prepared_chain(chain[0], chain[1], src, first_init, second_init)
            captures.append((graph, chain[1]))
        actual = []
        for graph, final in captures:
            graph.replay()
            cp.cuda.runtime.deviceSynchronize()
            actual.append(to_host(final.output))
    for case, value in zip(cases, actual, strict=True):
        np.testing.assert_allclose(value, case.expected, atol=tol(DATA_DTYPE)[0], rtol=tol(DATA_DTYPE)[1])


@pytest.mark.parametrize("direction", ["up", "down"])
@pytest.mark.parametrize("fmt", ["CSR", "CSC", "COO"])
@pytest.mark.parametrize("op_a", ["N", "T"])
@pytest.mark.parametrize("op_b", ["N", "T"])
@pytest.mark.parametrize("orders", [("ROW", "ROW"), ("ROW", "COL"), ("COL", "ROW"), ("COL", "COL")])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gpu_explicit_dense_view_plans(primary_artifact, primary_grg, gpu, direction, fmt, op_a, op_b, orders, dtype):
    backend, build_layout, Runtime, Plan, Pair = gpu
    store = "N" if (direction == "up") == (op_a == "N") else "T"
    plan = Plan(store, fmt, op_a, op_b, *orders, "DEFAULT")
    pair = Pair(plan if direction == "up" else None, plan if direction == "down" else None)
    layout = build_layout([primary_artifact], pair=pair, requirements=_requirements(4), dtype=dtype)
    rng = np.random.default_rng(841 if direction == "up" else 842)
    n = primary_grg.num_samples if direction == "up" else primary_grg.num_mutations
    x = rng.standard_normal((4, n), dtype=dtype)
    expected = np.asarray(pygrgl.matmul(primary_grg, x, _prepared_direction_enum(direction)))
    with Runtime(layout) as runtime:
        np.testing.assert_allclose(runtime.grgs[0].matmul(x, direction), expected,
                                   atol=tol(dtype)[0], rtol=tol(dtype)[1])


@pytest.mark.parametrize("order", [("up", "down"), ("down", "up")], ids=["up-down", "down-up"])
@pytest.mark.parametrize("runtime_k", [1, 2], ids=["k1", "k2"])
@pytest.mark.parametrize("mode", THREE_BLOCK_EXACTNESS_MODES, ids=[mode.name for mode in THREE_BLOCK_EXACTNESS_MODES])
def test_gpu_small_three_block_exactness(order, runtime_k, mode, gpu_small_stream_case, cusparse_small_stream_artifact, gpu):
    if gpu[0] == "rocsparse" and mode.requested_ring_buffer_size:
        pytest.skip("ROCm ring-buffer/streaming configuration is unsupported")
    clear_cupy_state()
    _run_exactness_case(cusparse_small_stream_artifact, gpu_small_stream_case, runtime_k=runtime_k, mode=mode, order=order, backend=gpu[0])
    clear_cupy_state()


@pytest.mark.stress
@pytest.mark.parametrize("order", [("up", "down"), ("down", "up")], ids=["up-down", "down-up"])
@pytest.mark.parametrize("runtime_k", [2], ids=["k2"])
@pytest.mark.parametrize("mode", THREE_BLOCK_EXACTNESS_MODES, ids=[mode.name for mode in THREE_BLOCK_EXACTNESS_MODES])
def test_gpu_large_three_block_exactness(order, runtime_k, mode, cusparse_stream_stress_case, cusparse_stream_stress_artifact, gpu):
    if gpu[0] == "rocsparse" and mode.requested_ring_buffer_size:
        pytest.skip("ROCm ring-buffer/streaming configuration is unsupported")
    clear_cupy_state()
    _run_exactness_case(cusparse_stream_stress_artifact, cusparse_stream_stress_case, runtime_k=runtime_k, mode=mode, order=order, backend=gpu[0])
    clear_cupy_state()


def test_gpu_caller_root_scope_uses_explicit_handoff_streams(primary_artifact, gpu):
    backend, build_layout, Runtime, *_ = gpu
    with cp.cuda.Device(0):
        requested = cp.cuda.Stream(non_blocking=True)
    layout = build_layout([primary_artifact], stream=requested, requirements=_requirements(1))
    runtime = Runtime(layout)
    log: list[tuple[object, ...]] = []

    class _SpyStream:
        def __init__(self, name: str):
            self.name = name

        def __enter__(self):
            log.append(("enter_stream", self.name))
            return self

        def __exit__(self, exc_type, exc, tb):
            log.append(("exit_stream", self.name))

        def wait_event(self, event) -> None:
            log.append(("wait_event", self.name, event.name))

    class _SpyEvent:
        def __init__(self, name: str):
            self.name = name

        def record(self, stream) -> None:
            log.append(("record", self.name, stream.name))

    runtime._caller_stream = _SpyStream("caller")
    runtime._root_stream = _SpyStream("root")
    runtime._caller_to_root_event = _SpyEvent("caller_to_root")
    runtime._root_to_caller_event = _SpyEvent("root_to_caller")
    if backend == "rocsparse":
        runtime._to_root = runtime._caller_to_root_event
        runtime._to_caller = runtime._root_to_caller_event
    ambient = cp.cuda.Stream(non_blocking=True)
    with ambient:
        ambient_ptr = int(cp.cuda.get_current_stream().ptr)
        log.append(("ambient_current_ptr", ambient_ptr))
        with runtime._caller_root_scope():
            log.append(("body_current_ptr", int(cp.cuda.get_current_stream().ptr)))
    expected = [
        ("ambient_current_ptr", ambient_ptr),
        ("enter_stream", "caller"),
        ("record", "caller_to_root", "caller"),
        ("exit_stream", "caller"),
        ("wait_event", "root", "caller_to_root"),
        ("body_current_ptr", ambient_ptr),
        ("enter_stream", "root"),
        ("record", "root_to_caller", "root"),
        ("exit_stream", "root"),
        ("wait_event", "caller", "root_to_caller"),
    ]
    if backend == "rocsparse":
        expected = [entry for entry in expected if entry[0] not in {"enter_stream", "exit_stream"}]
    assert log == expected


def test_gpu_executes_under_current_device_switch(primary_artifact, primary_grg, gpu):
    backend, build_layout, Runtime, *_ = gpu
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("needs at least two GPU devices")
    layout = build_layout([primary_artifact], device=0, requirements=_requirements(2))
    with cp.cuda.Device(1):
        with Runtime(layout) as runtime:
            (grg,) = runtime.grgs
            rng = np.random.default_rng(7301)
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


@pytest.mark.parametrize("streamed", [False, True], ids=["resident", "streamed"])
def test_gpu_small_int64_artifact(tmp_path, streamed, gpu):
    backend, build_layout, Runtime, *_ = gpu
    if streamed and backend == "rocsparse":
        pytest.skip("ROCm ring-buffer/streaming configuration is unsupported")
    artifact = write_three_level_band_artifact(tmp_path, "tiny-cusparse-i64", n=4, bandwidth=1, struct_dtype=np.int64)
    scan = scan_grg_spmv(artifact)
    assert all(block.indices_dtype == np.dtype(np.int64) and block.indptr_dtype == np.dtype(np.int64) for block in scan.blocks)
    if not streamed:
        layout = build_layout([artifact], requirements=_requirements(2))
        with Runtime(layout) as runtime, ReferenceRuntime(build_reference_layout([artifact])) as reference:
            for direction, n in (("up", runtime.grgs[0].num_samples), ("down", runtime.grgs[0].num_mutations)):
                x = np.arange(2*n, dtype=DATA_DTYPE).reshape(2, n)
                np.testing.assert_array_equal(runtime.grgs[0].matmul(x, direction), reference.grgs[0].matmul(x, direction))
        return
    from pygrgl_spmv.backends.cusparse import CusparseRuntime
    from pygrgl_spmv.tests.runtime._runtime_builders import build_cusparse_layout
    full = build_cusparse_layout([artifact], ring_buffer_size=0, vram_budget_bytes=1_000_000_000, requirements=_requirements(2))
    block_bytes = min(block.nbytes for a in full.artifacts for block in (*a.blocks_up, *a.blocks_down))
    streamed = build_cusparse_layout(
        [artifact],
        ring_buffer_size=1,
        vram_budget_bytes=full.required_budget_for_full_residency - block_bytes,
        requirements=_requirements(2),
    )
    assert streamed.allocated_ring_buffer_size == 1
    assert streamed.slot_plans[0].struct0_dtype == np.dtype(np.int32)
    assert streamed.slot_plans[0].struct1_dtype == np.dtype(np.int32)
    with CusparseRuntime(streamed) as runtime:
        assert runtime._slot_struct0[0].dtype == np.int32
        assert runtime._slot_struct1[0].dtype == np.int32
