"""Small adapters for the two vendors' test buffers and prepared entry points."""

from contextlib import contextmanager
import importlib

import numpy as np
import pytest

GPU_BACKENDS = tuple(pytest.param(name, id=name, marks=getattr(pytest.mark, name))
                     for name in ("cusparse", "rocsparse"))


def plan_types(backend):
    module = importlib.import_module(f"pygrgl_spmv.backends.{backend}")
    prefix = "Cusparse" if backend == "cusparse" else "Rocsparse"
    return getattr(module, prefix + "Plan"), getattr(module, prefix + "PlanPair")


def backend_factory(backend):
    import pygrgl_spmv
    return getattr(pygrgl_spmv, "make_backend_" + backend)


def prepare(grg, **kwargs):
    from pygrgl_spmv.backends.rocsparse import RocsparseRuntime
    method = "prepare_matmul_rocm" if isinstance(grg._runtime, RocsparseRuntime) else "prepare_matmul_cuda"
    return getattr(grg, method)(**kwargs)


def copy_to(target, value):
    import cupy as cp
    if isinstance(target, cp.ndarray):
        cp.copyto(target, cp.asarray(value))
    else:
        import torch
        target.copy_(torch.as_tensor(value, device=target.device))


def to_host(value):
    import cupy as cp
    cp.cuda.runtime.deviceSynchronize()
    return cp.asnumpy(value).copy() if isinstance(value, cp.ndarray) else value.cpu().numpy().copy()


def dense_entries(runtime, direction):
    from pygrgl_spmv.backends.types import Direction
    if not runtime._artifacts:
        return {}
    artifact = runtime._artifacts[0]
    if hasattr(artifact, "prepared"):
        return {k: (state, state.refs) for (d, k), state in artifact.prepared.items() if d == Direction(direction)}
    entries = artifact.up_dense_by_k if direction == "up" else artifact.down_dense_by_k
    return {k: (entry.state, entry.refs) for k, entry in entries.items()}


def _assert_cusparse_reset(runtime) -> None:
    assert runtime._cslib is None
    assert runtime._spmm_lib_by_stream_ptr == {}
    assert runtime._shared_ones is None
    assert runtime._alpha is None
    assert runtime._beta_zero is None
    assert runtime._beta_one is None
    assert runtime._caller_stream is None
    assert runtime._root_stream is None
    assert runtime._caller_to_root_event is None
    assert runtime._root_to_caller_event is None
    assert runtime._level_streams == []
    assert runtime._slot_copy_streams == []
    assert runtime._scratch_streams_up == []
    assert runtime._scratch_streams_down == []
    assert runtime._slot_struct0 == []
    assert runtime._slot_struct1 == []
    assert runtime._up_level_bufs == []
    assert runtime._down_level_bufs == []
    assert runtime._up_src_bufs is None
    assert runtime._down_src_bufs is None
    assert runtime._up_scratch == []
    assert runtime._down_scratch == []
    assert runtime._ext_main_up == []
    assert runtime._ext_main_down == []
    assert runtime._ext_scratch_up == []
    assert runtime._ext_scratch_down == []
    assert runtime._artifacts == ()
    assert runtime._grgs == ()
    assert runtime.stream is None
    assert not runtime._entered
    assert not runtime._active_call


def assert_reset(runtime):
    if hasattr(runtime, "_cslib"):
        _assert_cusparse_reset(runtime)
        return
    assert runtime._shared_ones is None
    assert runtime._caller_stream is None
    assert runtime._root_stream is None
    assert runtime._aux is None
    assert not runtime._artifacts
    assert runtime._grgs == ()
    assert runtime.stream is None
    assert not runtime._entered
    assert not runtime._active_call


def new_stream(backend):
    if backend == "rocsparse":
        import cupy as cp
        return cp.cuda.Stream(non_blocking=True)
    import torch
    return torch.cuda.Stream(device=0)


def to_device(backend, value):
    if backend == "rocsparse":
        import cupy as cp
        return cp.asarray(value)
    import torch
    return torch.from_numpy(value).to(device="cuda:0")


def empty_like(value):
    import cupy as cp
    if isinstance(value, cp.ndarray):
        return cp.empty_like(value)
    import torch
    return torch.empty_like(value)


class Graph:
    def __init__(self, backend, runtime):
        self.backend, self.runtime = backend, runtime
        self.graph = None

    @contextmanager
    def capture(self, stream):
        if self.backend == "cusparse":
            import torch
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                yield
        else:
            import cupy as cp
            cp.cuda.runtime.deviceSynchronize()
            # HIP captures the root wavefront; caller handoffs wrap replay.
            with self.runtime._root_stream as root:
                root.begin_capture()
                try:
                    yield
                finally:
                    self.graph = root.end_capture()

    def replay(self):
        if self.backend == "cusparse":
            self.graph.replay()
        else:
            self.graph.launch(self.runtime.stream)
