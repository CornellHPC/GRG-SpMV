"""Minimal runtime-centric benchmark runner."""

from __future__ import annotations

from time import perf_counter
from contextlib import contextmanager

import numpy as np

from pygrgl_spmv import RuntimeRequirements


def _synchronize_cuda_runtime(runtime) -> None:
    import torch

    torch.cuda.synchronize(torch.device("cuda", int(runtime.layout.device)))


def _runtime_torch_stream(runtime):
    stream = getattr(runtime, "_torch_caller_stream", None)
    return None if stream is None else stream()


def _copy_on_runtime_stream(runtime, dst, src) -> None:
    import torch

    stream = _runtime_torch_stream(runtime)
    if stream is None:
        dst.copy_(src)
        return
    with torch.cuda.device(dst.device), torch.cuda.stream(stream):
        dst.copy_(src)


@contextmanager
def _prepared_io(grg, runtime, matrix, direction, k):
    from pygrgl_spmv.backends.rocsparse import RocsparseRuntime

    if isinstance(runtime, RocsparseRuntime):
        import cupy as cp
        with runtime.device, runtime.stream, grg.prepare_matmul_rocm(direction=direction, k=k) as op:
            src = cp.asarray(matrix)
            yield op, lambda: cp.copyto(op.input, src), runtime.stream.synchronize
    else:
        import torch
        with grg.prepare_matmul_cuda(direction=direction, k=k) as op:
            stream = _runtime_torch_stream(runtime)
            if stream is None:
                src = torch.from_numpy(matrix).to(device=op.input.device)
            else:
                with torch.cuda.device(op.input.device), torch.cuda.stream(stream):
                    src = torch.from_numpy(matrix).to(device=op.input.device)
            yield (op, lambda: _copy_on_runtime_stream(runtime, op.input, src),
                   lambda: _synchronize_cuda_runtime(runtime))


def benchmark_runtime(*, runtime_cls, layout, direction: str, k: int, dtype: np.dtype, warmup: int, trials: int) -> None:
    with runtime_cls(layout) as runtime:
        (grg,) = runtime.grgs
        cols = grg.num_samples if direction == "up" else grg.num_mutations
        rng = np.random.default_rng(2026)
        matrix = rng.standard_normal((int(k), cols), dtype=dtype)
        times = []
        if getattr(runtime, "device", None) is not None:
            with _prepared_io(grg, runtime, matrix, direction, int(k)) as (op, copy, synchronize):
                synchronize()
                for _ in range(int(warmup)):
                    copy()
                    op()
                    synchronize()
                for _ in range(int(trials)):
                    t0 = perf_counter()
                    copy()
                    op()
                    synchronize()
                    times.append((perf_counter() - t0) * 1000.0)
        else:
            for _ in range(int(warmup)):
                _ = grg.matmul(matrix, direction)
            for _ in range(int(trials)):
                t0 = perf_counter()
                _ = grg.matmul(matrix, direction)
                times.append((perf_counter() - t0) * 1000.0)
        arr = np.asarray(times, dtype=np.float64)
        print(f"backend={runtime_cls.__name__} direction={direction} k={k} mean_ms={arr.mean():.4f} std_ms={arr.std():.4f}")


def baseline_requirements(*, direction: str, k: int) -> RuntimeRequirements:
    return RuntimeRequirements(
        max_k_up=max(int(k), 1) if direction == "up" else 1,
        max_k_down=max(int(k), 1) if direction == "down" else 1,
        need_down_miss_input=False,
        need_up_miss_output=False,
        need_init_vector=False,
        need_init_matrix=False,
        need_init_xtx=False,
    )
