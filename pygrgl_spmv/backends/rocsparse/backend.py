"""Resident single-device ROCm runtime with destination-stream wavefronts.

CUDA and HIP execution remain separate until both have hardware validation.
Artifact semantics and prepared-operation specifications are shared.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from numbers import Integral
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from pygrgl_spmv.backends.base import materialize_sparse_block, _SelectorLevels, _selector_levels
from pygrgl_spmv.backends.types import Direction, InitMode, SparseFormat
from pygrgl_spmv.grg import BoundGRG, RuntimeRequirements, _GpuMatmulSpec
from pygrgl_spmv.grg.artifact import _load_grg_spmv_host, iter_artifact_blocks, scan_grg_spmv
from .ffi import Handle, datatype
from .operations import SparseOperation
from .plan import RocsparsePlan, RocsparsePlanPair, DenseOrder, Operation
from .vmm import SharedOnes


def standard_pair():
    return RocsparsePlanPair(
        RocsparsePlan("N", "CSR", "N", "N", "ROW", "ROW", "CSR_ROW_SPLIT"),
        RocsparsePlan("T", "CSC", "N", "N", "ROW", "ROW", "DEFAULT"),
    )


@dataclass(frozen=True)
class RocsparseLayout:
    artifacts: tuple
    pair: RocsparsePlanPair
    dtype: np.dtype
    requirements: RuntimeRequirements
    device: int
    stream: object
    shared_ones: str

    @property
    def stream_ptr(self):
        return 0 if self.stream is None else int(self.stream.ptr)


def plan_rocsparse_layout(*, artifacts, pair=None, dtype=np.float64, requirements,
                         device=0, stream=None, shared_ones="vmm", allow_residency=True,
                         ring_buffer_size=0, vram_budget_bytes=0):
    import cupy as cp

    if not cp.cuda.runtime.is_hip:
        raise RuntimeError("rocSPARSE requires a HIP build of CuPy")
    if not allow_residency or ring_buffer_size or vram_budget_bytes:
        raise ValueError("rocSPARSE supports resident execution only; streaming/budgets are unsupported")
    if isinstance(device, bool) or not isinstance(device, Integral):
        raise TypeError("rocSPARSE requires a single integer device ordinal")
    device = int(device)
    if not 0 <= device < cp.cuda.runtime.getDeviceCount():
        raise ValueError(f"invalid ROCm device ordinal {device}")
    if shared_ones not in {"vmm", "materialized"}:
        raise ValueError("shared_ones must be 'vmm' or 'materialized'")
    datatype(dtype)
    if not isinstance(requirements, RuntimeRequirements):
        raise TypeError("requirements must be RuntimeRequirements")
    if pair is None:
        pair = standard_pair()
    if not isinstance(pair, RocsparsePlanPair):
        raise TypeError("pair must be RocsparsePlanPair")
    paths = tuple(Path(path) for path in artifacts)
    if not paths:
        raise ValueError("artifacts must be non-empty")
    for path in paths:
        if path.suffix != ".grg_spmv":
            raise ValueError(f"expected a .grg_spmv artifact, got {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
    if stream is not None and not isinstance(stream, cp.cuda.Stream):
        raise TypeError("ROCm stream must be a CuPy Stream or None")
    if stream is not None and stream.device_id not in {-1, device}:
        raise ValueError("ROCm stream belongs to a different device")
    return RocsparseLayout(tuple(scan_grg_spmv(path) for path in paths), pair,
                          np.dtype(dtype), requirements, device, stream, shared_ones)


@dataclass(frozen=True)
class _Block:
    struct0: object
    struct1: object
    nrows: int
    ncols: int
    nnz: int
    fmt: SparseFormat

    def alias(self, plan, owner):
        if plan.store == owner.store:
            return self
        return replace(self, nrows=self.ncols, ncols=self.nrows, fmt=plan.fmt)


@dataclass
class _DirectionState:
    levels: list
    sources: list
    operations: list = field(default_factory=list)
    refs: int = 0
    streams: list = field(default_factory=list)
    handles: list = field(default_factory=list)
    ready: list = field(default_factory=list)
    launch: object = None

    def close(self):
        for stream in self.streams:
            stream.synchronize()
        for _, _, operation in self.operations:
            operation.close()
        self.operations.clear()
        for handle in self.handles:
            handle.close()
        self.handles.clear()
        self.streams.clear()
        self.sources.clear()
        self.levels.clear()
        self.ready.clear()
        self.launch = None


@dataclass
class _ArtifactState:
    path: Path
    state: object
    blocks: dict[Direction, list]
    mut_selector: _SelectorLevels
    miss_selector: _SelectorLevels
    node_perm: object
    sample_to_individual: object
    xtx_bias: object
    init_vector_up_bias: object
    init_vector_down_bias: object
    init_xtx_up_bias: object
    init_xtx_down_bias: object
    prepared: dict[tuple[Direction, int], _DirectionState] = field(default_factory=dict)


class RocsparsePreparedMatmul:
    def __init__(self, runtime, artifact, spec):
        self._runtime, self._artifact, self._spec = runtime, artifact, spec
        self._dense_state = None
        cp, dtype = runtime._cp, runtime.layout.dtype
        with runtime.device, runtime._caller_stream:
            self._input_internal = cp.zeros((spec.input_cols, spec.k), dtype=dtype)
            self._output_internal = cp.zeros((spec.output_cols, spec.k), dtype=dtype)
            self.input, self.output = self._input_internal.T, self._output_internal.T
            if spec.use_miss:
                miss = cp.zeros((artifact.state.num_mutations, spec.k), dtype=dtype)
                if spec.direction == Direction.DOWN:
                    self._miss_input_internal, self.miss_input = miss, miss.T
                else:
                    self._miss_output_internal, self.miss_output = miss, miss.T
            if spec.init_mode == InitMode.VECTOR:
                self.init_vector = cp.zeros(spec.k, dtype=dtype)
                self._init_vector_internal = self.init_vector
            if spec.init_mode == InitMode.MATRIX:
                self._init_matrix_internal = cp.zeros((artifact.state.num_nodes, spec.k), dtype=dtype)
                self.init_matrix = self._init_matrix_internal.T

    def __enter__(self):
        if self._dense_state is not None:
            raise RuntimeError("ROCm prepared operation already entered")
        self._dense_state = self._runtime._acquire(self._artifact, self._spec.direction, self._spec.k)
        try:
            # Entered operations must be capture-ready, including CuPy's lazy
            # kernels for selectors, initialization and endpoint reductions.
            if not self._runtime._active_call:
                self._call_prelocked()
                self._runtime._caller_stream.synchronize()
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc):
        if self._dense_state is not None:
            self._runtime._release(self._artifact, self._spec.direction, self._spec.k, self._dense_state)
            self._dense_state = None

    def __call__(self):
        with self._runtime._call_scope():
            self._call_prelocked()

    def _call_prelocked(self):
        self._runtime._execute_prepared(self._artifact, self._spec, self)


class RocsparseRuntime:
    def __init__(self, layout):
        import cupy as cp

        self._cp, self.layout = cp, layout
        self.device = cp.cuda.Device(layout.device)
        self._entered = self._active_call = False
        self._artifacts, self._grgs = [], ()
        self._shared_ones = None
        self._graphs = []
        self._caller_stream = self._root_stream = None
        self._aux = None
        self.stream = None

    @property
    def stream_ptr(self):
        return 0 if self.stream is None else int(self.stream.ptr)

    @property
    def grgs(self):
        if not self._entered:
            raise RuntimeError("RocsparseRuntime must be entered before accessing grgs")
        return self._grgs

    def __enter__(self):
        if self._entered:
            raise RuntimeError("RocsparseRuntime is already entered")
        cp = self._cp
        try:
            with self.device:
                self._caller_stream = self.layout.stream or cp.cuda.get_current_stream()
                self.stream = self._caller_stream
                self._root_stream = cp.cuda.Stream(non_blocking=True)
                self._to_root = cp.cuda.Event(disable_timing=True)
                self._to_caller = cp.cuda.Event(disable_timing=True)
                with self._caller_stream:
                    self._shared_ones = SharedOnes(max(s.max_block_nnz for s in self.layout.artifacts),
                                                  self.layout.dtype, self.layout.device, self.layout.shared_ones)
                    max_k = max(self.layout.requirements.max_k_up if self.layout.pair.plan_up else 0,
                                self.layout.requirements.max_k_down if self.layout.pair.plan_down else 0)
                    max_rows = max(max(s.num_nodes, s.num_samples, s.num_mutations,
                                       s.selector_mut_nnz, s.selector_miss_nnz) for s in self.layout.artifacts)
                    self._aux = cp.zeros((max_rows, max_k), dtype=self.layout.dtype)
                    for scan in self.layout.artifacts:
                        self._artifacts.append(self._build_artifact(scan))
                self._caller_stream.synchronize()
            self._grgs = tuple(BoundGRG(self, i, a.state, a.path, self.layout.device)
                               for i, a in enumerate(self._artifacts))
            self._entered = True
            return self
        except BaseException:
            self.close()
            raise

    def __exit__(self, *exc):
        self.close()

    def close(self):
        with self.device:
            if self._caller_stream is not None:
                self._caller_stream.synchronize()
            if self._root_stream is not None:
                self._root_stream.synchronize()
            for owner in self._graphs:
                owner._invalidate()
            self._graphs.clear()
            for artifact in self._artifacts:
                for dense in artifact.prepared.values():
                    dense.close()
                    dense.refs = 0
                artifact.prepared.clear()
            self._artifacts.clear()
            self._grgs = ()
            if self._shared_ones is not None:
                self._shared_ones.close()
                self._shared_ones = None
            self._aux = None
            self._to_root = self._to_caller = None
            self._root_stream = self._caller_stream = None
            self.stream = None
            self._entered = False
            self._active_call = False

    @contextmanager
    def _call_scope(self):
        if not self._entered:
            raise RuntimeError("RocsparseRuntime must be entered before matmul")
        if self._active_call:
            raise RuntimeError("concurrent runtime.grgs calls are not supported")
        self._active_call = True
        try:
            with self.device:
                yield
        finally:
            self._active_call = False

    @contextmanager
    def _caller_root_scope(self):
        self._to_root.record(self._caller_stream)
        self._root_stream.wait_event(self._to_root)
        try:
            yield
        finally:
            self._to_caller.record(self._root_stream)
            self._caller_stream.wait_event(self._to_caller)

    def _build_artifact(self, scan):
        cp = self._cp
        state = _load_grg_spmv_host(scan.path, self.layout.dtype)
        pair = self.layout.pair
        blocks = {Direction.UP: [], Direction.DOWN: []}
        share = pair.plan_up is not None and pair.plan_down is not None and pair.plan_up.can_share_storage_with(pair.plan_down)
        for raw in iter_artifact_blocks(scan.path):
            # beta=1: an empty block is exactly a no-op. NNZ_SPLIT in
            # ROCm 10.1 otherwise attempts a zero-grid launch on this input.
            if raw.nnz == 0:
                continue
            base = sp.csr_matrix((np.ones(raw.nnz, dtype=np.bool_), raw.indices, raw.indptr), shape=raw.shape)
            owner = None
            for direction, plan in ((Direction.UP, pair.plan_up), (Direction.DOWN, pair.plan_down)):
                if plan is None:
                    continue
                if direction == Direction.DOWN and share:
                    block = owner.alias(plan, pair.plan_up)
                else:
                    mat = materialize_sparse_block(base, store=plan.store, fmt=plan.fmt)
                    a, b = (mat.row, mat.col) if plan.fmt == SparseFormat.COO else (mat.indptr, mat.indices)
                    block = _Block(cp.asarray(a), cp.asarray(b), *mat.shape, raw.nnz, plan.fmt)
                    owner = block
                dst, src = (raw.dst_level, raw.src_level) if direction == Direction.UP else (raw.src_level, raw.dst_level)
                blocks[direction].append((dst, src, block))
        biases = {}
        for name in ("init_vector_up_bias", "init_vector_down_bias", "init_xtx_up_bias", "init_xtx_down_bias"):
            value = getattr(state, name)
            biases[name] = None if value is None else cp.asarray(value, dtype=self.layout.dtype)
        return _ArtifactState(
            path=scan.path, state=state, blocks=blocks,
            mut_selector=_selector_levels(cp, state.sel_mut, state.level_offsets),
            miss_selector=_selector_levels(cp, state.sel_miss, state.level_offsets),
            node_perm=cp.asarray(state.node_perm),
            sample_to_individual=cp.asarray(state.sample_to_individual),
            xtx_bias=None if state.coalescence_counts is None else cp.asarray(
                2 * state.coalescence_counts, dtype=self.layout.dtype),
            **biases,
        )

    def _prepare_matmul_rocm(self, grg, spec):
        if not self._entered:
            raise RuntimeError("RocsparseRuntime must be entered before preparation")
        plan = self.layout.pair.plan_up if spec.direction == Direction.UP else self.layout.pair.plan_down
        if plan is None:
            raise ValueError(f"{spec.direction.value.upper()} plan is not configured")
        return RocsparsePreparedMatmul(self, self._artifacts[grg._artifact_index], spec)

    def _acquire(self, artifact, direction, k):
        if not self._entered:
            raise RuntimeError("RocsparseRuntime must be entered before preparation")
        key = (direction, k)
        if key not in artifact.prepared:
            cp = self._cp
            plan = self.layout.pair.plan_up if direction == Direction.UP else self.layout.pair.plan_down
            with self.device, self._caller_stream:
                levels = [cp.zeros((int(rows), k), dtype=self.layout.dtype,
                                  order="C" if plan.order_c == DenseOrder.ROW else "F")
                          for rows in np.diff(artifact.state.level_offsets)]
                sources = []
                for level in levels:
                    view = level if plan.op_b == Operation.N else level.T
                    # Preserve an alias when its order already matches the plan.
                    convert = cp.ascontiguousarray if plan.order_b == DenseOrder.ROW else cp.asfortranarray
                    sources.append(convert(view))
                # A stream that rejoined a HIP capture cannot safely be reused
                # for a second wavefront in that graph (hipGraphInstantiate can
                # crash). Each prepared state owns its streams and handles,
                # just as it owns descriptors and preprocessing workspaces.
                dense = _DirectionState(levels, sources,
                    streams=[cp.cuda.Stream(non_blocking=True) for _ in levels],
                    ready=[cp.cuda.Event(disable_timing=True) for _ in levels],
                    launch=cp.cuda.Event(disable_timing=True))
                self._caller_stream.synchronize()
                try:
                    for stream in dense.streams:
                        dense.handles.append(Handle(stream))
                    for dst, src, block in artifact.blocks[direction]:
                        with dense.streams[dst]:
                            operation = SparseOperation(dense.handles[dst], dense.streams[dst], block,
                                                        self._shared_ones, sources[src], levels[dst], plan)
                        dense.operations.append((dst, src, operation))
                except BaseException:
                    dense.close()
                    raise
            artifact.prepared[key] = dense
        dense = artifact.prepared[key]
        dense.refs += 1
        return dense

    def _release(self, artifact, direction, k, dense):
        key = (direction, k)
        if not self._entered:  # Runtime teardown already released every entry.
            return
        if artifact.prepared.get(key) is not dense or dense.refs <= 0:
            raise RuntimeError("ROCm prepared state reference count mismatch")
        dense.refs -= 1
        if dense.refs == 0:
            self._caller_stream.synchronize()
            dense.close()
            del artifact.prepared[key]

    def _enqueue_wavefront(self, artifact, direction, dense):
        cp = self._cp
        h = len(dense.levels)
        plan = self.layout.pair.plan_up if direction == Direction.UP else self.layout.pair.plan_down
        dense.launch.record(self._root_stream)
        for stream in dense.streams:
            stream.wait_event(dense.launch)
        order = range(h) if direction == Direction.UP else range(h - 1, -1, -1)
        for dst in order:
            stream = dense.streams[dst]
            with stream:
                for destination, src, operation in dense.operations:
                    if destination == dst:
                        stream.wait_event(dense.ready[src])
                        operation()
                view = dense.levels[dst] if plan.op_b == Operation.N else dense.levels[dst].T
                if view.data.ptr != dense.sources[dst].data.ptr:
                    cp.copyto(dense.sources[dst], view)
                dense.ready[dst].record(stream)
        for event in dense.ready:
            self._root_stream.wait_event(event)

    def _execute_prepared(self, artifact, spec, prepared):
        dense = prepared._dense_state
        if dense is None or dense.refs <= 0 or not self._entered:
            raise RuntimeError("ROCm prepared matmul must be entered before use")
        with self.device, self._caller_root_scope():
            self._enqueue_prepared(artifact, spec, prepared)

    def _enqueue_prepared(self, artifact, spec, prepared):
        # Capture starts at the root stream, so the graph contains only the
        # destination-stream wavefront. Caller/root handoffs belong to eager
        # calls; replay itself is ordered on the caller stream by graph launch.
        self._seed_prepared(artifact, spec, prepared)
        self._enqueue_wavefront(artifact, spec.direction, prepared._dense_state)
        self._write_prepared_output(artifact, spec, prepared)

    def _seed_prepared(self, artifact: object, spec: _GpuMatmulSpec, prepared: RocsparsePreparedMatmul) -> None:
        level_bufs = prepared._dense_state.levels
        assert level_bufs
        k = int(spec.k)
        with self._root_stream:
            for level in range(len(artifact.state.level_offsets) - 1):
                rows = int(artifact.state.level_offsets[level + 1] - artifact.state.level_offsets[level])
                level_bufs[level][:rows, :k].fill(0)
            if spec.backend_init_mode == InitMode.XTX:
                if artifact.xtx_bias is None:
                    raise ValueError("init_mode=xtx requires GRG coalescence counts")
                for level in range(len(artifact.state.level_offsets) - 1):
                    lo = int(artifact.state.level_offsets[level])
                    hi = int(artifact.state.level_offsets[level + 1])
                    level_bufs[level][: hi - lo, :k] += artifact.xtx_bias[lo:hi, None]
            elif spec.backend_init_mode == InitMode.VECTOR:
                for level in range(len(artifact.state.level_offsets) - 1):
                    rows = int(artifact.state.level_offsets[level + 1] - artifact.state.level_offsets[level])
                    level_bufs[level][:rows, :k] += prepared._init_vector_internal[None, :]
            elif spec.backend_init_mode == InitMode.MATRIX:
                for level in range(len(artifact.state.level_offsets) - 1):
                    lo = int(artifact.state.level_offsets[level])
                    hi = int(artifact.state.level_offsets[level + 1])
                    self._cp.take(prepared._init_matrix_internal, artifact.node_perm[lo:hi], axis=0, out=level_bufs[level][: hi - lo, :k])

            if spec.direction == Direction.UP:
                if spec.by_individual:
                    temp = self._aux[: artifact.state.num_samples, :k]
                    self._cp.take(prepared._input_internal, artifact.sample_to_individual, axis=0, out=temp)
                    level_bufs[0][: artifact.state.num_samples, :k] += temp
                else:
                    level_bufs[0][: artifact.state.num_samples, :k] += prepared._input_internal
                return

            for level, rows in enumerate(artifact.mut_selector.rows_by_level):
                if rows.size == 0:
                    continue
                count = int(rows.size)
                level_rows = int(artifact.state.level_offsets[level + 1] - artifact.state.level_offsets[level])
                temp = self._aux[:count, :k]
                self._cp.take(prepared._input_internal, rows, axis=0, out=temp)
                self._cp.add.at(level_bufs[level][:level_rows, :k], artifact.mut_selector.cols_by_level[level], temp)
            if spec.use_miss:
                for level, rows in enumerate(artifact.miss_selector.rows_by_level):
                    if rows.size == 0:
                        continue
                    count = int(rows.size)
                    level_rows = int(artifact.state.level_offsets[level + 1] - artifact.state.level_offsets[level])
                    temp = self._aux[:count, :k]
                    self._cp.take(prepared._miss_input_internal, rows, axis=0, out=temp)
                    self._cp.add.at(level_bufs[level][:level_rows, :k], artifact.miss_selector.cols_by_level[level], temp)

    def _apply_endpoint_bias_prepared(
        self,
        artifact: object,
        spec: _GpuMatmulSpec,
        prepared: RocsparsePreparedMatmul,
        output,
    ) -> None:
        if not spec.apply_endpoint_bias:
            return
        rows = int(output.shape[0])
        if spec.init_mode == InitMode.XTX:
            bias = artifact.init_xtx_up_bias if spec.direction == Direction.UP else artifact.init_xtx_down_bias
            if bias is None:
                raise RuntimeError(f"missing rocSPARSE {spec.direction.value} XTX endpoint bias")
            output += bias[:rows, None]
            return
        bias = artifact.init_vector_up_bias if spec.direction == Direction.UP else artifact.init_vector_down_bias
        if bias is None:
            raise RuntimeError(f"missing rocSPARSE {spec.direction.value} vector endpoint bias")
        temp = self._aux[:rows, : spec.k]
        temp.fill(0)
        temp += bias[:rows, None]
        temp *= prepared._init_vector_internal[None, :]
        output += temp

    def _write_prepared_output(self, artifact: object, spec: _GpuMatmulSpec, prepared: RocsparsePreparedMatmul) -> None:
        level_bufs = prepared._dense_state.levels
        assert level_bufs
        k = int(spec.k)
        with self._root_stream:
            if spec.emit_all_nodes:
                for level in range(len(artifact.state.level_offsets) - 1):
                    lo = int(artifact.state.level_offsets[level])
                    hi = int(artifact.state.level_offsets[level + 1])
                    prepared._output_internal[artifact.node_perm[lo:hi]] = level_bufs[level][: hi - lo, :k]
                return

            if spec.direction == Direction.UP:
                output = prepared._output_internal
                output.fill(0)
                for level, rows in enumerate(artifact.mut_selector.rows_by_level):
                    if rows.size == 0:
                        continue
                    count = int(rows.size)
                    level_rows = int(artifact.state.level_offsets[level + 1] - artifact.state.level_offsets[level])
                    temp = self._aux[:count, :k]
                    self._cp.take(level_bufs[level][:level_rows, :k], artifact.mut_selector.cols_by_level[level], axis=0, out=temp)
                    if artifact.mut_selector.row_unique:
                        output[rows] = temp
                    else:
                        self._cp.add.at(output, rows, temp)
                if spec.use_miss:
                    miss_output = prepared._miss_output_internal
                    miss_output.fill(0)
                    for level, rows in enumerate(artifact.miss_selector.rows_by_level):
                        if rows.size == 0:
                            continue
                        count = int(rows.size)
                        level_rows = int(artifact.state.level_offsets[level + 1] - artifact.state.level_offsets[level])
                        temp = self._aux[:count, :k]
                        self._cp.take(level_bufs[level][:level_rows, :k], artifact.miss_selector.cols_by_level[level], axis=0, out=temp)
                        if artifact.miss_selector.row_unique:
                            miss_output[rows] = temp
                        else:
                            self._cp.add.at(miss_output, rows, temp)
                self._apply_endpoint_bias_prepared(artifact, spec, prepared, output)
                return

            if spec.by_individual:
                output = level_bufs[0][: artifact.state.num_samples, :k]
                self._apply_endpoint_bias_prepared(artifact, spec, prepared, output)
                prepared._output_internal.fill(0)
                self._cp.add.at(prepared._output_internal, artifact.sample_to_individual, output)
            else:
                self._cp.copyto(prepared._output_internal[: artifact.state.num_samples, :k], level_bufs[0][: artifact.state.num_samples, :k])
                self._apply_endpoint_bias_prepared(artifact, spec, prepared, prepared._output_internal)
