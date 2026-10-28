"""CuPy HIP graph capture; CUDA's Torch graph path remains independent."""

import numpy as np

from pygrgl_spmv.adaptor import _CapturedBoundGRGBase, CaptureSpec, RunConfigs, _bare_req, _validate_capture_spec


class RocsparseCapturedBoundGRG(_CapturedBoundGRGBase):
    """Share the capture key and array-validation contract with the CUDA wrapper."""

    def _invalidate(self):
        with self._device_lock:
            if not self._closed:
                self._capture_stream.synchronize()
            super()._invalidate()

    def matmul(self, input, direction, emit_all_nodes=False, by_individual=False, init=None, miss=None):
        import cupy as cp

        direction = self._parse_direction(direction).value
        with self._device_lock:
            if self._closed:
                raise RuntimeError("CapturedBoundGRG.matmul() called after its runtime was released")
            if init is None:
                init_mode, payload = "none", None
            elif isinstance(init, str):
                if init != "xtx":
                    raise ValueError(f"unexpected init value: {init!r}")
                init_mode, payload = "xtx", None
            else:
                init_mode, payload = ("vector" if getattr(init, "ndim", None) == 1 else "matrix"), init
            if miss is not None and emit_all_nodes:
                raise RuntimeError('the "miss" parameter cannot be mixed with the "emit_all_nodes" parameter')
            if miss is not None and init is not None:
                raise ValueError('the "miss" parameter cannot be mixed with the "init" parameter')
            key = self._key(direction, bool(by_individual), init_mode, miss is not None, bool(emit_all_nodes))
            op = self._prepared_ops[key]
            device = self.device if self._native else None
            self._validate_array(input, name="input", role="input", expected_shape=op.input.shape,
                                 expected_dtype=op.input.dtype, expected_device=device)
            k = input.shape[0]
            for value, name, shape, role in (
                (payload, "init", (k,) if init_mode == "vector" else (k, self.num_nodes), "init"),
                (miss, "miss", (k, self.num_mutations), "miss-in" if direction == "down" else "miss-out"),
            ):
                if value is not None:
                    if hasattr(value, "dtype") and value.dtype != input.dtype:
                        raise TypeError(f"{name} must match the dtype of the input matrix")
                    self._validate_array(value, name=name, role=role, expected_shape=shape,
                                         expected_dtype=input.dtype, expected_device=device, exact_k=True)
            # Like CUDA: input arrays must be ready on the caller's current
            # stream. Output and in-place missingness are ready on return.
            with cp.cuda.Device(self.device):
                if self._native:
                    cp.cuda.get_current_stream().synchronize()
                with self._capture_stream:
                    def stage(dst, value):
                        dst.fill(0)
                        cp.copyto(dst[:k], cp.asarray(value, dtype=dst.dtype))

                    stage(op.input, input)
                    if payload is not None:
                        stage(op.init_vector if init_mode == "vector" else op.init_matrix, payload)
                    if miss is not None and direction == "down":
                        stage(op.miss_input, miss)
                    self._graphs[key].launch(self._capture_stream)
                    result = op.output[:k].copy()
                    if input.dtype == np.int32:
                        result = result.astype(cp.int32)
                    if miss is not None and direction == "up":
                        value = op.miss_output[:k].astype(miss.dtype, copy=False)
                        miss += value if self._native else cp.asnumpy(value)
                    if not self._native:
                        result = cp.asnumpy(result)
                self._capture_stream.synchronize()
                return result


def capture_grg(grg, req, cfg, stack, device_lock):
    runtime = grg._runtime
    stream = runtime._caller_stream
    bare = _bare_req(req)
    specs = req.capture_ops if isinstance(req, RunConfigs) and req.capture_ops else (
        CaptureSpec("up", bare.max_k_up), CaptureSpec("down", bare.max_k_down))
    prepared, graphs, inputs, inits, misses = {}, {}, {}, {}, {}
    graph = None
    try:
        for spec in specs:
            _validate_capture_spec(spec, bare)
            key = (spec.direction, spec.by_individual, spec.init_mode, spec.use_miss, spec.emit_all_nodes)
            if key in prepared:
                raise ValueError(f"Duplicate CaptureSpec key {key!r} in capture_ops")
            op = stack.enter_context(grg.prepare_matmul_rocm(
                direction=spec.direction, k=spec.k, by_individual=spec.by_individual,
                init_mode=spec.init_mode, use_miss=spec.use_miss, emit_all_nodes=spec.emit_all_nodes))
            prepared[key] = op
            # Preprocessing, valid allocations and CuPy kernel compilation precede capture.
            for _ in range(2):
                op()
            stream.synchronize()
            root = runtime._root_stream
            with root:
                root.begin_capture()
                try:
                    runtime._enqueue_prepared(op._artifact, op._spec, op)
                finally:
                    graph = root.end_capture()
            graphs[key] = graph
            inputs[key] = op.input
            if spec.init_mode in {"vector", "matrix"}:
                inits[key] = op.init_vector if spec.init_mode == "vector" else op.init_matrix
            if spec.use_miss and spec.direction == "down":
                misses[key] = op.miss_input
    except BaseException:
        # A traceback can retain these locals while the loader unwinds its
        # ExitStack. Destroy captured graphs before prepared descriptors.
        graph = None
        graphs.clear()
        raise
    captured = RocsparseCapturedBoundGRG(grg, prepared, graphs, inputs, inits, misses,
                                       stream, native=cfg.native, device_lock=device_lock)
    runtime._graphs.append(captured)
    stack.callback(captured._invalidate)
    return captured
