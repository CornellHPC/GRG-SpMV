"""Prepared sparse operations; each instance owns all mutable vendor state."""

import ctypes as C
import numpy as np

from . import ffi
from .plan import DenseOrder, Operation, SpMMAlgorithm, SpMVAlgorithm


class SparseOperation:
    def __init__(self, handle, stream, block, ones, source, destination, plan):
        import cupy as cp

        self.handle, self.stream, self.plan = handle, stream, plan
        self.sparse = self.source = self.destination = None
        self.mv = None
        self.analysis_workspace = None
        self.is_vector = plan.spmv is not None and destination.shape[1] == 1
        effective_csr_normal = ((block.fmt.value == "CSR" and plan.op_a == Operation.N)
                                or (block.fmt.value == "CSC" and plan.op_a == Operation.T))
        if self.is_vector and effective_csr_normal and plan.spmv == SpMVAlgorithm.CSR_NNZSPLIT:
            raise ValueError("ROCm 10.1 normal CSR_NNZSPLIT SpMV is incorrect; "
                             "select DEFAULT, CSR_ADAPTIVE, CSR_ROWSPLIT, or CSR_LRB")
        if (not self.is_vector and effective_csr_normal and destination.shape[1] <= 2
                and plan.algo in {SpMMAlgorithm.DEFAULT, SpMMAlgorithm.CSR, SpMMAlgorithm.CSR_NNZ_SPLIT}):
            raise ValueError("ROCm 10.1 CSR NNZ-split (including DEFAULT/CSR) is incorrect at widths 1/2; "
                             "select CSR_ROW_SPLIT, CSR_MERGE_PATH, or explicit native SpMV")
        self.workspace = None
        self.size = ffi.Z()
        self.scalar = np.ones(1, dtype=source.dtype)
        self.scalar_ptr = self.scalar.ctypes.data
        self.dtype = ffi.datatype(source.dtype)
        try:
            prefix = (block.nrows, block.ncols, block.nnz,
                      block.struct0.data.ptr, block.struct1.data.ptr, ones.ptr)
            indices = (ffi.indextype(block.struct0.dtype),)
            if block.fmt.value != "COO":
                indices += (ffi.indextype(block.struct1.dtype),)
            self.sparse = ffi.create("create_" + block.fmt.value.lower() + "_descr",
                                     *prefix, *indices, 0, self.dtype)
            if self.is_vector:
                if not (source.flags.c_contiguous or source.flags.f_contiguous):
                    raise ValueError("native SpMV requires contiguous source vectors")
                if not (destination.flags.c_contiguous or destination.flags.f_contiguous):
                    raise ValueError("native SpMV requires contiguous destination vectors")
                self.source = ffi.create("create_dnvec_descr", source.size, source.data.ptr, self.dtype)
                self.destination = ffi.create("create_dnvec_descr", destination.size, destination.data.ptr, self.dtype)
                self.mv = ffi.create("create_spmv_descr")
                for field, value in enumerate((plan.spmv, plan.op_a, self.dtype, self.dtype)):
                    scalar = ffi.I(int(value))
                    ffi.call("spmv_set_input", handle.ptr, self.mv, field, C.byref(scalar), C.sizeof(scalar), None)
                analysis_size = self._spmv_size(0)
                self.analysis_workspace = cp.empty(analysis_size, dtype=cp.uint8) if analysis_size else None
                self._spmv(0, analysis_size, self.analysis_workspace)
                self.size.value = self._spmv_size(1)
                self.workspace = cp.empty(self.size.value, dtype=cp.uint8) if self.size.value else None
            else:
                self.source = self._matrix(source, plan.order_b)
                self.destination = self._matrix(destination, plan.order_c)
                self._spmm(1)
                self.workspace = cp.empty(self.size.value, dtype=cp.uint8) if self.size.value else None
                self._spmm(2)
            stream.synchronize()
        except BaseException:
            self.close()
            raise

    def _matrix(self, array, order):
        axis = 0 if order == DenseOrder.ROW else 1
        # Singleton dimensions can have both C/F-contiguous flags with a stride
        # smaller than the vendor's required leading dimension.
        minimum = array.shape[1] if order == DenseOrder.ROW else array.shape[0]
        ld = max(1, minimum, array.strides[axis] // array.itemsize)
        return ffi.create("create_dnmat_descr", *array.shape, ld, array.data.ptr, self.dtype, int(order))

    def _spmm(self, stage):
        ffi.call("spmm", self.handle.ptr, int(self.plan.op_a), int(self.plan.op_b),
                 self.scalar_ptr, self.sparse, self.source, self.scalar_ptr,
                 self.destination, self.dtype, int(self.plan.algo), stage, C.byref(self.size),
                 None if self.workspace is None else self.workspace.data.ptr)

    def __call__(self):
        if self.is_vector:
            self._spmv(1, self.size.value, self.workspace)
        else:
            self._spmm(3)

    def _spmv_size(self, stage):
        size = ffi.Z()
        ffi.call("v2_spmv_buffer_size", self.handle.ptr, self.mv, self.sparse,
                 self.source, self.destination, stage, C.byref(size), None)
        return size.value

    def _spmv(self, stage, size, workspace):
        ffi.call("v2_spmv", self.handle.ptr, self.mv, self.scalar_ptr,
                 self.sparse, self.source, self.scalar_ptr, self.destination,
                 stage, size, None if workspace is None else workspace.data.ptr, None)

    def close(self):
        dense_kind = "dnvec" if self.is_vector else "dnmat"
        for attr, kind in (("mv", "spmv"), ("source", dense_kind), ("destination", dense_kind), ("sparse", "spmat")):
            desc = getattr(self, attr)
            if desc:
                ffi.call("destroy_" + kind + "_descr", desc)
                setattr(self, attr, None)
        self.workspace = None
        self.analysis_workspace = None
