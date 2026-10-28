"""Prepared sparse operations; each instance owns all mutable vendor state."""

import ctypes as C
import numpy as np

from . import ffi
from .plan import DenseOrder, Operation, SpMMAlgorithm


class SparseOperation:
    def __init__(self, handle, stream, block, ones, source, destination, plan):
        import cupy as cp

        self.handle, self.stream, self.plan = handle, stream, plan
        self.sparse = self.source = self.destination = None
        if plan.spmv is not None and destination.shape[1] == 1:
            raise NotImplementedError("explicit SpMV is not implemented")
        effective_csr_normal = ((block.fmt.value == "CSR" and plan.op_a == Operation.N)
                                or (block.fmt.value == "CSC" and plan.op_a == Operation.T))
        if (effective_csr_normal and destination.shape[1] <= 2
                and plan.algo in {SpMMAlgorithm.DEFAULT, SpMMAlgorithm.CSR, SpMMAlgorithm.CSR_NNZ_SPLIT}):
            raise ValueError("ROCm 10.1 CSR NNZ-split (including DEFAULT/CSR) is incorrect at widths 1/2; "
                             "select CSR_ROW_SPLIT, CSR_MERGE_PATH, or explicit native SpMV")
        self.workspace = None
        self.size = ffi.Z()
        self.scalar = np.ones(1, dtype=source.dtype)
        self.dtype = ffi.datatype(source.dtype)
        try:
            prefix = (block.nrows, block.ncols, block.nnz,
                      block.struct0.data.ptr, block.struct1.data.ptr, ones.ptr)
            indices = (ffi.indextype(block.struct0.dtype),)
            if block.fmt.value != "COO":
                indices += (ffi.indextype(block.struct1.dtype),)
            self.sparse = ffi.create("create_" + block.fmt.value.lower() + "_descr",
                                     *prefix, *indices, 0, self.dtype)
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
                 self.scalar.ctypes.data, self.sparse, self.source, self.scalar.ctypes.data,
                 self.destination, self.dtype, int(self.plan.algo), stage, C.byref(self.size),
                 None if self.workspace is None else self.workspace.data.ptr)

    def __call__(self):
        self._spmm(3)

    def close(self):
        for attr, kind in (("source", "dnmat"), ("destination", "dnmat"), ("sparse", "spmat")):
            desc = getattr(self, attr)
            if desc:
                ffi.call("destroy_" + kind + "_descr", desc)
                setattr(self, attr, None)
        self.workspace = None
