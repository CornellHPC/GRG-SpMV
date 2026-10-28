"""Typed rocSPARSE C ABI, verified against the installed ROCm 10.1 headers.

No library is loaded at import time. Handles and descriptors have explicit owners.
The v2 SpMV API separates analysis from compute and supports graph capture.
"""

import ctypes as C
from functools import cache
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

P, I, N, Z = C.c_void_p, C.c_int, C.c_int64, C.c_size_t


class RocsparseError(RuntimeError):
    def __init__(self, function, status):
        self.status = int(status)
        super().__init__(f"rocSPARSE {function} failed with status {status}")


@cache
def library():
    root = os.environ.get("ROCM_PATH")
    if root is None:
        try:
            root = subprocess.check_output(
                [sys.executable, "-m", "rocm_sdk", "path", "--root"],
                text=True, stderr=subprocess.PIPE, timeout=30,
            ).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise ImportError("Install the pip ROCm SDK or set ROCM_PATH to its root") from exc
    if not root:
        raise ImportError("ROCm SDK root is empty")
    path = Path(root) / "lib/librocsparse.so"
    if not path.is_file():
        raise ImportError(f"rocSPARSE library not found: {path}")
    lib = C.CDLL(str(path))
    signatures = {
        "create_handle": [C.POINTER(P)], "destroy_handle": [P],
        "set_stream": [P, P], "set_pointer_mode": [P, I],
        "get_version": [P, C.POINTER(I)],
        "create_csr_descr": [C.POINTER(P), N, N, N, P, P, P, I, I, I, I],
        "create_csc_descr": [C.POINTER(P), N, N, N, P, P, P, I, I, I, I],
        "create_coo_descr": [C.POINTER(P), N, N, N, P, P, P, I, I, I],
        "destroy_spmat_descr": [P],
        "create_dnmat_descr": [C.POINTER(P), N, N, N, P, I, I],
        "destroy_dnmat_descr": [P],
        "create_dnvec_descr": [C.POINTER(P), N, P, I],
        "destroy_dnvec_descr": [P],
        "spmm": [P, I, I, P, P, P, P, P, I, I, I, C.POINTER(Z), P],
        "create_spmv_descr": [C.POINTER(P)], "destroy_spmv_descr": [P],
        "spmv_set_input": [P, P, I, P, Z, P],
        "v2_spmv_buffer_size": [P, P, P, P, P, I, C.POINTER(Z), P],
        "v2_spmv": [P, P, P, P, P, P, P, I, Z, P, P],
    }
    for name, args in signatures.items():
        function = getattr(lib, "rocsparse_" + name)
        function.argtypes = args
        function.restype = I
    return lib


def call(name, *args):
    status = getattr(library(), "rocsparse_" + name)(*args)
    if status:
        raise RocsparseError(name, status)


def create(name, *args):
    result = P()
    call(name, C.byref(result), *args)
    return result


def datatype(dtype):
    dt = np.dtype(dtype)
    if dt == np.float32:
        return 151
    if dt == np.float64:
        return 152
    raise TypeError(f"rocSPARSE requires float32 or float64, got {dt}")


def indextype(dtype):
    dt = np.dtype(dtype)
    if dt == np.int32:
        return 2
    if dt == np.int64:
        return 3
    raise TypeError(f"rocSPARSE structural indices must be int32 or int64, got {dt}")


class Handle:
    def __init__(self, stream):
        self.ptr = create("create_handle")
        try:
            call("set_stream", self.ptr, stream.ptr)
            call("set_pointer_mode", self.ptr, 0)  # host scalar constants
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.ptr:
            call("destroy_handle", self.ptr)
            self.ptr = P()
