"""HIP-only ownership checks: CUDA VMM uses a different driver API."""

import numpy as np
import pytest

pytestmark = pytest.mark.rocsparse


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("count", [7, 5_000_000])
@pytest.mark.parametrize("mode", ["vmm", "materialized"])
def test_shared_ones(dtype, count, mode):
    import cupy as cp
    from pygrgl_spmv.backends.rocsparse.vmm import SharedOnes, TILE_BYTES

    storage = SharedOnes(count, dtype, 0, mode)
    try:
        assert float(storage.array.sum()) == count
        if mode == "vmm":
            assert storage.physical_nbytes <= max(4096, TILE_BYTES)
    finally:
        cp.cuda.get_current_stream().synchronize()
        storage.close()
        storage.close()
    assert storage.ptr == 0
    assert storage.addr == 0
    assert storage.handle is None


def test_partial_mapping_failure_cleans_up(monkeypatch):
    from rocm.bindings import hip
    from pygrgl_spmv.backends.rocsparse.vmm import SharedOnes

    original_map = hip.hipMemMap
    original_release = hip.hipMemRelease
    original_free = hip.hipMemAddressFree
    counts = dict(map=0, release=0, free=0)

    def fail_second(*args):
        counts["map"] += 1
        if counts["map"] == 2:
            raise RuntimeError("injected map failure")
        return original_map(*args)

    def release(*args):
        counts["release"] += 1
        return original_release(*args)

    def free(*args):
        counts["free"] += 1
        return original_free(*args)

    monkeypatch.setattr(hip, "hipMemMap", fail_second)
    monkeypatch.setattr(hip, "hipMemRelease", release)
    monkeypatch.setattr(hip, "hipMemAddressFree", free)
    with pytest.raises(RuntimeError, match="injected map failure"):
        SharedOnes(5_000_000, np.float64, 0)
    assert counts == dict(map=2, release=1, free=1)
