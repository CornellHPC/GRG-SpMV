"""Read-only shared-one storage using official HIP virtual memory bindings."""

import numpy as np

TILE_BYTES = 16 * 1024 * 1024


def checked(result):
    if int(result[0]):
        raise RuntimeError(f"HIP virtual memory call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def round_up(size, granularity):
    return ((size + granularity - 1) // granularity) * granularity


class SharedOnes:
    """One physical tile per runtime; errors never select materialized storage."""

    def __init__(self, count, dtype, device, mode="vmm"):
        import cupy as cp

        self.array = None
        self.addr = 0
        self.handle = None
        self.mapped = 0
        self.reserved = 0
        self.physical_nbytes = 0
        self.logical_nbytes = int(count) * np.dtype(dtype).itemsize
        self.device = cp.cuda.Device(device)
        if mode not in {"vmm", "materialized"}:
            raise ValueError("shared_ones must be 'vmm' or 'materialized'")
        self.vmm = mode == "vmm"
        if not count:
            return
        with self.device:
            if not self.vmm:
                self.array = cp.ones(count, dtype=dtype)
                self.physical_nbytes = self.logical_nbytes
                return
            from rocm.bindings import hip

            self.hip = hip
            supported = checked(hip.hipDeviceGetAttribute(
                hip.hipDeviceAttribute_t.hipDeviceAttributeVirtualMemoryManagementSupported, device))
            if not supported:
                raise RuntimeError("HIP virtual memory is unsupported on this device")
            prop = hip.hipMemAllocationProp()
            prop.type = hip.hipMemAllocationType.hipMemAllocationTypePinned
            prop.location.type = hip.hipMemLocationType.hipMemLocationTypeDevice
            prop.location.id = device
            gran = int(checked(hip.hipMemGetAllocationGranularity(
                prop, hip.hipMemAllocationGranularity_flags.hipMemAllocationGranularityMinimum)))
            self.physical_nbytes = round_up(min(self.logical_nbytes, TILE_BYTES), gran)
            self.reserved = round_up(self.logical_nbytes, self.physical_nbytes)
            try:
                self.handle = checked(hip.hipMemCreate(self.physical_nbytes, prop, 0))
                self.base = checked(hip.hipMemAddressReserve(self.reserved, gran, 0, 0))
                self.addr = int(self.base)
                for offset in range(0, self.reserved, self.physical_nbytes):
                    checked(hip.hipMemMap(self.addr + offset, self.physical_nbytes, 0, self.handle, 0))
                    self.mapped += self.physical_nbytes
                access = hip.hipMemAccessDesc()
                access.location.type = hip.hipMemLocationType.hipMemLocationTypeDevice
                access.location.id = device
                access.flags = hip.hipMemAccessFlags.hipMemAccessFlagsProtReadWrite
                checked(hip.hipMemSetAccess(self.addr, self.reserved, access, 1))
                # UnownedMemory keeps a lightweight token, avoiding an owner/array cycle.
                memory = cp.cuda.UnownedMemory(self.addr, self.reserved, object(), device_id=device)
                self.array = cp.ndarray((count,), dtype=dtype, memptr=cp.cuda.MemoryPointer(memory, 0))
                self.array[:min(count, self.physical_nbytes // np.dtype(dtype).itemsize)].fill(1)
                cp.cuda.get_current_stream().synchronize()
                access.flags = hip.hipMemAccessFlags.hipMemAccessFlagsProtRead
                checked(hip.hipMemSetAccess(self.addr, self.reserved, access, 1))
            except BaseException:
                self.close()
                raise

    @property
    def ptr(self):
        return 0 if self.array is None else self.array.data.ptr

    def close(self):
        # The runtime synchronizes users and destroys graphs/descriptors first.
        self.array = None
        with self.device:
            try:
                if self.addr:
                    while self.mapped:
                        offset = self.mapped - self.physical_nbytes
                        checked(self.hip.hipMemUnmap(self.addr + offset, self.physical_nbytes))
                        self.mapped = offset
            finally:
                try:
                    if self.addr and not self.mapped:
                        checked(self.hip.hipMemAddressFree(self.base, self.reserved))
                        self.addr = 0
                finally:
                    if self.handle is not None:
                        checked(self.hip.hipMemRelease(self.handle))
                        self.handle = None
