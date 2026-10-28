# pygrgl-spmv

`pygrgl-spmv` is the core of MIKADO's formulation, which enables lightning-fast and parallel-friendly GRG operations on devices including CPUs and GPUs.


## Hardware requirements

Currently, the following hardware is supported:
- CPU: Any x86 CPU is supported, though Intel MKL is not officially supported on AMD CPUs.
- NVIDIA GPU: CUDA Toolkit >= 13.3.1 is required for correctness. Older drivers may support [CUDA Forward Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html).
- AMD GPU: pip ROCm SDK 10.1.0 and HIP CuPy are required.

## Install

Requires Python >= 3.11. To install, use:

```bash
pip install .          # CPU dependencies; install MKL separately
```

To use the MKL-based CPU backend, you need to install MKL manually. We recommend using the conda package manager for this.
```bash
conda install -c conda-forge mkl mkl-devel mkl-static mkl-include
```
You can also install following Intel's official [instructions](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html).

Use a separate virtual environment for each GPU vendor. Both extras pin the same
CuPy source, including the [Lanczos breakdown fix](https://github.com/cupy/cupy/pull/10098)
and the separate [near-full-rank restart fix](https://github.com/cupy/cupy/issues/10220).
Prepare the shared build dependencies first:

```bash
uv venv --python 3.12.15 --seed .venv
source .venv/bin/activate
pip install numpy==2.5.4 Cython==3.3.0 setuptools==84.2.1 wheel==0.48.0 fastrlock==0.8.4 hatchling
export CUPY_NUM_BUILD_JOBS=8
```

For NVIDIA, install CUDA Toolkit >= 13.3.1 and set `CUDA_PATH` to its root:

```bash
unset CUPY_INSTALL_USE_HIP HCC_AMDGPU_TARGET ROCM_HOME ROCM_PATH HIP_PATH
export PATH="$CUDA_PATH/bin:$PATH"
pip install --no-build-isolation '.[cuda13]'
```

For AMD, this MI300X (gfx942) setup uses the pip SDK:

```bash
unset ROCM_HOME ROCM_PATH HIP_PATH LD_LIBRARY_PATH CMAKE_PREFIX_PATH
pip install 'rocm[libraries,devel,device-gfx942]==10.1.0' \
    --index-url https://stable.repo.amd.com/rocm/whl-next/
rocm-sdk init
export ROCM_HOME="$(rocm-sdk path --root)"
export ROCM_PATH="$ROCM_HOME"
export PATH="$(rocm-sdk path --bin):$PATH"
export CUPY_INSTALL_USE_HIP=1 HCC_AMDGPU_TARGET=gfx942
pip install --no-build-isolation '.[rocm10]'
rocm-sdk test
```

`[devel]` supplies SDK headers/build tools; `--no-build-isolation` uses the
prepared build environment. These switches are separate from the CuPy fixes.
Without `ROCM_PATH`, rocSPARSE is discovered through the active interpreter's
`python -m rocm_sdk path --root`. An invalid override or missing SDK is an error.

## Basic Usage

To use `pygrgl-spmv` with `grapp` and the supported applications (GWAS, PCA, BOLT-LMM), some adaptor functions have been provided, so that users don't need to touch the lower-level APIs.

### Converting

To obtain a `.grg` file from formats such as `.vcf.gz`, please refer to the [grgl docs](https://grgl.readthedocs.io/en/stable/) for instructions.

`pygrgl-spmv` consumes `.grg_spmv` artifacts only. Converting a `.grg` file into one is a **mandatory, separate step** — nothing in the library converts implicitly, so conversion time is never counted as GRG-SpMV runtime. Use `simple_convert`, which names both the input and the output file:

```python
from pygrgl_spmv import simple_convert

artifact = simple_convert("chr1.grg", "artifacts/chr1.grg_spmv")
```

Or from the command line, which also reports how long the conversion took:

```bash
python -m pygrgl_spmv convert chr1.grg artifacts/chr1.grg_spmv
# artifacts/chr1.grg_spmv  (1.5 MB, 0.55s)
```

`output_path` names the artifact file, not a directory; the `.grg_spmv` suffix is appended if absent, parent directories are created, and an existing artifact at that path is replaced atomically. `dtype` (`float32` or `float64`, default `float64`) sets the precision of the precomputed init-bias arrays baked into the artifact. Multi-processing can speed up conversion of several datasets.

### Running

The adaptor hides planning, layout and runtime setup for user behind three calls: 
- choose a backend (CPU MKL, GPU cuSPARSE or GPU rocSPARSE)
- choose a run configuration (the application)
- load artifacts into an `ExitStack` that owns the runtime.
The loaded GRG artifacts are then ready for execution with `grapp`.
A minimal PCA example:

Install `grapp` separately, or use the `dev` extra; neither GPU extra includes it.

```python
from contextlib import ExitStack

from pygrgl_spmv import make_backend_cusparse, make_backend_rocsparse, make_backend_mkl
from pygrgl_spmv import make_runconfig_pca, load_grg_spmv_multi
# grapp.linalg first: importing grapp.grg_calculator ahead of it hits a circular
# import inside grapp (grg_calculator -> util -> util.simple -> grg_calculator).
from grapp.linalg import PCs
from grapp.grg_calculator import GRGSpMVCalculator

artifacts = ["artifacts/chr1.grg_spmv", "artifacts/chr2.grg_spmv"]
# capture=True is required for threads > 1: two artifacts on one device share one
# runtime, and the eager path rejects concurrent calls on it.
backend = make_backend_cusparse(device=0, capture=True)
# AMD: make_backend_rocsparse(device=0, capture=True)
# CPU: make_backend_mkl(n_threads=0)
req = make_runconfig_pca()                  # kernel / pca / bolt / gwas

with ExitStack() as stack:
    grgs = [
        GRGSpMVCalculator(g)
        for g in load_grg_spmv_multi(artifacts, backend, req, stack)
    ]
    pcs_df = PCs(grgs, k=10, threads=4)
```

`PCs` returns a DataFrame, and clamps `k` to the total mutation count. `include_eig=True` additionally returns the eigenvalues and eigenvectors, but it drives a width-`k` product, so it needs `make_runconfig_pca(maxk=k)` and a `k` strictly below the mutation count.

The same three steps drive grapp's other entry points; only the run configuration changes. Summary statistics need `make_runconfig_gwas()`, whose captured set includes the missingness-carrying UP product:

```python
from pygrgl_spmv import make_backend_cusparse, make_runconfig_gwas, load_grg_spmv_single
from grapp.util import allele_counts, allele_frequencies
from grapp.grg_calculator import GRGSpMVCalculator

backend = make_backend_cusparse(device=0, capture=True)

with ExitStack() as stack:
    grg = GRGSpMVCalculator(
        load_grg_spmv_single("artifacts/chr1.grg_spmv", backend, make_runconfig_gwas(), stack)
    )
    freqs = allele_frequencies(grg, adjust_missing=True)
    counts, missing = allele_counts(grg, return_missing=True)
```

Single-variant association uses the same configuration; the phenotype has one entry per individual:

```python
import numpy as np
from grapp.assoc import linear_assoc_no_covar

with ExitStack() as stack:
    grg = GRGSpMVCalculator(
        load_grg_spmv_single("artifacts/chr1.grg_spmv", backend, make_runconfig_gwas(), stack)
    )
    Y = np.asarray(phenotype, dtype=np.float64)     # shape (grg.num_individuals,)
    results = linear_assoc_no_covar(grg, Y)
```

For covariate GWAS set `make_runconfig_gwas(maxk=n_covariates + 1)` so the `X^T Q` product fits. BOLT-LMM uses `make_runconfig_bolt()`.

### Supported Backends and Parameters

The adaptor exposes MKL on CPU and cuSPARSE/rocSPARSE on GPUs. Both GPU
backends support eager/prepared products, capture, native CuPy I/O, streams
and multiple artifacts on one device. ROCm supports resident single-device
execution; streaming, multi-GPU, profiling and scratch execution are unsupported.

`make_backend_mkl(n_threads=0, optimize=False)`

| Parameter | Default | Meaning |
|---|---|---|
| `n_threads` | `0` | Threads per file. `0` auto-detects `physical_cores // n_files`, minimum 1. Core topology comes from the kernel, counting only CPUs in the process's affinity mask and capped by any cgroup CPU quota, so `taskset`, a cpuset, and a `docker --cpus` / Kubernetes CPU limit are all respected. Also accepts a per-file dict. |
| `optimize` | `False` | Run MKL's inspector-executor `mkl_sparse_optimize()` on each matrix at load. Costs load time, can speed up repeated matmuls. |

`make_backend_cusparse(device=0, allow_residency=True, vram_budget_mb=0, capture=False, native=False)`

`make_backend_rocsparse(device=0, capture=False, native=False)`

| Parameter | Default | Meaning |
|---|---|---|
| `device` | `0` | Non-negative integer ordinal (NumPy integers accepted, booleans rejected). CUDA also accepts a per-file dict with the same ordinal contract and loads device groups in parallel. Files on one device share a layout and runtime. |
| `allow_residency` | `True` | Keep every sparse block resident in VRAM. `False` selects CUDA streaming and is rejected on ROCm. |
| `vram_budget_mb` | `0` | CUDA only: VRAM cap in MiB. Used only in streaming mode, where it must be `> 0`; ignored when resident. |
| `capture` | `False` | Capture vendor graphs after loading, so matmul replays a graph instead of re-issuing kernels. |
| `native` | `False` | Keep matmul I/O on device (CuPy in, CuPy out, no host copies). Requires `capture=True` — on its own it is silently ignored. In this mode `miss` is an in-place accumulator on the device: pass the array you will read afterwards, never a fresh `cupy.asarray(host_array)`, or the counts are written to a temporary and lost. |

MKL `n_threads` and CUDA `device` also accept mappings keyed by artifact file stem:

```json
{"chr1": {"cuda_device": 0},        "chr2": {"cuda_device": 1}}
{"chr1": {"mkl_threads": [2, 1]},   "chr2": {"mkl_threads": [4, 1]}}
```

`mkl_threads` is `[n_up, n_down]` and is passed through as written. Note that a `0` here does *not* mean the same thing as the scalar `n_threads=0`: it skips the per-file division and falls through to MKL's own default of `os.cpu_count()`, so every file would get the whole machine.

### Supported Applications and RunConfigs

Run configurations carry the `RuntimeRequirements` for an application, which decides which buffers the runtime allocates and which graphs get captured:

| Factory | Application | Parameters beyond `force_spmm` |
|---|---|---|
| `make_runconfig_kernel` | matmul microbenchmarks | `direction` (`"up"`/`"down"`) and `k`, both required |
| `make_runconfig_pca` | PCA | `maxk` |
| `make_runconfig_bolt` | BOLT-LMM-inf | — |
| `make_runconfig_gwas` | GWAS | `maxk`, `sample_variance` |

`force_spmm=False` captures at `k=1` (the SpMV path); `True` captures at `k=2` (SpMM). Either way `k=1` callers still work — `CapturedBoundGRG.matmul()` zero-pads and truncates.
**When you're forced to use an earlier CUDA version, setting `force_spmm=True` can ensure correctness, at the cost of significant performance degradation.** Note that `force_spmm` only has an effect when `maxk == 1`: at `maxk >= 2` the SpMM path is already in use, so the capture widths are identical either way.

For GWAS with covariates set `maxk = n_covariates + 1` so the `X^T Q` product fits, and set `sample_variance=False` for the binomial-variance-only workload, which drops the `diag(X^T X)` graph.

Each factory captures a fixed set of graphs, and **a call whose shape was not captured raises `ValueError`** naming both the requested and the available keys. It does not fall back to a different graph. If you hit that error, either use the run configuration that matches your application or add the missing `CaptureSpec` (together with the matching `need_*` flag on its `RuntimeRequirements`). Two consequences worth knowing:

- `make_runconfig_pca()` captures at `k=1`, so any solver whose block size *is* `k` needs an explicit `make_runconfig_pca(maxk=k)`.
- `make_runconfig_pca()` declares neither `need_init_vector` nor `need_init_xtx`, because no PCA path passes an array init and `grapp.util.simple.variance()` is a `custom_variance` you supply yourself. If you do supply one, add `CaptureSpec("up", k, init_mode="xtx")` and `need_init_xtx=True` to the PCA configuration — not `make_runconfig_gwas()`, which captures no DOWN graph and so cannot serve an eigensolver's reverse product.

## Advanced Usage

More freedom is provided when using the lower-level APIs directly. 
User may want to use them when fine-grained control or optimization is needed, or to develop new methods or applications.
An example and notes have been provided below.

### Core workflow

```python
import numpy as np

from pygrgl_spmv import RuntimeRequirements, simple_convert
from pygrgl_spmv.backends.cusparse import CusparsePlanPair, CusparseRuntime, plan_cusparse_layout

artifact = simple_convert("A.grg", "artifacts/A.grg_spmv")
req = RuntimeRequirements(
    max_k_up=8,
    max_k_down=8,
    need_down_miss_input=True,
    need_up_miss_output=False,
    need_init_vector=True,
    need_init_matrix=False,
    need_init_xtx=True,
)
pair = CusparsePlanPair.from_dicts(
    {"store": "N", "fmt": "CSR", "opA": "N", "opB": "N", "orderB": "ROW", "orderC": "ROW", "algo": "DEFAULT", "scratch": "none"},
    {"store": "T", "fmt": "CSC", "opA": "N", "opB": "N", "orderB": "ROW", "orderC": "ROW", "algo": "DEFAULT", "scratch": "none"},
)
layout = plan_cusparse_layout(
    artifacts=[artifact],
    pair=pair,
    dtype=np.float64,
    requirements=req,
    vram_budget_bytes=8_000_000_000,
    ring_buffer_size=4,
    device=0,
    stream=0,
)

with CusparseRuntime(layout) as runtime:
    (A,) = runtime.grgs
    with A.prepare_matmul_cuda(direction="up", k=1) as op:
        op.input.copy_(op.input.new_tensor(np.ones((1, A.num_samples), dtype=np.float64)))
        op()
        y = op.output.cpu().numpy().copy()
```

### Public surface

Everything in `pygrgl_spmv.__all__`, grouped by role:

- **Conversion** — `simple_convert(input_path, output_path, *, dtype=float64) -> Path`
- **Adaptor (the front door)** — `make_backend_mkl`, `make_backend_cusparse`, `make_backend_rocsparse`, `make_runconfig_kernel`, `make_runconfig_pca`, `make_runconfig_bolt`, `make_runconfig_gwas`, `load_grg_spmv_single`, `load_grg_spmv_multi`
- **Adaptor value objects** — `MklBackendConfig`, `CusparseBackendConfig`, `RocsparseBackendConfig`, `RunConfigs`, `CaptureSpec`, `CapturedBoundGRG`
- **Lower-level planning/execution** — `RuntimeRequirements`, `plan_reference_layout` + `ReferenceRuntime` + `ReferencePlan` + `ReferencePlanPair`, `plan_mkl_layout` + `MklRuntime` + `MklPlan` + `MklPlanPair`

Not re-exported from the root, on purpose:

- GPU planners and runtimes live in `pygrgl_spmv.backends.cusparse` and `pygrgl_spmv.backends.rocsparse`, so that `import pygrgl_spmv` stays CPU-safe and works with CuPy and torch absent.
- Test helpers live in `pygrgl_spmv.testing` (see [Testing Against pygrgl-spmv](#testing-against-pygrgl-spmv)).
- `BoundGRG` is the class every loaded GRG actually is, but it is only ever obtained from a runtime or a loader, never constructed directly.

GPU execution uses `prepare_matmul_cuda(...)` or `prepare_matmul_rocm(...)`; eager `matmul(...)` wraps the same prepared operation with NumPy I/O.

| Backend | Planner / runtime | Prepared buffers | Copy input / return output |
|---|---|---|---|
| cuSPARSE | `plan_cusparse_layout` / `CusparseRuntime` | Torch | `op.input.copy_(...)` / `op.output.cpu().numpy()` |
| rocSPARSE | `plan_rocsparse_layout` / `RocsparseRuntime` | CuPy | `cp.copyto(op.input, cp.asarray(x))` / `cp.asnumpy(op.output)` |

ROCm plans use the same store/format/op/dense-order fields as CUDA. Their optional
`spmv` algorithm applies only at width one; `spmv=None` always uses SpMM `algo`.
The default pair uses CSR row-split SpMM / default SpMV upward and shared CSC
SpMM / transpose NNZ-split SpMV downward. Unsupported vendor combinations raise;
ROCm 10.1's incorrect normal-CSR NNZ-split SpMV and widths 1/2 NNZ-split SpMM
(including DEFAULT/CSR) are rejected. Empty blocks are no-ops.

The parallel benchmark entry points `python -m scripts.bench.cusparse` and
`python -m scripts.bench.rocsparse` accept `--artifact`, `--direction`, `--k`,
`--warmup`, `--trials` and an optional JSON `--plan`.

## Testing Against pygrgl-spmv

If your own test suite needs to run against our backends, `pygrgl_spmv.testing` ships helpers so you don't have to hand-roll planner and runtime setup. It is importable without `pytest`.

```python
import pytest
from pygrgl_spmv import testing as spmv_test

@pytest.fixture(scope="session")
def artifact(tmp_path_factory):
    return spmv_test.artifact_for("chr1.grg", tmp_path_factory.mktemp("artifacts"))

@pytest.fixture(params=spmv_test.available_backends(), ids=lambda b: b)
def grg(request, artifact):
    with spmv_test.load(artifact, backend=request.param, max_k=4) as g:
        yield g          # the fixture's teardown owns the runtime's lifetime
```

`unittest` works the same way via `TestCase.enterContext`:

```python
grg = self.enterContext(spmv_test.load(artifact, backend=backend, max_k=4))
```

| Function | Purpose |
|---|---|
| `available_backends()` | Backends usable in this process, a subset of `reference`, `mkl`, `cusparse`, `rocsparse`. Parameterize over this instead of writing per-backend `skipif`s. |
| `is_mkl_available()`, `is_cusparse_available()`, `is_rocsparse_available()`, `is_available(backend)` | Individual predicates. Cached, and deliberately fork-safe — they do not initialise a GPU driver in the parent, so `multiprocessing` in a consumer's suite keeps working. Set `PYGRGL_SPMV_DISABLE_GPU=1` to force CPU-only, at any point. |
| `artifact_for(grg_path, out_dir, *, dtype)` | Convert a `.grg` for use in tests. A separate call from `load()` on purpose, so conversion stays visible; it does not cache, so call it from a session-scoped fixture. Names the artifact from the input's basename, so distinct sources need distinct `out_dir`s. |
| `requirements(*, max_k=..., **fields)` | An everything-enabled `RuntimeRequirements`. Keyword names mirror the dataclass field for field. |
| `load(artifact, *, backend, req, max_k, dtype, **kw)` | Context manager yielding one bound GRG. `load_reference` / `load_mkl` / `load_cusparse` / `load_rocsparse` are pinned shorthands. |
| `load_many(artifacts, *, backend, ...)` | Context manager yielding one bound GRG per artifact, in order. |

Backend-specific arguments go through `**kw` to the matching `make_backend_*` factory — `capture`, `native`, `device`, `allow_residency`, `vram_budget_mb` for cuSPARSE; `capture`, `native`, `device`, `pair`, `shared_ones`, `allow_residency=True` for rocSPARSE; `n_threads` and `optimize` for MKL, `plans` for reference. Passing one to a backend that cannot honour it raises `TypeError` rather than being ignored.

`capture=True` with only `max_k` captures just the default up/down pair. For a real application shape pass `req=make_runconfig_pca()` (or `_bolt` / `_gwas` / `_kernel`); otherwise a call like `by_individual=True` has no graph and raises.

There is no `exit_stack` parameter: the loaders are context managers, and your test framework already owns the stack. Do not use a GRG after its `with` block — for captured GRGs the underlying buffers are freed on exit, and calling `matmul` afterwards raises.

`reference` is a pure NumPy/SciPy backend, always available, and is the parity oracle the others are checked against — useful as the expected value in your own tests.

### Notes

- planners and runtimes consume `.grg_spmv` artifacts only
- runtime-owned buffers are allocated in `__enter__()`
- one runtime owns one shared execution arena across all `runtime.grgs`
- concurrent calls on one runtime fail fast
- both GPU backends support any declared `max_k >= 1`
- CUDA layouts can mix resident and streamed sparse blocks under a VRAM budget
- `ring_buffer_size=0` is valid for CUDA layouts only when the budget keeps every sparse block resident
- the package root intentionally stays CPU-safe and does not re-export GPU runtime symbols
- in `native=True` mode, `matmul` awaits only CuPy's *current* stream on the capture device, which is where a plain `cupy.asarray(...)` enqueues. Synchronize inputs yourself if you produce them on an explicit `cupy.cuda.Stream()`, or hand over a CuPy view of torch memory
