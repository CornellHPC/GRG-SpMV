Parent docs: [Project README](../../README.md)

# Tests

## Run Modes

- Default suite:
  - `python -m pytest -q pygrgl_spmv/tests --backend all`
- Backend-focused runs:
  - `python -m pytest -q pygrgl_spmv/tests --backend reference`
  - `python -m pytest -q pygrgl_spmv/tests --backend mkl`
  - `python -m pytest -q pygrgl_spmv/tests --backend cusparse`
  - `python -m pytest -q pygrgl_spmv/tests --backend rocsparse`
- Long streamed-GPU suite:
  - `python -m pytest -q pygrgl_spmv/tests --backend all --stress`

Unavailable backends skip under the default selection; explicitly requesting an unavailable backend fails. Activate the matching environment from the [installation instructions](../../README.md#install). Both GPUs use the same simulated fixtures, numerical assertions and tolerances. ROCm supports resident single-device execution, ordinary streams/events, multiple GRGs and capture; streaming, multi-device, profiling and scratch execution are unsupported. CUDA execution requires an NVIDIA host.

Default backend test files already include small-artifact streamed GPU mirrors for planner transitions, exactness, and overlap contracts.
`--stress` is the long run. It keeps the large streamed cuSPARSE band cases and can take around 30 minutes.
Streamed GPU tests skip only for insufficient host RAM while constructing the synthetic large artifacts; VRAM-budget failures are treated as correctness/accounting bugs.

## Shared Contracts

The suite protects seven things:

- compile and artifact correctness
- `BoundGRG` host-side API semantics
- planner byte/layout correctness
- entered-runtime ownership and lifecycle invariants
- backend numerical parity and end-to-end explicit-matrix equivalence
- adaptor run-configuration and capture-key correctness
- the shipped `pygrgl_spmv.testing` public contract

## File Map

- `test_compile_selectors.py`
  - Compile-time selector and stable-height block construction edge cases.
- `test_sparse_utils.py`
  - Canonical sparse helper invariants for structural dtype finalization and binary CSR rehydration.
- `test_import_surface.py`
  - CPU-safe package import surface under blocked optional GPU modules.
- `test_bench_scripts.py`
  - Minimal benchmark runner loop counts and reporting contract.
- `runtime/test_adaptor_surface.py`
  - Adaptor surface contracts: backend/run-config factory validation (including uniform `maxk` checking), `_physical_cores` reading kernel topology rather than logical CPUs and capping by CPU affinity, per-file `mkl_threads`/`cuda_device` dict resolution, `load_grg_spmv_*` argument validation, every accepted `direction` form on the captured path, and missingness agreement across reference/MKL/cuSPARSE/rocSPARSE plus captured copy and native modes.
- `runtime/test_adaptor_capture.py`
  - Capture-key correctness for the adaptor's CUDA/HIP graph paths: CPU-only snapshots of every `make_runconfig_*` capture set and `need_*` flag (self-consistency, uniqueness, no unused flag), plus GPU parity of every captured key against the eager path, loud failure on an uncaptured key, exact-k rejection of short init/miss, pad/truncate for short input, `emit_all_nodes`, and multi-artifact load ordering.
- `runtime/test_testing_helpers.py`
  - Contract for the shipped `pygrgl_spmv.testing` helpers: no `test_`-prefixed public name (pytest would collect it in the consumer's suite), importable with pytest blocked, fork-safe availability probes that leave the parent GPU driver uninitialised, context-manager lifetimes with no caller-supplied `ExitStack`, the post-release replay guard, and the strict per-backend keyword allowlist.
- `runtime/test_convert.py`
  - `simple_convert()` explicit-in/explicit-out contract: exact destination, suffix appending, parent creation, atomic overwrite, resolved return, rejected inputs/dtypes, validate-before-work (including both directory spellings and an unloadable GRG), the `python -m pygrgl_spmv convert` exit codes, byte reproducibility, and down-edge-only compilation.
- `runtime/test_artifacts.py`
  - `.grg_spmv` scan/load/block iteration correctness and direct artifact consumption by runtimes.
- `runtime/test_api.py`
  - Entered-runtime lifecycle, `runtime.grgs`, concurrency guard, fixed owned-buffer reuse, and one-runtime multi-GRG usage.
- `runtime/test_matmul_semantics.py`
  - Shared `BoundGRG.matmul()` semantics across the CPU and GPU runtimes: `by_individual`, init, miss, dtype, and fast-fail contract checks.
- `runtime/test_prepare_gpu.py`
  - Direct `prepare_matmul_cuda()`/`prepare_matmul_rocm()` contract checks: CPU rejection, GPU buffer semantics, and prepared-path correctness for init, missingness, and node emission.
- `runtime/test_emit_all_nodes.py`
  - `emit_all_nodes=True` semantics, including init modes, dtype variants, stability, and miss rejection.
- `runtime/test_traversal_correctness.py`
  - Shared traversal correctness, stable node ordering, exact binary cases, dtype coverage, and zero/stability behavior.
- `runtime/test_reference.py`
  - Reference runtime parity and exact byte accounting for resident sparse blocks, selectors, and workspaces.
- `runtime/test_mkl_lifecycle.py`
  - `MklRuntime.__enter__` failure-path ownership under a stubbed MKL FFI: a failure part way through one artifact's block loop must still destroy every handle built so far, before the shared values buffer is munmapped. Carries no `mkl` marker on purpose, so it runs on hosts without `libmkl_rt.so`.
- `runtime/test_mkl.py`
  - MKL runtime parity, format/thread-local behavior, separate-runtime threaded execution, LP64 ABI checks, `float32` dispatch, shared-value budgeting, and optimize-flag semantics.
- `runtime/test_plans.py`
  - CUDA device/stream parsing plus backend plan/pair validation and storage-sharing rules.
- `runtime/test_directional_layouts.py`
  - One-sided planner/runtime behavior across backends, including disabled-direction failures, reduced owned bytes, and runtime `k <= max_k` coverage.
- `runtime/test_gpu.py`
  - Shared prepared widths/refcounts, ownership, streams, dense views, resident exactness and multi-GRG capture.
- `runtime/test_cusparse.py`
  - CUDA ABI, planner budgets, shared-ones modes, streaming, overlap, slot compaction and scratch contracts.
- `runtime/test_rocsparse.py`
  - Explicit rocSPARSE algorithm combinations and width-one dispatch.
- `runtime/test_rocsparse_vendor_regressions.py`
  - Pip SDK discovery, vendor numerical exclusions and failed-capture graph ownership.
- `runtime/test_rocsparse_vmm.py`
  - HIP virtual shared values and partial-initialization cleanup.
- `runtime/_runtime_builders.py`
  - Shared layout-builder helpers used by the runtime-era suite.
- `runtime/_streaming_cases.py`
  - Synthetic streamed-artifact builders, three-block streamed mode tables, and analytic expected-value helpers for GPU streamed tests.
- `endtoend/conftest.py`
  - Explicit genotype-matrix and missingness helper functions used by end-to-end tests.
- `endtoend/test_matmul.py`
  - End-to-end equivalence against `pygrgl.dot_product`, explicit genotype matrices, diploid semantics, init modes, and GRG splitting.
- `endtoend/test_missing.py`
  - End-to-end missingness counts, mean-imputation semantics, and shared-site missingness equality.
- `endtoend/test_readme_examples.py`
  - The project README's grapp examples, executed as printed: that the documented blocks run at all, which reading them twice failed to establish.

## Maintenance Rule

If a new test file is added, add one line here describing the correctness contract it protects.
