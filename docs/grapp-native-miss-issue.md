# Issue for `grapp`: native-mode missingness output is silently discarded

**Where the bug is:** `grapp`, not `pygrgl-spmv`. Filed from the GRG-SpMV side because that
is where it was found and measured.

**Severity:** silent wrong numbers, no error, in the default GPU configuration for PCA.

**Affects:** any `GRGSpMVCalculator` loaded with `native=True` (which requires
`capture=True`), on a dataset that has missing genotypes.

**Which grapp:** the in-tree `mikado/grapp` copy. The released 0.4 on PyPI has no
`use_cupy`/`device` on `GRGSpMVCalculator` at all, so the branch quoted below does not
exist there and that build cannot hit this. Line numbers refer to the mikado copy.

---

## The defect

`GRGSpMVCalculator.matmul` (`grapp/grapp/grg_calculator.py`, the `if self.use_cupy:`
branch) converts every argument to CuPy before delegating:

```python
mm_miss = cupy.asarray(miss) if miss is not None else miss
result = self._op.matmul(mm_input, self._convert_dir(direction), ..., miss=mm_miss)
...
return result
```

`miss` is an **in-place accumulator**, not an input. `pygrgl-spmv` correctly adds the
missingness counts into whatever array it is handed — but when `miss` arrives as a host
NumPy array, `cupy.asarray(miss)` allocates a *device copy*. The counts are accumulated
into that copy, `mm_miss` goes out of scope, and the caller's `miss` array is never
touched. Every missingness count is silently lost.

`cupy.asarray` on an array that is already a same-device CuPy array is a no-op
(`cp.asarray(x) is x` → `True`), so the bug only fires for host input — which is exactly
what every `grapp` caller passes.

## Measured impact

On `test-200-samples.miss.final.grg` (400 samples, 10893 mutations, real missingness):

| path | `miss.sum()` |
|---|---|
| `pygrgl.matmul` (reference) | 27 |
| cuSPARSE, `capture=False` | 27 |
| cuSPARSE, `capture=True`, `native=False` | 27 |
| cuSPARSE, `capture=True`, `native=True`, caller-owned CuPy array | 27 |
| **via `GRGSpMVCalculator` with `native=True`** | **0** |

Downstream of that, `allele_frequencies(adjust_missing=True)` diverges by up to
**2.63e-3**, because `n_j = num_samples - miss_count` collapses to `num_samples`.

## Why this matters more for PCA than for GWAS

- `mikado/benchmark/evaluate.py` force-disables `--native` for the `gwas` application, so
  the GWAS benchmark path cannot hit it.
- The PCA scripts (`benchmark/examples/pca/grapp_cusparse.py`) forward `native=args.native`
  unconditionally, and `benchmark/evaluate.py` has it **on by default** via
  `CUSPARSE_TOGGLE_DEFAULTS`. To reproduce by hand, pass `--native` explicitly: the
  script's own flag is `store_true`, so running it directly does not.
- `grapp/grapp/linalg/__init__.py` calls `_allele_frequencies(..., adjust_missing=True)` on
  the single-GRG path, and those frequencies standardize the operator fed to the
  eigensolver.

So single-file PCA with default flags on a dataset with missingness has been standardizing
with unadjusted frequencies. The Mikado preprint's datasets have no missingness, so the
published numbers are not affected — but any user with missing genotypes is.

## The fix

Two lines in `GRGSpMVCalculator.matmul`, **inside the `if self.use_cupy:` branch**,
after its `self._op.matmul(...)` call:

```python
if miss is not None and mm_miss is not miss:
    miss[...] = cupy.asnumpy(mm_miss).astype(miss.dtype, copy=False)
```

Keep it inside that branch. `mm_miss` is bound only there, and `matmul` has a second
`self._op.matmul(...)` call in the `else` — so placed outside the `if` it raises
`NameError` on the first non-native call with missingness (every MKL or reference run),
and placed after the shared `return result` it is unreachable and fixes nothing.

The `mm_miss is not miss` guard makes this free for callers who already pass a device
array. Measured end to end through `make_runconfig_gwas` and `grapp`'s real
`allele_counts` / `allele_frequencies`: `miss.sum()` 0 → 27, `max|Δfreq|` 2.63e-3 → 0.

## Alternatives that were built and rejected

These were all prototyped and measured, so please don't re-litigate them without new
evidence:

- **Relax `pygrgl-spmv` native mode to accept a host `miss` and copy back.** Measured
  useless: it works standalone, but `grapp` converts unconditionally, so the result is
  still `0`. It needs the same `grapp` edit anyway, *plus* it reintroduces host↔device
  transfers into the one path whose documented contract is "no host transfers occur".
- **Change `matmul` to return `(result, miss)`.** Actively dangerous:
  `grapp/grapp/util/simple.py` does `grg.matmul(...)[0]` to take row 0. With a tuple
  return, `[0]` silently becomes the entire `(1, N)` result matrix — shape `(6,)` becomes
  `(1, 6)` with no exception, and it broadcasts onward through `allele_frequencies`. That
  would fix one silent-wrongness bug by introducing another.
- **Detect the temporary in `pygrgl-spmv`.** Rejected as unreliable, not as impossible.
  The two cases are in fact distinguishable by refcount — measured inside the converting
  frame, `sys.getrefcount(cupy.asarray(x))` is 2 when `x` is a host array (a fresh
  temporary) and 4 when `x` is already a same-device cupy array (`asarray` returns `x`
  itself and the caller still holds it). But that gap is an artifact of the call shape:
  it moves with frame depth, with whether the caller keeps its own reference, and with
  any future change to `asarray`'s fast path. Branching on a CPython refcount is not a
  contract we can offer callers, so the copy-back in `grapp` remains the fix.

## What changed on the `pygrgl-spmv` side

No behaviour change was needed — the accumulate was already correct. Two things were
hardened:

1. **The error message was the root cause.** It previously said only
   `expected cupy.ndarray on cuda:N (native mode), got ndarray`. The most natural way to
   satisfy that instruction is `cupy.asarray(miss)`, i.e. to write this bug. It now names
   `miss` as an in-place accumulator and explicitly forbids wrapping a host array. `grapp`
   did what our diagnostic told it to.
2. `hasattr(op, "miss_output")` in the accumulate branches was replaced with an explicit
   key check, so a missing `miss_output` raises rather than silently skipping the
   accumulate.

Regression tests now pin that all paths agree
(`pygrgl_spmv/tests/runtime/test_adaptor_surface.py`). **Note these cannot catch the bug
described here** — our side was never wrong, so the "all paths agree" test passes on
unfixed code. Only a `grapp`-side test can. Please do not treat the GRG-SpMV tests as
making this fix optional.

## Also worth fixing while in that file

`pygrgl_spmv.load(...)` was removed from `pygrgl-spmv` in `cb78b5e`. In this in-tree copy
exactly one caller is left: `grapp/test/testing_utils.py:192`. `WRAP_GRG_PARAMS.append` is
unconditional whenever `pygrgl_spmv` is importable, so **installing `pygrgl-spmv`
currently breaks `grapp`'s own test suite**.

`load_grg_calculator` is already fixed here — it registers `_raise_spmv_error` for
`.grg_spmv` (`grapp/grapp/grg_calculator.py:602-605`). Worth knowing anyway, because the
released 0.4 on PyPI still registers
`lambda f: GRGSpMVCalculator(pygrgl_spmv.load(f))` there, so on that build
`grapp pca foo.grg_spmv` and `grapp assoc foo.grg_spmv` raise `AttributeError`. Anyone
backporting to 0.4 needs both sites.

`pygrgl_spmv.testing` now ships a replacement designed for exactly this:

```python
from pygrgl_spmv import testing as spmv_test

# artifacts are a mandatory explicit step now; .grg is no longer accepted directly
artifact = spmv_test.artifact_for(grg_path, out_dir)

for backend in spmv_test.available_backends():        # ('reference', 'mkl', 'cusparse')
    grg = self.enterContext(spmv_test.load(artifact, backend=backend, max_k=4))
    ...
```

Note `test_ops.py::test_multi_ops` calls `split_and_load(..., CLEANUP)` which deletes the
split `.grg` files before wrapping them, so it will need `cleanup=False` (or to convert
before cleanup) to obtain artifacts.
