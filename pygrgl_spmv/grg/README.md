Parent docs: [Project README](../../README.md)

# GRG Artifacts

`pygrgl_spmv/grg/` contains:

- `simple_convert(input_path, output_path, *, dtype=float64) -> Path` — the only way to produce an artifact
- the `.grg_spmv` save/load/scan helpers in [artifact.py](artifact.py)
- the compile pipeline in [compile.py](compile.py)
- the internal `BoundGRG` host-side API logic in [__init__.py](__init__.py)

`.grg_spmv` artifacts are uncompressed (a `np.savez` zip with one member per array).
Header metadata alone is sufficient for **GPU** layout planning: `plan_cusparse_layout`
reads only the small `scan_*` side tables via `scan_grg_spmv`, never a sparse block or
selector array. The reference and MKL planners do not achieve this -- both call
`_load_grg_spmv_host` to size their selectors, which reads roughly half the archive.
