Parent docs: [Project README](../../README.md)

# GRG Artifacts

`pygrgl_spmv/grg/` contains:

- `simple_convert(input_path, output_path, *, dtype=float64) -> Path` — the only way to produce an artifact
- the `.grg_spmv` save/load/scan helpers in [artifact.py](artifact.py)
- the compile pipeline in [compile.py](compile.py)
- the internal `BoundGRG` host-side API logic in [__init__.py](__init__.py)

`.grg_spmv` artifacts are uncompressed. Their header metadata is sufficient for layout planning without loading sparse block or selector arrays.
