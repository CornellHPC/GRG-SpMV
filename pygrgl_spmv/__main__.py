"""Command line entry point: ``python -m pygrgl_spmv <command>``.

Conversion is a mandatory, explicit step before any GRG-SpMV runtime can be
used, so it gets a command line of its own. Running it separately -- and seeing
how long it takes -- is the point: conversion time is not GRG-SpMV runtime.
"""

from __future__ import annotations

import argparse
import sys
import time


def _cmd_convert(args: argparse.Namespace) -> int:
    import numpy as np

    from pygrgl_spmv import simple_convert

    dtype = np.float32 if args.dtype == "float32" else np.float64
    t0 = time.perf_counter()
    out = simple_convert(args.input, args.output, dtype=dtype)
    elapsed = time.perf_counter() - t0
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB, {elapsed:.2f}s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pygrgl_spmv",
        description="pygrgl-spmv command line utilities.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser(
        "convert",
        help="compile a .grg file into a .grg_spmv artifact",
        description=(
            "Compile a .grg file into a .grg_spmv artifact at an exact path. "
            "OUTPUT names the artifact file, not a directory."
        ),
    )
    convert.add_argument("input", help="path to the input .grg file")
    convert.add_argument("output", help="path to the output .grg_spmv file")
    convert.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float64",
        help="precision of the precomputed init-bias arrays (default: float64)",
    )
    convert.set_defaults(func=_cmd_convert)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    # Anything narrower let the write path escape as a traceback: PermissionError,
    # FileExistsError on a non-directory parent, ENOSPC mid-savez.
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
