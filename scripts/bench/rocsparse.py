"""Benchmark resident rocSPARSE plans with the shared workload and timer."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping

from pygrgl_spmv.backends.rocsparse import (
    RocsparsePlanPair,
    RocsparseRuntime,
    plan_rocsparse_layout,
)

from .cli import add_common_args, parse_args
from .run import baseline_requirements, benchmark_runtime


def parse_rocsparse_plan(value: str) -> RocsparsePlanPair:
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not isinstance(raw, Mapping):
        raise argparse.ArgumentTypeError("rocSPARSE JSON plan must be an object with plan_up and plan_down")
    allowed = {"plan_up", "plan_down"}
    extra = sorted(set(raw) - allowed)
    if extra:
        raise argparse.ArgumentTypeError(f"unknown rocSPARSE JSON plan field(s): {extra}")
    plan_up = raw.get("plan_up")
    plan_down = raw.get("plan_down")
    if plan_up is not None and not isinstance(plan_up, Mapping):
        raise argparse.ArgumentTypeError("rocSPARSE JSON plan_up must be an object or null")
    if plan_down is not None and not isinstance(plan_down, Mapping):
        raise argparse.ArgumentTypeError("rocSPARSE JSON plan_down must be an object or null")
    try:
        return RocsparsePlanPair.from_dicts(plan_up, plan_down)
    except (KeyError, TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RocsparseRuntime")
    add_common_args(parser, gpu=True)
    parser.add_argument(
        "--plan",
        type=parse_rocsparse_plan,
        metavar="PLAN",
        help="JSON object with plan_up/plan_down; defaults to the library pair",
    )
    raw_args = parser.parse_args()
    args = parse_args(raw_args, gpu=True)
    import cupy as cp

    with cp.cuda.Device(args.device):
        stream = None if args.stream == 0 else cp.cuda.ExternalStream(args.stream)
    layout = plan_rocsparse_layout(
        artifacts=[args.artifact],
        pair=raw_args.plan,
        dtype=args.dtype,
        requirements=baseline_requirements(direction=args.direction, k=args.k),
        vram_budget_bytes=args.vram_budget_bytes,
        ring_buffer_size=args.ring_buffer_size,
        allow_residency=args.allow_residency,
        device=args.device,
        stream=stream,
    )
    benchmark_runtime(
        runtime_cls=RocsparseRuntime,
        layout=layout,
        direction=args.direction,
        k=args.k,
        dtype=args.dtype,
        warmup=args.warmup,
        trials=args.trials,
    )


if __name__ == "__main__":
    main()
