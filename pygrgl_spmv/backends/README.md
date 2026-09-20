Parent docs: [Project README](../../README.md)

# Backends

Backends are concrete planner/runtime pairs.

The package root and `pygrgl_spmv.backends` re-export only the CPU-safe reference and MKL APIs.
The cuSPARSE GPU runtime is imported directly from `pygrgl_spmv.backends.cusparse`.

- `reference.py`: `ReferencePlan`, `ReferencePlanPair`, `ReferenceLayout`, `ReferenceRuntime`, `plan_reference_layout`
- `mkl/`: `MklPlan`, `MklPlanPair`, `MklLayout`, `MklRuntime`, `plan_mkl_layout`
- `cusparse/`: `CusparsePlan`, `CusparsePlanPair`, `CusparseLayout`, `CusparseRuntime`, `plan_cusparse_layout`
