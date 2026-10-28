# Contributing

These principles also guide changes to the application wrappers, experiments
and documentation that use GRG-SpMV.

## Keep the maintained surface small

- Read the existing implementation and follow the repository's conventions.
  Prefer a direct solution over extra abstractions, forwarding layers or knobs.
- Commit enduring code, focused tests, reproducible commands and useful results.
  Keep working plans, investigation scripts, delivery diaries and intermediate
  outputs outside the repositories. Bulky outputs need a reproduction or
  retrieval path, not a place in Git.
- Keep comments and documentation concise: explain a contract, a non-obvious
  decision or a limitation, and point to the relevant source.
- Maintain each fact at its natural owner; link to it rather than copying
  instructions or policy. Remove unused consumers when removing a feature.

## Preserve backend contracts

- Implement the same supported behavior across CPU, CUDA and ROCm. Audit API
  names, defaults, validation, error handling, documentation and experiments
  together; explain genuine vendor limitations explicitly.
- Share semantics and small adapters where useful. Keep vendor ABI, buffer,
  stream and graph ownership separate where their lifetimes or behavior differ.
  Duplication is preferable to an abstraction that hides those differences.
- Reject invalid configurations clearly. Do not silently change algorithms,
  fall back to another backend or continue after an impossible state.
- Prefer general fixes over dimension-specific guards. Vendor exclusions need
  a reproducer, a documented affected range and a focused regression test.
- Make resource and device ownership explicit, including partial initialization,
  asynchronous work and teardown. Optional GPU imports and availability checks
  must remain safe for CPU-only use and subsequent process creation.

## Verify correctness before performance

- Parameterize existing tests for shared features. Preserve their fixtures,
  CPU oracles, assertions and numerical tolerances. Keep only necessary vendor
  regressions separate; exclude unsupported cases rather than whole test files.
- Treat unavailable backends consistently: optional selection skips; explicit
  selection fails. Use paired backend markers such as `cusparse` and `rocsparse`.
- Reuse established application-specific numerical checks. Do not invent a
  stricter acceptance gate for one backend, relax assertions to pass, or change
  solver settings while presenting an optimization as equivalent.
- Run checks appropriate to the change. Distinguish passing, failing,
  unsupported and unexecuted coverage; compilation is not hardware validation.

## Make measurements and changes auditable

- Use isolated, reproducible environments and configurable dependency discovery.
  Avoid host-specific paths and implicit system-library dependencies.
- Compare matching inputs, precision, solver settings and timing boundaries.
  Record seeds, versions, hardware, device counts, warmups and repetitions.
  Distinguish repeated kernel calls from independent process runs.
- Derive plots from recorded measurements. Preserve input provenance and disclose
  configuration differences; keep duplicated deliverable figures byte-identical.
- Review the final reader experience as well as source and tests: labels, units,
  captions and hardware attribution must describe the selected observations.
  Prior measurements remain valid evidence; state verification limits precisely.
- Make a small number of purposeful commits. Fold review fixes into the relevant
  commit when history rewriting is authorized, resolve dependent conflicts and
  recheck affected behavior. Preserve unrelated work and upstream history; back
  up rewritten tips and use an explicit branch and lease when force-pushing.
- Audit the integrated result and each changed hunk against its purpose. Repair
  violations in their owning changes and stop once the required checks pass.
