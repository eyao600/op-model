# Autoresearch Agent

## Purpose

Run autonomous, iterative latency-model experiments for this repository. The
goal is to reduce full held-out GEMM latency MAPE from the validation harness
while preserving a general, analytical GPU performance model.

## Required Setup Before Experiments

Before starting an experiment loop:

1. Ensure the active branch is `autoresearch`.
2. Verify the repo has the GEMM latency validation harness:
   `src/opmodel/validation/gemm_latency.py`.
3. Verify the core model file exists:
   `src/opmodel/models/extended_roofline.py`.
4. Verify local GEMM data is available under `data/`.
5. Verify local Git identity is configured.
6. Run one baseline validation only after user confirmation:
   `env PYTHONPATH=src python3 -m opmodel.cli validate-gemm-latency --data-dir data`.
7. Create or append `results.tsv` with one row per validation pass.

Do not start the first validation or experiment loop until the user confirms
that setup is complete.

## Allowed Files

During experiments, only this file may be modified:

- `src/opmodel/models/extended_roofline.py`

The agent may create or append:

- `results.tsv`

No other repo file may be modified during the experiment loop.

## Validation Command

Each experiment must result in exactly one full validation pass:

```bash
env PYTHONPATH=src python3 -m opmodel.cli validate-gemm-latency --data-dir data
```

Use the reported full held-out `all/all` MAPE as the primary objective. Also
inspect per-class metrics and prioritize the currently worst class, especially
`small` and `vector_like` when they dominate error. Do not significantly
sacrifice any kernel class metric merely to improve aggregate MAPE; reject
changes that create a large per-class MAPE regression unless the regression is
small, well-understood, and outweighed by broad improvements across other
classes.

## Results Log

Append one tab-separated row to `results.tsv` for every validation pass.

Recommended columns:

```text
timestamp	commit	experiment	status	all_mape	a10_mape	a100_mape	worst_class	worst_class_mape	summary
```

If a candidate is worse than the previous kept row, record the candidate result
and discard the code change. If a candidate is kept, commit it before moving to
the next experiment.

## Git As Research Memory

Use Git to track research state:

- Start from branch `autoresearch`.
- One kept experiment equals one commit.
- Commit messages must summarize:
  - the analytical modeling change,
  - the GPU/CUTLASS concept behind it,
  - the validation result and MAPE change.
- If validation worsens, discard the candidate with Git and continue from the
  most recent kept commit.
- Rewinding is allowed sparingly if the current line of work is stuck.

## Modeling Constraints

Changes must remain analytical and hardware-grounded:

- Do not add empirical fitting knobs beyond the existing fixed-overhead
  calibration performed by the validation harness.
- Do not tune constants to the dataset.
- Do not add shape-specific lookup tables or per-kernel special cases.
- Do not introduce heuristic shape-regime switches such as `M < 64`,
  `min(M, N) <= threshold`, aspect-ratio buckets, or validation-class labels as
  modeling behavior unless there is clear documentation or explicit kernel
  metadata showing that real GPU kernels dispatch or execute differently on
  that boundary. Shape classifications may be used for reporting only, not as
  hidden calibration logic.
- Do not modify hardware YAMLs or validation code.
- Derive outputs from problem dimensions, kernel attrs, and `HardwareSpec`.
- Ground changes in GPU microarchitecture and NVIDIA CUTLASS GEMM behavior.
- Reject changes that are unrealistic, unfounded, or only improve metrics by
  overfitting.
- Reject changes that make the model less hardware-agnostic by encoding
  benchmark-specific or dataset-discovered shape regimes.

## Runtime Constraints

Keep the model a simple calculator:

- Avoid cycle-level simulation.
- Avoid major overhauls.
- Reject changes that make validation more than 5x slower.
- Prefer simple first-principles corrections.
- Removing false or unnecessary modeling is encouraged.

## Experiment Loop

For each iteration:

1. Inspect the latest validation metrics and identify the highest-error class.
2. Form one analytical hypothesis about a likely modeling error.
3. Modify only `src/opmodel/models/extended_roofline.py`.
4. Spend at most 10 minutes tuning code before validation.
5. Run exactly one full validation pass.
6. Append the result to `results.tsv`.
7. Keep and commit only if the full held-out MAPE improves, no kernel class
   metric regresses significantly, and the change is analytically defensible.
8. If kept, continue improving from the new commit.
9. If rejected, discard the change and try a different hypothesis from the
   most recent kept commit.

Do not wait for user input between iterations after the user confirms the
experiment loop should start.
