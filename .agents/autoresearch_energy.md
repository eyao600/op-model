# Energy Autoresearch Agent

## Purpose

Run autonomous, iterative fitted-energy experiments for this repository. The
goal is to reduce held-out GEMM energy MAPE while keeping timing behavior
unchanged. FLOP and byte-event energy coefficients must remain fixed from the
hardware configuration; fitted power coefficients are experiment artifacts only.

## Required Setup Before Experiments

Before starting an experiment loop:

1. Ensure the active branch is `energy-autoresearch`.
2. Verify the repo has the GEMM validation harness:
   `src/opmodel/validation/gemm_latency.py`.
3. Verify local GEMM data is available under `data/`.
4. Verify local Git identity is configured.
5. Ensure `trial_results/` and `results_energy.tsv` are excluded locally by
   `.git/info/exclude`, not by repo-tracked ignore files.
6. Run one baseline validation only after user confirmation:
   `env PYTHONPATH=src python3 -m opmodel.cli validate-gemm-latency --data-dir data`.
7. Create or append `results_energy.tsv` with one row per experiment.

Do not start the first validation or experiment loop until the user confirms
that setup is complete.

## Allowed Files

During experiments, code and model changes are limited to:

- `src/opmodel/energy.py`
- `src/opmodel/calibration.py`
- `src/opmodel/hardware.py`, only when a proposed model needs additional named
  nonnegative power coefficients in `EnergyModelPowerCoefficients`
- `src/opmodel/models/extended_roofline.py`, only for memory-traffic accounting
  formulas that are grounded in exposed kernel/hardware metadata or documented
  GPU/CUTLASS behavior, and only when timing behavior is unchanged

The agent may create or append untracked local research artifacts:

- `trial_results/<experiment_tag>.json`
- `results_energy.tsv`

Only kept model-code changes are committed. Rejected experiments are represented
only by their `trial_results/<experiment_tag>.json` artifact and
`results_energy.tsv` row.

## Forbidden Changes

Do not modify:

- hardware YAMLs to store fitted coefficients or calibration metadata
- `src/opmodel/validation/gemm_latency.py`
- `src/opmodel/models/extended_roofline.py`, except for the allowed
  evidence-grounded memory-traffic accounting formulas above
- FLOP energy coefficients in `compute.*_energy_j_per_flop`
- byte-event coefficients in `memory.levels[*].energy_j_per_byte`
- timing-model behavior or latency-calibration behavior

## Calibration And Validation Protocol

Use the existing GEMM validation harness for predictions. Do not modify the
harness to support calibration.

Build the calibration set per hardware from GEMM rows satisfying:

- latency APE `< 20%`
- positive measured energy
- positive measured latency
- stratified coverage across `large`, `regular`, `skinny`, `small`,
  `small_k`, and `vector_like`

Use a small calibration set with a fixed cap per hardware/class. Select rows by
log measured energy or log FLOPs to cover the range without making calibration
the benchmark.

Fit only residual or static power terms with nonnegative regression:

- fixed event energy remains
  `compute_j + hbm_j + l2_j + sram_j + register_j` from hardware config
  coefficients
- residual target is
  `measured_energy_j - fixed_event_energy_j(profile)`

Apply trial coefficients in memory for evaluation, or through temporary
untracked generated configs under `trial_results/`. Do not edit hardware YAMLs
for fitted values. Validate on all non-calibration rows.

## Objective And Guardrails

Primary objective:

- lowest held-out `all/all` energy MAPE

Guardrails:

- timing metrics must remain unchanged except for noise
- do not significantly regress any kernel class for a small aggregate gain
- reject shape-regime heuristics unless backed by real exposed kernel or
  hardware metadata
- reject memory-traffic formula changes that are not backed by diagnostics,
  kernel metadata, hardware metadata, or documented GPU/CUTLASS behavior
- reject changes that make validation more than 5x slower
- reject changes that hide fitted values in tracked config or source files

## Trial Artifact Schema

Each experiment writes one untracked JSON file:

```text
trial_results/<experiment_tag>.json
```

Required fields:

- `experiment_tag`
- `timestamp_utc`
- `concept`
- `model_changes_summary`
- `feature_order`
- `calibration_policy`
- `calibration_rows`
- `heldout_policy`
- `per_hardware_coefficients`
- `fit_metrics`
- `heldout_metrics`
- `validation_runtime_s`
- `git_commit_before`
- `git_commit_after`
- `status`
- `rejection_reason`

`calibration_rows` records row keys per hardware. `per_hardware_coefficients`
contains fitted values only for that experiment and is never copied into YAML
unless the user separately asks for a final export. `git_commit_after` is the
kept commit hash for accepted model-code changes, otherwise `null`. `status`
must be `kept` or `rejected`; `rejection_reason` is required when rejected.

## Results Log

Append one tab-separated row to `results_energy.tsv` for every experiment.

Recommended columns:

```text
timestamp_utc	commit_before	experiment_tag	status	all_energy_mape	a10_energy_mape	a100_energy_mape	worst_class	worst_class_energy_mape	runtime_s	summary
```

If a candidate is worse than the previous kept row, record the candidate result
and discard the code change. If a candidate is kept, commit it before moving to
the next experiment.

## Git As Research Memory

Use Git to track kept model-code state:

- start from branch `energy-autoresearch`
- one kept experiment equals one commit
- commit messages must summarize the fitted-energy model change, the physical
  interpretation, and the held-out energy MAPE result
- rejected code changes must not be committed
- local fitted coefficients remain in `trial_results/` only

## Experiment Loop

For each iteration:

1. Inspect the latest validation metrics and identify the highest-error class.
2. Form one energy-model hypothesis about residual/static power composition or
   feature-to-energy mapping, or about evidence-grounded memory-traffic
   accounting.
3. Modify only the allowed code files.
4. Fit nonnegative residual/static power coefficients on the calibration rows.
5. Evaluate on all non-calibration rows with the existing validation harness.
6. Write `trial_results/<experiment_tag>.json`.
7. Append one row to `results_energy.tsv`.
8. Keep and commit only if held-out `all/all` energy MAPE improves, timing is
   unchanged, per-class regressions are acceptable, and the model is physically
   defensible.
9. If rejected, discard the code change and continue from the most recent kept
   commit.

Do not wait for user input between iterations after the user confirms the
experiment loop should start.
