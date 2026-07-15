# opmodel

`opmodel` is a standalone Python package for per-operation accelerator performance prediction.

It predicts latency, memory traffic, FLOPs, execution engine use, global memory footprint, and energy for a single already-partitioned local operation.

The package intentionally has no Rapid-LLM dependency and no graph knowledge. Upstream systems are responsible for partitioning model dimensions and lowering work into local operation dimensions before calling:

```python
profile = model.predict(local_op, hardware)
```

The core contract is:

```text
LocalOp + HardwareSpec -> OpProfile
```

## Install

```bash
pip install -e .
pip install -e ".[test]"
```

## Python API

```python
from opmodel import DType, LocalOp, OpKind, Phase, TensorRole, TensorSpec
from opmodel.hardware import load_hardware
from opmodel.models.roofline import RooflineModel

hardware = load_hardware("src/opmodel/configs/hardware/gpu_generic.yaml")
model = RooflineModel()

op = LocalOp(
    name="gemm",
    kind=OpKind.GEMM,
    phase=Phase.TRAIN_FWD,
    tensors=(
        TensorSpec(TensorRole.INPUT, (4096, 8192), DType.BF16),
        TensorSpec(TensorRole.WEIGHT, (8192, 32768), DType.BF16),
        TensorSpec(TensorRole.OUTPUT, (4096, 32768), DType.BF16),
    ),
)

profile = model.predict(op, hardware)
print(profile.latency_s, profile.energy_j)
```

## CLI

```bash
opmodel predict \
  --hardware src/opmodel/configs/hardware/gpu_generic.yaml \
  --op examples/gemm.yaml
```

The command writes JSON to stdout.

An opt-in extended GEMM roofline model is also available:

```bash
opmodel predict \
  --model extended_roofline \
  --hardware src/opmodel/configs/hardware/a10.yaml \
  --op examples/gemm_extended_roofline.yaml
```

For GEMM and batched GEMM, `extended_roofline` estimates CTA tiling,
first-touch L2 reuse, SMEM/L2/DRAM active utilization, compute-memory overlap,
and likely bottlenecks. Kernel parameters such as `cta_tile_m`, `cta_tile_n`,
`cta_tile_k`, `warp_tile_m`, `warp_tile_n`, `pipeline_stages`,
`warps_per_cta`, and `registers_per_thread` may be provided in op `attrs`;
conservative defaults are used when they are omitted. Non-GEMM ops reuse the
standard `roofline` estimators.

The first-class `effective_roofline` model uses the same GEMM parsing, kernel
catalog, traffic, occupancy, energy, and non-GEMM estimators, but replaces the
detailed K-stage timeline with a constant-complexity phase-aware roofline:

```bash
opmodel predict \
  --model effective_roofline \
  --hardware src/opmodel/configs/hardware/a10.yaml \
  --op examples/gemm_extended_roofline.yaml
```

For each full or tail CTA wave it derives concurrency-limited tensor, SMEM, L2,
and HBM rates, overlaps the local and memory bodies, and retains explicit memory
prologue and output-epilogue phases. Diagnostics include raw/effective rates,
occupancy classes, exact SMSP warp distributions, memory windows, service and
phase cycles, tied limiting resources, tile efficiency, and useful throughput.
Automatic kernel selection evaluates its shortlist with the effective model;
when `gemm_selection_backend` is set, its value must be `effective_roofline`.

To validate the extended GEMM latency timeline against the local GEMM latency
CSVs in `data/`:

```bash
opmodel validate-gemm-latency --data-dir data
```

By default this command uses `extended_roofline` (pass
`--model effective_roofline` to validate the effective-ceiling model), loads
hardware inputs from `src/opmodel/configs/hardware/`, and calibrates only fixed
device overhead cycles. It selects a small deterministic set of `small` and
`vector_like` GEMMs for overhead calibration, then reports held-out latency
error for every other supported GEMM row overall, by hardware, and by GEMM size class. Use
`--no-calibrate-fixed-overhead` to report accuracy with the hardware config
overhead as-is, `--output-csv` for per-row predictions, and `--output-params`
for the resolved overhead inputs and training rows.

To compare bf16 predictions against the EnergAIzer artifact data and write a
normalized SVG scatter plot:

```bash
opmodel validate-artifact \
  --output-csv artifact_accuracy.csv \
  --output-plot artifact_accuracy.svg
```

The validation harness uses the A100 40GB PCIe and A10 hardware configs in
`src/opmodel/configs/hardware/`, compares latency and total energy, and uses
the artifact CSV `energy` value as measured per-op energy. The SVG plots show
predicted/measured latency and energy versus bf16 working-set bytes, with one
panel per op type and hardware shown by color. The harness also writes
workload-filtered normalized bar plots for square GEMM and softmax subsets next
to the requested plot path.

## Hardware Config Schema

Hardware configs describe an accelerator with:

- `name` and `kind`
- `compute.clock_hz`
- `compute.vector_flops_per_s` and `compute.tensor_flops_per_s`
- per-engine energy maps keyed by dtype
- `memory.levels`, including an `hbm` level
- optional utilization values for vector, tensor, and memory levels

See `src/opmodel/configs/hardware/gpu_generic.yaml` and `src/opmodel/configs/hardware/tpu_generic.yaml`.

## Current Limitations

- The roofline model is analytical and intentionally simple.
- Tensor shapes must use conventional layouts for v0.
- L2, SRAM, and register traffic are not modeled unless an estimator adds diagnostics.
- GPU and TPU behavior is controlled by hardware config values, not separate code paths.
- The package models local operations only; graph construction, parallelism, scheduling, and communication are out of scope.
