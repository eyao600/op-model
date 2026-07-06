from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import yaml

from opmodel.api import DType, LocalOp, OpKind, OpProfile, Phase, TensorRole, TensorSpec
from opmodel.calibration import (
    calibrate_energy_from_artifact_database,
    format_calibration_report,
    write_calibrated_hardware_config,
)
from opmodel.hardware import load_hardware
from opmodel.registry import create_model
from opmodel.validation.artifact_accuracy import (
    DEFAULT_ARTIFACT_DATA_DIR,
    DEFAULT_HARDWARE_DIR,
    format_text_report,
    run_artifact_validation,
    write_csv_report,
    write_performance_details_csv_report,
    write_validation_plots,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opmodel")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model", default="roofline")
    predict_parser.add_argument("--hardware", required=True)
    predict_parser.add_argument("--op", required=True)

    validate_parser = subparsers.add_parser("validate-artifact")
    validate_parser.add_argument("--data-dir", default=str(DEFAULT_ARTIFACT_DATA_DIR))
    validate_parser.add_argument("--hardware-dir", default=str(DEFAULT_HARDWARE_DIR))
    validate_parser.add_argument("--model", default="roofline")
    validate_parser.add_argument("--limit", type=int, help="maximum supported rows per CSV file")
    validate_parser.add_argument("--output-csv")
    validate_parser.add_argument("--output-details-csv")
    validate_parser.add_argument("--output-plot", default="artifact_accuracy.svg")
    validate_parser.add_argument("--no-plot", action="store_true")
    validate_parser.add_argument(
        "--workload-y-max",
        type=float,
        help="fixed y-axis maximum for generated workload bar plots",
    )

    calibrate_parser = subparsers.add_parser("calibrate-energy")
    calibrate_parser.add_argument("--data-dir", default=str(DEFAULT_ARTIFACT_DATA_DIR))
    calibrate_parser.add_argument("--hardware-dir", default=str(DEFAULT_HARDWARE_DIR))
    calibrate_parser.add_argument("--hardware", required=True)
    calibrate_parser.add_argument("--fit-fraction", type=float, default=0.7)
    calibrate_parser.add_argument("--limit", type=int, help="maximum GEMM rows to use")
    calibrate_parser.add_argument("--output-hardware", required=True)

    args = parser.parse_args(argv)
    if args.command == "predict":
        hardware = load_hardware(args.hardware)
        op = load_op(args.op)
        model = create_model(args.model)
        profile = model.predict(op, hardware)
        print(json.dumps(profile_to_dict(profile), indent=2, sort_keys=True))
        return 0
    if args.command == "validate-artifact":
        report = run_artifact_validation(
            data_dir=args.data_dir,
            hardware_dir=args.hardware_dir,
            model_name=args.model,
            limit=args.limit,
        )
        if args.output_csv:
            write_csv_report(report, args.output_csv)
        if args.output_details_csv:
            write_performance_details_csv_report(report, args.output_details_csv)
        plot_paths = ()
        if not args.no_plot and args.output_plot:
            plot_paths = write_validation_plots(
                report,
                args.output_plot,
                workload_y_max=args.workload_y_max,
            )
        print(format_text_report(report))
        if args.output_csv:
            print(f"Wrote CSV report: {args.output_csv}")
        if args.output_details_csv:
            print(f"Wrote performance details CSV report: {args.output_details_csv}")
        for plot_path in plot_paths:
            print(f"Wrote normalized plot: {plot_path}")
        return 0
    if args.command == "calibrate-energy":
        result = calibrate_energy_from_artifact_database(
            hardware_name=args.hardware,
            data_dir=args.data_dir,
            hardware_dir=args.hardware_dir,
            fit_fraction=args.fit_fraction,
            limit=args.limit,
        )
        write_calibrated_hardware_config(
            input_hardware_path=result.input_hardware_path,
            output_hardware_path=args.output_hardware,
            energy_model=result.energy_model,
        )
        print(format_calibration_report(result))
        print(f"Wrote calibrated hardware config: {args.output_hardware}")
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


def load_op(path: str | Path) -> LocalOp:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise ValueError("Op config must be a mapping")
    return parse_op(data)


def parse_op(data: Mapping[str, Any]) -> LocalOp:
    try:
        name = str(data["name"])
        kind = OpKind(str(data["kind"]))
        phase = Phase(str(data["phase"]))
    except KeyError as exc:
        raise ValueError(f"Missing required op field: {exc.args[0]}") from exc
    except ValueError as exc:
        raise ValueError(f"Invalid op enum value: {exc}") from exc

    tensors_data = data.get("tensors", [])
    if not isinstance(tensors_data, list):
        raise ValueError("tensors must be a list")
    tensors = tuple(_parse_tensor(item, index) for index, item in enumerate(tensors_data))
    attrs = data.get("attrs", {})
    if not isinstance(attrs, Mapping):
        raise ValueError("attrs must be a mapping")
    return LocalOp(name=name, kind=kind, phase=phase, tensors=tensors, attrs=dict(attrs))


def profile_to_dict(profile: OpProfile) -> dict[str, Any]:
    data = asdict(profile)
    data["engine"] = profile.engine.value
    data["energy_breakdown"] = asdict(profile.energy_breakdown)
    data["footprint"] = asdict(profile.footprint)
    data["memory_access"] = asdict(profile.memory_access)
    return data


def _parse_tensor(data: Any, index: int) -> TensorSpec:
    if not isinstance(data, Mapping):
        raise ValueError(f"tensors[{index}] must be a mapping")
    try:
        role = TensorRole(str(data["role"]))
        shape = tuple(int(dim) for dim in data["shape"])
        dtype = DType(str(data["dtype"]))
    except KeyError as exc:
        raise ValueError(f"Missing required tensors[{index}] field: {exc.args[0]}") from exc
    except ValueError as exc:
        raise ValueError(f"Invalid tensors[{index}] value: {exc}") from exc
    layout = data.get("layout")
    return TensorSpec(role=role, shape=shape, dtype=dtype, layout=None if layout is None else str(layout))


if __name__ == "__main__":
    raise SystemExit(main())
