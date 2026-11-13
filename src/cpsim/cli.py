from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from cpsim.engine.model_runner import ModelRunner
from cpsim.metrics import append_summary
from cpsim.plots import generate_plots
from cpsim.scheduler.base import Request
from cpsim.scheduler.chunked import ChunkedScheduler
from cpsim.scheduler.normal import NormalScheduler
from cpsim.simulator import Simulator, copy_requests
from cpsim.workload import WorkloadGenerator


def _read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration at {path} must be a YAML mapping")
    for section in ("model", "runtime", "workload", "scheduler"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Configuration is missing the '{section}' mapping")
    return config


def _configure_runtime(
    config: dict[str, Any], device: str | None, model_name: str | None
) -> None:
    if device:
        config["runtime"]["device"] = device
    if model_name:
        config["model"]["name"] = model_name
    if config["runtime"].get("device") == "cpu":
        if not model_name and config["model"]["name"] == "Qwen/Qwen2.5-0.5B-Instruct":
            config["model"]["name"] = "tiny-random"
        config["model"]["dtype"] = "float32"


def _make_runner(config: dict[str, Any], max_running: int | None = None) -> ModelRunner:
    model = config["model"]
    runtime = config["runtime"]
    return ModelRunner(
        model_name=model["name"],
        device=runtime.get("device", "auto"),
        dtype=model.get("dtype", "auto"),
        max_seq_len=int(model["max_seq_len"]),
        max_running=max(max_running or 1, int(runtime.get("max_running", 1))),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
    )


def _make_workload_generator(config: dict[str, Any]) -> WorkloadGenerator:
    return WorkloadGenerator(
        config["model"]["name"],
        trust_remote_code=bool(config["model"].get("trust_remote_code", False)),
    )


def _execute_policy(
    requests: list[Request],
    scheduler_name: str,
    config: dict[str, Any],
    runner: ModelRunner,
    output_root: Path,
    metadata: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    runtime = config["runtime"]
    settings = config["scheduler"]
    max_running = int(metadata.get("max_running", runtime["max_running"]))
    if scheduler_name == "normal":
        scheduler = NormalScheduler(max_running, int(config["model"]["max_seq_len"]))
        chunk_size = None
    else:
        chunk_size = int(metadata.get("chunk_size", settings["chunk_size"]))
        scheduler = ChunkedScheduler(
            max_running=max_running,
            chunk_size=chunk_size,
            max_seq_len=int(config["model"]["max_seq_len"]),
            token_budget=settings.get("token_budget"),
        )
    run_name = f"{label}-{scheduler_name}"
    result = Simulator(runner).run(
        copy_requests(requests),
        scheduler,
        scheduler_name,
        chunk_size,
        output_root / run_name,
        metadata={
            "model": config["model"]["name"],
            **metadata,
        },
    )
    print(json.dumps(result, indent=2, allow_nan=True))
    return result


def run_command(
    config: dict[str, Any],
    runner: ModelRunner,
    workload_generator: WorkloadGenerator,
    output_root: Path,
    policies: list[str] | None = None,
    workload_overrides: dict[str, Any] | None = None,
    metadata_overrides: dict[str, Any] | None = None,
    label: str | None = None,
) -> list[dict[str, Any]]:
    workload_settings = {**config["workload"], **(workload_overrides or {})}
    requests = workload_generator.generate(workload_settings)
    policy_mode = config["scheduler"].get("mode", "compare")
    selected = policies or (
        ["normal", "chunked"] if policy_mode == "compare" else [policy_mode]
    )
    if any(policy not in {"normal", "chunked"} for policy in selected):
        raise ValueError("scheduler.mode must be normal, chunked, or compare")
    run_label = label or datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    metadata = {
        "seed": workload_settings["seed"],
        "arrival_rate": workload_settings["arrival_rate"],
        "prompt_distribution": workload_settings["prompt_distribution"],
        "prompt_length": (
            workload_settings["prompt_length"]
            if workload_settings["prompt_distribution"] == "fixed"
            else None
        ),
        "max_running": int(
            (metadata_overrides or {}).get(
                "max_running", config["runtime"]["max_running"]
            )
        ),
        "experiment": (metadata_overrides or {}).get("experiment", "single_run"),
        **(metadata_overrides or {}),
    }
    if workload_settings["prompt_distribution"] == "fixed":
        max_prompt = int(workload_settings["prompt_length"])
        if max_prompt > int(config["model"]["max_seq_len"]):
            raise ValueError("prompt_length cannot exceed model.max_seq_len")
    results = []
    for policy in selected:
        results.append(
            _execute_policy(
                requests,
                policy,
                config,
                runner,
                output_root,
                metadata,
                run_label,
            )
        )
    return results


def _build_sweep_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    sweep = config.get("sweep", {})
    seeds = [int(value) for value in sweep.get("seeds", [11, 23, 37])]
    cases: list[dict[str, Any]] = []
    for rate in sweep.get("arrival_rates", [0.5, 1, 2, 4]):
        for seed in seeds:
            cases.append(
                {
                    "experiment": "E1",
                    "policies": ["normal", "chunked"],
                    "workload": {
                        "mode": "poisson",
                        "arrival_rate": float(rate),
                        "request_count": int(sweep.get("request_count", 12)),
                        "seed": seed,
                    },
                    "metadata": {"arrival_rate": float(rate)},
                    "label": f"E1-rate{rate}-seed{seed}",
                }
            )
    for chunk_size in config["scheduler"].get("chunk_sizes", [64, 128, 256, 512, 1024]):
        for seed in seeds:
            cases.append(
                {
                    "experiment": "E2",
                    "policies": ["chunked"],
                    "workload": {"request_count": int(sweep.get("request_count", 12)), "seed": seed},
                    "metadata": {"chunk_size": int(chunk_size)},
                    "label": f"E2-C{chunk_size}-seed{seed}",
                }
            )
    for prompt_length in sweep.get("prompt_lengths", [512, 1024, 2048, 3072]):
        for seed in seeds:
            cases.append(
                {
                    "experiment": "E3",
                    "policies": ["normal", "chunked"],
                    "workload": {
                        "prompt_distribution": "fixed",
                        "prompt_length": int(prompt_length),
                        "request_count": int(sweep.get("request_count", 12)),
                        "arrival_rate": float(sweep.get("fixed_arrival_rate", 1.0)),
                        "seed": seed,
                    },
                    "metadata": {
                        "prompt_length": int(prompt_length),
                        "arrival_rate": float(sweep.get("fixed_arrival_rate", 1.0)),
                    },
                    "label": f"E3-prompt{prompt_length}-seed{seed}",
                }
            )
    for max_running in sweep.get("max_running_values", [2, 4, 8, 16]):
        for seed in seeds:
            cases.append(
                {
                    "experiment": "E4",
                    "policies": ["normal", "chunked"],
                    "workload": {
                        "request_count": int(sweep.get("request_count", 12)),
                        "seed": seed,
                    },
                    "metadata": {"max_running": int(max_running)},
                    "label": f"E4-running{max_running}-seed{seed}",
                }
            )
    for seed in seeds:
        cases.append(
            {
                "experiment": "E5",
                "policies": ["normal", "chunked"],
                "workload": {
                    "mode": "burst",
                    "request_count": int(sweep.get("burst_request_count", 16)),
                    "seed": seed,
                },
                "metadata": {"arrival_rate": 0.0},
                "label": f"E5-burst-seed{seed}",
            }
        )
    return cases


def sweep_command(
    config: dict[str, Any],
    runner: ModelRunner,
    workload_generator: WorkloadGenerator,
    output_root: Path,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in _build_sweep_cases(config):
        overrides = {**case["metadata"], "experiment": case["experiment"]}
        results.extend(
            run_command(
                config,
                runner,
                workload_generator,
                output_root,
                policies=case["policies"],
                workload_overrides=case["workload"],
                metadata_overrides=overrides,
                label=case["label"],
            )
        )
    append_summary(output_root / "summary.csv", results)
    plots = generate_plots(output_root)
    print(f"Generated {len(plots)} plots under {output_root}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cpsim")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "sweep"):
        command = subparsers.add_parser(name)
        command.add_argument(
            "--config",
            type=Path,
            default=Path(os.environ.get("CONFIG", "configs/default.yaml")),
        )
        command.add_argument("--device", choices=("auto", "cpu", "cuda"))
        command.add_argument("--model", help="Override the model configured in YAML")
    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("--results", type=Path, default=Path("results"))

    args = parser.parse_args(argv)
    if args.command == "plot":
        generated = generate_plots(args.results)
        print("\n".join(str(path) for path in generated))
        return 0

    config = _read_config(args.config)
    _configure_runtime(config, args.device, args.model)
    output_root = Path(config["runtime"].get("output_dir", "results"))
    sweep = config.get("sweep", {})
    max_running = max(
        [int(config["runtime"]["max_running"])]
        + [int(value) for value in sweep.get("max_running_values", [])]
    )
    runner = _make_runner(config, max_running=max_running)
    runner.warmup(int(config["runtime"].get("warmup_iterations", 3)))
    workload_generator = _make_workload_generator(config)
    if args.command == "run":
        results = run_command(
            config,
            runner,
            workload_generator,
            output_root,
        )
        append_summary(output_root / "summary.csv", results)
        plots = generate_plots(output_root)
        print(f"Generated {len(plots)} plots under {output_root}")
    else:
        sweep_command(config, runner, workload_generator, output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
