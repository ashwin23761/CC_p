from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_plots(results_dir: Path) -> list[Path]:
    request_files = sorted(
        path for path in results_dir.rglob("requests.csv") if path.parent != results_dir
    )
    iteration_files = sorted(
        path for path in results_dir.rglob("iterations.csv") if path.parent != results_dir
    )
    summary_files = sorted(
        path for path in results_dir.rglob("summary.csv") if path.parent != results_dir
    )
    if not request_files or not iteration_files or not summary_files:
        raise FileNotFoundError(
            f"No complete simulator runs found under {results_dir}; run a benchmark first"
        )
    requests = pd.concat(
        [pd.read_csv(path).assign(run=path.parent.name) for path in request_files],
        ignore_index=True,
    )
    iterations = pd.concat(
        [pd.read_csv(path).assign(run=path.parent.name) for path in iteration_files],
        ignore_index=True,
    )
    summaries = pd.concat(
        [pd.read_csv(path).assign(run=path.parent.name) for path in summary_files],
        ignore_index=True,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    generated = [
        _ttft_cdf(requests, results_dir),
        _latency_vs_rate(summaries, results_dir, "p50_ttft", "p99_ttft", "TTFT vs arrival rate", "ttft_vs_rate.png"),
        _latency_vs_rate(summaries, results_dir, "p99_itl", None, "p99 inter-token latency vs arrival rate", "itl_p99_vs_rate.png"),
        _throughput_vs_rate(summaries, results_dir),
        _chunk_tradeoff(summaries, results_dir),
        _iteration_timeline(iterations, results_dir),
        _token_timeline(requests, results_dir),
    ]
    _write_measured_summary(summaries, results_dir)
    _write_comparison_summary(summaries, results_dir)
    return generated


def _policy_label(frame: pd.DataFrame) -> pd.Series:
    labels = frame["scheduler"].fillna("unknown").astype(str)
    if "chunk_size" in frame.columns:
        chunk = frame["chunk_size"].fillna("").astype(str)
        labels = labels.where(chunk.eq(""), labels + " (C=" + chunk + ")")
    return labels


def _ttft_cdf(requests: pd.DataFrame, output: Path) -> Path:
    fig, axis = plt.subplots(figsize=(8, 5))
    requests = requests.copy()
    requests["policy"] = _policy_label(requests)
    for label, frame in requests.groupby("policy"):
        values = frame["ttft"].sort_values().to_numpy()
        if len(values):
            axis.plot(values, np.arange(1, len(values) + 1) / len(values), label=label)
    axis.set(xlabel="TTFT (seconds)", ylabel="Cumulative fraction", title="TTFT CDF")
    axis.grid(alpha=0.25)
    axis.legend()
    return _save(fig, output / "ttft_cdf.png")


def _latency_vs_rate(
    summaries: pd.DataFrame,
    output: Path,
    first_metric: str,
    second_metric: str | None,
    title: str,
    filename: str,
) -> Path:
    fig, axis = plt.subplots(figsize=(8, 5))
    frame = summaries.copy()
    if "arrival_rate" not in frame.columns:
        frame["arrival_rate"] = 0.0
    frame["policy"] = _policy_label(frame)
    grouped = frame.groupby(["policy", "arrival_rate"], dropna=False)[first_metric].median()
    for policy, values in grouped.groupby(level=0):
        x = values.index.get_level_values(1).astype(float)
        axis.plot(x, values.to_numpy(), marker="o", label=f"{policy} {first_metric}")
    if second_metric:
        grouped = frame.groupby(["policy", "arrival_rate"], dropna=False)[second_metric].median()
        for policy, values in grouped.groupby(level=0):
            x = values.index.get_level_values(1).astype(float)
            axis.plot(x, values.to_numpy(), marker="x", linestyle="--", label=f"{policy} {second_metric}")
    axis.set(xlabel="Arrival rate (requests/s)", ylabel="Latency (seconds)", title=title)
    axis.grid(alpha=0.25)
    axis.legend(fontsize="small")
    return _save(fig, output / filename)


def _throughput_vs_rate(summaries: pd.DataFrame, output: Path) -> Path:
    fig, axis = plt.subplots(figsize=(8, 5))
    frame = summaries.copy()
    if "arrival_rate" not in frame.columns:
        frame["arrival_rate"] = 0.0
    frame["policy"] = _policy_label(frame)
    for policy, group in frame.groupby("policy"):
        grouped = group.groupby("arrival_rate")["output_tokens_per_second"].median()
        axis.plot(grouped.index, grouped.values, marker="o", label=policy)
    axis.set(
        xlabel="Arrival rate (requests/s)",
        ylabel="Generated output tokens/s",
        title="Output throughput vs arrival rate",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize="small")
    return _save(fig, output / "throughput_vs_rate.png")


def _chunk_tradeoff(summaries: pd.DataFrame, output: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    frame = summaries[summaries["chunk_size"].notna()].copy()
    metrics = [
        ("p50_ttft", "p50 TTFT (s)"),
        ("p99_itl", "p99 inter-token latency (s)"),
        ("output_tokens_per_second", "Output tokens/s"),
    ]
    if frame.empty:
        for axis in axes:
            axis.text(0.5, 0.5, "No chunk-size sweep data", ha="center", va="center")
            axis.set_axis_off()
    else:
        for axis, (column, label) in zip(axes, metrics, strict=True):
            for policy, group in frame.groupby("scheduler"):
                points = group.groupby("chunk_size")[column].median().sort_index()
                axis.plot(points.index, points.values, marker="o", label=policy)
            axis.set(xlabel="Prefill chunk size (tokens)", ylabel=label)
            axis.grid(alpha=0.25)
            axis.legend(fontsize="small")
    fig.suptitle("Chunk-size trade-offs")
    return _save(fig, output / "chunk_size_tradeoff.png")


def _iteration_timeline(iterations: pd.DataFrame, output: Path) -> Path:
    fig, axis = plt.subplots(figsize=(9, 5))
    grouped = list(iterations.groupby("run"))
    for run, frame in grouped[:8]:
        axis.plot(frame["iteration_id"], frame["duration"], marker=".", linewidth=1, label=run)
    axis.set(
        xlabel="Iteration",
        ylabel="Iteration duration (seconds)",
        title="Iteration duration and token composition",
    )
    axis.grid(alpha=0.25)
    if grouped:
        representative = grouped[0][1]
        token_axis = axis.twinx()
        token_axis.bar(
            representative["iteration_id"],
            representative["prefill_tokens"],
            alpha=0.15,
            color="tab:orange",
            label="prefill tokens (first run)",
        )
        token_axis.bar(
            representative["iteration_id"],
            representative["decode_tokens"],
            bottom=representative["prefill_tokens"],
            alpha=0.15,
            color="tab:green",
            label="decode tokens (first run)",
        )
        token_axis.set_ylabel("Tokens per iteration (first run)")
        handles, labels = axis.get_legend_handles_labels()
        token_handles, token_labels = token_axis.get_legend_handles_labels()
        axis.legend(handles + token_handles, labels + token_labels, fontsize="x-small")
    return _save(fig, output / "iteration_timeline.png")


def _token_timeline(requests: pd.DataFrame, output: Path) -> Path:
    fig, axis = plt.subplots(figsize=(9, 5))
    selected = requests.groupby("run", sort=False).head(2)
    for _, row in selected.iterrows():
        try:
            token_times = json.loads(row["token_times"])
        except (TypeError, json.JSONDecodeError):
            continue
        relative = [timestamp - float(row["arrival_time"]) for timestamp in token_times]
        axis.plot(range(1, len(relative) + 1), relative, marker=".", label=f"{row['run']} {row['request_id']}")
    axis.set(xlabel="Output token index", ylabel="Time since arrival (seconds)", title="Per-request token generation timeline")
    axis.grid(alpha=0.25)
    axis.legend(fontsize="x-small")
    return _save(fig, output / "token_timeline.png")


def _write_measured_summary(summaries: pd.DataFrame, output: Path) -> None:
    columns = [
        name
        for name in (
            "experiment",
            "scheduler",
            "chunk_size",
            "arrival_rate",
            "prompt_length",
            "max_running",
            "seed",
            "p50_ttft",
            "p99_ttft",
            "p99_itl",
            "output_tokens_per_second",
        )
        if name in summaries
    ]
    selected = summaries[columns].fillna("")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = "\n".join(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    )
    (output / "measured_summary.md").write_text(
        "# Measured benchmark results\n\n"
        "Values below are measured from completed model runs; no latency values are synthesized.\n\n"
        + header
        + "\n"
        + separator
        + "\n"
        + rows
        + "\n",
        encoding="utf-8",
    )


def _write_comparison_summary(summaries: pd.DataFrame, output: Path) -> None:
    keys = [
        name
        for name in (
            "experiment",
            "seed",
            "arrival_rate",
            "prompt_distribution",
            "prompt_length",
            "max_running",
        )
        if name in summaries
    ]
    metrics = ["p50_ttft", "p99_itl", "output_tokens_per_second"]
    normal = summaries[summaries["scheduler"] == "normal"][keys + metrics].copy()
    chunked = summaries[summaries["scheduler"] == "chunked"][keys + metrics + ["chunk_size"]].copy()
    normal = normal.rename(columns={metric: f"normal_{metric}" for metric in metrics})
    chunked = chunked.rename(columns={metric: f"chunked_{metric}" for metric in metrics})
    comparisons = normal.merge(chunked, on=keys, how="inner")
    for metric in metrics:
        base = comparisons[f"normal_{metric}"].replace(0, float("nan"))
        comparisons[f"{metric}_change_pct"] = (
            (comparisons[f"chunked_{metric}"] / base) - 1
        ) * 100
    comparisons.to_csv(output / "comparison_summary.csv", index=False)
    display_columns = keys + [
        "chunk_size",
        "p50_ttft_change_pct",
        "p99_itl_change_pct",
        "output_tokens_per_second_change_pct",
    ]
    header = "| " + " | ".join(display_columns) + " |"
    separator = "| " + " | ".join("---" for _ in display_columns) + " |"
    rows = "\n".join(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in comparisons[display_columns].fillna("").itertuples(
            index=False, name=None
        )
    )
    (output / "comparison_summary.md").write_text(
        "# Normal vs chunked comparison\n\n"
        "Percentage change is `(chunked / normal - 1) * 100`; negative latency change is an improvement, while positive throughput change is an improvement.\n\n"
        + header
        + "\n"
        + separator
        + "\n"
        + rows
        + "\n",
        encoding="utf-8",
    )


def _save(figure: plt.Figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
