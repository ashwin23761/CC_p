from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from cpsim.scheduler.base import Request


REQUEST_COLUMNS = [
    "request_id",
    "arrival_time",
    "first_token_time",
    "finish_time",
    "prompt_len",
    "output_len",
    "ttft",
    "end_to_end_latency",
    "token_times",
    "inter_token_latencies",
    "scheduler",
    "chunk_size",
]
ITERATION_COLUMNS = [
    "iteration_id",
    "start_time",
    "end_time",
    "duration",
    "prefill_tokens",
    "decode_tokens",
    "gpu_memory_bytes",
]


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(values, quantile)) if values else math.nan


def request_rows(requests: list[Request]) -> list[dict[str, Any]]:
    rows = []
    for request in requests:
        if request.first_token_time is None or request.finish_time is None:
            raise RuntimeError(f"Request {request.request_id} did not finish")
        intervals = [
            later - earlier
            for earlier, later in zip(request.token_times, request.token_times[1:])
        ]
        rows.append(
            {
                "request_id": request.request_id,
                "arrival_time": request.arrival_time,
                "first_token_time": request.first_token_time,
                "finish_time": request.finish_time,
                "prompt_len": len(request.prompt_ids),
                "output_len": len(request.output_ids),
                "ttft": request.first_token_time - request.arrival_time,
                "end_to_end_latency": request.finish_time - request.arrival_time,
                "token_times": json.dumps(request.token_times),
                "inter_token_latencies": json.dumps(intervals),
            }
        )
    return rows


def summarize(
    requests: list[Request],
    iterations: list[dict[str, Any]],
    scheduler: str,
    chunk_size: int | None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = request_rows(requests)
    ttfts = [float(row["ttft"]) for row in rows]
    itls = [
        float(value)
        for request in rows
        for value in json.loads(request["inter_token_latencies"])
    ]
    first_arrival = min(request.arrival_time for request in requests)
    final_finish = max(float(request.finish_time) for request in requests if request.finish_time is not None)
    duration = max(final_finish - first_arrival, 1e-12)
    total_output_tokens = sum(len(request.output_ids) for request in requests)
    total_prompt_tokens = sum(len(request.prompt_ids) for request in requests)
    return {
        **(metadata or {}),
        "scheduler": scheduler,
        "chunk_size": chunk_size,
        "request_count": len(requests),
        "mean_ttft": float(np.mean(ttfts)),
        "p50_ttft": percentile(ttfts, 50),
        "p90_ttft": percentile(ttfts, 90),
        "p99_ttft": percentile(ttfts, 99),
        "p50_itl": percentile(itls, 50),
        "p99_itl": percentile(itls, 99),
        "max_itl": max(itls, default=math.nan),
        "mean_end_to_end_latency": float(np.mean([row["end_to_end_latency"] for row in rows])),
        "output_tokens_per_second": total_output_tokens / duration,
        "total_tokens_per_second": (total_output_tokens + total_prompt_tokens) / duration,
        "requests_per_second": len(requests) / duration,
        "makespan_seconds": duration,
        "iteration_count": len(iterations),
    }


def write_run(
    output_dir: Path,
    requests: list[Request],
    iterations: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            **row,
            "scheduler": summary["scheduler"],
            "chunk_size": summary.get("chunk_size"),
        }
        for row in request_rows(requests)
    ]
    write_csv(output_dir / "requests.csv", rows, REQUEST_COLUMNS)
    write_csv(output_dir / "iterations.csv", iterations, ITERATION_COLUMNS)
    write_csv(output_dir / "summary.csv", [summary], list(summary))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(path, rows, columns)
