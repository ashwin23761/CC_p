from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from cpsim.engine.model_runner import ModelRunner
from cpsim.metrics import summarize, write_run
from cpsim.scheduler.base import Request, RequestState, Scheduler, Segment


class Simulator:
    def __init__(self, runner: ModelRunner) -> None:
        self.runner = runner

    def run(
        self,
        requests: list[Request],
        scheduler: Scheduler,
        scheduler_name: str,
        chunk_size: int | None,
        output_dir: Path,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if any(len(request.prompt_ids) > self.runner.max_seq_len for request in requests):
            raise ValueError(
                f"Every prompt must be <= max_seq_len ({self.runner.max_seq_len})"
            )
        ordered = sorted(requests, key=lambda request: request.arrival_time)
        next_request = 0
        iterations: list[dict[str, Any]] = []
        started = time.perf_counter()
        completed = 0

        while completed < len(ordered):
            elapsed = time.perf_counter() - started
            while (
                next_request < len(ordered)
                and ordered[next_request].arrival_time <= elapsed
            ):
                scheduler.enqueue(ordered[next_request])
                next_request += 1

            segments = scheduler.next_batch()
            if not segments:
                if next_request >= len(ordered):
                    raise RuntimeError("Scheduler has unfinished requests but produced no work")
                wait_for = ordered[next_request].arrival_time - elapsed
                time.sleep(min(max(wait_for, 0.0), 0.01))
                continue

            for segment in segments:
                self.runner.ensure_cache(segment.request.request_id)
            self.runner.synchronize()
            iteration_start = time.perf_counter()
            output = self.runner.forward(segments)
            self.runner.synchronize()
            iteration_end = time.perf_counter()
            start_offset = iteration_start - started
            end_offset = iteration_end - started
            iterations.append(
                {
                    "iteration_id": len(iterations),
                    "start_time": start_offset,
                    "end_time": end_offset,
                    "duration": iteration_end - iteration_start,
                    "prefill_tokens": sum(
                        len(segment.token_ids) for segment in segments if segment.kind == "prefill"
                    ),
                    "decode_tokens": sum(
                        len(segment.token_ids) for segment in segments if segment.kind == "decode"
                    ),
                    "gpu_memory_bytes": output.gpu_memory_bytes,
                }
            )

            token_time = end_offset
            for segment in segments:
                request = segment.request
                if segment.kind == "prefill":
                    request.prompt_cursor += len(segment.token_ids)
                    if request.prompt_cursor == len(request.prompt_ids):
                        request.state = RequestState.DECODING
                token_id = output.next_tokens.get(request.request_id)
                if token_id is not None:
                    request.output_ids.append(token_id)
                    request.token_times.append(token_time)
                    if request.first_token_time is None:
                        request.first_token_time = token_time
                    if len(request.output_ids) >= request.output_limit:
                        request.finish_time = token_time
                        scheduler.request_finished(request)
                        self.runner.release_cache(request.request_id)
                        completed += 1

        summary = summarize(
            ordered,
            iterations,
            scheduler=scheduler_name,
            chunk_size=chunk_size,
            metadata=metadata,
        )
        write_run(output_dir, ordered, iterations, summary)
        return summary


def copy_requests(requests: list[Request]) -> list[Request]:
    return [
        Request(
            request_id=request.request_id,
            prompt_ids=list(request.prompt_ids),
            arrival_time=request.arrival_time,
            output_limit=request.output_limit,
        )
        for request in requests
    ]
