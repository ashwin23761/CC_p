from __future__ import annotations

from cpsim.scheduler.base import RequestState, Scheduler, Segment


class ChunkedScheduler(Scheduler):
    """Runs all active decodes together with at most one prompt chunk."""

    def __init__(
        self,
        max_running: int,
        chunk_size: int,
        max_seq_len: int,
        token_budget: int | None = None,
    ) -> None:
        super().__init__(max_running)
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if token_budget is not None and token_budget < 2:
            raise ValueError("token_budget must be at least 2")
        self.chunk_size = chunk_size
        self.max_seq_len = max_seq_len
        self.token_budget = token_budget

    def next_batch(self) -> list[Segment]:
        segments = [
            Segment(
                request=request,
                token_ids=[request.output_ids[-1]],
                start_pos=len(request.prompt_ids) + len(request.output_ids) - 1,
                kind="decode",
                emits_logits=True,
            )
            for request in self.active.values()
            if request.state == RequestState.DECODING
        ]
        available = self.chunk_size
        if self.token_budget is not None:
            available = min(available, self.token_budget - len(segments))
        if available <= 0:
            return segments
        prefill = self._admit_prefill()
        if prefill is not None:
            end = min(
                len(prefill.prompt_ids),
                prefill.prompt_cursor + available,
                self.max_seq_len,
            )
            segments.append(
                Segment(
                    request=prefill,
                    token_ids=prefill.prompt_ids[prefill.prompt_cursor:end],
                    start_pos=prefill.prompt_cursor,
                    kind="prefill",
                    emits_logits=end == len(prefill.prompt_ids),
                )
            )
        return segments
