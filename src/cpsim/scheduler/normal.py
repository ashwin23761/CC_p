from __future__ import annotations

from cpsim.scheduler.base import RequestState, Scheduler, Segment


class NormalScheduler(Scheduler):
    """Prefill-prioritized scheduling; active decodes stall during each prefill."""

    def __init__(self, max_running: int, max_seq_len: int) -> None:
        super().__init__(max_running)
        self.max_seq_len = max_seq_len

    def next_batch(self) -> list[Segment]:
        prefill = self._admit_prefill()
        if prefill is not None:
            end = min(len(prefill.prompt_ids), prefill.prompt_cursor + self.max_seq_len)
            return [
                Segment(
                    request=prefill,
                    token_ids=prefill.prompt_ids[prefill.prompt_cursor:end],
                    start_pos=prefill.prompt_cursor,
                    kind="prefill",
                    emits_logits=end == len(prefill.prompt_ids),
                )
            ]
        return [
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
