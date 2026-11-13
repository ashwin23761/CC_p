from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class RequestState(str, Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODING = "decoding"
    FINISHED = "finished"


@dataclass
class Request:
    request_id: str
    prompt_ids: list[int]
    arrival_time: float
    output_limit: int
    prompt_cursor: int = 0
    state: RequestState = RequestState.WAITING
    output_ids: list[int] = field(default_factory=list)
    token_times: list[float] = field(default_factory=list)
    first_token_time: float | None = None
    finish_time: float | None = None


@dataclass(frozen=True)
class Segment:
    request: Request
    token_ids: list[int]
    start_pos: int
    kind: str
    emits_logits: bool


class Scheduler(ABC):
    def __init__(self, max_running: int) -> None:
        if max_running < 1:
            raise ValueError("max_running must be at least 1")
        self.max_running = max_running
        self.waiting: deque[Request] = deque()
        self.active: dict[str, Request] = {}

    def enqueue(self, request: Request) -> None:
        self.waiting.append(request)

    def _active_prefill(self) -> Request | None:
        return next(
            (request for request in self.active.values() if request.state == RequestState.PREFILL),
            None,
        )

    def _admit_prefill(self) -> Request | None:
        if self._active_prefill() is not None:
            return self._active_prefill()
        if len(self.active) >= self.max_running or not self.waiting:
            return None
        request = self.waiting.popleft()
        request.state = RequestState.PREFILL
        self.active[request.request_id] = request
        return request

    def request_finished(self, request: Request) -> None:
        request.state = RequestState.FINISHED
        self.active.pop(request.request_id, None)

    @abstractmethod
    def next_batch(self) -> list[Segment]:
        raise NotImplementedError
