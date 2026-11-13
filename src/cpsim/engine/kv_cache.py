from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CacheSlot:
    keys: torch.Tensor
    values: torch.Tensor


class KVCachePool:
    """Lazily allocates fixed-size KV buffers and reuses them across requests."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        max_slots: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.shape = (num_layers, num_kv_heads, max_seq_len, head_dim)
        self.max_slots = max_slots
        self.device = device
        self.dtype = dtype
        self.slots: list[CacheSlot | None] = [None] * max_slots
        self.request_slots: dict[str, int] = {}
        self.free_slots = list(range(max_slots - 1, -1, -1))

    def acquire(self, request_id: str) -> CacheSlot:
        if request_id in self.request_slots:
            slot = self.slots[self.request_slots[request_id]]
            if slot is None:
                raise RuntimeError("An allocated KV cache slot has no buffers")
            return slot
        if not self.free_slots:
            raise RuntimeError(
                f"KV cache capacity ({self.max_slots} requests) exhausted while admitting {request_id}"
            )
        slot_index = self.free_slots.pop()
        slot = self.slots[slot_index]
        if slot is None:
            slot = CacheSlot(
                keys=torch.empty(self.shape, device=self.device, dtype=self.dtype),
                values=torch.empty(self.shape, device=self.device, dtype=self.dtype),
            )
            self.slots[slot_index] = slot
        self.request_slots[request_id] = slot_index
        return slot

    def get(self, request_id: str) -> CacheSlot:
        try:
            slot_index = self.request_slots[request_id]
        except KeyError as exc:
            raise KeyError(f"No KV cache allocated for request {request_id}") from exc
        slot = self.slots[slot_index]
        if slot is None:
            raise RuntimeError(f"KV cache slot {slot_index} was not initialized")
        return slot

    def release(self, request_id: str) -> None:
        slot_index = self.request_slots.pop(request_id, None)
        if slot_index is not None:
            self.free_slots.append(slot_index)
