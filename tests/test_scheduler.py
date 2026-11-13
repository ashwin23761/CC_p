from cpsim.scheduler.base import Request, RequestState
from cpsim.scheduler.chunked import ChunkedScheduler
from cpsim.scheduler.normal import NormalScheduler


def _request(request_id: str, prompt_length: int = 8) -> Request:
    return Request(request_id, list(range(prompt_length)), 0.0, 3)


def test_normal_scheduler_stalls_decodes_while_prefilling() -> None:
    scheduler = NormalScheduler(max_running=2, max_seq_len=32)
    first = _request("first")
    scheduler.enqueue(first)

    batch = scheduler.next_batch()
    assert len(batch) == 1
    assert batch[0].kind == "prefill"
    assert len(batch[0].token_ids) == 8

    first.state = RequestState.DECODING
    first.output_ids.append(100)
    second = _request("second", 5)
    scheduler.enqueue(second)
    batch = scheduler.next_batch()
    assert len(batch) == 1
    assert batch[0].request is second
    assert batch[0].kind == "prefill"
    assert all(segment.kind != "decode" for segment in batch)

    second.state = RequestState.DECODING
    second.output_ids.append(101)
    batch = scheduler.next_batch()
    assert {segment.request.request_id for segment in batch} == {"first", "second"}
    assert all(segment.kind == "decode" for segment in batch)


def test_chunked_scheduler_piggybacks_all_decodes_with_one_bounded_chunk() -> None:
    scheduler = ChunkedScheduler(max_running=3, chunk_size=4, max_seq_len=32)
    first = _request("first", 8)
    second = _request("second", 9)
    scheduler.enqueue(first)
    batch = scheduler.next_batch()
    assert len(batch) == 1 and len(batch[0].token_ids) == 4
    first.prompt_cursor = 4
    scheduler.enqueue(second)
    batch = scheduler.next_batch()
    assert len(batch) == 1
    assert batch[0].request is first
    assert len(batch[0].token_ids) == 4

    first.state = RequestState.DECODING
    first.output_ids.append(100)
    batch = scheduler.next_batch()
    assert {segment.kind for segment in batch} == {"decode", "prefill"}
    decode = next(segment for segment in batch if segment.kind == "decode")
    prefill = next(segment for segment in batch if segment.kind == "prefill")
    assert decode.request is first
    assert prefill.request is second
    assert len(prefill.token_ids) <= 4


def test_token_budget_accounts_for_piggybacked_decodes() -> None:
    scheduler = ChunkedScheduler(
        max_running=3,
        chunk_size=8,
        max_seq_len=32,
        token_budget=3,
    )
    first = _request("first", 8)
    scheduler.enqueue(first)
    scheduler.next_batch()
    first.state = RequestState.DECODING
    first.output_ids.append(100)
    second = _request("second", 8)
    third = _request("third", 8)
    scheduler.enqueue(second)
    scheduler.enqueue(third)

    batch = scheduler.next_batch()
    decode_count = sum(len(segment.token_ids) for segment in batch if segment.kind == "decode")
    prefill_count = sum(len(segment.token_ids) for segment in batch if segment.kind == "prefill")
    assert decode_count == 1
    assert prefill_count == 2
