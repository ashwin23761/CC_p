import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from cpsim.engine.model_runner import ModelRunner
from cpsim.scheduler.base import Request, Segment


def _runner() -> ModelRunner:
    torch.manual_seed(17)
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        attention_dropout=0.0,
    )
    model = Qwen2ForCausalLM(config).eval()
    return ModelRunner.from_config(model, max_seq_len=128, max_running=4)


def _prefill(
    runner: ModelRunner,
    request: Request,
    chunk_size: int,
) -> torch.Tensor:
    final_logits = None
    for start in range(0, len(request.prompt_ids), chunk_size):
        end = min(start + chunk_size, len(request.prompt_ids))
        segment = Segment(
            request=request,
            token_ids=request.prompt_ids[start:end],
            start_pos=start,
            kind="prefill",
            emits_logits=end == len(request.prompt_ids),
        )
        output = runner.forward([segment])
        if segment.emits_logits:
            final_logits = output.next_logits[request.request_id]
    assert final_logits is not None
    return final_logits


def test_chunked_prefill_logits_and_greedy_tokens_match_hf() -> None:
    runner = _runner()
    prompt = [4, 12, 31, 7, 42, 18, 9, 3, 5, 11, 21]
    request = Request("chunked", prompt, 0.0, 32)
    with torch.inference_mode():
        reference = runner.model(input_ids=torch.tensor([prompt])).logits[0, -1]
    for chunk_size in (1, 4, 8, 11):
        runner.ensure_cache(f"prefill-{chunk_size}")
        probe = Request(f"prefill-{chunk_size}", prompt, 0.0, 1)
        actual = _prefill(runner, probe, chunk_size)
        torch.testing.assert_close(actual, reference, rtol=2e-4, atol=2e-4)
        runner.release_cache(probe.request_id)

    runner.ensure_cache(request.request_id)
    next_logits = _prefill(runner, request, len(prompt))
    generated = [int(next_logits.argmax().item())]
    for index in range(1, 32):
        segment = Segment(
            request=request,
            token_ids=[generated[-1]],
            start_pos=len(prompt) + index - 1,
            kind="decode",
            emits_logits=True,
        )
        actual = runner.forward([segment]).next_logits[request.request_id]
        with torch.inference_mode():
            full_context = torch.tensor([prompt + generated])
            expected = runner.model(input_ids=full_context).logits[0, -1]
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
        generated.append(int(actual.argmax().item()))
    runner.release_cache(request.request_id)


def test_mixed_prefill_and_decode_match_independent_forwards() -> None:
    runner = _runner()
    prompt = [4, 12, 31, 7, 42]
    decode_request = Request("decode", prompt, 0.0, 2)
    runner.ensure_cache(decode_request.request_id)
    first_logits = _prefill(runner, decode_request, len(prompt))
    decode_request.output_ids.append(int(first_logits.argmax().item()))
    new_request = Request("new-prefill", [8, 9, 10, 11, 12, 13], 0.0, 2)
    runner.ensure_cache(new_request.request_id)
    batch = [
        Segment(
            request=decode_request,
            token_ids=[decode_request.output_ids[-1]],
            start_pos=len(prompt),
            kind="decode",
            emits_logits=True,
        ),
        Segment(
            request=new_request,
            token_ids=new_request.prompt_ids,
            start_pos=0,
            kind="prefill",
            emits_logits=True,
        ),
    ]
    actual = runner.forward(batch).next_logits
    with torch.inference_mode():
        decode_expected = runner.model(
            input_ids=torch.tensor([prompt + decode_request.output_ids])
        ).logits[0, -1]
        prefill_expected = runner.model(
            input_ids=torch.tensor([new_request.prompt_ids])
        ).logits[0, -1]
    torch.testing.assert_close(actual["decode"], decode_expected, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual["new-prefill"], prefill_expected, rtol=2e-4, atol=2e-4)
    runner.release_cache(decode_request.request_id)
    runner.release_cache(new_request.request_id)
