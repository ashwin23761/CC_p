from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

from cpsim.engine.kv_cache import KVCachePool
from cpsim.scheduler.base import Request, Segment


@dataclass(frozen=True)
class RunnerOutput:
    next_tokens: dict[str, int]
    next_logits: dict[str, torch.Tensor]
    gpu_memory_bytes: int


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class ModelRunner:
    """Runs fused projections/MLPs with per-sequence causal attention and KV caches."""

    def __init__(
        self,
        model_name: str,
        device: str,
        dtype: str,
        max_seq_len: int,
        max_running: int,
        trust_remote_code: bool = False,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but this PyTorch installation has no CUDA support")
        self.device = torch.device(device)
        if dtype == "auto":
            self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        elif dtype in {"float16", "fp16"}:
            if self.device.type == "cpu":
                raise ValueError("float16 CPU inference is not supported; use dtype: float32")
            self.dtype = torch.float16
        elif dtype in {"bfloat16", "bf16"}:
            if self.device.type == "cpu":
                raise ValueError("bfloat16 CPU inference is not supported by this runner")
            self.dtype = torch.bfloat16
        elif dtype in {"float32", "fp32"}:
            self.dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype {dtype!r}; use float16, bfloat16, float32, or auto")

        if model_name == "tiny-random":
            torch.manual_seed(0)
            tiny_config = Qwen2Config(
                vocab_size=257,
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=max_seq_len,
                bos_token_id=1,
                eos_token_id=2,
            )
            self.model = Qwen2ForCausalLM(tiny_config).to(
                device=self.device, dtype=self.dtype
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                trust_remote_code=trust_remote_code,
            ).to(self.device)
        self.model.eval()
        self.config = self.model.config
        self.transformer = getattr(self.model, "model", None)
        if self.transformer is None or not all(
            hasattr(self.transformer, name) for name in ("layers", "embed_tokens", "norm")
        ):
            raise ValueError(
                f"Model {model_name!r} is not a supported Llama/Qwen2-style causal LM"
            )
        self.layers = self.transformer.layers
        self.num_heads = int(self.config.num_attention_heads)
        self.num_kv_heads = int(getattr(self.config, "num_key_value_heads", self.num_heads))
        self.head_dim = int(getattr(self.config, "head_dim", self.config.hidden_size // self.num_heads))
        self.max_seq_len = max_seq_len
        self.cache = KVCachePool(
            num_layers=len(self.layers),
            num_kv_heads=self.num_kv_heads,
            max_seq_len=max_seq_len,
            head_dim=self.head_dim,
            max_slots=max_running,
            device=self.device,
            dtype=self.dtype,
        )

    @classmethod
    def from_config(cls, model: torch.nn.Module, max_seq_len: int, max_running: int) -> ModelRunner:
        """Create a runner around an already-constructed compatible model (used by tests)."""
        runner = cls.__new__(cls)
        runner.model = model.to(next(model.parameters()).device)
        runner.model.eval()
        runner.device = next(runner.model.parameters()).device
        runner.dtype = next(runner.model.parameters()).dtype
        runner.config = runner.model.config
        runner.transformer = getattr(runner.model, "model", None)
        if runner.transformer is None or not all(
            hasattr(runner.transformer, name) for name in ("layers", "embed_tokens", "norm")
        ):
            raise ValueError("Expected a Llama/Qwen2-style causal LM")
        runner.layers = runner.transformer.layers
        runner.num_heads = int(runner.config.num_attention_heads)
        runner.num_kv_heads = int(getattr(runner.config, "num_key_value_heads", runner.num_heads))
        runner.head_dim = int(
            getattr(runner.config, "head_dim", runner.config.hidden_size // runner.num_heads)
        )
        runner.max_seq_len = max_seq_len
        runner.cache = KVCachePool(
            num_layers=len(runner.layers),
            num_kv_heads=runner.num_kv_heads,
            max_seq_len=max_seq_len,
            head_dim=runner.head_dim,
            max_slots=max_running,
            device=runner.device,
            dtype=runner.dtype,
        )
        return runner

    def ensure_cache(self, request_id: str) -> None:
        self.cache.acquire(request_id)

    def release_cache(self, request_id: str) -> None:
        self.cache.release(request_id)

    @torch.inference_mode()
    def forward(self, segments: list[Segment]) -> RunnerOutput:
        if not segments:
            raise ValueError("forward requires at least one segment")
        if any(not segment.token_ids for segment in segments):
            raise ValueError("segments must contain at least one token")

        flat_ids: list[int] = []
        positions: list[int] = []
        ranges: list[tuple[int, int]] = []
        cursor = 0
        for segment in segments:
            end = cursor + len(segment.token_ids)
            ranges.append((cursor, end))
            flat_ids.extend(segment.token_ids)
            positions.extend(range(segment.start_pos, segment.start_pos + len(segment.token_ids)))
            cursor = end
        if max(positions) >= self.max_seq_len:
            raise ValueError(
                f"Token position {max(positions)} exceeds max_seq_len={self.max_seq_len}"
            )

        input_ids = torch.tensor(flat_ids, device=self.device, dtype=torch.long)
        hidden = self.transformer.embed_tokens(input_ids)
        position_ids = torch.tensor(positions, device=self.device, dtype=torch.long).unsqueeze(0)
        total_tokens = len(flat_ids)

        for layer_index, layer in enumerate(self.layers):
            attention = layer.self_attn
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            query = attention.q_proj(normalized).view(
                total_tokens, self.num_heads, self.head_dim
            )
            key = attention.k_proj(normalized).view(
                total_tokens, self.num_kv_heads, self.head_dim
            )
            value = attention.v_proj(normalized).view(
                total_tokens, self.num_kv_heads, self.head_dim
            )

            cos, sin = attention.rotary_emb(query.unsqueeze(0), position_ids)
            cos = cos.squeeze(0).unsqueeze(1).to(dtype=query.dtype)
            sin = sin.squeeze(0).unsqueeze(1).to(dtype=query.dtype)
            query = query * cos + _rotate_half(query) * sin
            key = key * cos + _rotate_half(key) * sin

            attention_outputs: list[torch.Tensor] = []
            for segment, (start, end) in zip(segments, ranges, strict=True):
                slot = self.cache.get(segment.request.request_id)
                segment_key = key[start:end].transpose(0, 1)
                segment_value = value[start:end].transpose(0, 1)
                cache_start = segment.start_pos
                cache_end = cache_start + end - start
                slot.keys[layer_index, :, cache_start:cache_end].copy_(segment_key)
                slot.values[layer_index, :, cache_start:cache_end].copy_(segment_value)

                keys = slot.keys[layer_index, :, :cache_end]
                values = slot.values[layer_index, :, :cache_end]
                groups = self.num_heads // self.num_kv_heads
                if groups > 1:
                    keys = keys.repeat_interleave(groups, dim=0)
                    values = values.repeat_interleave(groups, dim=0)
                queries = query[start:end].transpose(0, 1)
                mask: torch.Tensor | None = None
                if segment.kind == "prefill":
                    query_positions = torch.arange(
                        cache_start, cache_end, device=self.device
                    )
                    key_positions = torch.arange(cache_end, device=self.device)
                    mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
                    mask = mask.unsqueeze(0).unsqueeze(0)
                result = F.scaled_dot_product_attention(
                    queries.unsqueeze(0),
                    keys.unsqueeze(0),
                    values.unsqueeze(0),
                    attn_mask=mask,
                    is_causal=False,
                )
                attention_outputs.append(result.squeeze(0).transpose(0, 1))

            attention_result = torch.cat(attention_outputs, dim=0).reshape(
                total_tokens, self.num_heads * self.head_dim
            )
            hidden = residual + attention.o_proj(attention_result)
            residual = hidden
            hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))

        output_segments = [
            (segment, end - 1)
            for segment, (_, end) in zip(segments, ranges, strict=True)
            if segment.emits_logits
        ]
        next_tokens: dict[str, int] = {}
        next_logits: dict[str, torch.Tensor] = {}
        if output_segments:
            indices = torch.tensor([index for _, index in output_segments], device=self.device)
            final_hidden = self.transformer.norm(hidden.index_select(0, indices))
            logits = self.model.lm_head(final_hidden)
            token_ids = torch.argmax(logits, dim=-1).tolist()
            for position, ((segment, _), token_id) in enumerate(
                zip(output_segments, token_ids, strict=True)
            ):
                next_tokens[segment.request.request_id] = int(token_id)
                next_logits[segment.request.request_id] = logits[position]

        memory = int(torch.cuda.memory_allocated(self.device)) if self.device.type == "cuda" else 0
        return RunnerOutput(
            next_tokens=next_tokens,
            next_logits=next_logits,
            gpu_memory_bytes=memory,
        )

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def warmup(self, iterations: int) -> None:
        if iterations < 0:
            raise ValueError("warmup_iterations cannot be negative")
        request_id = "__warmup__"
        self.ensure_cache(request_id)
        try:
            for _ in range(iterations):
                segment = Segment(
                    request=Request(request_id, [int(self.config.bos_token_id or 0)], 0.0, 1),
                    token_ids=[int(self.config.bos_token_id or 0)],
                    start_pos=0,
                    kind="prefill",
                    emits_logits=True,
                )
                self.forward([segment])
            self.synchronize()
        finally:
            self.release_cache(request_id)
