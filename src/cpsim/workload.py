from __future__ import annotations

import random
from typing import Any

from transformers import AutoTokenizer

from cpsim.scheduler.base import Request


class WorkloadGenerator:
    def __init__(self, model_name: str, trust_remote_code: bool = False) -> None:
        self.tiny_byte_mode = model_name == "tiny-random"
        self.tokenizer = None
        if not self.tiny_byte_mode:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust_remote_code
            )

    def generate(self, config: dict[str, Any], overrides: dict[str, Any] | None = None) -> list[Request]:
        settings = {**config, **(overrides or {})}
        count = int(settings["request_count"])
        if count < 1:
            raise ValueError("request_count must be at least 1")
        seed = int(settings["seed"])
        rng = random.Random(seed)
        corpus = settings.get("corpus") or [
            "A language model processes context one token at a time using learned parameters."
        ]
        if self.tiny_byte_mode:
            encoded_corpus = [[byte + 1 for byte in text.encode("utf-8")] for text in corpus]
        else:
            assert self.tokenizer is not None
            encoded_corpus = [
                self.tokenizer.encode(text, add_special_tokens=False)
                for text in corpus
            ]
        corpus_ids = [token for row in encoded_corpus for token in row]
        if not corpus_ids:
            raise ValueError("The configured prompt corpus tokenizes to an empty sequence")

        mode = settings.get("mode", "poisson")
        arrival_rate = float(settings.get("arrival_rate", 1.0))
        if mode not in {"poisson", "burst"}:
            raise ValueError("workload.mode must be 'poisson' or 'burst'")
        if arrival_rate <= 0:
            raise ValueError("arrival_rate must be greater than zero")
        arrivals = [0.0] * count
        if mode == "poisson":
            elapsed = 0.0
            arrivals = []
            for _ in range(count):
                elapsed += rng.expovariate(arrival_rate)
                arrivals.append(elapsed)

        requests: list[Request] = []
        for index, arrival in enumerate(arrivals):
            prompt_length = self._prompt_length(settings, rng)
            repeated = (corpus_ids * ((prompt_length // len(corpus_ids)) + 1))[:prompt_length]
            output_length = int(settings["output_length"])
            if output_length < 1:
                raise ValueError("output_length must be at least 1")
            requests.append(
                Request(
                    request_id=f"request-{index:05d}",
                    prompt_ids=repeated,
                    arrival_time=arrival,
                    output_limit=output_length,
                )
            )
        return requests

    @staticmethod
    def _prompt_length(settings: dict[str, Any], rng: random.Random) -> int:
        distribution = settings.get("prompt_distribution", "fixed")
        if distribution == "fixed":
            length = int(settings["prompt_length"])
        elif distribution == "uniform":
            low = int(settings["min_prompt_length"])
            high = int(settings["max_prompt_length"])
            length = rng.randint(low, high)
        elif distribution == "mixed":
            if rng.random() < float(settings.get("short_prompt_probability", 0.7)):
                low = int(settings["short_prompt_min"])
                high = int(settings["short_prompt_max"])
            else:
                low = int(settings["long_prompt_min"])
                high = int(settings["long_prompt_max"])
            length = rng.randint(low, high)
        else:
            raise ValueError(
                "prompt_distribution must be one of: fixed, uniform, mixed"
            )
        if length < 1:
            raise ValueError("prompt lengths must be at least 1")
        return length
