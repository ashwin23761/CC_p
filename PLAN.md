# Chunked Prefill Simulator: Implementation Plan ()

## 0. Goal
Build a **real-LLM serving benchmark** (no hard-coded or synthetic latencies) that compares:
1. **Normal prefill**: each request's whole prompt is processed in one iteration (Orca/vLLM-style, prefill-prioritized; running decodes stall while a prefill runs).
2. **Chunked prefill with decode-maximal batching** (SARATHI, arXiv 2308.16369): the prompt is split into fixed-size chunks; each iteration = one prefill chunk + all running decodes **piggybacked in the same fused forward pass**.

Report TTFT, TPOT/inter-token latency, throughput; produce graphs; package it in Docker + Docker Compose with README.

Target environment: **Windows laptop, NVIDIA GPU (assume 6-8 GB VRAM), Docker Desktop (WSL2 backend)**.

## 1. Key ideas from SARATHI to implement
- Prefill is compute-bound, decode is memory-bound. A decode-only batch under-uses the GPU.
- **Chunked prefill**: split a prompt of length P into ceil(P/C) chunks of size C. Chunk k attends to the KV cache of chunks 0..k-1 plus itself (causal mask offset). Results must be identical to unchunked prefill.
- **Decode-maximal batching**: each iteration holds ONE prefill chunk plus as many decodes as fit. The decodes ride along with the prefill's compute, so their weight-loading cost is nearly free.
- **Chunk size choice**: align to tile sizes (multiples of 64/128; avoid sizes like 130 that cause tile quantization). Sweep C.
- Token budget per iteration = chunk size + number of decodes.
- Benefit: removes decode stalls (lower tail inter-token latency), keeps iteration times uniform. Cost: slightly higher TTFT for long prompts and some overhead from re-reading the KV cache. The experiments should show these trade-offs honestly.

## 2. Tech choices (decided, don't re-litigate)
- Language: Python 3.11, PyTorch (CUDA), Hugging Face `transformers` for weights/tokenizer only.
- Model (pick by VRAM, configurable): `Qwen/Qwen2.5-0.5B-Instruct` (default, safe on 4-6 GB), `Qwen/Qwen2.5-1.5B-Instruct` or `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (6-8 GB). fp16, ungated.
- **Do not use `model.generate()` for the experiments.** Write our own small inference engine so we control batching. vLLM is optional only as an external sanity check (it needs WSL2/Linux; not part of the main deliverable).
- Metrics via `time.perf_counter()` with `torch.cuda.synchronize()` around each iteration.
- Plotting: matplotlib + pandas. CLI: argparse or typer. Config: YAML.

## 3. Repository layout
```
chunked-prefill-sim/
  Dockerfile
  docker-compose.yml
  README.md
  requirements.txt
  configs/
    default.yaml            # model, dtype, max_seq_len, chunk sizes, rates, seeds
    smoke.yaml              # tiny run (~2 min) to verify the setup
  src/cpsim/
    engine/
      model_runner.py       # fused forward over a flat token batch (see 4.1)
      kv_cache.py           # per-request preallocated KV buffers (see 4.2)
    scheduler/
      base.py               # Request, RequestState, Scheduler interface
      normal.py             # normal prefill scheduler
      chunked.py            # chunked prefill + decode-maximal scheduler
    workload.py             # request generator (Poisson arrivals, length distributions)
    simulator.py            # main loop: arrivals -> scheduler -> runner -> metrics
    metrics.py              # per-request + per-iteration records -> CSV
    plots.py                # all graphs
    cli.py                  # `python -m cpsim.cli run|sweep|plot`
  tests/
    test_equivalence.py     # chunked == unchunked == HF reference
    test_scheduler.py       # budget/ordering invariants
  results/                  # CSVs + PNGs (mounted volume)
```

## 4. Engine (the real-LLM part)

### 4.1 Fused mixed-batch forward
Load an HF causal LM (Llama/Qwen2 architecture). Write `ModelRunner.forward(batch)` where `batch` is a list of segments, each `(request_id, token_ids, start_pos)`:
- A prefill chunk is a segment with many tokens; a decode is a segment with 1 token.
- Concatenate all segment tokens into one flat tensor `[T]` -> embeddings `[T, H]`.
- For each layer: run `input_layernorm`, **q/k/v projections once over all T tokens**, **MLP once over all T tokens** (this fusion is the whole point of piggybacking).
- Apply RoPE with per-token absolute positions (`start_pos + offset`).
- Attention: write new K/V into each request's KV buffer, then loop over segments and call `F.scaled_dot_product_attention` with the segment's query against its cached keys `[0 : start_pos+len]`. For prefill chunks use a causal mask with offset (query i can see keys ≤ start_pos+i); for decode no mask needed. Handle GQA (repeat KV heads).
- Take the final-token hidden state of each segment that finishes its prompt, or each decode, apply final norm + `lm_head`, greedy argmax for next token.
- Reuse the HF module weights (`layer.self_attn.q_proj`, `layer.mlp`, etc.) directly; only the attention/batching glue is custom.

### 4.2 KV cache
Preallocate per-request tensors `[layers, 2, kv_heads, max_seq_len, head_dim]` from a pool (slot allocator). Fixed `max_seq_len` (e.g. 4096), `max_running` (e.g. 16) sized to fit VRAM. Free slot on finish. No paging needed, but document that.

### 4.3 Correctness gate (must pass before any benchmarking)
`tests/test_equivalence.py`:
- Prefill the same prompt (a) in one shot, (b) with chunk sizes 64/128/256/513, (c) via HF `model(...)`. Final logits must match within fp16 tolerance, and greedy outputs for 32 tokens must be identical.
- Mixed batch (1 prefill chunk + N decodes) outputs must equal running each request alone.

## 5. Schedulers
Common: FCFS queue, `max_running` cap, run until EOS or `max_new_tokens`.

**NormalScheduler** (prefill-prioritized): if any waiting request can be admitted, run an iteration containing the **entire prompt** of one waiting request (decodes do NOT run in that iteration, so they stall). Otherwise run a decode-only iteration with all running requests.

**ChunkedScheduler(chunk_size C)**: every iteration:
1. Add all running decodes (one token each).
2. If a request is mid-prefill, or one can be admitted, add its next chunk of up to C tokens (a single prefill chunk per iteration, as in SARATHI).
3. Execute as one fused forward.
Optionally a `token_budget` option (chunk = budget - num_decodes) as an ablation.

## 6. Workload and simulator loop
- Arrivals: Poisson at rate λ req/s (seeded); also a "burst" mode (all at t=0).
- Prompt lengths: configurable (fixed, e.g. 512/1024/2048, plus a mixed distribution such as 70% short 128-512 and 30% long 1536-3072). Build prompts from real text (e.g. a public-domain corpus or repeated Wikipedia-like text), tokenised to the exact target length.
- Output lengths: e.g. 64-128 tokens (use `min_new_tokens`-style forced length for comparability; ignore EOS in benchmark mode).
- Simulator runs in **real time**: it keeps a clock, releases requests when their arrival time passes, and calls the scheduler -> runner each iteration. Record wall-clock timestamps. Warm up the GPU (3-5 dummy iterations) and discard them. Run each config for ≥3 seeds.

## 7. Metrics
Per request: arrival, first_token_time, finish_time, prompt_len, output_len, all token timestamps.
- **TTFT** = first_token_time − arrival (p50/p90/p99, mean)
- **TPOT / inter-token latency (ITL)** (p50/p99, and max "generation stall")
- **Throughput**: output tokens/s, total tokens/s (prefill + decode), requests/s
- **End-to-end latency**
- Per-iteration log: iteration id, start/end, #prefill tokens, #decode tokens, duration, GPU memory.
- Save everything as CSV under `results/<run_id>/`.

## 8. Experiments (the evaluation section)
| # | Experiment | Sweep |
|---|-----------|-------|
| E1 | Main comparison | normal vs chunked(C=256), mixed workload, λ ∈ {0.5,1,2,4} req/s |
| E2 | Chunk-size sensitivity | C ∈ {64,128,256,512,1024}; plot TTFT, ITL p99, throughput |
| E3 | Prompt-length scaling | prompt ∈ {512,1024,2048,3072}, fixed λ |
| E4 | Decode-batch size | max_running ∈ {2,4,8,16} (shows piggyback benefit growing with decode batch) |
| E5 | Burst / saturation | all requests at t=0, measure makespan and throughput |
| E6 (optional) | Non-aligned chunk (e.g. 200 vs 256) | demonstrates tile quantization |

## 9. Graphs (matplotlib, saved PNG + CSV)
1. TTFT CDF: normal vs chunked
2. p50/p99 TTFT vs arrival rate
3. p99 inter-token latency vs arrival rate (decode stalls; this is where chunked should win)
4. Throughput (tokens/s) vs arrival rate
5. Chunk size vs {TTFT, ITL p99, throughput} (trade-off curve)
6. Iteration timeline: per-iteration duration with prefill/decode token composition, showing normal's spikes vs chunked's flat profile
7. Per-token generation timeline for a few requests showing stalls
Include a summary table (markdown + CSV) of speedups/regressions. Discuss results honestly, including cases where chunked is worse (e.g. TTFT for long prompts, small-GPU effects).

## 10. Docker (Windows + GPU)
- `Dockerfile`: base `pytorch/pytorch:2.x-cuda12.x-cudnn9-runtime` (or `nvidia/cuda` + pip torch), install `requirements.txt`, copy `src/`, set `ENTRYPOINT ["python","-m","cpsim.cli"]`.
- `docker-compose.yml`:
  - service `sim` with GPU reservation:
    ```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ```
  - volumes: `./results:/app/results`, `./configs:/app/configs`, named volume `hf_cache:/root/.cache/huggingface` (so the model downloads once).
  - env: `HF_TOKEN` (optional), `CONFIG=configs/default.yaml`.
  - extra services/profiles: `smoke` (runs smoke.yaml), `bench` (full sweep), `plots` (generate graphs from results CSV, no GPU needed).
- Also provide a `--device cpu` fallback with a tiny model so reviewers without a GPU can run the smoke test.
- README prerequisites for Windows: latest NVIDIA driver, WSL2, Docker Desktop with WSL2 backend and GPU support enabled; verify with `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.

## 11. README must contain
Overview and architecture diagram; Windows/WSL2/Docker GPU setup; quick start (`docker compose run smoke`, `docker compose run bench`, `docker compose run plots`); config reference; how to choose model/chunk size for your VRAM; how to run tests; results section with embedded graphs and analysis; limitations (single GPU, no paged attention, per-sequence attention loop, greedy decoding, small model vs paper's A6000/A100 + LLaMA-13B results); reference to SARATHI.

## 12. Milestones (build in this order, commit per milestone)
1. Repo skeleton, config, Dockerfile/compose, GPU `nvidia-smi` + torch.cuda check in container.
2. ModelRunner + KV cache + equivalence tests passing (**gate**).
3. Normal scheduler + simulator + metrics CSV; smoke run.
4. Chunked scheduler; verify mixed batches in the iteration log.
5. Workload generator, warm-up, multi-seed sweeps (E1-E5).
6. Plots + summary table.
7. README, final Docker verification from a clean clone, results written up.

## 13. Acceptance criteria
- `docker compose run smoke` completes on a Windows GPU laptop in under ~5 minutes and writes CSV + at least 2 graphs.
- Equivalence tests pass.
- No latency numbers are hard-coded; all come from real GPU execution.
- Iteration logs confirm: normal = prefill-only or decode-only iterations; chunked = mixed iterations with ≤C prefill tokens plus all running decodes.
- README results contain graphs for E1-E5 and a written analysis.
