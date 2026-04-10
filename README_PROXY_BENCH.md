# Proxy Bench — lm-evaluation-harness Usage

How we use this fork of `lm-evaluation-harness` to run a fixed set of models across a fixed set of benchmarks for the Proxy Bench project. Driver script: `scripts/run_eval_harness_all.sh` (in the parent `proxy-bench` repo).

## Models (11)

All models are reached through the **CMU LiteLLM proxy** (`https://cmu.litellm.ai/v1/chat/completions`), except Nemotron which is served from RunPod.

| Provider (prefix) | Model |
|---|---|
| `azure_ai/` (Azure AI Foundry) | `Kimi-K2.5`, `Kimi-K2-Thinking`, `DeepSeek-V3.2` |
| `azure/` | `gpt-5.2`, `gpt-5.2-codex` |
| `fireworks_ai/` | `minimax-m2p5` (MiniMax-M2.5), `glm-4p7` (GLM-4.7), `minimax-m2p1` (MiniMax-M2.1), `glm-5` (GLM-5) |
| `anthropic/` | `claude-opus-4-6` |
| RunPod (`nvidia/...`) | `nvidia-nemotron-3-nano-30b-a3b-bf16` |

## Benchmarks (7)

```
ifeval
acp_gen_2shot
mbpp_chat
humaneval_chat
gpqa_diamond_cot_zeroshot,gpqa_main_cot_zeroshot,gpqa_extended_cot_zeroshot
aime25
logiqa_cot_zeroshot
```

## Example invocation

```bash
python -m lm_eval run \
  --model openai-chat-completions \
  --model_args "model=azure_ai/Kimi-K2.5,base_url=https://cmu.litellm.ai/v1/chat/completions,disable_seed=true,num_concurrent=20" \
  --tasks ifeval \
  --output_path $DATA_DIR/ifeval \
  --apply_chat_template \
  --confirm_run_unsafe_code \
  --log_samples \
  --gen_kwargs max_gen_toks=16384 \
  --use_cache $DATA_DIR/.cache/azure_ai__Kimi-K2.5
```

Fixed flags on every run:
- `--model openai-chat-completions` — all models speak the OpenAI chat API via LiteLLM / RunPod.
- `--apply_chat_template` — apply the task's chat template (required for `*_chat` tasks and for correct instruction formatting).
- `--confirm_run_unsafe_code` — needed by `mbpp_chat` / `humaneval_chat` which execute generated code.
- `--log_samples` — write per-sample outputs for later analysis.
- `--use_cache <dir>/<safe_model_name>` — SQLite response cache keyed per model (`/` in model name replaced with `__`). Lets re-runs skip already-completed prompts.

## Per-model / per-benchmark differences

The driver script only varies two things across runs:

**1. `model_args` — `disable_seed` for non-Azure models**

```
# Azure / Azure AI Foundry (seeds supported)
model=<...>,base_url=<proxy>,num_concurrent=20

# Everything else (Fireworks, Anthropic, RunPod): seed rejected by backend
model=<...>,base_url=<proxy>,disable_seed=true,num_concurrent=20
```

`num_concurrent=20` is used everywhere. For RunPod the `base_url` switches to `https://api.runpod.ai/v2/<endpoint>/openai/v1/chat/completions` and `OPENAI_API_KEY` is overridden with `$RUNPOD_API_KEY` inline.

**2. `gen_kwargs` — bigger budget for reasoning/thinking models**

```
max_gen_toks=32768   # if model name matches *Thinking* / *DeepSeek* / *thinking* / *deepseek*
max_gen_toks=16384   # everything else
```

Benchmarks are **not** varied per model — the same 7 tasks run for every model. The only per-benchmark change is the `--output_path` and `--tasks` value.

## Environment

- `conda activate proxy`
- `PYTHONPATH` prepended with this local `lm-evaluation-harness` checkout (we use a patched fork — see fixes below).
- `HF_ALLOW_CODE_EVAL=1` (humaneval / mbpp code execution).
- `OPENAI_API_KEY` → CMU LiteLLM proxy key (from `~/.bashrc`). Swapped to `RUNPOD_API_KEY` for the Nemotron loop.

## Notable local patches in this fork

This fork contains fixes that matter when running through proxies / reasoning models:

- **Async + sync cache**: don't cache empty responses (`lm_eval/models/api_models.py`, `lm_eval/api/model.py`).
- **`asyncio.gather` poison**: one failed request no longer wipes the whole batch — wrapped in a per-request `safe_call` (`api_models.py`).
- **Stop sequences**: whitespace-only stops filtered out (Anthropic rejects them) (`openai_completions.py`).
- **Reasoning-model detection**: narrowed to `gpt-5` / `o1` / `o3` / `o4` so it doesn't falsely match `K2.5`, `GLM-5`, `M2.5` (`openai_completions.py`).
- **Nemotron**: temp=1.0, repetition_penalty=1.1, strip `</think>` (`openai_completions.py`).
- **GPQA `\boxed{}` extraction** for reasoning-model answers (`lm_eval/filters/extraction.py`).

## Parallelism

The driver launches one background shell per model; inside each, benchmarks run **sequentially**. All model shells run in parallel, then `wait`.
