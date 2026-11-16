# Serving (vLLM, OpenAI-compatible)

Minimal Docker scaffold for serving the function-calling SLM with vLLM 0.15.1.
Default deployment serves V1 (base `Qwen/Qwen2.5-3B-Instruct`); SFT/GRPO LoRA
adapters are pulled from HF Hub at container start when `ADAPTER_REPO` is set.

## Build

```bash
docker build -t info290-serving src/serving
```

## Run V1 (base, no adapter)

```bash
docker run --gpus all -p 8000:8000 info290-serving
```

## Run with an adapter

The adapter repos (`arnursb/info290-{v2-sft,v3-sft-masked,v4-grpo,v5-grpo-masked}`)
are private, so `HF_TOKEN` is required.

```bash
docker run --gpus all -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  -e ADAPTER_REPO=arnursb/info290-v3-sft-masked \
  info290-serving
```

When the adapter is loaded, requests select it by setting `"model": "adapter"`.
Sending `"model": "Qwen/Qwen2.5-3B-Instruct"` against the same server hits the
unadapted base.

## Environment variables

| Var | Default | Notes |
|-----|---------|-------|
| `MODEL_NAME` | `Qwen/Qwen2.5-3B-Instruct` | Base model HF id or local path |
| `ADAPTER_REPO` | unset | HF repo id; if set, fetched and registered as LoRA `adapter` |
| `MAX_LORA_RANK` | `64` | Must be >= adapter's actual rank (ours are 64) |
| `MAX_MODEL_LEN` | `4096` | vLLM context window |
| `PORT` | `8000` | |
| `HOST` | `0.0.0.0` | |
| `HF_TOKEN` | unset | Required when `ADAPTER_REPO` is set |

## Test request

OpenAI-compatible `/v1/chat/completions` with a `tools` parameter. Generation
parameters match `configs/bfcl.yaml` (`temperature=0.0`, `top_p=1.0`,
`max_tokens=512`).

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "adapter",
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 512,
    "messages": [
      {"role": "user", "content": "What is the weather in Berkeley right now?"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location.",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City, state"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
          },
          "required": ["location"]
        }
      }
    }]
  }'
```

For V1 (base), swap `"model": "adapter"` to `"model": "Qwen/Qwen2.5-3B-Instruct"`.

## Hardware

Requires NVIDIA GPU + nvidia-container-toolkit. The image is built on
`vllm/vllm-openai:v0.15.1`, which targets CUDA 12.x. There is no CPU fallback
in this scaffold; for non-GPU smoke tests, swap the base image for a CPU vLLM
build and expect substantially slower inference.

## Why this scaffold (vs. issue #7's `server.py` + `client_demo.py`)

`vllm serve` is already an OpenAI-compatible FastAPI server, so wrapping it in
a hand-rolled `server.py` would duplicate functionality without adding behavior
this project needs (no auth, no custom routes, no batching layer). The scaffold
keeps the surface area small: one Dockerfile, one entrypoint, env-var config.
A `client_demo.py` can be added later if reviewers want a Python-side example;
the curl above exercises the same path.

## Promoting a variant to public release

After full BFCL V4 evaluation, pick the best variant and promote its private
staging adapter (`arnursb/info290-vX-...`) to a public release repo. The
`upload_to_hub.py` script snapshots the source repo, generates a model card
seeded with scores from `results/<variant>/scores.json`, creates the public
target repo if needed, and verifies the upload landed.

```bash
HF_TOKEN=hf_xxx python -m src.serving.upload_to_hub \
  --source-repo arnursb/info290-v3-sft-masked \
  --target-repo arnursb/info290-function-calling-slm-3b \
  --scores-json results/v3-sft-masked/scores.json \
  --variant-name "SFT + function masking"
```

Use `--dry-run` to preview the model card without pushing. The script fails
fast if `HF_TOKEN` is missing, and verifies `adapter_config.json` and
`README.md` are present on the hub before returning success.
