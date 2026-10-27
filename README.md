# Torchcast Decision 31B (v1.0)

A text-based decision model developed by **Torchcast AI**, based on Gemma-4-31B-it. It returns yes/no probabilities
(`noul`), choice distributions (`choice`) and ordinal score distributions (`score`) through a TypeSafe-compatible API.

[Model card and weights](https://huggingface.co/yangchen-torchcast/torchcast-decision-31b/tree/v1.0.0) (private; access on request)

## Model

- **Weights:** a full fine-tune of `google/gemma-4-31B-it` (all layers except the input embeddings) on a mix of
  public datasets (official training splits) and synthetic decision data covering choice, yes/no and score questions.
- **Attention:** hybrid. The sliding-window layers are causal as in the base model. The global layers attend to the whole
  prompt in both directions (`decision_config.json`: `attention_mode` `hybrid_bidir_global`).
- **Prompt format:** `d12_gemma4_h2o_hdr`, fixed in the bundled package.
- **Serving settings** (`decision_config.json`): calibration temperature 0.8 (choice), 0.6 (yes/no), 0.6 (score), and a
  yes/no commit floor of 0.801. Neither changes the answer.

## Results

Board results pending: the model has not yet been measured on the JevBench leaderboard.

## Run for evaluation

You need Linux with one NVIDIA GPU with 80 GB or more (tested on one H100 80 GB, BF16). The weights repository contains the
serving package (`torchcast_decision/`) and its configuration; use both unchanged. Only the bundled PyTorch server
implements the hybrid attention. Do not serve the weights with vLLM, llama.cpp, MLX or a generic `transformers`
causal-LM pipeline: they apply a causal mask to every layer and give different probabilities.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu130
pip install "transformers==5.17.0" "kernels==0.17.2" safetensors tokenizers jinja2 numpy "huggingface_hub[cli]"

hf download yangchen-torchcast/torchcast-decision-31b --revision v1.0.0 --local-dir torchcast-decision-31b
cd torchcast-decision-31b
sha256sum -c SHA256SUMS
D12_SLIDE=fa3 python -m torchcast_decision.server --model . --host 127.0.0.1 --port 8011
```

`D12_SLIDE=fa3` selects the FlashAttention-3 kernel for the sliding-window layers. It is fetched through `kernels` on first
start, and the server fails at start-up if FA3 is unavailable rather than silently using a slower path.
The pinned artifact is tag `v1.0.0` of `yangchen-torchcast/torchcast-decision-31b`. Its weight SHA-256 values are in
`SHA256SUMS`.

Check the served model name, then send a warm-up request before timing:

```bash
curl -s http://127.0.0.1:8011/v1/models        # "name": "torchcast-decision-31b-v1.0"
curl -s http://127.0.0.1:8011/health           # "fast_kernels": true, "max_length": 262144
curl --fail http://127.0.0.1:8011/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"A customer reports two charges for one order.","questions":{"decision":{"type":"choice","instructions":"Choose the support team.","criteria":{"billing":"Payments and refunds","technical":"Product malfunctions","other":"Other requests"}}}}'
```

For JevBench, use the `typesafe` adapter against `http://127.0.0.1:8011` with model name `torchcast-decision-31b-v1.0`.
The runner requires the output paths to be outside the checkout:

```bash
J=datasets/public
python -m jevbench.cli run \
  --tasks $J/easy.jsonl,$J/original.jsonl,$J/hard.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:8011 --key-env "" \
  --model torchcast-decision-31b-v1.0 --cost-basis self_hosted_gpu --reserve-usd 0 \
  --results ../runs/results.jsonl --raw-dir ../runs/raw --ledger ../runs/ledger.jsonl \
  --manifest ../runs/manifest.json --delay-s 0
```

Prompts longer than 16,384 tokens keep the head and the tail of the state. Requests are served one at a time. Choice
questions take 2 or more options (more than 26 are decided in groups). Score questions take 2 to 10 levels; others are
answered with HTTP 422. See the weights repository's `NOTICE`.

The server listens on 127.0.0.1 by default. If the benchmark client runs on another machine, pass `--host 0.0.0.0`
and restrict access at the network level. This checkpoint has no published provider tariff; apply the benchmark's
documented estimated-cost method.

## Numerical reproducibility

Repeated reads with this release on one NVIDIA H100 80 GB (CUDA 13.0, torch 2.13.0, transformers 5.17.0, BF16) give
identical probabilities: re-reads of the same items on different pods matched to every reported digit. Different GPU
models or software versions use different kernels and can give slightly different probabilities for the same request;
compare runs on the same GPU type and software.

## Licence

Model weights: Apache License 2.0. Base model: Gemma-4-31B-it by Google, Apache License 2.0. Serving package: Apache
License 2.0. See the licence files in the weights repository. This project is independent of TypeSafe AI and the
JevBench maintainers.
