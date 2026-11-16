# MetaDecision-31B-Rain-v2.3d

A self-hosted decision service for JevBench. It serves `POST /v1/systemone` for the unchanged
`typesafe` adapter of the official harness, and `GET /health`.

The model weights are exactly those of
[`MeTask-Rain/metask-jev-rain-31B-trained`](https://huggingface.co/MeTask-Rain/metask-jev-rain-31B-trained/tree/2feff076f07711fe2fefe7a6fdcedfe4132b773e)
(the same files, pinned by SHA-256 in `MODEL_MANIFEST.json`): Google's official W4A16 checkpoint
`google/gemma-4-31B-it-qat-w4a16-ct` (unchanged) and our LoRA adapter, loaded together and never
merged. `MODEL_MANIFEST.json` names that repository's earlier revision `9392335f5a17395ecd70049f85b1909d53500144`, which
holds the same model files.

Data and attribution are in [DISCLOSURE.md](DISCLOSURE.md).

## Requirements

- One NVIDIA GPU used only by this service: compute capability 8.0 or newer, NVIDIA driver 580 or
  newer, 80 GB or more of GPU memory. Validated hardware: two NVIDIA RTX 4090 with `tensor_parallel_size=2`. Single-GPU 80 GB deployment remains unverified. The service starts vLLM with
  `gpu_memory_utilization=0.94` (it reserves 94% of the GPU memory), `max_model_len=6656`,
  `enable_lora` and `max_lora_rank=16`. The model files take 23.3 GB (base) plus 495 MB (adapter).
  The commands use the host's GPU 0 (`--gpus '"device=0"'`).
- Docker with the NVIDIA Container Toolkit.
- On the host: Git, Python and the Hugging Face command line tool `hf`.
- Runtime image `vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b`
  (Python 3.12.3, vLLM 0.31.0, torch 2.13.0+cu130), the linux/amd64 image of vLLM release
  `v0.31.0`. Nothing is installed and the service has no network access.
- Model folder: the model files of https://huggingface.co/MeTask-Rain/metask-jev-rain-31B-trained/tree/2feff076f07711fe2fefe7a6fdcedfe4132b773e
  as regular files (step 1). The repository root holds the official base files; `adapter/` holds
  the adapter (`adapter_config.json`, `adapter_model.safetensors`). A Hugging Face cache
  `snapshots/<revision>` directory holds links and cannot be mounted on its own. At startup the
  service checks the SHA-256 of the small model files, the size of the weight file and the full
  SHA-256 of both adapter files (`MODEL_MANIFEST.json`), and refuses other weight, tokenizer,
  processor or template files in the folder; it never writes to the folder.
- Code: `service/` from the exact `CODE_COMMIT` in the Run variables.
- Harness: https://github.com/fstandhartinger/jevbench/tree/bb05a335bc809e61b20c0f745d25499a82b326fc

## Run

Use one Bash terminal in a writable parent directory, with Git, Python, the `hf` download tool, Docker and NVIDIA Container Toolkit installed on the host. Use a fresh parent directory for each run. The single-GPU 80 GB configuration below remains unverified for this release.

```bash
CODE_COMMIT='d31313238e4cbe5584253971aec5370e4929a2a0'
HF_SERVICE_REVISION='0205a1c3994cf1201a4f8595621b2ee8bb2ea9d3'
MODEL_REVISION=2feff076f07711fe2fefe7a6fdcedfe4132b773e
HARNESS_COMMIT=bb05a335bc809e61b20c0f745d25499a82b326fc
IMAGE=vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b
RUN_ROOT="$PWD/metadecision-run-$(date +%Y%m%dT%H%M%S)-$$"
```

1. Run: fetch the fixed code, harness and model, then start the container.

   ```bash
   set -euo pipefail
   [[ "$CODE_COMMIT" =~ ^[[:digit:]a-f]{40}$ && "$HF_SERVICE_REVISION" =~ ^[[:digit:]a-f]{40}$ ]]
   mkdir "$RUN_ROOT"
   git clone https://github.com/chenyuwang166-oss/MetaDecision-31B-Rain-v2.3d.git "$RUN_ROOT/code"
   git -C "$RUN_ROOT/code" checkout --detach "$CODE_COMMIT"
   test "$(git -C "$RUN_ROOT/code" rev-parse HEAD)" = "$CODE_COMMIT"
   git clone https://github.com/fstandhartinger/jevbench.git "$RUN_ROOT/jevbench"
   git -C "$RUN_ROOT/jevbench" checkout --detach "$HARNESS_COMMIT"
   PKG="$RUN_ROOT/code/service"; MODEL="$RUN_ROOT/model"; WORK="$RUN_ROOT/work"; JEVBENCH="$RUN_ROOT/jevbench"
   mkdir "$MODEL" "$WORK"
   hf download MeTask-Rain/metask-jev-rain-31B-trained --revision "$MODEL_REVISION" --local-dir "$MODEL" --exclude 'service/*'
   docker pull "$IMAGE"
   docker run -d --name metadecision31b --init --gpus '"device=0"' --network none --read-only --shm-size 8g \
     -v "$PKG":/pkg:ro -v "$MODEL":/model:ro -v "$WORK":/work -v "$JEVBENCH":/jevbench:ro \
     --entrypoint bash "$IMAGE" /pkg/serve.sh --model /model --state-dir /work/state --run-id run1
   ```

2. Logs and health: follow the log until a line starting with `{"status": "ok"` appears, press Ctrl-C to end the log display, then require a healthy response.

   ```bash
   docker logs -f metadecision31b || { log_exit=$?; test "$log_exit" -eq 130; }
   docker exec metadecision31b python3 -c 'import json,urllib.request; h=json.load(urllib.request.urlopen("http://localhost:8000/health")); print(json.dumps(h)); assert h["status"] == "ok" and not h["warnings"]'
   ```

   If startup exits, read `docker logs metadecision31b` and `$WORK/state/run1-startup/attempt-*.log`, then perform step 4. After correcting the cause, restart with a fresh run directory.

3. Exec harness: run the official harness in the same container and summarize its results.

   ```bash
   TASKS=/jevbench/datasets/public/easy.jsonl,/jevbench/datasets/public/original.jsonl,/jevbench/datasets/public/hard.jsonl
   docker exec -e PYTHONPATH=/jevbench metadecision31b python3 -B -m jevbench.cli run --adapter typesafe \
     --endpoint http://localhost:8000 --model MetaDecision-31B-Rain-v2.3d --key-env '' --tasks "$TASKS" \
     --results /work/run1/results.jsonl --ledger /work/run1/ledger.jsonl --raw-dir /work/run1/raw \
     --manifest /work/run1/manifest.json --cap-usd 15 --price-in-per-m 0.059819178473256074 --price-out-per-m 0.059819178473256074
   docker exec -e PYTHONPATH=/jevbench metadecision31b python3 -B -m jevbench.cli summarize --tasks "$TASKS" \
     --results /work/run1/results.jsonl --ledger /work/run1/ledger.jsonl
   ```

4. Stop and remove the container.

   ```bash
   docker stop -t 40 metadecision31b && docker rm metadecision31b
   ```

The service writes only below `$WORK/state` (plus `/dev/shm`); the harness writes below `$WORK/run1`. The package and model mounts are read-only. The runtime container installs nothing and has no network. Each new evaluation needs a new run directory and container; a reused state run is refused.

## Service checks

`GET /health` must report `status: ok` and empty warnings. Inspect service warnings and investigate HTTP 422/503 or health errors before continuing.

Run `python3 -B pkg_check.py` and `python3 -B selftest.py --offline` from the service directory before deployment.
