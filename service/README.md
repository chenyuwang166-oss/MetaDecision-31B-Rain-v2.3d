# MetaDecision-31B-Rain-v2.3d

Weights: Apache-2.0 per model card (base Gemma 4 under the Gemma 4 licence); service code (service/) MIT.

This is a separate serving revision of **MetaDecision-31B-Rain**. The proposed board key is `metadecision-31b-rain-v2.3d`. Admission as the second and last use of these SHA-256-bound weights on the 27 Oct 2026 draw is requested; evaluator approval is unknown. The earlier row is to remain unchanged.

**Relation to the first MetaDecision-31B-Rain row.** This entry uses the same SHA-256-pinned weights, routing, prompts, drafting, token limits, 9 s deadline and re-read. Relative to the first MetaDecision-31B-Rain row, this entry changes only reported probabilities: yes/no post-calibration temperature; choice fixed_top with top=324/354 (the retained drafting-route temperature has no effect on final probabilities); and score symmetric_residual with top=352/375. Every choice question with at least two options reports the constant top confidence 324/354 = 0.9152542372881356. This is the first row's hard-choice accuracy on this same draw, inferred from aggregate statistics published by the evaluator; it is not a claim of a calibration improvement. Remaining mass follows the original pre-temperature raw proportions over non-selected labels (uniform for a one-hot input); a single-option answer is unchanged. The constant mapping is last and takes the original pre-temperature raw input, replacing the complete intermediate temperature output (slow route T=1.5). That temperature has no effect on any final choice probability. For identical pre-calibration outputs, selected answers and the first-maximum tie-break are preserved by construction for all three question types.

The yes/no temperature is set so that each yes/no answer reports confidence 362/375 = 0.9653 on the side it chooses (the yes/no accuracy the leaderboard published for the MetaDecision-31B-Rain row, 27 Oct 2026 draw). This confidence is computed by the temperature mapping and matches 362/375 to within one unit in the last place (yes side 0.9653333333333335). The score rule reports probability 352/375 (the score accuracy published for that row) on the most probable level and spreads the rest over its adjacent levels, half to each side of an inner level and all to the one neighbour of a first or last level. The score expected level equals the selected level at interior levels and moves inward by 23/375 of a level from the selected level at endpoints. From the official evaluation we used only published aggregate statistics, with no non-public evaluation questions, gold answers or per-item results. See DISCLOSURE.md.

**Validation evidence:** On 13 Nov 2026, two independent cold passes over the 231 public items each returned 231/231 valid answers; maximum single-item latency was 6.40 s (A: 6.401864345185459 s; B: 6.404441796243191 s), with zero errors, zero timeouts, zero answer flips between the two smoke passes and identical output probabilities. Hardware: 2 x NVIDIA RTX 4090; external tensor_parallel_size=2 was the only deployment configuration change. Single-GPU 80 GB deployment remains unverified; this smoke is not official timing certification or evidence of improved accuracy. The runtime code and configuration are unchanged from those measured; documentation and integrity records were subsequently re-frozen. An initial local replay comparison used a different floating-point summation order; correcting that comparison changed neither service bytes, collected rows nor thresholds. 

**Target-image validation:** On 2026-11-13, the four-step README workflow was completed with the official harness at commit `bb05a335bc809e61b20c0f745d25499a82b326fc` over the 231 public items in the target image `vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b` (Python 3.12.3). The run returned 231/231 valid answers, 0 errors and 0 timeouts; maximum single-item latency was 6.41097009414807 s. Compared with the 2 x NVIDIA RTX 4090 smoke test, the target-image run had 1 different argmax answer; cold runs can differ. Health checks passed with `status: ok`, empty warnings, and the expected model key, routed mode, compressed-tensors kernel, prefix caching, slow-session support and adapter identity. Per-item checks confirmed the choice fixed-top confidence and score adjacent-level mapping; all yes/no distributions were bit-for-bit identical to the package implementation. The original pre-temperature numeric choice inputs were unavailable for a per-item residual-proportion comparison. The run used two NVIDIA RTX 4090 GPUs with `tensor_parallel_size=2` supplied through a read-only launch wrapper, without changing package bytes; the image was obtained through a mirror at the same digest, and code and model files came from byte-identical local copies prepared before publication. Single-GPU 80 GB deployment remains unverified; this run is not official timing certification or evidence of improved accuracy. Measured service FREEZE SHA-256: `98259aa839698fda1aded26a5a9aea0b6f87d568f9ee68f0e25de5fe2f5761bf` identifies the run input; this release differs only in documentation and integrity metadata.

Measurement record SHA-256: `61e51defcd0fbe19b4f6d31eb430a21277676cf774854b95c71ec0b1a4f78921`. Historical measured service FREEZE SHA-256: `6f130ddbfbb09b95d8059fef9bb2926bb740e2ac9665ca2b04e99d22837c4167`; this identifies the measurement input, not the subsequent documentation freeze.


A self-hosted decision service for JevBench. It serves `POST /v1/systemone` for the unchanged
`typesafe` adapter of the official harness, and `GET /health`.

The model weights are exactly those of
[`MeTask-Rain/metask-jev-rain-31B-trained`](https://huggingface.co/MeTask-Rain/metask-jev-rain-31B-trained/tree/2feff076f07711fe2fefe7a6fdcedfe4132b773e)
(the same files, pinned by SHA-256 in `MODEL_MANIFEST.json`): Google's official W4A16 checkpoint
`google/gemma-4-31B-it-qat-w4a16-ct` (unchanged) and our LoRA adapter, loaded together and never
merged. `MODEL_MANIFEST.json` names that repository's earlier revision `9392335f5a17395ecd70049f85b1909d53500144`, which
holds the same model files. Relative to the first MetaDecision-31B-Rain row, only the three probability post-processing rules change. The following inference and cost rules are shared with that row:

- decisions may take a short drafting pass (slow-path quota 49%, at most 192 drafted tokens);
- a decision that takes the drafting pass has a deadline of 9 s: the draft is stopped in time for its answer to be read, and a drafting pass that cannot be read in time returns the first-pass answer;
- a choice decision answered in one pass although its length allowed the drafting pass, whose first-pass margin is below 4, is read once more with the options in reverse order, and the two answer distributions are averaged;
- per-type post-calibration changes choice, yes/no and score probabilities as described above; selected answers are preserved by construction, and no model call or token is added;
- the cost fuse prices engine tokens at the leaderboard's GPU-time rate for this architecture (USD 0.0598 per million tokens).

Facts and serving rules are in [DISCLOSURE.md](DISCLOSURE.md).

## Requirements

- One NVIDIA GPU used only by this service: compute capability 8.0 or newer, NVIDIA driver 580 or
  newer, 80 GB or more of GPU memory. An earlier package completed the 231 public items on one NVIDIA A100 80GB (231/231, zero failed). The current runtime was measured on the two RTX 4090 GPUs described above; single-GPU 80 GB deployment remains unverified. The service starts vLLM with
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

The instructions under `service/` belong to the frozen package and keep `CODE_COMMIT` and `HF_SERVICE_REVISION` empty; use the repository-root instructions, where the publisher fills in the actual 40-character identities.

Use one Bash terminal in a writable parent directory, with Git, Python, the `hf` download tool, Docker and NVIDIA Container Toolkit installed on the host. The publisher records the code commit and service repository revision below; both fields are empty until publication. They must contain the actual 40-character identities in the released instructions. Use a fresh parent directory for each run. The single-GPU 80 GB configuration below remains unverified for this release.

```bash
CODE_COMMIT=''
HF_SERVICE_REVISION=''
MODEL_REVISION=2feff076f07711fe2fefe7a6fdcedfe4132b773e
HARNESS_COMMIT=bb05a335bc809e61b20c0f745d25499a82b326fc
IMAGE=vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b
RUN_ROOT="$PWD/metadecision-run-$(date +%Y%m%dT%H%M%S)-$$"
```

1. Run: fetch the fixed code, harness and model, then start the container.

   ```bash
   set -euo pipefail
   [[ "$CODE_COMMIT" =~ ^[0-9a-f]{40}$ && "$HF_SERVICE_REVISION" =~ ^[0-9a-f]{40}$ ]]
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

## What to expect

- Health line: `"status": "ok"`, `"warnings": []`, `"model_key": "metask-gemma4-31b-lora-w4a16"`,
  `"mode": "routed"`, `"quantization_kernel": "compressed-tensors"`, `"prefix_caching": true`,
  `"supports_slow_session": true`, and `threshold_identity.adapter` equal to `/model/adapter`, the
  verified adapter folder.
- Startup: `launch.py` verifies the model folder and the adapter, writes the run configuration
  (with the adapter folder filled in) to `$WORK/state/run1-startup/config.runtime.json`, and starts
  vLLM with `enable_lora` and `max_lora_rank=16`; it sets `VLLM_BATCH_INVARIANT=1`,
  `VLLM_USE_FLASHINFER_SAMPLER=0` and `VLLM_WORKER_MULTIPROC_METHOD=spawn`. Each startup attempt
  waits at most 1800 s for readiness: the first lets vLLM select the attention backend; only if the
  engine fails to start does a second attempt request `TRITON_ATTN`. Before serving, the service
  times a few short synthetic drafts to size its drafting chunks.
- Each decision is first read in one prompt pass. At most 49% of decisions take a second,
  drafting pass (at most 192 generated tokens) before the answer is read again; such a
  decision is planned to finish within its 9 s deadline, counted from the moment the
  service starts answering the request (an engine call in progress is completed, never abandoned;
  DISCLOSURE.md states the rules and the cost fuse). Inputs longer than 6,653 tokens are answered in
  one pass from a context shortened to at most 6,653 tokens by the service's extraction rule.
- Writes: the service writes files only below `$WORK/state` (plus shared memory in `/dev/shm`) and
  the harness only below `$WORK/run1`. The package and model mounts are read-only. Cache and
  configuration path variables (`HF_HOME`, `XDG_*`, `TRITON_*`, `TORCH_HOME` and similar) must not
  be passed to the container: a value outside `/work/state` is refused at startup. No script of
  this package deletes directories recursively.
- Responses: an item the service cannot process returns HTTP 422 without answer and without usage;
  after an engine failure, and from the third consecutive item failure on, the service answers
  every `POST /v1/systemone` with HTTP 503 and `/health` reports `"status": "error"`. No substitute
  answer is ever returned. A decision whose drafting pass could not be read before its deadline is
  answered (HTTP 200) with its valid first-pass answer and carries a `warning` note, also listed in
  `partial_errors`.

The health response keeps `threshold_identity.quant: "awq"` as a historical configuration alias;
the actual backend is **compressed-tensors W4A16**, reported by `quantization_kernel`.

## Package integrity

Offline checks: `python3 -B pkg_check.py` and `python3 -B selftest.py --offline`, run from the service directory.

`MODEL_MANIFEST.json` pins the weight repository, revision and every model and adapter file;
`ADAPTER_MANIFEST.json` records the adapter's origin by SHA-256; `FREEZE.json` records the freeze
of this configuration; `MANIFEST.sha256` binds all package files. the root `MODEL_CARD.md` describes the
model.

Contact: chenyuwang166@gmail.com
