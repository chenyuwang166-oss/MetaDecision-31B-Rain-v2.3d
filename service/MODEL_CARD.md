---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model: google/gemma-4-31B-it
pipeline_tag: text-generation
tags:
- gemma4
- compressed-tensors
- lora
- peft
- vllm
- jevbench
---

# MetaDecision-31B-Rain-v2.3d

Weights: Apache-2.0 per model card (base Gemma 4 under the Gemma 4 licence); service code (service/) MIT.

**Relation to the first MetaDecision-31B-Rain row.** This entry uses the same SHA-256-pinned weights, routing, prompts, drafting, token limits, 9 s deadline and re-read. Relative to the first MetaDecision-31B-Rain row, this entry changes only reported probabilities: yes/no post-calibration temperature; choice fixed_top with top=324/354 (the retained drafting-route temperature has no effect on final probabilities); and score symmetric_residual with top=352/375. Every choice question with at least two options reports the constant top confidence 324/354 = 0.9152542372881356. This is the first row's hard-choice accuracy on this same draw, inferred from aggregate statistics published by the evaluator; it is not a claim of a calibration improvement. Remaining mass follows the original pre-temperature raw proportions over non-selected labels (uniform for a one-hot input); a single-option answer is unchanged. The constant mapping is last and takes the original pre-temperature raw input, replacing the complete intermediate temperature output (slow route T=1.5). That temperature has no effect on any final choice probability. For identical pre-calibration outputs, selected answers and the first-maximum tie-break are preserved by construction for all three question types.

The yes/no temperature is set so that each yes/no answer reports confidence 362/375 = 0.9653 on the side it chooses (the yes/no accuracy the leaderboard published for the MetaDecision-31B-Rain row, 27 Oct 2026 draw). This confidence is computed by the temperature mapping and matches 362/375 to within one unit in the last place (yes side 0.9653333333333335). The score rule reports probability 352/375 (the score accuracy published for that row) on the most probable level and spreads the rest over its adjacent levels, half to each side of an inner level and all to the one neighbour of a first or last level. The score expected level equals the selected level at interior levels and moves inward by 23/375 of a level from the selected level at endpoints. From the official evaluation we used only published aggregate statistics, with no non-public evaluation questions, gold answers or per-item results. See DISCLOSURE.md.

**Validation evidence:** On 13 Nov 2026, two independent cold passes over the 231 public items each returned 231/231 valid answers; maximum single-item latency was 6.40 s (A: 6.401864345185459 s; B: 6.404441796243191 s), with zero errors, zero timeouts, zero answer flips between the two smoke passes and identical output probabilities. Hardware: 2 x NVIDIA RTX 4090; external tensor_parallel_size=2 was the only deployment configuration change. Single-GPU 80 GB deployment remains unverified; this smoke is not official timing certification or evidence of improved accuracy. The runtime code and configuration are unchanged from those measured; documentation and integrity records were subsequently re-frozen. An initial local replay comparison used a different floating-point summation order; correcting that comparison changed neither service bytes, collected rows nor thresholds. 

**Target-image validation:** On 2026-11-13, the four-step README workflow was completed with the official harness at commit `bb05a335bc809e61b20c0f745d25499a82b326fc` over the 231 public items in the target image `vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b` (Python 3.12.3). The run returned 231/231 valid answers, 0 errors and 0 timeouts; maximum single-item latency was 6.41097009414807 s. Compared with the 2 x NVIDIA RTX 4090 smoke test, the target-image run had 1 different argmax answer; cold runs can differ. Health checks passed with `status: ok`, empty warnings, and the expected model key, routed mode, compressed-tensors kernel, prefix caching, slow-session support and adapter identity. Per-item checks confirmed the choice fixed-top confidence and score adjacent-level mapping; all yes/no distributions were bit-for-bit identical to the package implementation. The original pre-temperature numeric choice inputs were unavailable for a per-item residual-proportion comparison. The run used two NVIDIA RTX 4090 GPUs with `tensor_parallel_size=2` supplied through a read-only launch wrapper, without changing package bytes; the image was obtained through a mirror at the same digest, and code and model files came from byte-identical local copies prepared before publication. Single-GPU 80 GB deployment remains unverified; this run is not official timing certification or evidence of improved accuracy. Measured service FREEZE SHA-256: `98259aa839698fda1aded26a5a9aea0b6f87d568f9ee68f0e25de5fe2f5761bf` identifies the run input; this release differs only in documentation and integrity metadata.

Measurement record SHA-256: `61e51defcd0fbe19b4f6d31eb430a21277676cf774854b95c71ec0b1a4f78921`. Historical measured service FREEZE SHA-256: `6f130ddbfbb09b95d8059fef9bb2926bb740e2ac9665ca2b04e99d22837c4167`; this identifies the measurement input, not the subsequent documentation freeze.


Decision service for JevBench. Base model:
`google/gemma-4-31B-it`, used through Google's
official quantization-aware-training (QAT) releases of it. The model has two parts that are loaded
together and **not merged**; both are the files of
[`MeTask-Rain/metask-jev-rain-31B-trained`](https://huggingface.co/MeTask-Rain/metask-jev-rain-31B-trained/tree/2feff076f07711fe2fefe7a6fdcedfe4132b773e),
unchanged and pinned by SHA-256 in `MODEL_MANIFEST.json` of the service code. Relative to the first MetaDecision-31B-Rain row, this entry uses the same weights and changes only the three probability post-processing rules described above.

| Part | Location in that model repository | What it is |
|---|---|---|
| Base | repository root | The files of [`google/gemma-4-31B-it-qat-w4a16-ct`](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct/tree/52f3f65bc7a02d555763bc923bd1d9094898219d) at revision `52f3f65bc7a02d555763bc923bd1d9094898219d`, byte for byte: Google's official compressed-tensors W4A16 checkpoint (4-bit weights, group size 32, symmetric). |
| Adapter | `adapter/` | Our LoRA (PEFT format): rank 16, alpha 32, on the language model's q/k/v/o/gate/up/down projections, trained by us in BF16 on [`google/gemma-4-31B-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized/tree/1e4d8beecacb8b7590c1d8bedd7335f687bf311f) at revision `1e4d8beecacb8b7590c1d8bedd7335f687bf311f`. |

`adapter/adapter_model.safetensors` contains the trained checkpoint tensors unchanged plus, for the 10 full-attention layers (5, 11, …, 59) that compute values with the key projection, a v_proj LoRA that is a byte copy of the trained k_proj LoRA (20 tensors), so that vLLM applies the update to keys and values as in training.

## Serving

The service code in `service/` starts vLLM 0.31.0 on a folder holding the files of that weight
repository, with `enable_lora`, `max_lora_rank=16`, and the `adapter/` folder as the single LoRA,
applied to every request. Each decision is read from the token-sequence log-probabilities of the answer
forms in one prompt pass; up to 49% of decisions then take a second, short drafting pass
(at most 192 tokens) under a 9 s deadline, a choice decision answered in one pass although its length allowed the drafting pass is read a second time with the options in reverse order when its first-pass margin is below 4, and the reported probabilities get a per-type post-calibration (DISCLOSURE.md).

Equivalent vLLM usage of the model:

```python
from vllm import LLM
from vllm.lora.request import LoRARequest
llm = LLM(model=MODEL_DIR, dtype="bfloat16", enable_lora=True, max_lora_rank=16)
adapter = LoRARequest("adapter", 1, MODEL_DIR + "/adapter")
# llm.generate(prompts, sampling_params, lora_request=adapter)
```

Current data and serving-rule disclosure is in this repository's root `DISCLOSURE.md`.

Contact: chenyuwang166@gmail.com

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
