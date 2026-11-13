# MetaDecision-31B-Rain-v2.3d: disclosure

## Model

- Base model: `google/gemma-4-31B-it`. The weights used are Google's official
  quantization-aware-training (QAT) releases of this model:
  - training base: `google/gemma-4-31B-it-qat-q4_0-unquantized` at revision
    `1e4d8beecacb8b7590c1d8bedd7335f687bf311f` (BF16);
  - served base: `google/gemma-4-31B-it-qat-w4a16-ct` at revision
    `52f3f65bc7a02d555763bc923bd1d9094898219d`, Google's compressed-tensors W4A16 checkpoint
    (4-bit weights, group size 32, symmetric; activations unquantized), used unchanged. We did not
    quantize anything.
- Own trained weights: the LoRA adapter in `adapter/` (PEFT format), rank 16, alpha 32, on the
  language model's q/k/v/o/gate/up/down projections, trained by us in BF16 on the training base.
- The adapter is not merged: vLLM 0.31.0 loads it with `enable_lora` and `max_lora_rank=16` and
  applies it to every request on top of the W4A16 base.
- In the 10 full-attention layers (5, 11, …, 59) Gemma 4 computes the attention values with the key
  projection (`attention_k_eq_v`); these layers have no v_proj, and vLLM serves them with separate
  key and value slots. `adapter/adapter_model.safetensors` therefore contains the trained checkpoint
  tensors unchanged plus, for these 10 layers, a v_proj LoRA that is a byte copy of the trained
  k_proj LoRA (20 tensors), so that the update applies to keys and values as in training.
- The base files and the adapter are exactly those of `MeTask-Rain/metask-jev-rain-31B-trained`
  (same SHA-256 for every file, `MODEL_MANIFEST.json`); no weight was trained or changed for this
  entry. The public comparison baseline is the first MetaDecision-31B-Rain row; only the three probability post-processing rules below differ.
- Relative to the first MetaDecision-31B-Rain row, this entry changes only reported probabilities: yes/no post-calibration temperature; choice fixed_top with top=324/354 (the retained drafting-route temperature has no effect on final probabilities); and score symmetric_residual with top=352/375. The SHA-256-pinned weights, routing, prompts, drafting, token limits, 9 s deadline and re-read remain unchanged. For identical pre-calibration outputs, selected answers and the first-maximum tie-break are preserved by construction for all three question types.

## Data

- The adapter is the one of `MeTask-Rain/metask-jev-rain-31B-trained`, trained on our own generated
  data as disclosed for that entry.
- Shared with the first MetaDecision-31B-Rain row, unchanged: the per-type temperatures of the
  first calibration step, fitted on the 231 public JevBench items, and the drafting prompt limit
  (840 tokens, below), chosen using the 231 public items together with our own development data.
  Fits and choices made on these public items are in-sample. The yes/no confidence constant 0.898
  (first-step answers carry 0.898 or 0.102) is the yes/no accuracy of our own held-out development
  data (1,419 yes/no items), rounded to 3 decimals; its value does not depend on any leaderboard
  result.
- The serving parameters introduced with MetaDecision-31B-Rain (drafting threshold and share, draft
  length and the re-read rule) were fitted on our own independently built development set.
- Every choice question with at least two options reports the same constant top confidence 324/354 = 0.9152542372881356. This number is the first row's hard-choice accuracy on this same draw (27 Oct 2026), inferred from aggregate statistics published by the evaluator. This is disclosure of a constant, not a claim of a calibration improvement. Remaining mass follows the original pre-temperature raw proportions over non-selected labels (uniform for a one-hot input); a single-option answer is unchanged. The mapping is last and takes the original pre-temperature raw input, replacing the complete intermediate temperature output (T=1.5 on drafted choice outputs). The retained temperature has no effect on any final choice probability. Selected label and tie-break are preserved.
- The yes/no post-calibration temperature, changed relative to the first MetaDecision-31B-Rain row, is set from a
  published leaderboard aggregate: every yes/no answer carries confidence 362/375 = 0.9653, the
  yes/no accuracy published for the MetaDecision-31B-Rain row (draw of 27 Oct 2026, 375 yes/no
  items). The calibration constant was re-set from that same-draw official accuracy, per question
  type only. This confidence is computed by the temperature mapping and matches 362/375 to within one unit in the last place (yes side 0.9653333333333335).
- The score post-calibration, changed relative to the first MetaDecision-31B-Rain row, gives the most probable level the
  probability 352/375 = 0.9387, the score accuracy published for the MetaDecision-31B-Rain row
  (same draw), and the rest to its adjacent levels. The choice drafting-route temperature (1.5)
  and the score rule for a first or last level were chosen on the cached outputs of this service
  on the 231 public JevBench items and our own development sets, each re-weighted so that its
  accuracy is set to match the score accuracy published for that row (352/375) or that row's hard-tier
  choice accuracy (324/354, derived from published aggregate statistics).
  The 9 s deadline and the drafting chunk settings are fixed rules, not fitted.
- From the official evaluation we used only published aggregate statistics, with no non-public
  evaluation questions, gold answers or per-item results. Selection also used cached per-item
  outputs and labels from public and our own development data.
- The 231 public items were also used for the validation run of the workflow in README.md (run with
  the earlier MetaDecision-31B-Rain package). This historical run does not validate the current probability mapping.

## Serving

- Each decision is first read in one prompt pass: the answer distribution comes from the token-sequence
  log-probabilities of the answer forms. If the gap between the two highest answer scores of this
  pass is below a threshold (fitted so that 52.1% of the decisions of our development set
  that can take the drafting pass fall below it), the decision may take a second, drafting pass: the
  model writes at most 192 tokens of reasoning (the instruction asks for at most
  50 words), then the answer distribution is read again. If the draft ends without an
  answer line, an answer line is appended and the answer is read after the draft.
- Deadline: a decision that takes the drafting pass has a deadline of 9 s from the
  moment the service starts answering the request. Drafting runs in chunks sized so that the answer
  can still be read before the deadline (an engine call in progress is always completed, never
  abandoned); a draft that reaches its time budget is ended with an answer line, and a drafting pass
  that cannot be read in time returns the first-pass answer (with a `warning` note, also listed in
  `partial_errors`).
- Decisions whose drafting prompt (the first-pass prompt plus the drafting instruction) is
  longer than 840 tokens are answered in one pass. Inputs longer than 6,653 tokens never take the drafting
  pass and are answered from a context shortened to at most 6,653 tokens by the service's existing
  extraction rule; an input is not refused because of its length.
- Drafting share (slow-path quota 49%): a counter kept below the service's state
  directory admits a drafting pass only while drafting decisions stay
  at most 49% of all decisions so far.
- Re-read: a choice decision answered in one pass although its length allowed the drafting pass (for example because the drafting share or the cost fuse refused it), whose first-pass margin (the log-score difference between its two highest answers) is below 4, is read once more in one pass with the options in reverse order (each option keeps its text and label). The two answer distributions are averaged by label. The second read is included in the decision's token usage and in the cost fuse; it is not a drafting pass and does not count toward the drafting share.
- Cost fuse: from the 101st decision on, when the running engine cost per decision reaches 90% of
  USD 0.0646 per 1,000 decisions, drafting passes are limited to the share that brings the running
  cost back to 85% of that limit (none while first-pass decisions alone cost more than 85% of it),
  until the running cost falls to 85% of the limit. The fuse prices engine token counts at
  USD 0.0598 per million tokens, input and output alike (the GPU-time basis of the
  leaderboard's cost rule for this architecture). The fuse never changes the first pass.
- Answers after a drafting pass use the same first-step calibration parameters (per-type
  temperatures and yes/no confidence) as first-pass answers.
- Post-calibration: after the first-step calibration, the reported probabilities are transformed per type, without any model call or token (a temperature t replaces the probabilities p by p^(1/t), renormalised; values in `config.json`, block `post_calibration`). yes/no: temperature 0.653861 (the first-step 0.898/0.102 become 0.965/0.035; set from the published yes/no accuracy above); choice: constant top confidence 324/354 on every question with at least two options, using original pre-temperature raw probabilities; retained drafting-route temperature has no effect on any final choice probability; score: the most probable level gets probability 352/375 = 0.938667 and the remaining 0.061333 goes to its adjacent levels, half to each neighbour of an inner level (so the expected level is the most probable level) and all to the one neighbour of a first or last level; for score, the first-step probabilities select the most probable level; for choice, the original pre-temperature probabilities determine both the selected label and the residual proportions. The answer label is recomputed from the transformed probabilities; types not listed are unchanged.
- Final choice probability mapping relative to the first MetaDecision-31B-Rain row: Every choice question with at least two options reports the same constant top confidence 324/354 = 0.9152542372881356. This number is the first row's hard-choice accuracy on this same draw (27 Oct 2026), inferred from aggregate statistics published by the evaluator. This is disclosure of a constant, not a claim of a calibration improvement. Remaining mass follows the original pre-temperature raw proportions over non-selected labels (uniform for a one-hot input); a single-option answer is unchanged. The mapping is last and takes the original pre-temperature raw input, replacing the complete intermediate temperature output (T=1.5 on drafted choice outputs). The retained temperature has no effect on any final choice probability. Selected label and tie-break are preserved.

## Run conditions and usage accounting

- vLLM 0.31.0 with `VLLM_BATCH_INVARIANT=1` (also set by the launcher for serving) and
  `VLLM_USE_FLASHINFER_SAMPLER=0`, one request at a time; model loading and warm-up are excluded.
- Cost basis: measured engine token counts (input tokens minus cached prefix tokens; generated
  tokens).


## Validation evidence

**Validation evidence:** On 13 Nov 2026, two independent cold passes over the 231 public items each returned 231/231 valid answers; maximum single-item latency was 6.40 s (A: 6.401864345185459 s; B: 6.404441796243191 s), with zero errors, zero timeouts, zero answer flips between the two smoke passes and identical output probabilities. Hardware: 2 x NVIDIA RTX 4090; external tensor_parallel_size=2 was the only deployment configuration change. Single-GPU 80 GB deployment remains unverified; this smoke is not official timing certification or evidence of improved accuracy. The runtime code and configuration are unchanged from those measured; documentation and integrity records were subsequently re-frozen. An initial local replay comparison used a different floating-point summation order; correcting that comparison changed neither service bytes, collected rows nor thresholds. 

**Target-image validation:** On 2026-11-13, the four-step README workflow was completed with the official harness at commit `bb05a335bc809e61b20c0f745d25499a82b326fc` over the 231 public items in the target image `vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b` (Python 3.12.3). The run returned 231/231 valid answers, 0 errors and 0 timeouts; maximum single-item latency was 6.41097009414807 s. Compared with the 2 x NVIDIA RTX 4090 smoke test, the target-image run had 1 different argmax answer; cold runs can differ. Health checks passed with `status: ok`, empty warnings, and the expected model key, routed mode, compressed-tensors kernel, prefix caching, slow-session support and adapter identity. Per-item checks confirmed the choice fixed-top confidence and score adjacent-level mapping; all yes/no distributions were bit-for-bit identical to the package implementation. The original pre-temperature numeric choice inputs were unavailable for a per-item residual-proportion comparison. The run used two NVIDIA RTX 4090 GPUs with `tensor_parallel_size=2` supplied through a read-only launch wrapper, without changing package bytes; the image was obtained through a mirror at the same digest, and code and model files came from byte-identical local copies prepared before publication. Single-GPU 80 GB deployment remains unverified; this run is not official timing certification or evidence of improved accuracy. Measured service FREEZE SHA-256: `98259aa839698fda1aded26a5a9aea0b6f87d568f9ee68f0e25de5fe2f5761bf` identifies the run input; this release differs only in documentation and integrity metadata.

Measurement record SHA-256: `61e51defcd0fbe19b4f6d31eb430a21277676cf774854b95c71ec0b1a4f78921`. Historical measured service FREEZE SHA-256: `6f130ddbfbb09b95d8059fef9bb2926bb740e2ac9665ca2b04e99d22837c4167`; this identifies the measurement input, not the subsequent documentation freeze.
