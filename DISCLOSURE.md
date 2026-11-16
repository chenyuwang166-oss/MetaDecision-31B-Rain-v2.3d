# MetaDecision-31B-Rain-v2.3d: disclosure

## Model

- Base model: `google/gemma-4-31B-it`, used through Google's official quantization-aware-training
  (QAT) releases of it:
  - training base: `google/gemma-4-31B-it-qat-q4_0-unquantized` at revision
    `1e4d8beecacb8b7590c1d8bedd7335f687bf311f`;
  - served base: `google/gemma-4-31B-it-qat-w4a16-ct` at revision
    `52f3f65bc7a02d555763bc923bd1d9094898219d`, used unchanged.
- Own trained weights: the LoRA adapter in `adapter/` (PEFT format). It is loaded on top of the
  served base and is not merged.
- The base files and the adapter are exactly those of `MeTask-Rain/metask-jev-rain-31B-trained`
  (same SHA-256 for every file, `MODEL_MANIFEST.json`); no weight was trained or changed for this
  entry.

## Data

- The adapter is the one of `MeTask-Rain/metask-jev-rain-31B-trained`, trained on our own generated
  data as disclosed for that entry.
- Some serving parameters were fitted or chosen using the 231 public JevBench items; fits on these
  items are in-sample. The other serving parameters were fitted on our own development data.

## Usage accounting

- The token usage reported for each decision is the engine's measured token count (cached prefix
  tokens excluded) and includes every engine call made for that decision.
