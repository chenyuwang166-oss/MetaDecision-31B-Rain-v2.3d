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

Decision service for JevBench. Base model:
[`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it), used through Google's
official quantization-aware-training (QAT) releases of it. The model has two parts that are loaded
together and **not merged**; both are the files of
[`MeTask-Rain/metask-jev-rain-31B-trained`](https://huggingface.co/MeTask-Rain/metask-jev-rain-31B-trained/tree/2feff076f07711fe2fefe7a6fdcedfe4132b773e),
unchanged and pinned by SHA-256 in `MODEL_MANIFEST.json` of the service code.

| Part | Location in that model repository | What it is |
|---|---|---|
| Base | repository root | The files of [`google/gemma-4-31B-it-qat-w4a16-ct`](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct) at revision `52f3f65bc7a02d555763bc923bd1d9094898219d`, byte for byte (Google's official W4A16 checkpoint). |
| Adapter | `adapter/` | Our LoRA adapter (PEFT format), trained by us on [`google/gemma-4-31B-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized) at revision `1e4d8beecacb8b7590c1d8bedd7335f687bf311f`. |

## Serving

The service code (`service/` in the Hugging Face repository `MeTask-Rain/MetaDecision-31B-Rain-v2.3d`,
mirrored at https://github.com/chenyuwang166-oss/MetaDecision-31B-Rain-v2.3d) serves the model with
vLLM 0.31.0, with the adapter applied to every request.

Equivalent vLLM usage of the model:

```python
from vllm import LLM
from vllm.lora.request import LoRARequest
llm = LLM(model=MODEL_DIR, dtype="bfloat16", enable_lora=True, max_lora_rank=16)
adapter = LoRARequest("adapter", 1, MODEL_DIR + "/adapter")
# llm.generate(prompts, sampling_params, lora_request=adapter)
```

Data and attribution are in `DISCLOSURE.md` of the service code (`service/DISCLOSURE.md` in the
Hugging Face repository).
