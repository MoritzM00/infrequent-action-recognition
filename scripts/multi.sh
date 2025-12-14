#!/bin/bash

# hydra multirun does not work well with vLLM yet, start them separately

python scripts/vllm_inference.py model.params=2B experiment=zeroshot
python scripts/vllm_inference.py model.params=4B experiment=zeroshot
python scripts/vllm_inference.py model.params=8B experiment=zeroshot
python scripts/vllm_inference.py model.params=32B experiment=zeroshot

# mixture of expert
python scripts/vllm_inference.py model=qwen/moe experiment=zeroshot
