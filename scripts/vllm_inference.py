import os

import torch
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from infreqact.data.utils import load_test_omnifall_dataset
from infreqact.inference.base import parse_llm_outputs, prepare_inputs_for_vllm
from infreqact.inference.zeroshot import build_prompts
from infreqact.metrics.base import compute_metrics

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def main():
    dataset = load_test_omnifall_dataset()

    checkpoint_path = "Qwen/Qwen3-VL-4B-Instruct"
    processor = AutoProcessor.from_pretrained(checkpoint_path)

    # TODO: parallelize this, with DataLoader?
    messages, samples = build_prompts(n_samples=200, dataset=dataset)
    inputs = [prepare_inputs_for_vllm([message], processor) for message in messages]

    llm = LLM(
        model=checkpoint_path,
        tensor_parallel_size=2,
        mm_encoder_tp_mode="data",
        mm_processor_cache_gb=0,
        seed=0,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.9,
        mm_processor_kwargs={"min_pixels": 16 * 32 * 32, "max_pixels": 400 * 32 * 32},
        # max_num_batched_tokens=2048,
        # max_num_seqs=1,  # Process one video at a time
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=512,
        top_k=-1,
        stop_token_ids=[],
    )

    # TODO: iterate in batches (with message processing above)
    outputs = llm.generate(inputs, sampling_params=sampling_params)

    predictions, predicted_labels, true_labels = parse_llm_outputs(outputs, samples, verbose=5)

    with open("outputs/vllm_inference_predictions.json", "w") as f:
        import json

        json.dump(predictions, f, indent=4)

    # Compute comprehensive metrics
    print("\n" + ">" * 40)
    print("EVALUATION METRICS")
    print(">" * 40)

    metrics = compute_metrics(y_pred=predicted_labels, y_true=true_labels)
    with open("outputs/vllm_inference_metrics.json", "w") as f:
        import json

        json.dump(metrics, f, indent=4)

    # Print key metrics
    print("\n📊 Overall Performance:")
    print(f"  Accuracy:          {metrics['accuracy']:.3f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"  Macro F1:          {metrics['macro_f1']:.3f}")

    print("\n🚨 Fall Detection (Binary):")
    print(f"  Sensitivity:  {metrics['fall_sensitivity']:.3f}")
    print(f"  Specificity:  {metrics['fall_specificity']:.3f}")
    print(f"  F1 Score:     {metrics['fall_f1']:.3f}")

    print("\n🤕 Fallen Detection (Binary):")
    print(f"  Sensitivity:  {metrics['fallen_sensitivity']:.3f}")
    print(f"  Specificity:  {metrics['fallen_specificity']:.3f}")
    print(f"  F1 Score:     {metrics['fallen_f1']:.3f}")

    print("\n⚠️  Fall ∪ Fallen (Binary):")
    print(f"  Sensitivity:  {metrics['fall_union_fallen_sensitivity']:.3f}")
    print(f"  Specificity:  {metrics['fall_union_fallen_specificity']:.3f}")
    print(f"  F1 Score:     {metrics['fall_union_fallen_f1']:.3f}")

    # Print per-class F1 scores for classes present in the test set
    print("\n📈 Per-Class F1 Scores:")
    for key, value in sorted(metrics.items()):
        if key.endswith("_f1") and not key.startswith(("fall_", "fallen_", "fall_union")):
            class_name = key.replace("_f1", "")
            print(f"  {class_name:15s}: {value:.3f}")

    print("\n📦 Sample Counts:")
    print(f"  Total: {metrics['sample_count']}")
    for key, value in sorted(metrics.items()):
        if key.startswith("sample_count_"):
            class_name = key.replace("sample_count_", "")
            print(f"  {class_name:15s}: {value}")


if __name__ == "__main__":
    main()
