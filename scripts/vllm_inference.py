import os

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from infreqact.data.utils import load_test_omnifall_dataset

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # qwen_vl_utils 0.0.14+ reqired
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    print(f"video_kwargs: {video_kwargs}")

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {"prompt": text, "multi_modal_data": mm_data, "mm_processor_kwargs": video_kwargs}


def create_sample_prompts(dataset, n_samples=5):
    # loads the omnifall test set and creates prompts with video samples
    prompt = "Analyze the video and describe the action happening in it."
    messages = []
    for i in range(n_samples):
        video_sample = dataset[i]["video"]
        # convert to PIL image
        video_sample_pil = [Image.fromarray(frame) for frame in video_sample]
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_sample_pil,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        )
    return messages


if __name__ == "__main__":
    dataset = load_test_omnifall_dataset()
    messages = create_sample_prompts(n_samples=1, dataset=dataset)
    checkpoint_path = "Qwen/Qwen3-VL-4B-Instruct"
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    # Fix: messages is already a list of message dicts, iterate over it directly
    inputs = [prepare_inputs_for_vllm(message, processor) for message in messages]

    # Get GPU count, but use 1 for single GPU or debugging
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")

    llm = LLM(
        model=checkpoint_path,
        mm_encoder_tp_mode="data",
        tensor_parallel_size=1,
        seed=0,
        dtype=torch.bfloat16,
        # Add max_num_batched_tokens to prevent OOM
        max_num_batched_tokens=4096,
        # Reduce max model length if still running into issues
        max_model_len=8192,
        mm_processor_kwargs={"min_pixels": 16 * 32 * 32, "max_pixels": 400 * 32 * 32},
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1024,
        top_k=-1,
        stop_token_ids=[],
    )

    for i, input_ in enumerate(inputs):
        print()
        print("=" * 40)
        print(f"Inputs[{i}]: {input_['prompt']=!r}")
    print("\n" + ">" * 40)

    outputs = llm.generate(inputs, sampling_params=sampling_params)
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        print()
        print("=" * 40)
        print(f"Generated text: {generated_text!r}")
