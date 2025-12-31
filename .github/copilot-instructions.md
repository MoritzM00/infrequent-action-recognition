# Copilot Instructions for Infrequent Action Recognition

## Project Overview

This is a research codebase for **fall detection and human activity recognition using Multimodal Large Language Models (MLLMs)**. The project evaluates vision-language models (primarily Qwen3-VL) on temporal action recognition with emphasis on infrequent events like falls.

**Core Architecture**: Video classification using vLLM for batched inference on temporal segmentation tasks. Videos are loaded in clips of 2-5 seconds at 8 FPS and annotated with a 16-class taxonomy (10 core actions + 6 extended). MLLMs are then prompted using different strategies (zero-shot, few-shot, chain-of-thought) to classify each clip.

**Implementation State**: Currently, only zero-shot inference is fully implemented and only Qwen3-VL is supported. It must be changed to support additional models and ICL paradigms. Later, training and fine-tuning pipelines will be added.

## Environment & Dependencies

### Setup Commands (Critical Order)
```bash
conda env create -f environment.yml && conda activate cuda-129
uv pip install vllm==0.12.0 --torch-backend=cu129
uv pip install psutil ninja
MAX_JOBS=4 uv pip install flash-attn==2.8.3 --no-build-isolation
uv pip install -r requirements.txt
```

**Key Dependencies**: Python 3.11, CUDA 12.9, PyTorch 2.9, vLLM 0.12, Transformers 4.57, Hydra 1.3

**Environment Variables Required**:
- `OMNIFALL_ROOT`: Path to Omnifall dataset root directory
- `WANFALL_ROOT`: Path to WanFall dataset root directory (optional)

## Project Structure

```
src/infreqact/          # Main package
├── inference/          # MLLM inference (base.py: HF models, zeroshot.py: prompts)
├── data/               # Dataset loaders (video_dataset.py, multi_video_dataset.py)
├── evaluation/         # Metrics computation (base.py) and visualization (visual.py)
├── metrics/            # Core metrics (compute_metrics function with 30+ metrics)
└── utils/              # Logging, wandb, latex table generation

config/                 # Hydra configuration hierarchy
├── inference_config.yaml
├── experiment/         # Experiment configs (debug.yaml, zeroshot.yaml)
├── dataset/            # Dataset configs (omnifall/, wanfall/, combined/)
├── model/qwen/         # Model configs (instruct.yaml)
├── vllm/               # vLLM settings (default.yaml)
└── sampling/           # Sampling parameters (greedy.yaml)

scripts/
└── vllm_inference.py   # Main inference script (@hydra.main entry point)
```

## Development Patterns

### 1. Hydra Configuration System

**Critical**: Use `@hydra.main` decorator with specific config structure:
```python
@hydra.main(version_base=None, config_path="../config", config_name="inference_config")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)  # ALWAYS resolve interpolations first
```

**Dataset Path Resolution**:
- Supports HuggingFace Hub: `hf://simplexsigil2/omnifall/labels/OOPS.csv`
- Supports env vars: `${oc.env:OMNIFALL_ROOT}/OOPS/video`
- See `src/infreqact/data/hf_utils.py::resolve_annotations_file()` for implementation

**Running with Hydra Overrides**:
```bash
python scripts/vllm_inference.py \
    -cn inference_config \
    experiment=zeroshot \
    batch_size=50 \
    num_samples=100
```

### 2. Action Recognition Taxonomy

**16-class system** (10 core + 6 extended WanFall classes):
**label2idx mapping** in `src/infreqact/data/video_dataset.py`

**Critical Distinction**: `fall` (uncontrolled descent) vs `sit_down`/`lie_down` (controlled). See `src/infreqact/inference/zeroshot.py::get_system_prompt()` for detailed definitions.

### 3. Metrics System

**Function**: `compute_metrics(y_pred, y_true)` in `src/infreqact/metrics/base.py`

**Returns 30+ metrics including**:
- Multi-class: `accuracy`, `balanced_accuracy`, `macro_f1`
- Binary fall detection: `fall_sensitivity`, `fall_specificity`, `fall_f1`
- Binary fallen detection: `fallen_sensitivity`, `fallen_specificity`, `fallen_f1`
- Union metrics: `fall_union_fallen_*` (treats fall OR fallen as positive)
- Per-class: `{class}_accuracy`, `{class}_precision`, `{class}_sensitivity`, `{class}_specificity`, `{class}_npv`, `{class}_f1`
- Distributions: `true_dist_{class}`, `pred_dist_{class}`, `sample_count_{class}`

**Flexible Input**: Accepts string labels (`["walk", "fall"]`) or numeric indices (`[0, 1]`)

### 4. vLLM Inference Pipeline

**Entry Point**: `scripts/vllm_inference.py`

**Key Functions**:
- `prepare_inputs_for_vllm()`: Converts messages to vLLM format using `qwen_vl_utils.process_vision_info`
- `parse_llm_outputs()`: JSON repair and label extraction (handles CoT reasoning)
- `collate_fn()`: Batches video samples with frame sampling at `model_fps`

**vLLM Configuration** (`config/vllm/default.yaml`):
```yaml
tensor_parallel_size: null  # Auto-detects GPUs
mm_encoder_tp_mode: all-to-all
mm_processor_cache_gb: 4.0
dtype: bfloat16
gpu_memory_utilization: 0.9
```

**Tensor Parallelism**: Set to `null` for automatic GPU detection, or specify explicit count.

### 5. Dataset Factory Pattern

**Factory**: `get_video_datasets()` in `src/infreqact/data/video_dataset_factory.py`

**Creates different dataset types**:
- `OmnifallVideoDataset`: Temporal segmentation (start/end times)
- `WanfallVideoDataset`: Full video classification with demographic metadata
- `MultiVideoDataset`: Combines multiple datasets with domain weighting

**Split Handling** (lines 85-100):
- Extracts split from `split_root` if contains `config=cs` or `config=cv`
- Validates split consistency (CRITICAL: fail fast on mismatch)
- Always includes split suffix in dataset key for evaluation groups

### 6. Weights & Biases Integration

**Initialization**: `initialize_run_from_config()` in `src/infreqact/utils/wandb.py`

**Pattern**:
```python
run = initialize_run_from_config(cfg)
reconfigure_logging_after_wandb(rich_handler, file_handler)  # Fix logging after W&B init
```

**Auto-naming**: Creates run name from `{model_name}-F{num_frames}@{fps} {unique_id}`

**Video Logging**: `log_videos_with_predictions()` logs sample predictions to W&B Media panel

### 7. Testing Strategy

**Test Location**: `tests/metrics/test_base.py`

**Coverage**:
- Handles string and numeric labels
- Tests binary fall/fallen metrics
- Tests union metrics (fall ∪ fallen)
- Tests empty inputs and edge cases
- Tests class distributions and sample counts

**Run Tests**: `pytest tests/`

**Linting**: `ruff check src/ tests/` (config in `pyproject.toml`)

## Common Workflows

### Run Zero-shot Inference
```bash
python scripts/vllm_inference.py experiment=zeroshot num_samples=100
```

### Debug with Subset
```bash
python scripts/vllm_inference.py experiment=debug batch_size=4 num_samples=10
```

### Enable Chain-of-Thought
```bash
python scripts/vllm_inference.py cot=true
```

### Change Model Size
```bash
python scripts/vllm_inference.py model.params=8B  # Options: 2B, 4B, 8B
```

### Custom Dataset
Create `config/dataset/omnifall/video/my_dataset.yaml`:
```yaml
video_datasets:
  - name: "MyDataset"
    video_root: "${oc.env:DATA_ROOT}/videos"
    annotations_file: "hf://org/repo/labels/annotations.csv"
    split_root: "hf://org/repo/splits"
    dataset_fps: 30.0
```

## Critical Conventions

1. **Always resolve OmegaConf first**: `OmegaConf.resolve(cfg)` before accessing nested configs
2. **HuggingFace path format**: `hf://{org}/{repo}/{filepath}` for remote files
3. **Split validation**: Dataset factory validates split consistency between config and split_root
4. **Logging order**: Initialize W&B, then call `reconfigure_logging_after_wandb()`
5. **Metrics input**: `compute_metrics()` accepts both string labels and numeric indices
6. **Environment variables**: Check `OMNIFALL_ROOT` is set before running dataset code
7. **vLLM spawning**: Set `VLLM_WORKER_MULTIPROC_METHOD=spawn` for stability
8. **Frame sampling**: Use `target_fps` for model input, `dataset_fps` for original video FPS

## Key Files to Reference

- **Hydra config entry**: [config/inference_config.yaml](config/inference_config.yaml)
- **Zero-shot prompts**: [src/infreqact/inference/zeroshot.py](src/infreqact/inference/zeroshot.py#L6-L53)
- **Metrics computation**: [src/infreqact/metrics/base.py](src/infreqact/metrics/base.py#L20-L240)
- **Dataset factory**: [src/infreqact/data/video_dataset_factory.py](src/infreqact/data/video_dataset_factory.py#L19-L198)
- **vLLM inference**: [scripts/vllm_inference.py](scripts/vllm_inference.py#L1-L221)
- **HF path resolution**: [src/infreqact/data/hf_utils.py](src/infreqact/data/hf_utils.py#L13-L196)

## Debugging Tips

- **Hydra config issues**: Run with `hydra.verbose=true` to see config composition
- **Dataset path errors**: Check environment variables with `echo $OMNIFALL_ROOT`
- **vLLM OOM**: Reduce `gpu_memory_utilization` in vllm config or decrease `batch_size`
- **Slow inference**: Increase `batch_size` (default: 50), reduce `num_workers` (default: 8)
- **JSON parsing errors**: Enable `json-repair` library debug mode in `parse_llm_outputs()`

---

**Note**: The `fall-da/` directory is a separate subproject and should be ignored for this codebase.
