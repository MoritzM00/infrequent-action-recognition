#!/usr/bin/env python
"""Run one-at-a-time prompt component ablation experiments.

Fixes output_format=text and uses the minimal baseline (baseline.yaml values)
as the reference point. For each prompt component (role, task, labels,
definitions), sweeps over all its variants while holding the others at baseline.

Baseline config (from baseline.yaml):
    role_variant=null, task_variant=standard, labels_variant=bulleted,
    definitions_variant=null, output_format=text

Ablation sweeps:
    1. Role variant:        null*, standard, specialized, video_specialized
    2. Task variant:        standard*, extended
    3. Labels variant:      bulleted*, numbered, grouped, comma
    4. Definitions variant: null*, standard, extended

    (* = baseline value)

Total unique experiments: 10 (13 configs minus 3 duplicated baselines).
"""

import argparse
import subprocess

# ---------------------------------------------------------------------------
# Baseline and sweep definitions
# ---------------------------------------------------------------------------

BASELINE = {
    "prompt.output_format": "text",
    "prompt.role_variant": None,
    "prompt.task_variant": "standard",
    "prompt.labels_variant": "bulleted",
    "prompt.definitions_variant": None,
}

# Each entry maps a config key to the list of values to sweep.
# The baseline value is always included so that it appears as the first run
# in the sweep and is later deduplicated across sweeps.
COMPONENT_SWEEPS: dict[str, list[str | None]] = {
    "prompt.role_variant": [None, "standard", "specialized", "video_specialized"],
    "prompt.task_variant": ["standard", "extended"],
    "prompt.labels_variant": ["bulleted", "numbered", "grouped", "comma"],
    "prompt.definitions_variant": [None, "standard", "extended"],
}


# ---------------------------------------------------------------------------
# Experiment generation
# ---------------------------------------------------------------------------


def _config_key(config: dict) -> tuple:
    """Return a hashable key for deduplication."""
    return tuple(sorted(config.items()))


def generate_experiment_configs() -> list[dict]:
    """Generate deduplicated one-at-a-time ablation configs.

    For each component, produce configs that differ from the baseline only in
    that component's value.  Deduplicate so the pure-baseline config appears
    only once.
    """
    seen: set[tuple] = set()
    configs: list[dict] = []

    for sweep_key, sweep_values in COMPONENT_SWEEPS.items():
        for value in sweep_values:
            config = dict(BASELINE)
            config[sweep_key] = value
            key = _config_key(config)
            if key not in seen:
                seen.add(key)
                configs.append(config)

    return configs


# ---------------------------------------------------------------------------
# Tag & command building
# ---------------------------------------------------------------------------

# Friendly short names for tags (key -> tag prefix)
_TAG_PREFIX = {
    "prompt.role_variant": "role",
    "prompt.task_variant": "task",
    "prompt.labels_variant": "labels",
    "prompt.definitions_variant": "defs",
    "prompt.output_format": "format",
}


def build_tags(config: dict) -> list[str]:
    """Build descriptive W&B tags from a config dict."""
    tags = ["ablation", "component"]

    for key, prefix in _TAG_PREFIX.items():
        value = config.get(key)
        tag_value = "none" if value is None else str(value)
        tags.append(f"{prefix}-{tag_value}")

    return tags


def _hydra_value(value) -> str:
    """Convert a Python value to its Hydra CLI representation."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def build_command(config: dict, model: str = "qwenvl", params: str = "8B") -> list[str]:
    """Build the Hydra CLI command for a single experiment."""
    cmd = [
        "python",
        "scripts/vllm_inference.py",
        "experiment=zeroshot",
        "data.mode=test",
        f"model={model}",
        f"model.params={params}",
        "sampling=qwen3_instruct",
        "vllm.tensor_parallel_size=1",
        "log_videos=0",
    ]

    # Ablation overrides
    for key, value in config.items():
        cmd.append(f"{key}={_hydra_value(value)}")

    # W&B tags
    tags = build_tags(config)
    cmd.append(f"wandb.tags={tags}")

    return cmd


def format_command_for_display(cmd: list[str]) -> str:
    """Format a command list for human-readable display."""
    parts = []
    for part in cmd:
        if part.startswith("wandb.tags="):
            parts.append(f'"{part}"')
        else:
            parts.append(part)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_experiment(config: dict, dry_run: bool = False, model: str = "qwenvl", params: str = "8B"):
    """Run a single experiment (or print the command in dry-run mode)."""
    cmd = build_command(config, model=model, params=params)

    if dry_run:
        print(f"[DRY-RUN] {format_command_for_display(cmd)}\n")
        return

    print(f"Running: {format_command_for_display(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Experiment failed with exit code {e.returncode}")
        raise


def _describe_config(config: dict) -> str:
    """Return a short human-readable description of what differs from baseline."""
    diffs = []
    for key, baseline_val in BASELINE.items():
        if config[key] != baseline_val:
            diffs.append(f"{_TAG_PREFIX.get(key, key)}={_hydra_value(config[key])}")
    return ", ".join(diffs) if diffs else "baseline"


def main():
    parser = argparse.ArgumentParser(
        description="Run one-at-a-time prompt component ablation experiments"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Skip first N experiments (for resuming, 0-indexed)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwenvl",
        help="Model name for experiments (default: qwenvl)",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="8B",
        help="Model parameters for experiments (default: 8B)",
    )
    args = parser.parse_args()

    configs = generate_experiment_configs()
    print(f"Total unique experiments: {len(configs)}")
    print()

    # Print the full experiment plan
    print("Experiment plan:")
    for i, config in enumerate(configs):
        marker = "(skip)" if i < args.start_from else ""
        print(f"  {i + 1:>3}. {_describe_config(config):40s} {marker}")
    print()

    if args.start_from > 0:
        print(f"Resuming from experiment {args.start_from + 1}")

    for i, config in enumerate(configs[args.start_from :], start=args.start_from):
        print(f"\n{'=' * 60}")
        print(f"Experiment {i + 1}/{len(configs)}: {_describe_config(config)}")
        print(f"Config: {config}")
        print(f"{'=' * 60}")
        run_experiment(config, args.dry_run, model=args.model, params=args.params)

    if not args.dry_run:
        print(f"\n{'=' * 60}")
        print("All experiments completed!")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
