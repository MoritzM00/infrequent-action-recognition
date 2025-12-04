"""
Utilities for working with HuggingFace datasets in the context of video datasets.
"""

import logging
import os

logger = logging.getLogger(__name__)


def resolve_annotations_file(annotations_path: str) -> str:
    """
    Resolve an annotations file path, supporting both local paths and HuggingFace dataset references.

    Supports the following formats:
    - Local absolute path: "/path/to/file.csv"
    - Local path with env vars: "${WANFALL_ROOT}/labels/wanfall.csv"
    - HuggingFace direct file: "hf://simplexsigil2/wanfall/labels/wanfall.csv"
    - HuggingFace direct file: "hf://simplexsigil2/omnifall/labels/cmdfall.csv"

    For HuggingFace references, the format is:
        hf://{repo_id}/{filepath}

    The function will download the file from HuggingFace Hub and return the local cached path.

    Args:
        annotations_path: Path or HuggingFace reference to annotations file

    Returns
    -------
        Local path to the annotations file

    Raises
    ------
        ImportError: If HuggingFace Hub is not installed when needed
        ValueError: If the HuggingFace reference format is invalid
        FileNotFoundError: If local file does not exist
    """
    # Handle HuggingFace dataset references
    if annotations_path.startswith("hf://"):
        logger.info(f"Resolving HuggingFace dataset reference: {annotations_path}")

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required to load datasets from HuggingFace. "
                "Install it with: pip install huggingface_hub"
            )

        # Parse the HF reference
        # Format: hf://{repo_id}/{filepath}
        # Example: hf://simplexsigil2/wanfall/labels/wanfall.csv
        hf_path = annotations_path.replace("hf://", "")

        # Split into repo_id (first two parts) and filepath (rest)
        parts = hf_path.split("/", 2)

        if len(parts) < 3:
            raise ValueError(
                f"Invalid HuggingFace reference format: {annotations_path}\n"
                "Expected format: hf://{{org}}/{{repo}}/{{filepath}}\n"
                "Example: hf://simplexsigil2/wanfall/labels/wanfall.csv"
            )

        org, repo, filepath = parts
        repo_id = f"{org}/{repo}"

        logger.info(f"Downloading {filepath} from {repo_id}")

        try:
            local_path = hf_hub_download(repo_id=repo_id, filename=filepath, repo_type="dataset")
            logger.info(f"Successfully downloaded to: {local_path}")
            return local_path

        except Exception as e:
            logger.error(f"Failed to download from HuggingFace: {e}")
            raise

    # Handle local paths with environment variable expansion
    expanded_path = os.path.expandvars(annotations_path)

    # Verify the file exists
    if not os.path.exists(expanded_path):
        raise FileNotFoundError(
            f"Annotations file not found: {expanded_path} (original: {annotations_path})"
        )

    return expanded_path


def resolve_split_file(
    split_path: str, mode: str, dataset_name: str | None = None, split_type: str | None = None
) -> str:
    """
    Resolve a split file path, supporting both local paths and HuggingFace dataset references.

    For HuggingFace references, downloads split CSV files directly using hf_hub_download.

    Supports different split structures:
    - HF direct file path: "hf://simplexsigil2/wanfall/splits/random/train.csv"
    - HF base path: "hf://simplexsigil2/omnifall/splits" (appends /{split_type}/{dataset_name}/{mode}.csv)
    - Local path: "/path/to/splits" (appends /{split_type}/{dataset_name}/{mode}.csv)

    Args:
        split_path: Base path or HuggingFace reference to split file/directory
        mode: Split mode ("train", "val", "test") - only used if split_path is a directory
        dataset_name: Optional dataset name for subdirectory structure (e.g., "cmdfall")
        split_type: Optional split type (e.g., "cs" for cross-subject, "cv" for cross-view)

    Returns
    -------
        Local path to the split file

    Examples
    --------
        >>> # WanFall random split - direct file path
        >>> resolve_split_file("hf://simplexsigil2/wanfall/splits/random/train.csv", "train")
        >>> # Omnifall cross-subject split - base path
        >>> resolve_split_file("hf://simplexsigil2/omnifall/splits", "train",
        ...                     dataset_name="cmdfall", split_type="cs")
    """
    if split_path.startswith("hf://"):
        logger.info(f"Resolving HuggingFace split reference: {split_path} (mode={mode})")

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required to load datasets from HuggingFace. "
                "Install it with: pip install huggingface_hub"
            )

        # Parse the HF reference
        # Format: hf://{org}/{repo}/{filepath}
        hf_path = split_path.replace("hf://", "")

        # Split into repo_id (first two parts) and filepath (rest)
        parts = hf_path.split("/", 2)

        if len(parts) < 3:
            raise ValueError(
                f"Invalid HuggingFace reference format: {split_path}\n"
                "Expected format: hf://{{org}}/{{repo}}/{{filepath}}\n"
                "Example: hf://simplexsigil2/wanfall/splits/random/train.csv"
            )

        org, repo, base_path = parts
        repo_id = f"{org}/{repo}"

        # Check if base_path is already a complete file path (ends with .csv)
        if base_path.endswith(".csv"):
            # Direct file path
            file_path = base_path
        else:
            # Base directory - construct full path
            if split_type and dataset_name:
                file_path = f"{base_path}/{split_type}/{dataset_name}/{mode}.csv"
            elif dataset_name:
                file_path = f"{base_path}/{dataset_name}/{mode}.csv"
            else:
                file_path = f"{base_path}/{mode}.csv"

        logger.info(f"Downloading {file_path} from {repo_id}")

        try:
            local_path = hf_hub_download(repo_id=repo_id, filename=file_path, repo_type="dataset")
            logger.info(f"Successfully downloaded split to: {local_path}")
            return local_path

        except Exception as e:
            logger.error(f"Failed to download split from HuggingFace: {e}")
            raise

    # Handle local paths
    expanded_path = os.path.expandvars(split_path)

    # Check if it's already a complete file path
    if expanded_path.endswith(".csv"):
        split_file = expanded_path
    else:
        # Construct full path based on dataset structure
        if split_type and dataset_name:
            split_file = os.path.join(expanded_path, split_type, dataset_name, f"{mode}.csv")
        elif dataset_name:
            split_file = os.path.join(expanded_path, dataset_name, f"{mode}.csv")
        else:
            split_file = os.path.join(expanded_path, f"{mode}.csv")

    # Verify the file exists
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file} (original: {split_path})")

    return split_file
