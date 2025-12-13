import os
from pathlib import Path

from infreqact.data.video_dataset import OmnifallVideoDataset


def load_test_omnifall_dataset():
    """Test loading OmnifallVideoDataset and inspect its contents."""

    OMNIFALL_ROOT = os.getenv("OMNIFALL_ROOT")
    if not OMNIFALL_ROOT:
        raise ValueError("OMNIFALL_ROOT environment variable not set")

    omnifall_path = Path(OMNIFALL_ROOT)
    dataset_config = {
        "video_root": str(omnifall_path / "OOPS" / "video"),
        "annotations_file": "hf://simplexsigil2/omnifall/labels/OOPS.csv",
        "split_root": "hf://simplexsigil2/omnifall/splits",
        "dataset_name": "OOPS",
        "mode": "test",  # Start with test set (smaller)
        "split": "cs",  # Cross-subject split
        "target_fps": 8.0,  # Low FPS for quick testing
        "vid_frame_count": 16,
        "data_fps": 30.0,  # OOPS videos are 30 FPS
        "ext": ".mp4",
        "fast": True,
    }
    return OmnifallVideoDataset(**dataset_config)
