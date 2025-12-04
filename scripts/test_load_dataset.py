"""
Test script to load and inspect OmnifallVideoDataset.
Usage: python test_omnifall_dataset.py
"""

import logging
import os
import sys
from pathlib import Path

# Add fall-da to path
sys.path.insert(0, str(Path(__file__).parent))

from infreqact.data.video_dataset import OmnifallVideoDataset, idx2label

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def test_omnifall_dataset():
    """Test loading OmnifallVideoDataset and inspect its contents."""
    # Configuration
    OMNIFALL_ROOT = os.getenv("OMNIFALL_ROOT")
    if not OMNIFALL_ROOT:
        raise ValueError("OMNIFALL_ROOT environment variable not set")

    print(f"OMNIFALL_ROOT: {OMNIFALL_ROOT}")
    print("=" * 80)

    # Dataset parameters
    dataset_config = {
        "video_root": f"{OMNIFALL_ROOT}/cmdfall/video",
        "annotations_file": "hf://simplexsigil2/omnifall/labels/cmdfall.csv",
        "split_root": "hf://simplexsigil2/omnifall/splits",
        "dataset_name": "cmdfall",
        "mode": "test",  # Start with test set (smaller)
        "split": "cs",  # Cross-subject split
        "target_fps": 2.0,  # Low FPS for quick testing
        "vid_frame_count": None,
        "data_fps": 20.0,  # CMDFall videos are 20 FPS
        "ext": ".mp4",
        "fast": True,
    }

    print("\nDataset Configuration:")
    for key, value in dataset_config.items():
        print(f"  {key}: {value}")
    print("=" * 80)

    # Create dataset
    print("\nCreating OmnifallVideoDataset...")
    try:
        dataset = OmnifallVideoDataset(**dataset_config)
        print("✓ Dataset created successfully!")
        print(f"\n{dataset}")
        print("=" * 80)

    except Exception as e:
        print(f"✗ Failed to create dataset: {e}")
        import traceback

        traceback.print_exc()
        return

    # Dataset statistics
    print("\nDataset Statistics:")
    print(f"  Total videos: {len(dataset.samples)}")
    print(f"  Total segments: {len(dataset.video_segments)}")
    print(f"  Mode: {dataset.mode}")
    print(f"  Split: {dataset.split}")
    print("=" * 80)

    # Class distribution
    print("\nClass Distribution:")
    class_counts = {}
    for segment in dataset.video_segments:
        label = segment["label"]
        label_str = idx2label.get(label, f"unknown_{label}")
        class_counts[label_str] = class_counts.get(label_str, 0) + 1

    for label_str, count in sorted(class_counts.items()):
        percentage = (count / len(dataset.video_segments)) * 100
        print(f"  {label_str:15s}: {count:4d} ({percentage:5.1f}%)")
    print("=" * 80)

    # Sample segments
    print("\nFirst 5 Segments:")
    for i, segment in enumerate(dataset.video_segments[:5]):
        print(f"\n  Segment {i}:")
        print(f"    Video: {segment['video_path']}")
        print(f"    Label: {segment['label']} ({segment['label_str']})")
        print(
            f"    Time: {segment['start']:.2f}s - {segment['end']:.2f}s (duration: {segment['duration']:.2f}s)"
        )
        print(f"    Subject: {segment['subsect']}, Camera: {segment['cam']}")
    print("=" * 80)

    # Try loading one sample
    print("\nLoading Sample Segment...")
    try:
        sample = dataset[0]
        print("✓ Sample loaded successfully!")
        print(f"\nSample keys: {list(sample.keys())}")

        if "pixel_values" in sample:
            print(f"  pixel_values shape: {sample['pixel_values'].shape}")
            print(f"  pixel_values dtype: {sample['pixel_values'].dtype}")
            print(
                f"  pixel_values range: [{sample['pixel_values'].min():.3f}, {sample['pixel_values'].max():.3f}]"
            )

        print(f"  label: {sample['label']} ({sample.get('label_str', 'N/A')})")
        print(f"  video_path: {sample['video_path']}")
        print(f"  start_time: {sample['start_time']:.2f}s")
        print(f"  end_time: {sample['end_time']:.2f}s")
        print(f"  dataset: {sample['dataset']}")

    except Exception as e:
        print(f"✗ Failed to load sample: {e}")
        import traceback

        traceback.print_exc()

    print("=" * 80)
    print("✓ Test complete!")


if __name__ == "__main__":
    test_omnifall_dataset()
