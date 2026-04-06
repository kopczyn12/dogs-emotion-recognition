"""
Augment the dog emotion dataset to increase training data.

Usage:
    python scripts/augment_dataset.py
"""

from pathlib import Path

from src.config import DATASET_PATH, PROJECT_ROOT
from src.utils.augmentation import augment_dataset, balance_dataset


def main():
    """Main data augmentation pipeline."""
    print("=" * 60)
    print("Data Augmentation for Dog Emotion Dataset")
    print("=" * 60)
    
    output_dir = PROJECT_ROOT / "Dataset_Augmented"
    
    print(f"\nInput dataset: {DATASET_PATH}")
    print(f"Output dataset: {output_dir}")
    
    print("\n[1/2] Augmenting dataset...")
    augment_dataset(
        input_dir=DATASET_PATH,
        output_dir=output_dir,
        num_augmentations=3,
        augmentations={
            'horizontal_flip': True,
            'rotation_range': 20,
            'color_jitter': True,
            'random_crop': False
        }
    )
    
    print("\n[2/2] Balancing dataset...")
    balanced_dir = PROJECT_ROOT / "Dataset_Balanced"
    balance_dataset(
        input_dir=DATASET_PATH,
        output_dir=balanced_dir,
        target_count=None
    )
    
    print("\n" + "=" * 60)
    print("Data augmentation complete!")
    print(f"Augmented dataset: {output_dir}")
    print(f"Balanced dataset: {balanced_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
