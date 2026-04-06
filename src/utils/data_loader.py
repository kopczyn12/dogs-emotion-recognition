"""
Data loading utilities for dog emotion images.
"""

import random
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset

from ..config import EMOTION_CLASSES


def load_dataset(dataset_path: Path) -> Tuple[List[Path], List[str]]:
    """
    Load image paths and their corresponding emotion labels from the dataset directory.
    
    Args:
        dataset_path: Path to the dataset root directory containing emotion subdirectories
        
    Returns:
        Tuple of (image_paths, labels) where:
            - image_paths: List of Path objects pointing to image files
            - labels: List of emotion label strings corresponding to each image
    """
    image_paths = []
    labels = []
    
    for emotion in EMOTION_CLASSES:
        emotion_path = dataset_path / emotion
        if not emotion_path.exists():
            print(f"Warning: {emotion_path} not found, skipping...")
            continue
            
        for img_file in emotion_path.iterdir():
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                image_paths.append(img_file)
                labels.append(emotion)
    
    print(f"Found {len(image_paths)} images across {len(set(labels))} emotion classes")
    return image_paths, labels


class DogEmotionDataset(Dataset):
    """
    PyTorch Dataset for dog emotion images.
    
    Loads images from emotion-labeled subdirectories and applies transformations.
    """
    
    def __init__(self, root_path: Path, transform=None):
        """
        Initialize the dataset.
        
        Args:
            root_path: Path to dataset root directory
            transform: Optional torchvision transforms to apply to images
        """
        self.samples = []
        self.transform = transform
        
        for label_idx, emotion in enumerate(EMOTION_CLASSES):
            emotion_path = root_path / emotion
            if emotion_path.exists():
                for img_file in emotion_path.iterdir():
                    if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        self.samples.append((img_file, label_idx))
        
        random.shuffle(self.samples)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (image_tensor, label, image_path_str)
        """
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, str(img_path)
