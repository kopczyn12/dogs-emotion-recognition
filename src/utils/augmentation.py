"""
Data augmentation utilities for dog emotion images.
"""

import os
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
from tqdm import tqdm


def augment_image_pil(
    image: Image.Image,
    augmentations: Dict
) -> list:
    """
    Apply augmentations to a PIL image.
    
    Args:
        image: PIL Image object
        augmentations: Dictionary of augmentation parameters
        
    Returns:
        List of augmented PIL images
    """
    from torchvision import transforms
    
    augmented_images = []
    
    transform_list = []
    
    if augmentations.get('horizontal_flip', False):
        transform_list.append(transforms.RandomHorizontalFlip(p=1.0))
    
    if augmentations.get('rotation_range', 0) > 0:
        degrees = augmentations['rotation_range']
        transform_list.append(transforms.RandomRotation(degrees))
    
    if augmentations.get('color_jitter', False):
        transform_list.append(transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ))
    
    if augmentations.get('random_crop', False):
        size = image.size
        transform_list.append(transforms.RandomResizedCrop(
            size=min(size),
            scale=(0.8, 1.0)
        ))
    
    if transform_list:
        transform = transforms.Compose(transform_list)
        augmented_images.append(transform(image))
    
    return augmented_images


def augment_dataset(
    input_dir: Path,
    output_dir: Path,
    num_augmentations: int = 3,
    augmentations: Dict = None
):
    """
    Augment entire dataset directory.
    
    Args:
        input_dir: Input directory containing emotion subdirectories
        output_dir: Output directory for augmented images
        num_augmentations: Number of augmented versions per image
        augmentations: Dictionary of augmentation parameters
    """
    if augmentations is None:
        augmentations = {
            'horizontal_flip': True,
            'rotation_range': 20,
            'color_jitter': True,
            'random_crop': False
        }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for emotion_dir in input_dir.iterdir():
        if not emotion_dir.is_dir():
            continue
        
        emotion_name = emotion_dir.name
        output_emotion_dir = output_dir / emotion_name
        output_emotion_dir.mkdir(exist_ok=True)
        
        image_files = list(emotion_dir.glob('*.jpg')) + \
                     list(emotion_dir.glob('*.jpeg')) + \
                     list(emotion_dir.glob('*.png'))
        
        print(f"\nAugmenting {emotion_name} ({len(image_files)} images)...")
        
        for img_path in tqdm(image_files):
            try:
                img = Image.open(img_path).convert('RGB')
                
                output_path = output_emotion_dir / img_path.name
                img.save(output_path)
                
                for i in range(num_augmentations):
                    augmented_imgs = augment_image_pil(img, augmentations)
                    
                    for j, aug_img in enumerate(augmented_imgs):
                        aug_filename = f"aug_{i}_{j}_{img_path.name}"
                        aug_path = output_emotion_dir / aug_filename
                        aug_img.save(aug_path)
                        
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    print(f"\nAugmentation complete! Augmented dataset saved to: {output_dir}")


def balance_dataset(
    input_dir: Path,
    output_dir: Path,
    target_count: int = None
):
    """
    Balance dataset by augmenting underrepresented classes.
    
    Args:
        input_dir: Input directory containing emotion subdirectories
        output_dir: Output directory for balanced dataset
        target_count: Target number of images per class (if None, uses max class count)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class_counts = {}
    for emotion_dir in input_dir.iterdir():
        if emotion_dir.is_dir():
            count = len(list(emotion_dir.glob('*.jpg')) + 
                       list(emotion_dir.glob('*.jpeg')) + 
                       list(emotion_dir.glob('*.png')))
            class_counts[emotion_dir.name] = count
    
    if target_count is None:
        target_count = max(class_counts.values())
    
    print(f"Balancing dataset to {target_count} images per class...")
    print(f"Current distribution: {class_counts}")
    
    for emotion_dir in input_dir.iterdir():
        if not emotion_dir.is_dir():
            continue
        
        emotion_name = emotion_dir.name
        current_count = class_counts[emotion_name]
        needed = target_count - current_count
        
        output_emotion_dir = output_dir / emotion_name
        output_emotion_dir.mkdir(exist_ok=True)
        
        image_files = list(emotion_dir.glob('*.jpg')) + \
                     list(emotion_dir.glob('*.jpeg')) + \
                     list(emotion_dir.glob('*.png'))
        
        for img_path in image_files:
            output_path = output_emotion_dir / img_path.name
            img = Image.open(img_path)
            img.save(output_path)
        
        if needed > 0:
            print(f"\n{emotion_name}: augmenting {needed} images...")
            augmentations_per_image = (needed // current_count) + 1
            
            for img_path in tqdm(image_files[:needed]):
                try:
                    img = Image.open(img_path).convert('RGB')
                    
                    for i in range(augmentations_per_image):
                        augmented_imgs = augment_image_pil(img, {
                            'horizontal_flip': True,
                            'rotation_range': 20,
                            'color_jitter': True
                        })
                        
                        for j, aug_img in enumerate(augmented_imgs):
                            aug_filename = f"balanced_aug_{i}_{j}_{img_path.name}"
                            aug_path = output_emotion_dir / aug_filename
                            aug_img.save(aug_path)
                            
                            if len(list(output_emotion_dir.iterdir())) >= target_count:
                                break
                        
                        if len(list(output_emotion_dir.iterdir())) >= target_count:
                            break
                            
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
    
    print(f"\nDataset balanced! Saved to: {output_dir}")
