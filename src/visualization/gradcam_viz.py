"""
Grad-CAM visualization utilities.
"""

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from ..config import EMOTION_CLASSES, GRADCAM_DPI, IMAGENET_MEAN, IMAGENET_STD


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize an ImageNet-normalized image tensor for visualization.
    
    Args:
        image_tensor: Normalized image tensor of shape (C, H, W)
        
    Returns:
        Denormalized numpy array of shape (H, W, C) with values in [0, 1]
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    
    img = image_tensor.cpu() * std + mean
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    
    return img


def visualize_gradcam(
    image_tensor: torch.Tensor,
    cam: np.ndarray,
    true_label: int,
    pred_label: int,
    confidence: float,
    save_path: Path
):
    """
    Create a comprehensive Grad-CAM visualization with three panels.
    
    Args:
        image_tensor: Input image tensor
        cam: Grad-CAM heatmap
        true_label: Ground truth label index
        pred_label: Predicted label index
        confidence: Prediction confidence
        save_path: Path to save the visualization
    """
    img = denormalize_image(image_tensor)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img)
    axes[0].set_title(f"Original Image\nTrue: {EMOTION_CLASSES[true_label]}", fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap\n(Model Attention)", fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(img)
    axes[2].imshow(cam, cmap='jet', alpha=0.5)
    axes[2].set_title(
        f"Overlay\nPredicted: {EMOTION_CLASSES[pred_label]} ({confidence:.1%})", 
        fontsize=12
    )
    axes[2].axis('off')
    
    if true_label != pred_label:
        fig.suptitle(
            "❌ MISCLASSIFICATION - Model Failure", 
            fontsize=14, 
            fontweight='bold', 
            color='red'
        )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=GRADCAM_DPI, bbox_inches='tight', facecolor='white')
    plt.close()


def visualize_bad_attention(
    image_tensor: torch.Tensor,
    cam: np.ndarray,
    pred_label: int,
    confidence: float,
    save_path: Path,
    title_extra: str = ""
):
    """
    Visualize cases where the model attention is on wrong areas.
    
    Args:
        image_tensor: Input image tensor
        cam: Grad-CAM heatmap
        pred_label: Predicted label index
        confidence: Prediction confidence
        save_path: Path to save the visualization
        title_extra: Additional text for the title
    """
    img = denormalize_image(image_tensor)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap\n(Where Model is Looking)", fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(img)
    axes[2].imshow(cam, cmap='jet', alpha=0.5)
    axes[2].set_title(
        f"Overlay\nPredicted: {EMOTION_CLASSES[pred_label]} ({confidence:.1%})", 
        fontsize=12
    )
    axes[2].axis('off')
    
    fig.suptitle(
        f"Neural Network Failure - Focusing on Wrong Areas{title_extra}", 
        fontsize=14, 
        fontweight='bold', 
        color='red'
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=GRADCAM_DPI, bbox_inches='tight', facecolor='white')
    plt.close()


def save_gradcam_pair(
    img_array: np.ndarray,
    cam: np.ndarray,
    save_path: Path
):
    """
    Save original image and Grad-CAM overlay side by side.
    
    Args:
        img_array: Image array of shape (H, W, C) with values in [0, 1]
        cam: Grad-CAM heatmap
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    axes[0].imshow(img_array)
    axes[0].axis('off')
    
    axes[1].imshow(img_array)
    axes[1].imshow(cam, cmap='jet', alpha=0.5)
    axes[1].axis('off')
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
    plt.savefig(save_path, dpi=GRADCAM_DPI, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close()


def save_clean_gradcam(
    img_array: np.ndarray,
    cam: np.ndarray,
    output_dir: Path,
    emotion: str
):
    """
    Save separate clean images (original and Grad-CAM) without text or borders.
    
    Args:
        img_array: Image array of shape (H, W, C) with values in [0, 1]
        cam: Grad-CAM heatmap
        output_dir: Directory to save the images
        emotion: Emotion label for filename
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_array)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(
        output_dir / f"{emotion}_original.png", 
        dpi=GRADCAM_DPI, 
        bbox_inches='tight', 
        pad_inches=0, 
        facecolor='white'
    )
    plt.close()
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_array)
    ax.imshow(cam, cmap='jet', alpha=0.5)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(
        output_dir / f"{emotion}_gradcam.png", 
        dpi=GRADCAM_DPI, 
        bbox_inches='tight', 
        pad_inches=0, 
        facecolor='white'
    )
    plt.close()
