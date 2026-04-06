"""
Confusion matrix visualization utilities.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ..config import EMOTION_CLASSES, COLORMAP, CONFUSION_MATRIX_DPI


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list = EMOTION_CLASSES,
    normalize: bool = False,
    title: str = None,
    cmap: str = COLORMAP
) -> tuple:
    """
    Create a confusion matrix visualization.
    
    Args:
        cm: Confusion matrix array of shape (num_classes, num_classes)
        classes: List of class names
        normalize: Whether to normalize the confusion matrix
        title: Optional title for the plot
        cmap: Matplotlib colormap name
        
    Returns:
        Tuple of (figure, axes) matplotlib objects
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11, rotation=90, va='center')
    
    thresh = cm.max() / 2.
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm[i, j] > thresh else "black"
            text = f"{cm[i, j]:.2f}" if normalize else str(int(cm[i, j]))
            ax.text(j, i, text, ha="center", va="center", 
                   color=color, fontsize=12)
    
    ax.set_xlabel('Predicted Labels', fontsize=12)
    ax.set_ylabel('True Labels', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.tick_params(left=False, bottom=False)
    
    plt.tight_layout()
    
    return fig, ax


def save_confusion_matrix(
    cm: np.ndarray,
    output_path: Path,
    classes: list = EMOTION_CLASSES,
    title: str = None
):
    """
    Create and save a confusion matrix visualization to file.
    
    Args:
        cm: Confusion matrix array
        output_path: Path where to save the figure
        classes: List of class names
        title: Optional title for the plot
    """
    fig, ax = plot_confusion_matrix(cm, classes, title=title)
    
    plt.savefig(
        output_path, 
        dpi=CONFUSION_MATRIX_DPI, 
        bbox_inches='tight', 
        facecolor='white', 
        edgecolor='none'
    )
    
    print(f"Saved confusion matrix to: {output_path}")
    plt.close()
