"""
Evaluation metrics and reporting utilities.
"""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

from ..config import EMOTION_CLASSES


def evaluate_classifier(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder
) -> dict:
    """
    Evaluate a classifier and return comprehensive metrics.
    
    Args:
        model: Trained classifier with predict method
        X_test: Test features
        y_test: Test labels
        label_encoder: Label encoder for class names
        
    Returns:
        Dictionary containing accuracy, predictions, and confusion matrix
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'confusion_matrix': cm,
        'classification_report': classification_report(
            y_test, y_pred, target_names=label_encoder.classes_
        )
    }


def print_classification_report(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder
):
    """
    Print a detailed classification report.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        label_encoder: Label encoder for class names
    """
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    
    print("\nClassification Report:")
    print("-" * 60)
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_
    ))
    
    print("\nConfusion Matrix:")
    print("-" * 60)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"{'':>10}", end="")
    for cls in label_encoder.classes_:
        print(f"{cls:>10}", end="")
    print()
    
    for i, cls in enumerate(label_encoder.classes_):
        print(f"{cls:>10}", end="")
        for j in range(len(label_encoder.classes_)):
            print(f"{cm[i, j]:>10}", end="")
        print()
