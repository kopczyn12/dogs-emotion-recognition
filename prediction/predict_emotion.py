"""
Predict dog emotion from a single image using the trained DINO + SVM model.

Usage:
    python predict_emotion.py path/to/dog/image.jpg
"""

import sys
from pathlib import Path
import joblib

from src.config import SVM_MODEL_PATH
from src.models import DinoFeatureExtractor


def predict_emotion(image_path: str) -> tuple:
    """
    Predict the emotion of a dog in an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (predicted_emotion, confidence_dict)
    """
    if not SVM_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {SVM_MODEL_PATH}. "
            "Please run train_classifier.py first to train the model."
        )
    
    saved = joblib.load(SVM_MODEL_PATH)
    svm = saved["svm"]
    label_encoder = saved["label_encoder"]
    
    print(f"Loading DINO model...")
    feature_extractor = DinoFeatureExtractor()
    
    print(f"Extracting features from image...")
    features = feature_extractor.extract_single(Path(image_path))
    
    prediction = svm.predict(features)[0]
    predicted_emotion = label_encoder.inverse_transform([prediction])[0]
    
    decision_scores = svm.decision_function(features)[0]
    
    import numpy as np
    exp_scores = np.exp(decision_scores - np.max(decision_scores))
    probabilities = exp_scores / exp_scores.sum()
    
    confidence = {
        emotion: float(prob) 
        for emotion, prob in zip(label_encoder.classes_, probabilities)
    }
    
    return predicted_emotion, confidence


def main():
    """Command-line interface for emotion prediction."""
    if len(sys.argv) < 2:
        print("Usage: python predict_emotion.py <image_path>")
        print("Example: python predict_emotion.py Dataset/happy/dog001.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    
    print(f"\nAnalyzing: {image_path}")
    print("-" * 40)
    
    emotion, confidence = predict_emotion(image_path)
    
    print(f"\nPredicted emotion: {emotion.upper()}")
    print("\nConfidence scores:")
    for emo, conf in sorted(confidence.items(), key=lambda x: -x[1]):
        bar = "█" * int(conf * 30)
        print(f"  {emo:>10}: {conf:.1%} {bar}")


if __name__ == "__main__":
    main()
