"""
Predict dog emotion using trained YOLO model.

Usage:
    python predict_yolo.py path/to/dog/image.jpg
"""

import sys
from pathlib import Path

from src.config import MODELS_DIR
from src.models.yolo_classifier import YOLOEmotionClassifier


def main():
    """Command-line interface for YOLO emotion prediction."""
    if len(sys.argv) < 2:
        print("Usage: python predict_yolo.py <image_path>")
        print("Example: python predict_yolo.py Dataset/happy/dog001.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    
    model_path = MODELS_DIR / "yolo_emotion_classifier.pt"
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please run train_yolo.py first to train the model.")
        sys.exit(1)
    
    print(f"\nAnalyzing with YOLO: {image_path}")
    print("-" * 40)
    
    classifier = YOLOEmotionClassifier(str(model_path))
    
    emotion, confidence, all_probs = classifier.predict(image_path)
    
    print(f"\nPredicted emotion: {emotion.upper()}")
    print(f"Confidence: {confidence:.1%}")
    
    print("\nAll probabilities:")
    for emo, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {emo:>10}: {prob:.1%} {bar}")


if __name__ == "__main__":
    main()
