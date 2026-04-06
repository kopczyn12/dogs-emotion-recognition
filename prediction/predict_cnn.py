"""
Predict dog emotion using trained CNN model.

Usage:
    python predict_cnn.py --model resnet50 --image Dataset/happy/dog001.jpg
"""

import argparse
import sys
from pathlib import Path

from src.config import MODELS_DIR
from src.models.cnn_classifier import CNNEmotionClassifier


def main():
    """Command-line interface for CNN emotion prediction."""
    parser = argparse.ArgumentParser(description='Predict dog emotion with CNN')
    parser.add_argument('--model', type=str, required=True,
                       choices=['resnet50', 'resnet18', 'densenet121', 'mobilenet_v2',
                               'inception_v3', 'efficientnet_b0', 'efficientnet_b1',
                               'vgg16', 'convnext_tiny', 'convnext_small'],
                       help='Model architecture to use')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image file')
    
    args = parser.parse_args()
    
    image_path = args.image
    
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    
    model_path = MODELS_DIR / f"{args.model}_emotion_classifier.pth"
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print(f"Please run: python train_cnn.py --model {args.model}")
        sys.exit(1)
    
    print(f"\nAnalyzing with {args.model.upper()}: {image_path}")
    print("-" * 40)
    
    classifier = CNNEmotionClassifier(model_name=args.model)
    classifier.load(model_path)
    
    emotion, confidence, all_probs = classifier.predict(image_path)
    
    print(f"\nPredicted emotion: {emotion.upper()}")
    print(f"Confidence: {confidence:.1%}")
    
    print("\nAll probabilities:")
    for emo, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {emo:>10}: {prob:.1%} {bar}")


if __name__ == "__main__":
    main()
