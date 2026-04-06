"""
Predict dog emotion using ensemble of multiple models.

Combines predictions from MobileNet, EfficientNet, and ResNet50
using majority voting for more robust predictions.

Usage:
    python prediction/predict_ensemble.py Dataset/happy/dog001.jpg
    python prediction/predict_ensemble.py --models resnet50 densenet121 efficientnet_b0 --image dog.jpg
"""

import argparse
import sys
from pathlib import Path

from src.config import MODELS_DIR
from src.models.ensemble import EnsembleClassifier


def main():
    """Command-line interface for ensemble prediction."""
    parser = argparse.ArgumentParser(description='Predict dog emotion with ensemble')
    parser.add_argument('image', nargs='?', type=str,
                       help='Path to image file')
    parser.add_argument('--image', dest='image_flag', type=str,
                       help='Path to image file (alternative syntax)')
    parser.add_argument('--models', nargs='+', 
                       default=['mobilenet_v2', 'efficientnet_b0', 'resnet50'],
                       help='Models to include in ensemble (default: mobilenet_v2 efficientnet_b0 resnet50)')
    parser.add_argument('--method', choices=['voting', 'average'], default='voting',
                       help='Ensemble method: voting (majority) or average (probabilities)')
    
    args = parser.parse_args()
    
    # Handle both positional and --image argument
    image_path = args.image or args.image_flag
    
    if not image_path:
        print("Error: Image path required")
        print("Usage: python prediction/predict_ensemble.py <image_path>")
        print("   or: python prediction/predict_ensemble.py --image <image_path>")
        sys.exit(1)
    
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    
    print(f"\nAnalyzing with Ensemble ({len(args.models)} models): {image_path}")
    print("=" * 60)
    
    # Initialize ensemble
    ensemble = EnsembleClassifier(model_names=args.models, models_dir=MODELS_DIR)
    
    print("\nLoading models...")
    try:
        ensemble.load_models()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Individual Model Predictions:")
    print("=" * 60)
    
    # Make prediction
    if args.method == 'voting':
        emotion, confidence, details = ensemble.predict_majority_voting(image_path)
        
        print("\n" + "=" * 60)
        print("Ensemble Result (Majority Voting):")
        print("=" * 60)
        print(f"\nPredicted emotion: {emotion.upper()}")
        print(f"Confidence: {confidence:.1%}")
        
        print("\nVote counts:")
        for emo, count in sorted(details['vote_counts'].items(), key=lambda x: -x[1]):
            bar = "█" * (count * 10)
            print(f"  {emo:>10}: {count} vote(s) {bar}")
    
    else:  # average
        emotion, confidence, probs = ensemble.predict_average_probabilities(image_path)
        
        print("\n" + "=" * 60)
        print("Ensemble Result (Average Probabilities):")
        print("=" * 60)
        print(f"\nPredicted emotion: {emotion.upper()}")
        print(f"Confidence: {confidence:.1%}")
        
        print("\nAveraged probabilities:")
        for emo, prob in sorted(probs.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 30)
            print(f"  {emo:>10}: {prob:.1%} {bar}")
    
    print("\n" + "=" * 60)
    print("Ensemble prediction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
