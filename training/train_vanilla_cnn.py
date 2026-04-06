"""
Train Vanilla CNN for dog emotion classification.

This custom CNN serves as a baseline to compare against pretrained models,
demonstrating the value of transfer learning.

Usage:
    python training/train_vanilla_cnn.py
    python training/train_vanilla_cnn.py --epochs 50 --batch-size 32
"""

import argparse
from pathlib import Path

from src.config import DATASET_PATH, MODELS_DIR
from src.models.vanilla_cnn import VanillaCNNClassifier


def main():
    """Main Vanilla CNN training pipeline."""
    parser = argparse.ArgumentParser(description='Train Vanilla CNN for dog emotion classification')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio (default: 0.2)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Dog Emotion Classification with Vanilla CNN")
    print("=" * 60)
    print("\nThis is a custom CNN trained from scratch (no pretrained weights)")
    print("Serves as baseline to compare with transfer learning models")
    
    MODELS_DIR.mkdir(exist_ok=True)
    
    print("\n[1/4] Initializing Vanilla CNN...")
    classifier = VanillaCNNClassifier()
    
    print(f"\n[2/4] Preparing data...")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Batch size: {args.batch_size}")
    print(f"Validation split: {args.val_split}")
    
    train_loader, val_loader = classifier.prepare_data(
        data_dir=DATASET_PATH,
        batch_size=args.batch_size,
        val_split=args.val_split
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Classes: {classifier.class_names}")
    
    print(f"\n[3/4] Training model...")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print("\nNote: Training from scratch takes longer than transfer learning")
    
    classifier.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )
    
    print(f"\n[4/4] Saving model...")
    model_save_path = MODELS_DIR / "vanilla_cnn_emotion_classifier.pth"
    classifier.save(model_save_path)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {model_save_path}")
    print("\nCompare this baseline with transfer learning models:")
    print("  python training/train_cnn.py --model resnet50")
    print("=" * 60)


if __name__ == "__main__":
    main()
