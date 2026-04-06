"""
Train CNN classifiers for dog emotion recognition using transfer learning.

Supports multiple architectures: ResNet, DenseNet, MobileNet, Inception, EfficientNet, VGG.

Usage:
    python train_cnn.py --model resnet50
    python train_cnn.py --model densenet121 --epochs 30
"""

import argparse
from pathlib import Path

from src.config import DATASET_PATH, MODELS_DIR
from src.models.cnn_classifier import CNNEmotionClassifier


def main():
    """Main CNN training pipeline."""
    parser = argparse.ArgumentParser(description='Train CNN for dog emotion classification')
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'resnet18', 'densenet121', 'mobilenet_v2',
                               'inception_v3', 'efficientnet_b0', 'efficientnet_b1',
                               'vgg16', 'convnext_tiny', 'convnext_small'],
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Dog Emotion Classification with {args.model.upper()}")
    print("=" * 60)
    
    MODELS_DIR.mkdir(exist_ok=True)
    
    print("\n[1/4] Initializing model...")
    classifier = CNNEmotionClassifier(
        model_name=args.model,
        pretrained=True,
        freeze_features=True
    )
    
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
    classifier.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )
    
    print(f"\n[4/4] Saving model...")
    model_save_path = MODELS_DIR / f"{args.model}_emotion_classifier.pth"
    classifier.save(model_save_path)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {model_save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
