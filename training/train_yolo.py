"""
Train YOLO classifier for dog emotion recognition.

Usage:
    python train_yolo.py
"""

from pathlib import Path

from src.config import DATASET_PATH, MODELS_DIR, NUM_EPOCHS
from src.models.yolo_classifier import YOLOEmotionClassifier


def main():
    """Main YOLO training pipeline."""
    print("=" * 60)
    print("Dog Emotion Classification with YOLOv8")
    print("=" * 60)
    
    MODELS_DIR.mkdir(exist_ok=True)
    
    print("\n[1/3] Initializing YOLO classifier...")
    classifier = YOLOEmotionClassifier()
    
    print("\n[2/3] Training YOLO model...")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Epochs: {NUM_EPOCHS}")
    
    results = classifier.train(
        data_path=DATASET_PATH,
        epochs=NUM_EPOCHS,
        imgsz=224,
        batch=16,
        patience=10,
        save=True,
        plots=True
    )
    
    print("\n[3/3] Validating model...")
    metrics = classifier.validate()
    
    print("\n" + "=" * 60)
    print("Training Results:")
    print("=" * 60)
    print(f"Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"Top-5 Accuracy: {metrics.top5:.4f}")
    
    model_save_path = MODELS_DIR / "yolo_emotion_classifier.pt"
    classifier.save(model_save_path)
    print(f"\nModel saved to: {model_save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
