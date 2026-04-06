"""
Train a dog emotion classifier using DINO features and SVM.

This script:
1. Loads dog emotion images from the Dataset folder
2. Extracts feature vectors using DINO (Vision Transformer)
3. Trains an SVM classifier on the extracted features
4. Evaluates and reports classification performance
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

from src.config import (
    DATASET_PATH, FEATURES_CACHE, SVM_MODEL_PATH, 
    BATCH_SIZE, TEST_SIZE, RANDOM_STATE, DEVICE,
    MODELS_DIR, CACHE_DIR
)
from src.utils import load_dataset
from src.models import DinoFeatureExtractor, SVMClassifier
from src.evaluation import print_classification_report


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Dog Emotion Classification with DINO + SVM")
    print("=" * 60)
    print(f"\nUsing device: {DEVICE}")
    
    CACHE_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    
    print("\n[1/5] Loading dataset...")
    image_paths, labels = load_dataset(DATASET_PATH)
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    print(f"Classes: {list(label_encoder.classes_)}")
    
    print("\n[2/5] Loading DINO model...")
    feature_extractor = DinoFeatureExtractor()
    
    print("\n[3/5] Extracting features...")
    
    if FEATURES_CACHE.exists():
        print(f"Loading cached features from {FEATURES_CACHE}")
        cached = np.load(FEATURES_CACHE)
        features = cached["features"]
        
        if len(features) != len(image_paths):
            print("Cache size mismatch, re-extracting features...")
            features = feature_extractor.extract_features(image_paths, BATCH_SIZE)
            np.savez(FEATURES_CACHE, features=features, labels=encoded_labels)
    else:
        features = feature_extractor.extract_features(image_paths, BATCH_SIZE)
        print(f"Saving features to cache: {FEATURES_CACHE}")
        np.savez(FEATURES_CACHE, features=features, labels=encoded_labels)
    
    print(f"Feature matrix shape: {features.shape}")
    
    print("\n[4/5] Splitting data and training SVM...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, encoded_labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=encoded_labels
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    classifier = SVMClassifier()
    classifier.label_encoder = label_encoder
    classifier.fit(X_train, y_train)
    
    print(f"\nSaving model to {SVM_MODEL_PATH}")
    joblib.dump({
        "svm": classifier.svm,
        "label_encoder": label_encoder
    }, SVM_MODEL_PATH)
    
    print("\n[5/5] Evaluating model...")
    y_pred = classifier.predict(X_test)
    print_classification_report(y_test, y_pred, label_encoder)
    
    print("\n" + "=" * 60)
    print("Done! Model saved to:", SVM_MODEL_PATH)
    print("=" * 60)


if __name__ == "__main__":
    main()
