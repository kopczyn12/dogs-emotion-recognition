"""
Generate Grad-CAM visualizations for model interpretation.

Creates visualizations showing where the neural network focuses its attention
when making predictions, useful for understanding model failures.
"""

import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models

from src.config import (
    DATASET_PATH, OUTPUT_DIR, DEVICE, BATCH_SIZE, 
    RANDOM_STATE, NUM_CLASSES, IMAGE_SIZE,
    IMAGENET_MEAN, IMAGENET_STD
)
from src.utils import DogEmotionDataset
from src.models import GradCAM
from src.visualization import visualize_gradcam

random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

OUTPUT_DIR.mkdir(exist_ok=True)


def train_simple_model(train_loader, num_epochs: int = 8):
    """
    Train a simple ResNet18 model for Grad-CAM visualization.
    
    Args:
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        
    Returns:
        Trained model
    """
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels, _ in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f} Acc: {acc:.2f}%")
    
    return model


def find_and_visualize_failures(model, test_loader, gradcam, num_visualizations: int = 10):
    """
    Find misclassified examples and generate Grad-CAM visualizations.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        gradcam: GradCAM instance
        num_visualizations: Number of failures to visualize
    """
    import torch.nn.functional as F
    from src.config import EMOTION_CLASSES
    
    model.eval()
    failures = []
    
    with torch.no_grad():
        for images, labels, paths in test_loader:
            images_gpu = images.to(DEVICE)
            outputs = model(images_gpu)
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            if predicted.item() != labels.item():
                confidence = probs[0, predicted.item()].item()
                failures.append({
                    'image': images[0],
                    'true_label': labels.item(),
                    'pred_label': predicted.item(),
                    'confidence': confidence,
                    'path': paths[0]
                })
    
    print(f"Found {len(failures)} misclassified samples")
    
    num_to_visualize = min(num_visualizations, len(failures))
    print(f"Generating Grad-CAM for {num_to_visualize} failure cases...")
    
    failures.sort(key=lambda x: x['confidence'], reverse=True)
    
    for i, failure in enumerate(failures[:num_to_visualize]):
        image_tensor = failure['image'].unsqueeze(0).to(DEVICE)
        image_tensor.requires_grad = True
        
        cam, _ = gradcam.generate(image_tensor, failure['pred_label'])
        
        save_path = OUTPUT_DIR / f"failure_{i+1}_{EMOTION_CLASSES[failure['true_label']]}_as_{EMOTION_CLASSES[failure['pred_label']]}.png"
        
        visualize_gradcam(
            failure['image'],
            cam,
            failure['true_label'],
            failure['pred_label'],
            failure['confidence'],
            save_path
        )
        
        print(f"  Saved: {save_path.name}")


def main():
    """Main Grad-CAM generation pipeline."""
    print("=" * 60)
    print("Grad-CAM Visualization for Model Failures")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    print("\n[1/4] Loading dataset...")
    full_dataset = DogEmotionDataset(DATASET_PATH, transform=train_transform)
    
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(RANDOM_STATE)
    )
    
    test_dataset.dataset.transform = test_transform
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    print("\n[2/4] Training model...")
    model = train_simple_model(train_loader)
    
    print("\n[3/4] Setting up Grad-CAM...")
    target_layer = model.layer4[-1].conv2
    gradcam = GradCAM(model, target_layer)
    
    print("\n[4/4] Finding failures and generating visualizations...")
    find_and_visualize_failures(model, test_loader, gradcam)
    
    print(f"\nDone! Check {OUTPUT_DIR} for Grad-CAM visualizations")


if __name__ == "__main__":
    main()
