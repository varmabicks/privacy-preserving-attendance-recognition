import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from face_recognition_model import FaceRecognitionModel, FaceDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

def load_and_evaluate_model():
    print("Starting model evaluation...")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the trained model
    model = FaceRecognitionModel(num_classes=1).to(device)
    try:
        model.load_state_dict(torch.load('face_recognition_model.pth'))
        print("Model loaded successfully!")
    except:
        print("Error: Could not find saved model file!")
        return
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    print("\nLoading test dataset...")
    image_dir = "dataset/student_1"
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
    labels = [0] * len(image_paths)
    
    # Split for testing
    _, test_paths, _, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )
    
    test_dataset = FaceDataset(test_paths, test_labels, transform)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Evaluate model
    model.eval()
    correct = 0
    total = 0
    predictions = []
    
    print("\nEvaluating model performance...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predictions.extend(predicted.cpu().numpy())
    
    accuracy = 100 * correct / total
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Total test images: {total}")
    print(f"Correctly classified: {correct}")
    print(f"Model Accuracy: {accuracy:.2f}%")
    
    # Plot sample predictions
    print("\nPlotting sample predictions...")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    
    for idx in range(min(6, len(test_paths))):
        img = Image.open(test_paths[idx])
        axes[idx].imshow(img)
        axes[idx].set_title(f'Predicted: {"Correct" if predictions[idx] == test_labels[idx] else "Incorrect"}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    load_and_evaluate_model() 