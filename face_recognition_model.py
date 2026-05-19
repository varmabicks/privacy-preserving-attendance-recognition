import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the FaceDataset class
class FaceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((260, 260)),  # EfficientNet-B2 optimal size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes=1):
        super(FaceRecognitionModel, self).__init__()
        # Load EfficientNet-B2
        self.model = models.efficientnet_b2(pretrained=True)
        
        # Enhanced classifier for better feature extraction
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1408, 512),  # B2 features to embedding
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

    def get_embedding(self, x):
        """Extract embeddings before final classification"""
        # Get features up to the embedding layer
        for i, layer in enumerate(self.model.classifier):
            x = layer(x)
            if isinstance(layer, nn.Linear) and x.size(1) == 512:
                return x
        return x

class ModelTrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=0.0001,
            weight_decay=0.01
        )
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(self.device)
            labels = labels.float().to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            outputs = outputs.squeeze()
            
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            self.optimizer.step()
            
            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.float().to(self.device)
                
                outputs = self.model(inputs)
                outputs = outputs.squeeze()
                
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        return val_loss, val_acc

    def plot_metrics(self):
        """Plot training and validation metrics"""
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics['train_loss'], label='Training Loss')
        plt.plot(self.metrics['val_loss'], label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.metrics['train_acc'], label='Training Accuracy')
        plt.plot(self.metrics['val_acc'], label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.close()

# Training configuration
class ModelConfig:
    BATCH_SIZE = 16  # Adjusted for 2GB GPU
    LEARNING_RATE = 1e-4
    EPOCHS = 30
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_dataloaders(dataset_path, batch_size=ModelConfig.BATCH_SIZE):
    """Create train and validation dataloaders"""
    # Collect all image paths and labels
    image_paths = []
    labels = []
    for student_id in os.listdir(dataset_path):
        student_path = os.path.join(dataset_path, student_id)
        if os.path.isdir(student_path):
            for img_name in os.listdir(student_path):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(student_path, img_name))
                    labels.append(int(student_id.split('_')[1]))  # Assuming format: student_1

    # Split data
    indices = np.arange(len(image_paths))
    np.random.shuffle(indices)
    split = int(0.9 * len(indices))
    
    train_paths = [image_paths[i] for i in indices[:split]]
    train_labels = [labels[i] for i in indices[:split]]
    val_paths = [image_paths[i] for i in indices[split:]]
    val_labels = [labels[i] for i in indices[split:]]

    # Create datasets
    train_dataset = FaceDataset(train_paths, train_labels)
    val_dataset = FaceDataset(val_paths, val_labels)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=2)

    return train_loader, val_loader

if __name__ == "__main__":
    print(f"Using device: {ModelConfig.DEVICE}")
    print(f"Batch size: {ModelConfig.BATCH_SIZE}")
    
    # Test model
    model = FaceRecognitionModel().to(ModelConfig.DEVICE)
    print("Model created successfully!")
    
    # Test with random input
    x = torch.randn(1, 3, 260, 260).to(ModelConfig.DEVICE)
    y = model(x)
    print(f"Output shape: {y.shape}")
    
    embedding = model.get_embedding(x)
    print(f"Embedding shape: {embedding.shape}")