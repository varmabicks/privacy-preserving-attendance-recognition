import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import flwr as fl
from face_recognition_model import FaceRecognitionModel, FaceDataset, ModelTrainer, create_dataloaders, ModelConfig
from federated_client import FederatedClient
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

def main():
    print("Initializing Face Recognition System...")
    print(f"Using device: {ModelConfig.DEVICE}")

    # 1. Load the model
    print("\nLoading EfficientNet-B2 model...")
    model = FaceRecognitionModel().to(ModelConfig.DEVICE)
    
    # 2. Create data loaders
    print("\nPreparing datasets...")
    train_loader, val_loader = create_dataloaders(
        dataset_path='dataset',
        batch_size=ModelConfig.BATCH_SIZE
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # 3. Initialize trainer
    trainer = ModelTrainer(
        model=model,
        device=ModelConfig.DEVICE
    )

    # 4. Create federated client
    client = FederatedClient(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        trainer=trainer
    )

    # 5. Start federated learning
    print("\nStarting Federated Learning...")
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client
    )

if __name__ == "__main__":
    # 1. First make sure server is running
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure the server is running first!")
        print("Run 'python server.py' in a separate terminal") 