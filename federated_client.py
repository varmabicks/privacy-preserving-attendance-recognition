import flwr as fl
import torch
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple

class FederatedClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, trainer):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.trainer = trainer
        self.best_acc = 0.0
        
    def get_parameters(self, config) -> List[np.ndarray]:
        """Get model parameters as a list of numpy arrays"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from a list of numpy arrays"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the model on the local dataset"""
        self.set_parameters(parameters)
        
        # Training loop
        for epoch in range(1, 31):  # 30 epochs per round
            train_loss, train_acc = self.trainer.train_epoch(self.train_loader)
            val_loss, val_acc = self.trainer.validate(self.val_loader)
            
            # Update metrics
            self.trainer.metrics['train_loss'].append(train_loss)
            self.trainer.metrics['val_loss'].append(val_loss)
            self.trainer.metrics['train_acc'].append(train_acc)
            self.trainer.metrics['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                  f"train_acc={train_acc:.2f}%, val_acc={val_acc:.2f}%")
            
            # Save best model
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        
        # Plot training metrics
        self.trainer.plot_metrics()
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        """Evaluate the model on the local validation dataset"""
        self.set_parameters(parameters)
        
        loss = 0
        correct = 0
        total = 0
        
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(next(self.model.parameters()).device)
                labels = labels.float().to(next(self.model.parameters()).device)
                
                outputs = self.model(inputs)
                outputs = outputs.squeeze()
                
                loss += self.trainer.criterion(outputs, labels).item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return loss, len(self.val_loader.dataset), {"accuracy": accuracy}

    def get_embeddings(self, image):
        """Generate embeddings for a single image"""
        self.model.eval()
        with torch.no_grad():
            image = image.unsqueeze(0)  # Add batch dimension
            image = image.to(next(self.model.parameters()).device)
            embedding = self.model(image)
            return embedding.cpu().numpy() 