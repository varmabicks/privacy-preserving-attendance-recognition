from flask import Flask, request, jsonify
import torch
import numpy as np
from face_recognition_model import FaceRecognitionModel
import io
import logging
import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics
from flwr.server.client_proxy import ClientProxy
import threading

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize global model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
global_model = FaceRecognitionModel(num_classes=1).to(device)
global_model_state = {"embeddings": {}}

class FederatedServer:
    def __init__(self):
        self.strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,  # Use all available clients for training
            fraction_evaluate=1.0,  # Evaluate all clients
            min_fit_clients=1,  # Minimum number of clients to train
            min_evaluate_clients=1,  # Minimum number of clients to evaluate
            min_available_clients=1,  # Minimum number of available clients
            evaluate_metrics_aggregation_fn=self.weighted_average,
            initial_parameters=None
        )
        
    def weighted_average(self, metrics: List[Tuple[int, Metrics]]) -> Metrics:
        """Aggregate evaluation metrics weighted by number of samples"""
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        
        return {
            "accuracy": sum(accuracies) / sum(examples)
        }

    def start_server(self, port=8080):
        """Start the Flower server"""
        fl.server.start_server(
            server_address=f"0.0.0.0:{port}",
            config=fl.server.ServerConfig(num_rounds=3),
            strategy=self.strategy
        )

@app.route('/get_model', methods=['GET'])
def get_model():
    # Serialize model parameters
    buffer = io.BytesIO()
    torch.save(global_model.state_dict(), buffer)
    return buffer.getvalue()

@app.route('/update_model', methods=['POST'])
def update_model():
    # Get model update from client
    model_data = request.files['model'].read()
    buffer = io.BytesIO(model_data)
    client_state_dict = torch.load(buffer, map_location=device)
    
    # Update global model (simple FedAvg)
    global_model.load_state_dict(client_state_dict)
    logging.info("Global model updated")
    
    return jsonify({"status": "success"})

@app.route('/update_embeddings', methods=['POST'])
def update_embeddings():
    """Update student embeddings"""
    data = request.json
    student_id = data.get('student_id')
    embedding = data.get('embedding')
    
    if student_id and embedding:
        global_model_state["embeddings"][student_id] = np.array(embedding)
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Invalid data"}), 400

@app.route('/compare_embeddings', methods=['POST'])
def compare_embeddings():
    """Compare face embeddings with stored embeddings"""
    data = request.json
    new_embedding = np.array(data.get('embedding'))
    threshold = data.get('threshold', 0.85)
    
    matches = []
    for student_id, stored_embedding in global_model_state["embeddings"].items():
        similarity = np.dot(new_embedding, stored_embedding) / (
            np.linalg.norm(new_embedding) * np.linalg.norm(stored_embedding)
        )
        if similarity > threshold:
            matches.append({
                "student_id": student_id,
                "similarity": float(similarity)
            })
    
    return jsonify({
        "status": "success",
        "matches": matches
    })

def start_flask(host='0.0.0.0', port=5000):
    """Start Flask server"""
    app.run(host=host, port=port)

if __name__ == "__main__":
    # Create and start the federated learning server
    fed_server = FederatedServer()
    
    # Start Flask server in a separate thread
    flask_thread = threading.Thread(
        target=start_flask,
        kwargs={'host': '0.0.0.0', 'port': 5000}
    )
    flask_thread.start()
    
    # Start Flower server
    print("Starting Federated Learning Server...")
    fed_server.start_server(port=8080)