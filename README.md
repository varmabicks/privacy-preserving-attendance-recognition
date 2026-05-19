# privacy-preserving-attendance-recognition
A decentralized, privacy-preserving Face Recognition pipeline using Flower (flwr), PyTorch, and FastAPI. Employs an EfficientNet-B2 core architecture to optimize local model training via Federated Averaging (FedAvg) while serving secure real-time edge streaming inference.
## 📌 Project Overview

Traditional biometric recognition architectures rely heavily on centralized repositories to collect and store sensitive facial imagery. This creates structural compliance risks and data security vulnerabilities. 

This project implements a **Privacy-Preserving, Decentralized Face Recognition System** utilizing **Federated Learning**. By utilizing a hub-and-spoke configuration, clients train an optimized computer vision model locally using their own localized datasets. Instead of uploading raw imagery, edge nodes broadcast only mathematical model parameters and encrypted embeddings back to a centralized engine via a lightweight **FastAPI/Flask REST orchestration interface**. 

The system leverages **EfficientNet-B2** for robust feature extraction and embedding generation, coordinated via **Flower (flwr)** for seamless Federated Averaging (`FedAvg`) cycles, and interfaces with **OpenCV** to drive zero-latency edge verification.

### ⚙️ Core Technical Highlights

* **Federated Learning Framework:** Orchestrated via Flower (`flwr`) to manage multi-client model synchronization, employing custom weighted metric evaluation routines (`FedAvg`) to scale cross-device training.
* **Efficient Backbone Feature Extraction:** Swaps massive, unmanageable image processing steps for a lean **EfficientNet-B2** architecture, customized with dynamic dropout and normalization heads to output high-density 512-dimensional face embeddings.
* **Hybrid Communication Architecture:** Integrates a multi-threaded **Flask REST API** alongside a persistent **Flower Server** allowing edge nodes to fetch the global model state, register fresh face embeddings, and run localized calculations simultaneously.
* **Live Edge Inference Engine:** Includes a real-time deployment script featuring OpenCV Haar Cascade windows, calculating dynamic cosine similarities against mathematical threshold boundaries to identify individuals without raw central data cross-checks.
