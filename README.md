
# Graph Neural Networks for Lung Airway Flow Simulation

## Project Overview
This repository presents a robust, research-driven implementation of Graph Neural Networks (GNNs) for simulating airflow in human lung airways. The project leverages the natural graph structure of the bronchial tree to predict pressure and flow rates at each airway segment, aiming to advance computational tools for respiratory health and personalized medicine.

## Why GNNs for Lungs?
Traditional models of lung airflow often struggle to balance anatomical realism and computational efficiency. GNNs, by design, excel at learning on graph-structured data—making them ideal for modeling the complex, branching architecture of the lungs. This project demonstrates how GNNs can deliver accurate, scalable predictions for lung ventilation, with the potential to generalize to unseen patient data.

## Key Features & Architecture
- **Graph-Based Data**: Each lung is represented as a graph (nodes = airways, edges = airflow paths), with node features including spatial coordinates, radius, pressure, and flow rate.
- **Modern GNN Model**: Implements a multi-layer (3-layer) Graph Convolutional Network (GCN) using PyTorch Geometric, with ReLU activations and robust training strategies.
- **Data Pipeline**: Modular data loading and preprocessing via `data_container.py` and `data_loader.py`, supporting scalable experimentation and reproducibility.
- **Training & Evaluation**: Scripts for training (`GNN_model_train_val.py`) and testing (`GNN_model_test.py`) the model, with support for early stopping and MAPE loss.
- **Extensible Dataset**: Easily add new airway graph CSVs to `GNN-AS01/` for further research or clinical adaptation.

## Directory Structure
```
GNN-AS01/                  # Raw and processed airway graph data (CSV)
GNNmodel.py                # GNN model architecture (PyTorch Geometric)
GNN_model_train_val.py     # Model training & validation script
GNN_model_test.py          # Model evaluation on test data
data_container.py          # Data container/wrapper class
data_loader.py             # Data loading & preprocessing utilities
Trained models and wrapped datasets/ # Saved models and datasets
requirements.txt           # Python dependencies
```

## Getting Started
1. **Install dependencies** (recommended: use a virtual environment):
	```sh
	python3 -m venv .venv
	source .venv/bin/activate
	pip install --upgrade pip
	pip install -r requirements.txt
	```
2. **Train the model:**
	```sh
	python GNN_model_train_val.py
	```
3. **Test the model:**
	```sh
	python GNN_model_test.py
	```
4. **Add new data:**
	Place new CSVs in `GNN-AS01/` and update loader logic if the format changes.

## Results & Performance
- Initial models had high error (~1000%), but through systematic optimization (MAPE loss, early stopping, learning rate tuning), the final model achieved ~50% error on unseen data.
- The codebase is structured for rapid experimentation and extension to new datasets or architectures.

## Extensibility & Best Practices
- **Modular Design**: All data access is routed through container/loader classes for consistency and maintainability.
- **Reproducibility**: Model checkpoints and datasets are versioned in `Trained models and wrapped datasets/`.
- **Open Science**: The code is well-commented and ready for adaptation to new research or clinical use cases.

## Dependencies
- Python 3.8+
- numpy, matplotlib, networkx
- torch, torch_geometric

## Contact
For questions, collaboration, or feedback, please contact: [aleksandergornik@gmail.com]
