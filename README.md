# Application of Graph Neural Networks for Simulating Airway Flow in the Lungs

## Overview
This repository contains the code and data for my final year dissertation titled **"Application of Graph Neural Networks for Simulating Airway Flow in the Lungs"**. The project focuses on predicting lung ventilation using a novel approach that applies Graph Neural Networks (GNNs). This work explores how effectively GNNs can simulate lung airways and predict pressure and flow rate at various points within the lung's airway tree.

## Project Motivation
Accurate simulations of airflow in lungs are essential for diagnosing and treating lung diseases. Traditional models often compromise between anatomical realism and computational efficiency. This project aims to bridge this gap by using GNNs, which can naturally model the lung's tree-like structure. The ultimate goal is to predict pressure and flow rates within a 30% error margin for unseen patient data.

## Key Features
- **Graph Representation**: Lungs modeled as a graph where nodes represent airways, and edges represent airflow paths.
- **Data**: The model uses a dataset generated from a full-scale airway network model, which includes features such as pressure, radius, and spatial coordinates.
- **Model**: A 6-layer Convolutional Graph Neural Network (GNN) was implemented with ReLU activation functions.
- **Improvements**: The final model achieved a percentage error of 50% after experimentation with learning rates, loss functions (Mean Absolute Percentage Error - MAPE), and early stopping.

## Project Files
- `GNN-AS01/`: Contains the raw data and processed graphs representing lung structures.
- `GNNModel.py`: Python script that holds the model architecture.
- `GNN_model_train_val.py`: Run this file to see how the model is learning about the airway flow of the lungs!
- `GNN_model_test.py`: Run this file to see how the model faces data it has never seen before!
- `data_container.py` and `data_loader.py`: These files work to together wrap all the data in one class. Here is a quick look at what they do:
![image](https://github.com/user-attachments/assets/014243ba-784c-48df-bab0-38db9d07bc7b)
 

## Libraries used
- **Data Preparation**: `NetworkX` is used to create graph representations of lung airways.
- **Model Training**: `Pytorch geometric` is used for GNN model

## Results
Initial models produced high errors (~1000%) but through various optimizations (MAPE loss, early stopping, etc.), the final model reduced errors to approximately 50%.

## Future Work
Future improvements could include:
- Developing a physics-informed GNN by incorporating known equations of airflow.
- Scaling up the dataset and using more complex neural network architectures.

## Contact
For any questions, please contact me at: [aleksandergornik@gmail.com].
