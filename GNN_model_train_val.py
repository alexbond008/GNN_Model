# Import necessary modules
from torch.utils.data import DataLoader  # Importing DataLoader for batch processing
from data_container import load_data_and_split_sets # Importing datasets from data_container module
import torch  # Importing PyTorch library
from GNNmodel import GraphGNN  # Importing the GraphGNN model from GNNmodel module
import torch.nn as nn  # Importing neural network modules from PyTorch
import torch.optim as optim  # Importing optimization algorithms from PyTorch
import matplotlib.pyplot as plt  # Importing matplotlib for visualization

def plot_loss_curve(train_losses, val_losses, num_graphs, filename='loss_curve.png'):
    """
    Plot the training and validation loss curves.

    Parameters:
    - train_losses (list): List of training losses for each epoch.
    - val_losses (list): List of validation losses for each epoch.
    - num_graphs (int): Number of graphs the model was trained on.
    - filename (str): Name of the file to save the plot.

    Returns:
    - None
    """
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title(f'Training and Validation Loss Over Epochs\n(Logarithmic Scale) - Trained on {num_graphs} Graphs')
    plt.legend()
    plt.savefig(filename)
    plt.show()

# Define a function to train the model
def train_model(model, train_set, criterion, optimizer, num_epochs=25):
    """
    Train the model.

    Parameters:
    - model (GraphGNN): Graph neural network model.
    - train_set (list): List of training data.
    - criterion (torch.nn.Module): Loss function.
    - optimizer (torch.optim.Optimizer): Optimization algorithm.
    - num_epochs (int): Number of epochs for training.

    Returns:
    - list: List of training losses for each epoch.
    """
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for graph in train_set:
            
            # Extract necessary data tensors
            true_values, combined_tensor = graph.torch_tensor_conversion(graph.Pressure, graph.FlowRate, graph.NodeX, graph.NodeY, graph.NodeZ, graph.Radius)

            # Get edges and edge weights
            edges, edge_weights = graph.calculate_edges_and_weights(graph.G)

            # Get AcinarID
            acinar_id = graph.get_AcinarID(graph.AcinarID)
            
            # Forward pass
            predictions = model(combined_tensor.view(-1, 1), edges.t(), edge_weights)

            # Apply acinarID filtering
            for i in range(len(acinar_id)):
                if acinar_id[i] == 1:
                    predictions[i] = true_values[i]

            # Compute loss
            loss = criterion(predictions, true_values).float()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_set)
        losses.append(epoch_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}')

    # Save the trained model
    torch.save(model.state_dict(), 'multi_graph_trained_model.pth')
    return losses

def train_model_with_validation(model, train_set, val_set, criterion, optimizer, num_epochs=25, lr_scheduler=None, early_stopping=None, patience=5):
    """
    Train the model with validation.

    Parameters:
    - model (GraphGNN): Graph neural network model.
    - train_set (list): List of training data.
    - val_set (list): List of validation data.
    - criterion (torch.nn.Module): Loss function.
    - optimizer (torch.optim.Optimizer): Optimization algorithm.
    - num_epochs (int): Number of epochs for training.
    - lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
    - early_stopping (bool): Whether to perform early stopping.
    - patience (int): Patience parameter for early stopping.

    Returns:
    - tuple: Lists of training and validation losses for each epoch.
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    no_improvement_count = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for graph in train_set:
            true_values, combined_tensor = graph.torch_tensor_conversion(graph.Pressure, graph.FlowRate, graph.NodeX, graph.NodeY, graph.NodeZ, graph.Radius)
            edges, edge_weights = graph.calculate_edges_and_weights(graph.G)
            acinar_id = graph.get_AcinarID(graph.AcinarID)

            optimizer.zero_grad()
            predictions = model(combined_tensor.view(-1, 1), edges.t(), edge_weights)

            for i in range(len(acinar_id)):
                if acinar_id[i] == 1:
                    predictions[i] = true_values[i]

            loss = criterion(predictions, true_values)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_set)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for graph in val_set:
                true_values, combined_tensor = graph.torch_tensor_conversion(graph.Pressure, graph.FlowRate, graph.NodeX, graph.NodeY, graph.NodeZ, graph.Radius)
                edges, edge_weights = graph.calculate_edges_and_weights(graph.G)
                acinar_id = graph.get_AcinarID(graph.AcinarID)

                predictions = model(combined_tensor.view(-1, 1), edges.t(), edge_weights)

                for i in range(len(acinar_id)):
                    if acinar_id[i] == 1:
                        predictions[i] = true_values[i]

                val_loss += criterion(predictions, true_values).item()

            val_loss /= len(val_set)
            val_losses.append(val_loss)

            # Early stopping
            if early_stopping and val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                print(f'Early stopping after {epoch + 1} epochs due to no improvement in validation loss.')
                break

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}')

        # Learning rate adjustment
        if lr_scheduler:
            lr_scheduler.step(val_loss)

    # Save the trained model
    torch.save(model.state_dict(), 'Trained models and wrapped datasets/train_model_with_validation.pth')
    return train_losses, val_losses

folder_path = "GNN-AS01_mini"

# load dataset
train_set, val_set, test_set, num_loaded_graphs = load_data_and_split_sets(folder_path)

# Example usage
model = GraphGNN()  # Instantiate the GraphGNN model
criterion = nn.MSELoss()  # Define the loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define the optimization algorithm
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.5, patience=3)# Define the learning rate scheduler
early_stopping = True  # Enable early stopping

# Train the model with validation
train_losses, val_losses = train_model_with_validation(model, train_set, val_set, criterion, optimizer, lr_scheduler=lr_scheduler, early_stopping=early_stopping)

# Plot the training and validation loss curves
plot_loss_curve(train_losses,val_losses,num_loaded_graphs)
