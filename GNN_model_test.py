from GNNmodel import GraphGNN
from data_container import GraphContainer
import torch
import torch.nn as nn


def test_model(model, test_set, criterion):
    """
    Test the trained model on the test set.

    Args:
    - model: The trained model to be tested.
    - test_set: The dataset containing the test samples.
    - criterion: The loss criterion used for evaluation.

    Returns:
    - avg_test_loss: The average test loss calculated over all samples in the test set.
    """
    # Set model to evaluation mode
    model.eval()

    # Initialize list to store test losses
    test_losses = []

    # Iterate through the test set
    for graph in test_set:
        # Extract necessary data tensors
        true_values, combined_tensor = graph.torch_tensor_conversion(graph.Pressure, graph.FlowRate, graph.NodeX, graph.NodeY, graph.NodeZ, graph.Radius)
        edges, edge_weights = graph.calculate_edges_and_weights(graph.G)
        acinar_id = graph.get_AcinarID(graph.AcinarID)
        
        # Forward pass
        with torch.no_grad():
            predictions = model(combined_tensor.view(-1, 1), edges.t(), edge_weights)
            
            # Apply acinarID filtering if necessary
            for i in range(len(acinar_id)):
                if acinar_id[i] == 1:
                    predictions[i] = true_values[i]
            
            # Calculate loss
            test_loss = criterion(predictions, true_values).item()
            test_losses.append(test_loss)

    avg_test_loss = sum(test_losses) / len(test_losses)

    return avg_test_loss

container = GraphContainer.load_container('graph_container.pkl')

train_set, val_set, test_set = container.split_data(train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)

model = GraphGNN()
model.load_state_dict(torch.load('Trained models and wrapped datasets/train_model_with_validation.pth'))

criterion = nn.MSELoss()

avg_test_loss = test_model(model, test_set, criterion)

print(f'Average Test Loss: {avg_test_loss}')


