# Importing necessary modules
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn

# Define input, hidden, and output dimensions
input_dim = 1
hidden_dim = 16  # You can adjust this value based on your requirements
output_dim = 2  # Output features for the second layer (a and b)

# Define the Graph GNN model class
class GraphGNN(nn.Module):
    def __init__(self):
        """
        Initialize the GraphGNN model.

        Parameters:
        - None

        Returns:
        - None
        """
        super(GraphGNN, self).__init__()
        
        # Define graph convolutional layers
        self.conv1 = GCNConv(input_dim, hidden_dim)  
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)  

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the GraphGNN model.

        Parameters:
        - x (Tensor): Node feature matrix.
        - edge_index (LongTensor): Graph edge indices.
        - edge_attr (Tensor): Edge feature matrix.

        Returns:
        - x (Tensor): Output tensor after passing through the GNN layers.

        Note all matricies are to be torch tensors
        """
        # Forward pass through each layer with ReLU activation
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        
        # Output layer without activation (linear transformation)
        x = self.conv3(x, edge_index, edge_attr)

        return x.float()  # Convert the output to float
