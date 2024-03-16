# Import necessary modules
import os
from data_loader import AirwayGraph, read_graph_data  # Importing necessary functions from data_loader module
import pickle  # Importing pickle module for serialization
import re  # Importing re module for regular expressions

# Define the folder path where the data is located
folder_path = "GNN-AS01_mini"

# Define a class to contain graphs
class GraphContainer:
    def __init__(self):
        """
        Initialize the GraphContainer class.

        Parameters:
        - None

        Returns:
        - None
        """
        self.graphs = []

    def add_graph(self, graph):
        """
        Add a graph to the container.

        Parameters:
        - graph (AirwayGraph): Graph to add to the container.

        Returns:
        - None
        """
        self.graphs.append(graph)

    def num_graphs(self):
        """
        Get the number of graphs in the container.

        Parameters:
        - None

        Returns:
        - int: Number of graphs in the container.
        """
        return len(self.graphs)

    def save_container(self, filename):
        """
        Save the container to a file using pickle serialization.

        Parameters:
        - filename (str): Name of the file to save the container.

        Returns:
        - None
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.graphs, f)

    @classmethod
    def load_container(cls, filename):
        """
        Load a container from a file.

        Parameters:
        - filename (str): Name of the file to load the container from.

        Returns:
        - GraphContainer: Loaded container object.
        """
        with open(filename, 'rb') as f:
            container = cls()
            container.graphs = pickle.load(f)
        
        print("Number of graphs in container is: ",len(container.graphs))
        return container
    
    def split_data(self, train_ratio, val_ratio, test_ratio):
        """
        Split the data into training, validation, and test sets.

        Parameters:
        - train_ratio (float): Ratio of data for training.
        - val_ratio (float): Ratio of data for validation.
        - test_ratio (float): Ratio of data for testing.

        Returns:
        - tuple: Three lists containing training, validation, and test sets.
        """
        total_size = len(self.graphs)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size

        # Split the shuffled list into training, validation, and test sets
        train_set = self.graphs[:train_size]
        val_set = self.graphs[train_size:train_size + val_size]
        test_set = self.graphs[train_size + val_size:]

        return train_set, val_set, test_set
    

def load_data_and_split_sets(folder_path):
    """
    Load data from CSV files in the given folder and split it into training, validation, and test sets.

    Parameters:
    - folder_path (str): Path to the folder containing CSV files.

    Returns:
    - list: Three lists containing training, validation, and test sets.
    - num_of_graphs (int)
    """
    # Initialize a container to hold graphs
    container = GraphContainer()
    num_loaded_graphs = 0  # Initialize a counter for loaded graphs

    # Define a function to extract the numeric part after "AS01_"
    def extract_number(filename):
        match = re.search(r'AS01_(\d+)\.csv', filename)
        if match:
            return int(match.group(1))
        else:
            return -1  # Return a large negative number if no number is found

    # Get a list of CSV file names in the folder and sort them based on the numeric part after "AS01_"
    csv_files = sorted([file_name for file_name in os.listdir(folder_path) if file_name.endswith(".csv")], key=extract_number)

    # Iterate through each CSV file
    for csv_file_name in csv_files:
        try:
            # Construct the full path of the CSV file
            csv_file_path = os.path.join(folder_path, csv_file_name)
            
            # Construct the full path of the corresponding .edge file
            edge_file_path = os.path.join(folder_path, "generated_airways.edge")

            print("Graph with files of :", csv_file_path, " and ", edge_file_path, " is being loaded")
            
            # Read node and edge data from CSV and .edge files
            node_data, edge_data = read_graph_data(csv_file_path, edge_file_path)
            
            # Create an AirwayGraph object
            graph = AirwayGraph(node_data, edge_data)
            
            # Add the graph to the container
            container.add_graph(graph)

            print("Successfully added to container")
            num_loaded_graphs += 1  # Increment the counter for loaded graphs

        except Exception as e:
            print("Error occurred while loading graph:", e)

    # Split the data into training, validation, and test sets
    train_set, val_set, test_set = container.split_data(train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)

    # Save all the graphs in the container
    container.save_container("graph_container.pkl")
    print("Graph container saved.")

    # Output the number of graphs loaded
    print("Number of graphs saved:", num_loaded_graphs)

    return train_set, val_set, test_set, num_loaded_graphs



