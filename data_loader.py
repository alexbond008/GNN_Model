import csv
import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt

def read_graph_data(node_data, edge_data):
    columns = {}
    with open(node_data, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for key, value in row.items():
                if key not in columns:
                    columns[key] = [value]
                else:
                    columns[key].append(value)

    for column in columns:
        columns[column] = [float(value) for value in columns[column]]

    num_nodes = len(columns['Points:0'])

    # Extract node attributes
    node_data = {
        'NodeX': np.array(columns['Points:0']),
        'NodeY': np.array(columns['Points:1']),
        'NodeZ': np.array(columns['Points:2']),
        'Radius': np.array(columns['AwayRadius_mm']),
        'Pressure': np.array(columns['Pressure_Pa']),
        'FlowRate': np.array(columns['FlowRate_mL']),
        'AcinarID': np.array(columns['AcinarID'])
    }

    # Read edge data from file
    edgsta = []
    edgend = []
    with open(edge_data, "r") as edgefile:
        edgefile.readline()  # Skip the header
        for line in edgefile:
            edgerow = line.strip().split("\t")
            edgsta.append(int(edgerow[1]))
            edgend.append(int(edgerow[2]))

    edge_data = {
        'Edgsta': edgsta,
        'Edgend': edgend
    }

    return node_data, edge_data
    
class AirwayGraph:
    def __init__(self, node_data, edge_data):
        self.NodeX = node_data['NodeX']
        self.NodeY = node_data['NodeY']
        self.NodeZ = node_data['NodeY']
        self.Radius = node_data['Radius']
        self.Pressure = node_data['Pressure']
        self.FlowRate = node_data['FlowRate']
        self.AcinarID = node_data['AcinarID']
        self.G = self.create_nxgraph(edge_data)
        self.edges, self.edge_weights = self.calculate_edges_and_weights(self.G)
        self.true_vales,self.combined_tensor = self.torch_tensor_conversion(self.Pressure,self.FlowRate,self.NodeX,self.NodeY,self.NodeZ,self.Radius)

    def get_AcinarID(self,AcinarID):
        return AcinarID

    def create_nxgraph(self,edge_data):
        G = nx.Graph()

        edgsta = edge_data['Edgsta']
        edgend = edge_data['Edgend']

        nodenum = [i + 1 for i in range(len(self.NodeX))]
        G.add_nodes_from(nodenum)

        edge_array = [(edgsta[i] + 1, edgend[i] + 1) for i in range(len(edgsta))]
        G.add_edges_from(edge_array)

        return G

    def calculate_edges_and_weights(self, G):

        A = nx.adjacency_matrix(G)
        adj_tensor = torch.tensor(A.todense())
        upper_triangular = torch.triu(adj_tensor)
        # Find non-zero elements in the upper triangular part
        edges = torch.nonzero(upper_triangular, as_tuple=False)
        node_radius_dict = {i: radius for i, radius in enumerate(self.Radius)}

        edge_weights = torch.sqrt(torch.tensor([node_radius_dict[edge[0].item()]**2 + node_radius_dict[edge[1].item()]**2 for edge in edges])).float()
        return edges, edge_weights

    def visualize_graph(self,G):
        pos = {i + 1: [self.NodeX[i], self.NodeY[i]] for i in range(len(self.NodeX))}
        nx.draw_networkx(G, pos, node_size=50)
        nx.draw_networkx_labels(G, pos, font_size=0.1)
        plt.show()

    def torch_tensor_conversion(self,Pressure,Flowrate,NodeX,NodeY,NodeZ,Radius):
        P_tensor = torch.tensor(Pressure)
        Q_tensor = torch.tensor(Flowrate)

        # Concatenate pressure and flow rate tensors along the second dimension
        true_values = torch.cat((P_tensor.unsqueeze(1), Q_tensor.unsqueeze(1)), dim=1).float()

        # Convert x, y, z, and R to torch tensors
        x_tensor = torch.tensor(NodeX)
        y_tensor = torch.tensor(NodeY)
        z_tensor = torch.tensor(NodeZ)
        R_tensor = torch.tensor(Radius)

        # Stack x, y, and z tensors along the second dimension
        pos_tensor = torch.stack((x_tensor, y_tensor, z_tensor), dim=1)

        # Combine R and pos tensors along the second dimension
        combined_features = torch.cat([R_tensor.unsqueeze(1), pos_tensor], dim=1).float()

        # Calculate combined tensor by taking the product along the second dimension
        combined_tensor = torch.prod(combined_features, dim=1)
        
        return true_values,combined_tensor



    

