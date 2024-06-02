import h5
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ACANN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.05):
        """Builds ACANN network with an arbitrary number of hidden layers.

        Arguments
        ----------
        input_size : integer, size of the input
        output_size : integer, size of the output layer
        hidden_layers: list of integers, the sizes of the hidden layers
        drop_p: float in (0,1), value of the dropout probability
        """
        super().__init__()
        # Add the first layer: input_size into the first hidden layer
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0]).to(device)])
        self.normalizations = nn.ModuleList([nn.BatchNorm1d(input_size).to(device)])

        # Add the other layers
        layers_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.layers.extend([nn.Linear(h1, h2).to(device) for h1, h2 in layers_sizes])
        self.normalizations.extend([nn.BatchNorm1d(size).to(device) for size in hidden_layers])

        self.output = nn.Linear(hidden_layers[-1], output_size).to(device)
        self.dropout = nn.Dropout(drop_p).to(device)

    def forward(self, x):
        # Pass through each layer
        for layer, normalization in zip(self.layers, self.normalizations):
            x = normalization(x)
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.output(x)
        return x  # No activation function on the output layer for regression

def load_data(files):
    dlr_list = []
    A_list = []
    dlr_shape = None
    A_shape = None
    statistics = None

    for data in files:
        with h5.HDFArchive(data, "r") as B:
            dlr_data = B["dlr"]
            A_data = B["A"]

            if dlr_shape is None:
                dlr_shape = dlr_data.shape[1:]
            if A_shape is None:
                A_shape = A_data.shape[1:]
            if statistics is None:
                statistics = B["statistics"]

            if dlr_data.shape[1:] != dlr_shape:
                raise ValueError(f"Inconsistent shapes for dlr data: expected {dlr_shape}, got {dlr_data.shape[1:]}")
            if A_data.shape[1:] != A_shape:
                raise ValueError(f"Inconsistent shapes for A data: expected {A_shape}, got {A_data.shape[1:]}")
            if B["statistics"] != statistics:
                raise ValueError("Inconsistent statistics")

            dlr_list.append(dlr_data)
            A_list.append(A_data)

    dlr_combined = torch.from_numpy(np.vstack(dlr_list)).double().to(device)
    A_combined = torch.from_numpy(np.vstack(A_list)).double().to(device)

    return TensorDataset(dlr_combined, A_combined), statistics
