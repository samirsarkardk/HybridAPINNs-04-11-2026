import torch
import torch.nn as nn
import numpy as np
from Aconfig import (Config, DEVICE)

config = Config()

class PINN1(nn.Module):
    def __init__(self):
        super(PINN1, self).__init__()
        self.input_dim = config.input_dim_1
        self.hidden_dim = config.hidden_dim_1
        self.output_dim = config.output_dim_1
        self.num_hidden_layers = config.num_hidden_layers_1

        # Define the layers of the neural network
        layers = []
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(self.num_hidden_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        
        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x, t):
      inputs = torch.cat([x, t], dim=1)
      return t * self.model(inputs)


class PINN2(nn.Module):
    def __init__(self):
        super(PINN2, self).__init__()
        self.input_dim = config.input_dim_2
        self.hidden_dim = config.hidden_dim_2
        self.output_dim = config.output_dim_2
        self.num_hidden_layers = config.num_hidden_layers_2

        # Define the layers of the neural network
        layers = []
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(self.num_hidden_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        
        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x, t):
       inputs = torch.cat([x, t], dim=1)
       return 1.0 + t * self.model(inputs)


class PINN3(nn.Module):
    def __init__(self):
        super(PINN3, self).__init__()
        self.input_dim = config.input_dim_3
        self.hidden_dim = config.hidden_dim_3
        self.output_dim = config.output_dim_3
        self.num_hidden_layers = config.num_hidden_layers_3

        # Define the layers of the neural network
        layers = []
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(self.num_hidden_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        
        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x, t):
      inputs = torch.cat([x, t], dim=1)
      return t * self.model(inputs)




class GateNetwork(nn.Module):
    def __init__(self):
        super(GateNetwork, self).__init__()
        self.input_dim = config.input_dim_gate
        self.hidden_dim = config.hidden_dim_gate
        self.output_dim = config.output_dim_gate
        self.num_hidden_layers = config.num_hidden_layers_gate

        # Define the layers of the neural network
        layers = []
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(self.num_hidden_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        
        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x, t):
        # Concatenate x and t before passing to the network
        inputs = torch.cat([x, t], dim=1)
        logits = self.model(inputs)
        gates = torch.softmax(logits, dim=1)  # Apply softmax to get gate values
        return gates

    
# ==========================================================
# APINN 1
# ==========================================================
class APINN1(nn.Module):
    def __init__(self, model1, model2, gate1):
        super().__init__()

        self.model1 = model1
        self.model2 = model2
        self.gate1 = gate1

    def forward(self, x, t):
        gates = self.gate1(x, t)

        G1 = gates[:, 0:1]
        G2 = gates[:, 1:2]

        u1 = self.model1(x, t)
        u2 = self.model2(x, t)

        return G1 * u1 + G2 * u2

class APINN2(nn.Module):
    def __init__(self, model2, model3, gate2):
        super().__init__()

        self.model2 = model2
        self.model3 = model3
        self.gate2 = gate2

    def forward(self, x, t):
        gates = self.gate2(x, t)

        G1 = gates[:, 0:1]
        G2 = gates[:, 1:2]

        u2 = self.model2(x, t)
        u3 = self.model3(x, t)

        return G1 * u2 + G2 * u3