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


class PINN12(nn.Module):
    def __init__(self):
        super(PINN12, self).__init__()
        self.input_dim = config.input_dim_12
        self.hidden_dim = config.hidden_dim_12
        self.output_dim = config.output_dim_12
        self.num_hidden_layers = config.num_hidden_layers_12

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
      
      xi = (x - (-0.2 + config.c * t)) / config.delta
      return self.model(torch.cat([xi, t], dim=1))


class PINN23(nn.Module):
    def __init__(self):
        super(PINN23, self).__init__()
        self.input_dim = config.input_dim_23
        self.hidden_dim = config.hidden_dim_23
        self.output_dim = config.output_dim_23
        self.num_hidden_layers = config.num_hidden_layers_23

        # Define the layers of the neural
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
       xi = (x - (0.2 + config.c * t)) / config.delta
       return self.model(torch.cat([xi, t], dim=1))
