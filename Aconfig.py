import numpy as np 
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
torch.set_default_dtype(torch.float64)


class Config:
    def __init__(self):
        # set random seed for reproducibility
        self.seed = 42
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)


        # Training Parameters
        self.num_epochs = 15000
        self.learning_rate = 8e-4

        # Model Parameters for main pinn1 networks
        self.input_dim_1 = 2
        self.hidden_dim_1 = 20
        self.output_dim_1 = 1
        self.num_hidden_layers_1 = 2

        # Model Parameters for main pinn12 networks
        self.input_dim_12 = 2
        self.hidden_dim_12 = 20
        self.output_dim_12 = 1
        self.num_hidden_layers_12 = 6

        # Model Parameters for main pinn2 networks
        self.input_dim_2 = 2
        self.hidden_dim_2 = 20
        self.output_dim_2 = 1
        self.num_hidden_layers_2 = 2


        # Model Parameters for main pinn23 networks
        self.input_dim_23 = 2
        self.hidden_dim_23 = 20
        self.output_dim_23 = 1
        self.num_hidden_layers_23 = 6    
       

        # Model Parameters for main pinn3 networks
        self.input_dim_3 = 2
        self.hidden_dim_3 = 20
        self.output_dim_3 = 1
        self.num_hidden_layers_3 = 2
        
        # Domain Specifications
        self.x_min = -1.0
        self.x_max = 1.0
        self.t_min = 0.0
        self.t_max = 1.0
        self.delta = 0.000005

        # PDE Equation needs
        self.c = 0.5