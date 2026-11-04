import torch
import torch.nn as nn
import torch.optim as optim
from Aconfig import (Config, DEVICE)
import numpy as np
from Cdomain import interface_left, interface_right

delta = Config().delta

def strip_mass_loss(model, x_center_fn, delta, n_t=50, n_x=200):
    t = torch.rand(n_t, 1, device=DEVICE)
    off = torch.linspace(-delta, delta, n_x, device=DEVICE).view(1, -1)
    x = x_center_fn(t) + off                              # (n_t, n_x)
    u = model(x.reshape(-1, 1), t.repeat(1, n_x).reshape(-1, 1)).reshape(n_t, n_x)
    mass = torch.trapz(u, x, dim=1)
    return torch.mean((mass / delta - 1.0) ** 2)             # normalized, since delta is tiny

def PDE_Residual(model, x, t, c):
    # Ensure x and t require gradients for autograd
    x.requires_grad_(True)
    t.requires_grad_(True)

    # Forward pass through the model
    u = model(x, t)

    # Compute gradients
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    # PDE residual for the advection equation: u_t + c * u_x = 0
    residual = u_t + c * u_x

    return residual

def Omega1_Loss(model, x, t, x_i, c):
    # Compute the PDE residual for Omega1
    residual = PDE_Residual(model, x, t, c)
    
    # Compute the loss as the mean squared error of the residual
    ResidualLoss = nn.MSELoss()(residual, torch.zeros_like(residual))
    
    # compute the initial condition loss
    u_i = model(x_i, torch.zeros_like(x_i))
    InitialConditionLoss = nn.MSELoss()(u_i, torch.zeros_like(u_i))

    t_bc = torch.rand(200, 1, device=DEVICE)
    x_bc = -torch.ones_like(t_bc)
    BC_loss = nn.MSELoss()(model(x_bc, t_bc), torch.zeros_like(t_bc))

    return ResidualLoss + InitialConditionLoss + BC_loss

def strip1_Loss(model1, model2, model12, x, t, x_i_1, x_i_2, x_I_left, t_I_left, x_I_right, t_I_right, c):
    # Compute the PDE residual for Strip1
    residual = PDE_Residual(model12, x, t, c)
    
    # Compute the loss as the mean squared error of the residual
    ResidualLoss = nn.MSELoss()(residual, torch.zeros_like(residual))

    # compute the initial condition loss
    u_i_1 = model12(x_i_1, torch.zeros_like(x_i_1))
    InitialConditionLoss_1 = nn.MSELoss()(u_i_1, torch.zeros_like(u_i_1))

    u_i_2 = model12(x_i_2, torch.zeros_like(x_i_2))
    InitialConditionLoss_2 = nn.MSELoss()(u_i_2, torch.ones_like(u_i_2))

    # interface residual loss
    ResLeftStripLine_1 = PDE_Residual(model1, x_I_left, t_I_left, c)
    ResLeftStripLine_12 = PDE_Residual(model12, x_I_left, t_I_left, c)
    ResRightStripLine_2 = PDE_Residual(model2, x_I_right, t_I_right, c)
    ResRightStripLine_12 = PDE_Residual(model12, x_I_right, t_I_right, c)
    InterfaceResLoss = nn.MSELoss()(ResLeftStripLine_1, ResLeftStripLine_12) + nn.MSELoss()(ResRightStripLine_2, ResRightStripLine_12)

    # interface average loss
    AveLeftStripLine_1 = model1(x_I_left, t_I_left)
    AveLeftStripLine_12 = model12(x_I_left, t_I_left)
    AveRightStripLine_2 = model2(x_I_right, t_I_right)
    AveRightStripLine_12 = model12(x_I_right, t_I_right)
    InterfaceAveLoss = nn.MSELoss()(AveLeftStripLine_1, AveLeftStripLine_12) + nn.MSELoss()(AveRightStripLine_2, AveRightStripLine_12)

    MassLoss = strip_mass_loss(model12, interface_left, delta)



    return (ResidualLoss + InitialConditionLoss_1 + InitialConditionLoss_2 + InterfaceResLoss + InterfaceAveLoss + MassLoss)


def Omega2_Loss(model, x, t, x_i, c):
    # Compute the PDE residual for Omega2
    residual = PDE_Residual(model, x, t, c)
    
    # Compute the loss as the mean squared error of the residual
    ResidualLoss = nn.MSELoss()(residual, torch.zeros_like(residual))
    
    # compute the initial condition loss
    u_i = model(x_i, torch.zeros_like(x_i))
    InitialConditionLoss = nn.MSELoss()(u_i, torch.ones_like(u_i))

    return ResidualLoss + InitialConditionLoss


def strip2_Loss(model2, model3, model23, x, t, x_i_1, x_i_2,x_I_left, t_I_left, x_I_right, t_I_right, c):
    # Compute the PDE residual for Strip2
    residual = PDE_Residual(model23, x, t, c)
    
    # Compute the loss as the mean squared error of the residual
    ResidualLoss = nn.MSELoss()(residual, torch.zeros_like(residual))

    # compute the initial condition loss
    u_i_1 = model23(x_i_1, torch.zeros_like(x_i_1))
    InitialConditionLoss_1 = nn.MSELoss()(u_i_1, torch.ones_like(u_i_1))

    u_i_2 = model23(x_i_2, torch.zeros_like(x_i_2))
    InitialConditionLoss_2 = nn.MSELoss()(u_i_2, torch.zeros_like(u_i_2))

    # interface residual loss
    ResLeftStripLine_2 = PDE_Residual(model2, x_I_left, t_I_left, c)
    ResLeftStripLine_23 = PDE_Residual(model23, x_I_left, t_I_left, c)
    ResRightStripLine_3 = PDE_Residual(model3, x_I_right, t_I_right, c)
    ResRightStripLine_23 = PDE_Residual(model23, x_I_right, t_I_right, c)
    InterfaceResLoss = nn.MSELoss()(ResLeftStripLine_2, ResLeftStripLine_23) + nn.MSELoss()(ResRightStripLine_3, ResRightStripLine_23)

    # interface average loss
    AveLeftStripLine_2 = model2(x_I_left, t_I_left)
    AveLeftStripLine_23 = model23(x_I_left, t_I_left)
    AveRightStripLine_3 = model3(x_I_right, t_I_right)
    AveRightStripLine_23 = model23(x_I_right, t_I_right)
    InterfaceAveLoss = nn.MSELoss()(AveLeftStripLine_2, AveLeftStripLine_23) + nn.MSELoss()(AveRightStripLine_3, AveRightStripLine_23)

    MassLoss = strip_mass_loss(model23, interface_right, delta)

    return (ResidualLoss + InitialConditionLoss_1 + InitialConditionLoss_2 + InterfaceAveLoss + InterfaceResLoss + MassLoss)



def Omega3_Loss(model, x, t, x_i, c):
    # Compute the PDE residual for Omega3
    residual = PDE_Residual(model, x, t, c)
    
    # Compute the loss as the mean squared error of the residual
    ResidualLoss = nn.MSELoss()(residual, torch.zeros_like(residual))
    
    # compute the initial condition loss
    u_i = model(x_i, torch.zeros_like(x_i))
    InitialConditionLoss = nn.MSELoss()(u_i, torch.zeros_like(u_i))

    return ResidualLoss + InitialConditionLoss

