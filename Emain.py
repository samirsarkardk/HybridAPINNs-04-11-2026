import torch
import torch.nn as nn
from Aconfig import (Config, DEVICE)
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.optim as optim
from Bmodel import (PINN1, PINN2, PINN3, PINN12, PINN23)
from Dloss import (PDE_Residual, Omega1_Loss, strip1_Loss, Omega2_Loss, strip2_Loss, Omega3_Loss)
from Cdomain import (sample_omega1, sample_omega2, sample_strip1, sample_strip2, 
                     sample_omega3, strip1_lower, strip1_upper, strip2_lower, strip2_upper)
import time
time_start = time.time()
c = Config().c
delta = Config().delta
x_max = Config().x_max
x_min = Config().x_min
t_min = Config().t_min
t_max = Config().t_max


# ==========================================================
# RESIDUAL POINTS
# ==========================================================

Nr1 = 2000
Ns1 = 3000

Nr2 = 2000
Ns2 = 3000

Nr3 = 2000

x_r1 , t_r1 = sample_omega1(Nr1)

x_s1 , t_s1 = sample_strip1(Ns1)

x_r2 , t_r2 = sample_omega2(Nr2)

x_s2 , t_s2 = sample_strip2(Ns2)

x_r3 , t_r3 = sample_omega3(Nr3)

x_r1 = torch.tensor(x_r1, dtype=torch.float64, device=DEVICE)
t_r1 = torch.tensor(t_r1, dtype=torch.float64, device=DEVICE)
x_s1 = torch.tensor(x_s1, dtype=torch.float64, device=DEVICE)
t_s1 = torch.tensor(t_s1, dtype=torch.float64, device=DEVICE)
x_r2 = torch.tensor(x_r2, dtype=torch.float64, device=DEVICE)
t_r2 = torch.tensor(t_r2, dtype=torch.float64, device=DEVICE)
x_s2 = torch.tensor(x_s2, dtype=torch.float64, device=DEVICE)
t_s2 = torch.tensor(t_s2, dtype=torch.float64, device=DEVICE)
x_r3 = torch.tensor(x_r3, dtype=torch.float64, device=DEVICE)
t_r3 = torch.tensor(t_r3, dtype=torch.float64, device=DEVICE)

omega1_x_i = torch.linspace(-1, -0.2 - delta, 200, device=DEVICE).view(-1, 1)
strip1_x_i_1 = torch.linspace(-0.2 - delta, -0.20, 100, device=DEVICE).view(-1, 1)
strip1_x_i_2 = torch.linspace(-0.20, -0.2 + delta, 100, device=DEVICE).view(-1, 1)
omega2_x_i = torch.linspace(-0.2 + delta, 0.2 - delta, 200, device=DEVICE).view(-1, 1)
strip2_x_i_1 = torch.linspace(0.2 - delta, 0.20, 100, device=DEVICE).view(-1, 1)
strip2_x_i_2 = torch.linspace(0.20, 0.2 + delta, 100, device=DEVICE).view(-1, 1)
omega3_x_i = torch.linspace(0.2 + delta, 1, 200, device=DEVICE).view(-1, 1)

# now collect points on the interfaces in the strip1
t_1_I_left = torch.linspace(Config().t_min, Config().t_max, 1000, device=DEVICE).view(-1, 1)
x_1_I_left = -0.2 + c * t_1_I_left  - delta
t_1_I_right = torch.linspace(Config().t_min, Config().t_max, 1000, device=DEVICE).view(-1, 1)
x_1_I_right = -0.2 + c * t_1_I_right  + delta


# now collect points on the interfaces in the strip2
t_2_I_left = torch.linspace(Config().t_min, Config().t_max, 1000, device=DEVICE).view(-1, 1)
x_2_I_left = 0.2 + c * t_2_I_left  - delta
t_2_I_right = torch.linspace(Config().t_min, Config().t_max, 1000, device=DEVICE).view(-1, 1)
x_2_I_right = 0.2 + c * t_2_I_right  + delta


model1 = PINN1().to(DEVICE)
model2 = PINN2().to(DEVICE)
model3 = PINN3().to(DEVICE)
model12 = PINN12().to(DEVICE)
model23 = PINN23().to(DEVICE)


optimizer1 = optim.Adam(model1.parameters(), lr=Config().learning_rate)
optimizer2 = optim.Adam(model2.parameters(), lr=Config().learning_rate)
optimizer3 = optim.Adam(model3.parameters(), lr=Config().learning_rate) 
optimizer12 = optim.Adam(model12.parameters(), lr=Config().learning_rate)
optimizer23 = optim.Adam(model23.parameters(), lr=Config().learning_rate)

for epoch in range(Config().num_epochs):
    optimizer1.zero_grad()
    loss1 = Omega1_Loss(model1, x_r1, t_r1, omega1_x_i, Config().c)
    loss1.backward()
    optimizer1.step()

    optimizer2.zero_grad()
    loss2 = Omega2_Loss(model2, x_r2, t_r2, omega2_x_i, Config().c)
    loss2.backward()
    optimizer2.step()


    optimizer3.zero_grad()
    loss3 = Omega3_Loss(model3, x_r3, t_r3, omega3_x_i, Config().c)
    loss3.backward()    
    optimizer3.step()

    optimizer12.zero_grad()
    loss_strip1 = strip1_Loss(model1, model2, model12, x_s1, t_s1, strip1_x_i_1, strip1_x_i_2,x_1_I_left, t_1_I_left, x_1_I_right, t_1_I_right, Config().c )
    loss_strip1.backward()
    optimizer12.step()   


    optimizer23.zero_grad()
    loss_strip2 = strip2_Loss(model2, model3, model23, x_s2, t_s2, strip2_x_i_1, strip2_x_i_2,x_2_I_left, t_2_I_left, x_2_I_right, t_2_I_right, Config().c)
    loss_strip2.backward()  
    optimizer23.step()

    if epoch % 500 == 0 or epoch == Config().num_epochs - 1:
        print(
            f"Epoch = {epoch:6d} | "
            f"Loss1 = {loss1.item():.6e} | "
            f"Loss2 = {loss2.item():.6e} | "
            f"Loss3 = {loss3.item():.6e} | "
            f"Loss_strip1 = {loss_strip1.item():.6e} | "
            f"Loss_strip2 = {loss_strip2.item():.6e}"
        )


time_end = time.time()
print(f"Training time: {time_end - time_start:.2f} seconds")


os.makedirs("saved_models", exist_ok=True)

torch.save({
    "model1": model1.state_dict(),
    "model2": model2.state_dict(),
    "model3": model3.state_dict(),
    "model12": model12.state_dict(),
    "model23": model23.state_dict(),
}, "saved_models/hybrid_apinn.pth")

print("All trained models saved successfully.")