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

# Residual points in the strip1 and strip2
Ns1 = 3000
Ns2 = 3000

x_s1 , t_s1 = sample_strip1(Ns1)
x_s2 , t_s2 = sample_strip2(Ns2)

x_s1 = torch.tensor(x_s1, dtype=torch.float64, device=DEVICE)
t_s1 = torch.tensor(t_s1, dtype=torch.float64, device=DEVICE)

x_s2 = torch.tensor(x_s2, dtype=torch.float64, device=DEVICE)
t_s2 = torch.tensor(t_s2, dtype=torch.float64, device=DEVICE)

# initial points in both the strips

strip1_x_i_1 = torch.linspace(-0.2 - delta, -0.20, 100, device=DEVICE).view(-1, 1)
strip1_x_i_2 = torch.linspace(-0.20, -0.2 + delta, 100, device=DEVICE).view(-1, 1)

strip2_x_i_1 = torch.linspace(0.2 - delta, 0.20, 100, device=DEVICE).view(-1, 1)
strip2_x_i_2 = torch.linspace(0.20, 0.2 + delta, 100, device=DEVICE).view(-1, 1)

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

# models

model1 = PINN1().to(DEVICE)
model2 = PINN2().to(DEVICE)
model3 = PINN3().to(DEVICE)
model12 = PINN12().to(DEVICE)
model23 = PINN23().to(DEVICE)

checkpoint_path = "saved_models/hybrid_apinn.pth"

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=True
)

model1.load_state_dict(checkpoint["model1"])
model2.load_state_dict(checkpoint["model2"])
model3.load_state_dict(checkpoint["model3"])
model12.load_state_dict(checkpoint["model12"])
model23.load_state_dict(checkpoint["model23"])


print("Stage 1 parameters loaded successfully.")

# ==========================================================
# FREEZE PINN1 AND PINN2 AND PINN3
# ==========================================================

for parameter in model1.parameters():
    parameter.requires_grad_(False)

for parameter in model2.parameters():
    parameter.requires_grad_(False)

for parameter in model3.parameters():
    parameter.requires_grad_(False)

model1.eval()
model2.eval()
model3.eval()

# PINN12 and PINN23 is the only trainable network in Stage 2.
for parameter in model12.parameters():
    parameter.requires_grad_(True)

model12.train()

for parameter in model23.parameters():
    parameter.requires_grad_(True)

model23.train()

optimizer12 = optim.Adam(
    model12.parameters(),
    lr=Config().learning_rate
)

optimizer23 = optim.Adam(
    model23.parameters(),
    lr=Config().learning_rate
)


# # ==========================================================
# # STAGE 2: OPTIMIZE ONLY PINN12 AND PINN23
# # ==========================================================

# num_epochs_stage2 = Config().num_epochs

# for epoch in range(num_epochs_stage2):

#     optimizer12.zero_grad()
#     loss_strip1 = strip1_Loss(model1, model2, model12, x_s1, t_s1, strip1_x_i_1, strip1_x_i_2,x_1_I_left, t_1_I_left, x_1_I_right, t_1_I_right, Config().c )
#     loss_strip1.backward()
#     optimizer12.step()   


#     optimizer23.zero_grad()
#     loss_strip2 = strip2_Loss(model2, model3, model23, x_s2, t_s2, strip2_x_i_1, strip2_x_i_2,x_2_I_left, t_2_I_left, x_2_I_right, t_2_I_right, Config().c)
#     loss_strip2.backward()  
#     optimizer23.step()

#     if epoch % 50 == 0 or Config().num_epochs - 1 == 0:
#         print(
#             f"EPOCH = {epoch:6d} | "
#             f"Loss_strip2 = {loss_strip2.item():.6e} | "
#             f"Loss_strip2 = {loss_strip2.item():.6e} | "
#         )


# time_end = time.time()
# print(f"Training time: {time_end - time_start:.2f} seconds")

# START LBFGS

# ==========================================================
# STAGE 2: OPTIMIZE ONLY PINN12 AND PINN23 USING L-BFGS
# ==========================================================

num_epochs_stage2 = 1000

# Initialize L-BFGS optimizers
optimizer12 = torch.optim.LBFGS(
    model12.parameters(),
    lr=1.0,
    max_iter=20,
    history_size=100,
    line_search_fn="strong_wolfe"
)

optimizer23 = torch.optim.LBFGS(
    model23.parameters(),
    lr=1.0,
    max_iter=20,
    history_size=100,
    line_search_fn="strong_wolfe"
)

for epoch in range(num_epochs_stage2):

    # ------------------------------------------------------
    # Optimize PINN12
    # ------------------------------------------------------
    def closure12():
        optimizer12.zero_grad()

        loss_strip1 = strip1_Loss(
            model1, model2, model12,
            x_s1, t_s1,
            strip1_x_i_1, strip1_x_i_2,
            x_1_I_left, t_1_I_left,
            x_1_I_right, t_1_I_right,
            Config().c
        )

        loss_strip1.backward()
        return loss_strip1

    optimizer12.step(closure12)

    # Evaluate loss after the optimizer update
    with torch.enable_grad():
        loss_strip1_value = strip1_Loss(
            model1, model2, model12,
            x_s1, t_s1,
            strip1_x_i_1, strip1_x_i_2,
            x_1_I_left, t_1_I_left,
            x_1_I_right, t_1_I_right,
            Config().c
        ).item()

    # ------------------------------------------------------
    # Optimize PINN23
    # ------------------------------------------------------
    def closure23():
        optimizer23.zero_grad()

        loss_strip2 = strip2_Loss(
            model2, model3, model23,
            x_s2, t_s2,
            strip2_x_i_1, strip2_x_i_2,
            x_2_I_left, t_2_I_left,
            x_2_I_right, t_2_I_right,
            Config().c
        )

        loss_strip2.backward()
        return loss_strip2

    optimizer23.step(closure23)

    # Evaluate loss after the optimizer update
    with torch.enable_grad():
        loss_strip2_value = strip2_Loss(
            model2, model3, model23,
            x_s2, t_s2,
            strip2_x_i_1, strip2_x_i_2,
            x_2_I_left, t_2_I_left,
            x_2_I_right, t_2_I_right,
            Config().c
        ).item()

    # ------------------------------------------------------
    # Print training losses
    # ------------------------------------------------------
    if epoch % 50 == 0 or epoch == num_epochs_stage2 - 1:
        print(
            f"EPOCH = {epoch:6d} | "
            f"Loss_strip1 = {loss_strip1_value:.6e} | "
            f"Loss_strip2 = {loss_strip2_value:.6e}"
        )

time_end = time.time()
print(f"Training time: {time_end - time_start:.2f} seconds")

# END LBFGS


os.makedirs("saved_models", exist_ok=True)

torch.save({
    "model1": model1.state_dict(),
    "model2": model2.state_dict(),
    "model3": model3.state_dict(),
    "model12": model12.state_dict(),
    "model23": model23.state_dict(),
}, "saved_models/hybrid_apinn_stage2.pth")

print("All trained models saved successfully.")