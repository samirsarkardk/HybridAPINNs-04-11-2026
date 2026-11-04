import torch
import torch.nn as nn
import numpy as np
import os
from Aconfig import (Config, DEVICE)
from Bmodel import (PINN1, PINN2, PINN3, PINN12, PINN23)
import matplotlib.pyplot as plt


config = Config()
x_min = config.x_min
x_max = config.x_max    
t_min = config.t_min
t_max = config.t_max
c = config.c
delta = config.delta


def exact_solution(x, t):

    x_shift = x - c*t

    u = np.where(
        (x_shift >= -0.2) &
        (x_shift <=  0.2),
        1.0,
        0.0
    )

    return u

model1 = PINN1().to(DEVICE)
model2 = PINN2().to(DEVICE) 
model3 = PINN3().to(DEVICE)
model12 = PINN12().to(DEVICE)
model23 = PINN23().to(DEVICE)

# 2. Load the saved checkpoint
checkpoint_path = "saved_models/hybrid_apinn_stage2.pth"

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=True
)

# print the keys in the checkpoint to verify what has been saved
print(checkpoint.keys())

# Restore the saved model states
model1.load_state_dict(checkpoint["model1"])
model2.load_state_dict(checkpoint["model2"])
model3.load_state_dict(checkpoint["model3"])
model12.load_state_dict(checkpoint["model12"])
model23.load_state_dict(checkpoint["model23"])

print("All five trained models loaded successfully.")

model1.eval()
model2.eval()
model3.eval()
model12.eval()
model23.eval()


def hybrid_predict(x, t):

    s1_lower = -0.2 + c*t - delta
    s1_upper = -0.2 + c*t + delta

    s2_lower =  0.2 + c*t - delta
    s2_upper =  0.2 + c*t + delta

    u = torch.zeros_like(x)

    # Omega 1
    idx = (x < s1_lower).squeeze()

    if idx.any():
        u[idx, :] = model1(x[idx], t[idx])

    # Strip 1
    idx = (
        (x > s1_lower) &
        (x < s1_upper)
    ).squeeze()

    if idx.any():
        u[idx, :] = model12(x[idx], t[idx])

    # strip1 lower line solution
    idx = (x == s1_lower).squeeze()

    if idx.any():
        u[idx, :] = ( model12(x[idx], t[idx]) +  model1(x[idx], t[idx])) / 2.0

    # strip1 upper line solution
    idx = (x == s1_upper).squeeze()

    if idx.any():
        u[idx, :] = ( model12(x[idx], t[idx]) +  model2(x[idx], t[idx])) / 2.0

    # Omega 2
    idx = (
        (x > s1_upper) &
        (x < s2_lower)
    ).squeeze()

    if idx.any():
        u[idx, :] = model2(x[idx], t[idx])

    # Strip 2
    idx = (
        (x > s2_lower) &
        (x < s2_upper)
    ).squeeze()

    if idx.any():
        u[idx, :] = model23(x[idx], t[idx])

    # strip2 lower line solution
    idx = (x == s2_lower).squeeze()
    
    if idx.any():
        u[idx, :] = ( model23(x[idx], t[idx]) +  model2(x[idx], t[idx])) / 2.0

    # strip2 upper line solution
    idx = (x == s2_upper).squeeze()

    if idx.any():
        u[idx, :] = ( model23(x[idx], t[idx]) +  model3(x[idx], t[idx])) / 2.0
    
    
    

    # Omega 3
    idx = (x > s2_upper).squeeze()

    if idx.any():
        u[idx, :] = model3(x[idx], t[idx])

    return u

x_test = torch.tensor([[-0.2 - delta/2], [-0.2 + delta/2]], device=DEVICE)
t_test = torch.zeros_like(x_test)
print(model12(x_test, t_test))
# =========================================================
# 1. Generate the space-time grid
# =========================================================
Nx = 3000
Nt = 3000

x_values = np.linspace(x_min, x_max, Nx)
t_values = np.linspace(t_min, t_max, Nt)

X_grid, T_grid = np.meshgrid(x_values, t_values)

# Convert the grid to PyTorch tensors
x_tensor = torch.tensor(
    X_grid.reshape(-1, 1),
    dtype=torch.float64,
    device=DEVICE
)

t_tensor = torch.tensor(
    T_grid.reshape(-1, 1),
    dtype=torch.float64,
    device=DEVICE
)

# =========================================================
# 2. Calculate the hybrid APINN prediction
# =========================================================
with torch.no_grad():
    U_pred_tensor = hybrid_predict(x_tensor, t_tensor)

U_pred = U_pred_tensor.detach().cpu().numpy().reshape(Nt, Nx)

# =========================================================
# 3. Calculate the exact solution
# =========================================================
U_exact = exact_solution(X_grid, T_grid)

# =========================================================
# 4. Calculate L2 errors
# =========================================================
error = U_pred - U_exact

# Absolute discrete L2 error
absolute_L2 = np.linalg.norm(error.ravel())

# Relative discrete L2 error
exact_L2 = np.linalg.norm(U_exact.ravel())

if exact_L2 > 0:
    relative_L2 = absolute_L2 / exact_L2
else:
    relative_L2 = np.nan

print("\n========== Hybrid APINN Errors ==========")
print(f"Absolute L2 error : {absolute_L2:.6e}")
print(f"Relative L2 error : {relative_L2:.6e}")
print(f"Relative L2 error (%) : {100 * relative_L2:.4f}%")


# =========================================================
# 6. Plot solutions at particular time instants
# =========================================================

# Time instants at which we want to plot the solution
time_instants = [0.1, 0.5, 1.0]

# Generate spatial points
x_plot = np.linspace(x_min, x_max, 2000)

# Convert x to PyTorch tensor
x_plot_tensor = torch.tensor(
    x_plot.reshape(-1, 1),
    dtype=torch.float64,
    device=DEVICE
)

fig, axes = plt.subplots(
    1, len(time_instants),
    figsize=(18, 5),
    sharey=True
)

# Display global errors above the plots
fig.suptitle(
    f"Hybrid APINN | Absolute L2 Error = {absolute_L2:.6e} | "
    f"Relative L2 Error = {relative_L2:.6e} "
    f"({100 * relative_L2:.4f}%)",
    fontsize=14,
    fontweight="bold",
    y=1.03
)

for i, time_value in enumerate(time_instants):

    # Create a time tensor with the same shape as x
    t_plot_tensor = torch.full_like(
        x_plot_tensor,
        fill_value=time_value
    )

    # Hybrid APINN prediction
    with torch.no_grad():
        u_pred_tensor = hybrid_predict(
            x_plot_tensor,
            t_plot_tensor
        )

    u_pred = u_pred_tensor.cpu().numpy().flatten()

    # Exact solution
    t_plot = np.full_like(x_plot, time_value)
    u_exact = exact_solution(x_plot, t_plot)

    # Plot exact and predicted solutions
    axes[i].plot(
        x_plot,
        u_exact,
        'k-',
        linewidth=2,
        label="Exact solution"
    )

    axes[i].plot(
        x_plot,
        u_pred,
        'r--',
        linewidth=2,
        label="Hybrid APINN"
    )

    axes[i].set_title(f"Solution at t = {time_value}")
    axes[i].set_xlabel("x")
    axes[i].grid(True, alpha=0.3)
    axes[i].legend()

axes[0].set_ylabel("u(x,t)")

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

# =========================================================
# 5. Plot exact solution, prediction, and absolute error
# =========================================================
U_error = np.abs(U_pred - U_exact)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Exact solution
contour1 = axes[0].contourf(
    X_grid, T_grid, U_exact,
    levels=50, cmap="viridis"
)
axes[0].set_title("Exact Solution")
axes[0].set_xlabel("x")
axes[0].set_ylabel("t")
fig.colorbar(contour1, ax=axes[0])

# Hybrid APINN prediction
contour2 = axes[1].contourf(
    X_grid, T_grid, U_pred,
    levels=50, cmap="viridis"
)
axes[1].set_title("Hybrid APINN Prediction")
axes[1].set_xlabel("x")
axes[1].set_ylabel("t")
fig.colorbar(contour2, ax=axes[1])

# Absolute pointwise error
contour3 = axes[2].contourf(
    X_grid, T_grid, U_error,
    levels=50, cmap="magma"
)
axes[2].set_title("Absolute Error")
axes[2].set_xlabel("x")
axes[2].set_ylabel("t")
fig.colorbar(contour3, ax=axes[2])

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

# =========================================================
# Plot exact solution vs prediction with L2 errors on top
# =========================================================

time_instants = [0.1, 0.5, 1.0]

x_plot = np.linspace(x_min, x_max, 2000)

x_plot_tensor = torch.tensor(
    x_plot.reshape(-1, 1),
    dtype=torch.float64,
    device=DEVICE
)

fig, axes = plt.subplots(
    1, len(time_instants),
    figsize=(18, 6),
    sharey=True
)

# Reserve space at the top for the error information
fig.subplots_adjust(top=0.78)

# Print both errors above the graphs
fig.text(
    0.5, 0.96,
    f"Absolute L2 Error = {absolute_L2:.6e}",
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold"
)

fig.text(
    0.5, 0.90,
    f"Relative L2 Error = {relative_L2:.6e} "
    f"({100 * relative_L2:.4f}%)",
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold"
)

fig.text(
    0.5, 0.85,
    f"Points taken x = {Nx} "
    f", t = {Nt} "
    f", Delta = {Config().delta}",
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold"
)

for i, time_value in enumerate(time_instants):

    t_plot_tensor = torch.full_like(
        x_plot_tensor,
        fill_value=time_value
    )

    with torch.no_grad():
        u_pred_tensor = hybrid_predict(
            x_plot_tensor,
            t_plot_tensor
        )

    u_pred = u_pred_tensor.cpu().numpy().flatten()

    t_plot = np.full_like(x_plot, time_value)
    u_exact = exact_solution(x_plot, t_plot)

    axes[i].plot(
        x_plot, u_exact,
        "k-", linewidth=2,
        label="Exact solution"
    )

    axes[i].plot(
        x_plot, u_pred,
        "r--", linewidth=2,
        label="Hybrid APINN"
    )

    axes[i].set_title(f"Solution at t = {time_value}")
    axes[i].set_xlabel("x")
    axes[i].grid(True, alpha=0.3)
    axes[i].legend()

axes[0].set_ylabel("u(x,t)")

plt.show()