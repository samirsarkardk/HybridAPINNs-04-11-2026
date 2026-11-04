import torch
import torch.nn as nn
import numpy as np
from Aconfig import (Config, DEVICE)
import matplotlib.pyplot as plt

config = Config()
delta = config.delta
c = config.c
x_min = config.x_min
x_max = config.x_max    
t_min = config.t_min
t_max = config.t_max

def interface_left(t):

    return -0.2 + c * t


def interface_right(t):

    return 0.2 + c * t



# ==========================================================
# STRIP BOUNDARIES
# ==========================================================

def strip1_lower(t):

    return interface_left(t) - delta


def strip1_upper(t):

    return interface_left(t) + delta


def strip2_lower(t):

    return interface_right(t) - delta


def strip2_upper(t):

    return interface_right(t) + delta


# ==========================================================
# OMEGA 1
# ==========================================================

def sample_omega1(N):

    t = np.random.uniform(
        t_min,
        t_max,
        (N,1)
    )

    right = strip1_lower(t)

    x = x_min + (right - x_min)*np.random.rand(N,1)

    return x,t


# ==========================================================
# STRIP 1
# ==========================================================

def sample_strip1(N):

    t = np.random.uniform(
        t_min,
        t_max,
        (N,1)
    )

    left = strip1_lower(t)

    right = strip1_upper(t)

    x = left + (right-left)*np.random.rand(N,1)

    return x,t


# ==========================================================
# OMEGA 2
# ==========================================================

def sample_omega2(N):

    t = np.random.uniform(
        t_min,
        t_max,
        (N,1)
    )

    left = strip1_upper(t)

    right = strip2_lower(t)

    x = left + (right-left)*np.random.rand(N,1)

    return x,t


# ==========================================================
# STRIP 2
# ==========================================================

def sample_strip2(N):

    t = np.random.uniform(
        t_min,
        t_max,
        (N,1)
    )

    left = strip2_lower(t)

    right = strip2_upper(t)

    x = left + (right-left)*np.random.rand(N,1)

    return x,t


# ==========================================================
# OMEGA 3
# ==========================================================

def sample_omega3(N):

    t = np.random.uniform(
        t_min,
        t_max,
        (N,1)
    )

    left = strip2_upper(t)

    x = left + (x_max-left)*np.random.rand(N,1)

    return x,t


# # ==========================================================
# # RESIDUAL POINTS
# # ==========================================================

# Nr1 = 2000
# Ns1 = 5000

# Nr2 = 2000
# Ns2 = 5000

# Nr3 = 2000

# x_r1 , t_r1 = sample_omega1(Nr1)

# x_s1 , t_s1 = sample_strip1(Ns1)

# x_r2 , t_r2 = sample_omega2(Nr2)

# x_s2 , t_s2 = sample_strip2(Ns2)

# x_r3 , t_r3 = sample_omega3(Nr3)


# # ==========================================================
# # CHECK SAMPLING
# # ==========================================================

# plt.figure(figsize=(10,6))

# plt.scatter(
#     x_r1,
#     t_r1,
#     s=1,
#     label='Omega1'
# )

# plt.scatter(
#     x_s1,
#     t_s1,
#     s=1,
#     label='Strip1'
# )

# plt.scatter(
#     x_r2,
#     t_r2,
#     s=1,
#     label='Omega2'
# )

# plt.scatter(
#     x_s2,
#     t_s2,
#     s=1,
#     label='Strip2'
# )

# plt.scatter(
#     x_r3,
#     t_r3,
#     s=1,
#     label='Omega3'
# )

# plt.legend()

# plt.xlabel("x")
# plt.ylabel("t")

# plt.title("Residual Sampling")

# plt.show()
