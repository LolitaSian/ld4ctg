import math

import numpy as np
import torch


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def sqrt(timesteps, max_beta=0.999):
    betas = []
    def alpha_bar(t):
        return 1 - np.sqrt(t + 0.0001)
    for i in range(timesteps):
        t1 = i / timesteps
        t2 = (i + 1) / timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def sqrt_beta_schedule(timesteps):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = 1 - torch.sqrt(x / timesteps + 0.0001)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)



timestep = 5

print(sqrt_beta_schedule(timestep))

print(sqrt(timestep))
