import torch
import torch.nn as nn
import random
import numpy as np


# self.beta[t] -> Gaussian noise added at step t
# linearly increases from 1e-4 to 0.02

# alpha[t] -> cumulative product of (1 - self.beta) up to step t
# alpha + self.beta = 1

# self.alpha[t] -> It tells you how much of the original clean image x0 remains after noising up to timestep t



def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps: int=1000):
        super().__init__()
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        alpha_bar_prev = torch.cat([torch.ones(1), self.alpha_bar[:-1]])
        self.beta_tilde = self.beta * (1 - alpha_bar_prev) / (1 - self.alpha_bar)

    def forward(self, t):
        return self.beta[t], self.alpha_bar[t]
