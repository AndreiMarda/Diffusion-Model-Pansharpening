import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List
import random
import math
import pdb
import numpy as np

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
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False)
        alpha_prev = torch.cat([torch.ones(1), self.alpha[:-1]])  # /**/
        self.beta_tilde = (self.beta * (1 - alpha_prev) / (1 - self.alpha)).requires_grad_(False)  # /**/

    def forward(self, t):
        return self.beta[t], self.alpha[t]

def main():
    scheduler = DDPM_Scheduler(num_time_steps=1000)
    print(scheduler(999))

if __name__ == '__main__':
    main()
