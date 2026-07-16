import torch
import torch.nn.functional as F
import numpy as np

# epsilon == noise

_CDF23_HALF = np.array([
    0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942,
    0,  -0.005192756653, 0,  0.000807762146, 0, -0.000060081482,
])
_CDF23_KERNEL = np.concatenate([_CDF23_HALF[::-1][:-1], 2.0 * _CDF23_HALF])



def interp23(ms, ratio):
    """Upsample ms (B,C,H,W) by an integer power-of-2 ratio using the 23-tap"""
    if ratio < 1 or (ratio & (ratio - 1)) != 0:
        raise ValueError(f"ratio must be a power of 2, got {ratio}")
    if ratio == 1:
        return ms
    device, dtype = ms.device, ms.dtype
    B, C, H, W = ms.shape
    k   = torch.tensor(_CDF23_KERNEL, dtype=dtype, device=device)
    K   = k.shape[0]
    pad = K // 2
    image = ms
    for step in range(int(round(np.log2(ratio)))):
        _, _, h, w = image.shape
        BC = B * C
        up = torch.zeros(BC, 1, h*2, w*2, dtype=dtype, device=device)
        src = image.reshape(BC, 1, h, w)
        if step == 0:
            up[:, :, 1::2, 1::2] = src
        else:
            up[:, :, 0::2, 0::2] = src
        k_r = k.view(1,1,1,K).expand(BC,1,1,K).contiguous()
        k_c = k.view(1,1,K,1).expand(BC,1,K,1).contiguous()
        x = F.pad(up, (pad,pad,0,0), mode="circular")
        x = F.conv2d(x, k_r, groups=BC)
        x = F.pad(x,  (0,0,pad,pad), mode="circular")
        x = F.conv2d(x, k_c, groups=BC)
        image = x.view(B, C, h*2, w*2)
    return image

def sample_timesteps(batch_size, num_time_steps, device):
    return torch.randint(0, num_time_steps, (batch_size,), device=device)

# forward diffusion (adds noise)
def q_sample(x_start, t, noise, scheduler):
    alpha_bar_t = scheduler.alpha_bar.to(x_start.device)[t].view(-1, 1, 1, 1)
    return torch.sqrt(alpha_bar_t) * x_start + torch.sqrt(1.0 - alpha_bar_t) * noise


# this estimates the clean image x0 from the noisy image xt, and predicted noise
def predict_x0_from_eps(x_t, t, eps_pred, scheduler):
    alpha_bar_t = scheduler.alpha_bar.to(x_t.device)[t].view(-1, 1, 1, 1)
    return (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)


# this is the reverse diffusion step, which estimates xt-1 from xt, predicted noise, and scheduler parameters
@torch.no_grad()
def p_sample(x_t, t, denoiser, cond, scheduler):
    beta_t = scheduler.beta.to(x_t.device)[t].view(-1, 1, 1, 1)
    alpha_t = 1.0 - beta_t
    alpha_bar_t = scheduler.alpha_bar.to(x_t.device)[t].view(-1, 1, 1, 1)
    beta_tilde_t = scheduler.beta_tilde.to(x_t.device)[t].view(-1, 1, 1, 1)

    eps_pred = denoiser(x_t, t, cond)
    mean = (x_t - beta_t * eps_pred / torch.sqrt(1.0 - alpha_bar_t)) / torch.sqrt(alpha_t)

    nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)
    noise = torch.randn_like(x_t)
    return mean + nonzero_mask * torch.sqrt(beta_tilde_t) * noise


# this runs the full reverse diffusion process to generate a clean image from pure noise, given the denoiser and scheduler (iteratively, x_T->x_(T-1)->x_(T-2)->... x_0)
# when snapshot_steps is given, also returns a {step: x_t} dict captured at those timesteps (plus the final x_0)
@torch.no_grad()
def reverse_diffusion_sample(shape, denoiser, cond, scheduler, num_time_steps, device, snapshot_steps=None):
    x_t = torch.randn(shape, device=device)
    snapshot_targets = set(snapshot_steps) if snapshot_steps is not None else None
    snapshots = {} if snapshot_steps is not None else None

    for step in reversed(range(num_time_steps)):
        if snapshot_targets is not None and step in snapshot_targets:
            snapshots[step] = x_t.clone()

        t = torch.full((shape[0],), step, device=device, dtype=torch.long)
        x_t = p_sample(
            x_t=x_t,
            t=t,
            denoiser=denoiser,
            cond=cond,
            scheduler=scheduler,
        )

    if snapshot_targets is not None and 0 in snapshot_targets:
        snapshots[0] = x_t.clone()

    if snapshot_steps is not None:
        return x_t, snapshots
    return x_t
