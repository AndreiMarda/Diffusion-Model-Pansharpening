import torch
import torch.nn.functional as F

# epsilon == noise

def bicubic_upsample(ms, size):
    return F.interpolate(ms, size=size, mode="bicubic", align_corners=False)


def sample_timesteps(batch_size, num_time_steps, device):
    return torch.randint(0, num_time_steps, (batch_size,), device=device)


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
@torch.no_grad()
def reverse_diffusion_sample(shape, denoiser, cond, scheduler, num_time_steps, device):
    x_t = torch.randn(shape, device=device)

    for step in reversed(range(num_time_steps)):
        t = torch.full((shape[0],), step, device=device, dtype=torch.long)
        x_t = p_sample(
            x_t=x_t,
            t=t,
            denoiser=denoiser,
            cond=cond,
            scheduler=scheduler,
        )

    return x_t
