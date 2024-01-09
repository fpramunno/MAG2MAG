# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 14:45:59 2023

@author: pio-r
"""

import torch
from tqdm import tqdm
import torch.nn as nn
import logging
import numpy as np

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion_cond:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, img_channel=1, device="cuda"):
        self.noise_steps = noise_steps # timestesps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_channel = img_channel
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alphas_prev = torch.cat([torch.tensor([1.0]).to(device), self.alpha[:-1]], dim=0)
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alpha_hat[:-1]], dim=0)
        # self.alphas_cumprod_prev = torch.from_numpy(np.append(1, self.alpha_hat[:-1].cpu().numpy())).to(device)
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps) # linear variance schedule as proposed by Ho et al 2020

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ # equation in the paper from Ho et al that describes the noise processs

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, y, labels, cfg_scale=3, eta=1, sampling_mode='ddpm'):
        logging.info(f"Sampling {n} new images....")
        model.eval() # evaluation mode
        with torch.no_grad(): # algorithm 2 from DDPM
            x = torch.randn((n, self.img_channel, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0): # reverse loop from T to 1
                t = (torch.ones(n) * i).long().to(self.device) # create timesteps tensor of length n
                predicted_noise = model(x, y, labels, t)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, y, None, t)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                 
                
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None] # this is noise, created in one
                alpha_prev = self.alphas_cumprod_prev[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                # SAMPLING adjusted from Stable diffusion
                sigma = (
                            eta
                            * torch.sqrt((1 - alpha_prev) / (1 - alpha_hat)
                            * (1 - alpha_hat / alpha_prev))
                        )
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                # pred_x0 = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise)
                pred_x0 = (x - torch.sqrt(1 - alpha_hat) * predicted_noise) / torch.sqrt(alpha_hat)
                if sampling_mode == 'ddpm':
                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                elif sampling_mode == 'ddim':
                    noise = torch.randn_like(x)
                    nonzero_mask = (
                                        (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                                    )
                    x = ( 
                         torch.sqrt(alpha_prev) * pred_x0 +
                         torch.sqrt(1 - alpha_prev - sigma ** 2) * predicted_noise +
                         nonzero_mask * sigma * noise
                        )
                else:
                    print('The sampler {} is not implemented'.format(sampling_mode))
                    break
        model.train() # it goes back to training mode
        # x = (x.clamp(-1, 1) + 1) / 2 # to be in [-1, 1], the plus 1 and the division by 2 is to bring back values to [0, 1]
        # x = (x * 255).type(torch.uint8) # to bring in valid pixel range
        return x

mse = nn.MSELoss()

def psnr(input: torch.Tensor, target: torch.Tensor, max_val: float) -> torch.Tensor:
    r"""Create a function that calculates the PSNR between 2 images.

    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    Given an m x n image, the PSNR is:

    .. math::

        \text{PSNR} = 10 \log_{10} \bigg(\frac{\text{MAX}_I^2}{MSE(I,T)}\bigg)

    where

    .. math::

        \text{MSE}(I,T) = \frac{1}{mn}\sum_{i=0}^{m-1}\sum_{j=0}^{n-1} [I(i,j) - T(i,j)]^2

    and :math:`\text{MAX}_I` is the maximum possible input value
    (e.g for floating point images :math:`\text{MAX}_I=1`).

    Args:
        input: the input image with arbitrary shape :math:`(*)`.
        labels: the labels image with arbitrary shape :math:`(*)`.
        max_val: The maximum value in the input tensor.

    Return:
        the computed loss as a scalar.

    Examples:
        >>> ones = torch.ones(1)
        >>> psnr(ones, 1.2 * ones, 2.) # 10 * log(4/((1.2-1)**2)) / log(10)
        tensor(20.0000)

    Reference:
        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor but got {type(target)}.")

    if not isinstance(target, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor but got {type(input)}.")

    if input.shape != target.shape:
        raise TypeError(f"Expected tensors of equal shapes, but got {input.shape} and {target.shape}")

    return 10.0 * torch.log10(max_val**2 / mse(input, target))