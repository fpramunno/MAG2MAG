# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:38:51 2024

@author: pio-r
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, Normalize
import pandas as pd
# Preprocessing

from torchvision.transforms import Compose
import math

CHANNEL_PREPROCESS = {
    "94A": {"min": 0.1, "max": 800, "scaling": "log10"},
    "131A": {"min": 0.7, "max": 1900, "scaling": "log10"},
    "171A": {"min": 5, "max": 3500, "scaling": "log10"},
    "193A": {"min": 20, "max": 5500, "scaling": "log10"},
    "211A": {"min": 7, "max": 3500, "scaling": "log10"},
    "304A": {"min": 0.1, "max": 3500, "scaling": "log10"},
    "335A": {"min": 0.4, "max": 1000, "scaling": "log10"},
    "1600A": {"min": 10, "max": 800, "scaling": "log10"},
    "1700A": {"min": 220, "max": 5000, "scaling": "log10"},
    "4500A": {"min": 4000, "max": 20000, "scaling": "log10"},
    "continuum": {"min": 0, "max": 65535, "scaling": None},
    "magnetogram": {"min": -250, "max": 250, "scaling": None},
    "Bx": {"min": -250, "max": 250, "scaling": None},
    "By": {"min": -250, "max": 250, "scaling": None},
    "Bz": {"min": -250, "max": 250, "scaling": None},
}


def get_default_transforms(target_size=256, channel="171", mask_limb=False, radius_scale_factor=1.0):
    """Returns a Transform which resizes 2D samples (1xHxW) to a target_size (1 x target_size x target_size)
    and then converts them to a pytorch tensor.

    Apply the normalization necessary for the SDO ML Dataset. Depending on the channel, it:
      - masks the limb with 0s
      - clips the "pixels" data in the predefined range (see above)
      - applies a log10() on the data
      - normalizes the data to the [0, 1] range
      - normalizes the data around 0 (standard scaling)

    Args:
        target_size (int, optional): [New spatial dimension of the input data]. Defaults to 256.
        channel (str, optional): [The SDO channel]. Defaults to 171.
        mask_limb (bool, optional): [Whether to mask the limb]. Defaults to False.
        radius_scale_factor (float, optional): [Allows to scale the radius that is used for masking the limb]. Defaults to 1.0.
    Returns:
        [Transform]
    """

    transforms = []

    # also refer to
    # https://pytorch.org/vision/stable/transforms.html
    # https://github.com/i4Ds/SDOBenchmark/blob/master/dataset/data/load.py#L363
    # and https://gitlab.com/jdonzallaz/solarnet-thesis/-/blob/master/solarnet/data/transforms.py
    preprocess_config = CHANNEL_PREPROCESS[channel]

    if preprocess_config["scaling"] == "log10":
        # TODO does it make sense to use vflip(x) in order to align the solar North as in JHelioviewer?
        # otherwise this has to be done during inference
        def lambda_transform(x): return torch.log10(torch.clamp(
            x,
            min=preprocess_config["min"],
            max=preprocess_config["max"],
        ))
        mean = math.log10(preprocess_config["min"])
        std = math.log10(preprocess_config["max"]) - \
            math.log10(preprocess_config["min"])
    elif preprocess_config["scaling"] == "sqrt":
        def lambda_transform(x): return torch.sqrt(torch.clamp(
            x,
            min=preprocess_config["min"],
            max=preprocess_config["max"],
        ))
        mean = math.sqrt(preprocess_config["min"])
        std = math.sqrt(preprocess_config["max"]) - \
            math.sqrt(preprocess_config["min"])
    else:
        def lambda_transform(x): return torch.clamp(
            x,
            min=preprocess_config["min"],
            max=preprocess_config["max"],
        )
        mean = preprocess_config["min"]
        std = preprocess_config["max"] - preprocess_config["min"]

    def limb_mask_transform(x):
        h, w = x.shape[1], x.shape[2]  # C x H x W

        # fixed disk size of Rs of 976 arcsec, pixel size in the scaled image (512x512) is ~4.8 arcsec
        original_resolution = 4096
        scaled_resolution = h
        pixel_size_original = 0.6
        radius_arcsec = 976.0
        radius = (radius_arcsec / pixel_size_original) / \
            original_resolution * scaled_resolution

        mask = create_circular_mask(
            h, w, radius=radius*radius_scale_factor)
        mask = torch.as_tensor(mask, device=x.device)
        return torch.where(mask, x, torch.tensor(0.0))

    if mask_limb:
        def mask_lambda_func(x):
            return limb_mask_transform(x)
        transforms.append(mask_lambda_func)
        # transforms.append(Lambda(lambda x: limb_mask_transform(x)))

    transforms.append(Resize((target_size, target_size)))
    # TODO find out if these transforms make sense
    def test_lambda_func(x):
        return lambda_transform(x)
    transforms.append(test_lambda_func)
    # transforms.append(Lambda(lambda x: lambda_transform(x)))
    transforms.append(Normalize(mean=[mean], std=[std]))
    # required to remove strange distribution of pixels (everything too bright)
    transforms.append(Normalize(mean=(0.5), std=(0.5)))

    return Compose(transforms)

def create_circular_mask(h, w, center=None, radius=None):
    # TODO investigate the use of a circular mask to prevent focussing to much on the limb
    # https://gitlab.com/jdonzallaz/solarnet-app/-/blob/master/src/prediction.py#L9

    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))

    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask




def save_images(img, path):
    V = []
    imgs = []
    for i in range(img.shape[0]):
        imgs.append(img[i])
    for images in imgs:
        images = images.permute(1, 2, 0)
        images = np.squeeze(images.cpu().numpy())
        # v = vis(images, channel_to_map(171))
        v = Image.fromarray(images)
        V.append(v)
    for value in V:
        value.save(path)

def save_tensor_as_png(tensor, filename):
    # Make sure the tensor is in the CPU and detach it from the computational graph
    tensor = tensor.detach().cpu()

    # Convert the tensor to a PIL image
    if tensor.shape[0] == 1:
        # If the input tensor has only 1 channel, convert it to a grayscale image
        image = Image.fromarray((tensor.squeeze(0).numpy() * 255).astype('uint8'), mode='L')
    else:
        # If the input tensor has 3 channels, convert it to an RGB image
        image = Image.fromarray((tensor.permute(1, 2, 0).numpy() * 255).astype('uint8'))

    # Save the image to the specified filename
    image.save(filename)


# Data loading

from torch.utils.data import Dataset

mapping = {'A': 0, 'B': 1, 'C': 2, 'M': 3, 'X': 4}


class PairedJP2Dataset(Dataset):
    def __init__(self, dir2, dir3, labels_list, time_mag_list, transform2=None):
        super(PairedJP2Dataset, self).__init__()
        
        # Ensure directories exist
        assert os.path.isdir(dir2), f"{dir2} is not a directory."
        assert os.path.isdir(dir3), f"{dir3} is not a directory."

        self.dir2_files = sorted([os.path.join(dir2, fname) for fname in os.listdir(dir2) if fname.endswith('.jp2')])
        self.dir3_files = sorted([os.path.join(dir3, fname) for fname in os.listdir(dir3) if fname.endswith('.jp2')])
        assert len(self.dir2_files) == len(self.dir3_files), "Directories have different number of .jp2 files."
        self.transform2 = transform2
        self.labels = labels_list
        self.time_mag = time_mag_list

    def __len__(self):
        return len(self.dir2_files)

    def __getitem__(self, idx):
        data2 = Image.open(self.dir2_files[idx])
        data3 = Image.open(self.dir3_files[idx])
        
        data2 = to_tensor(data2)
        data3 = to_tensor(data3)
        
        # Apply any transformations if provided
        if self.transform2:
            data2 = self.transform2(data2)
        if self.transform2:
            data3 = self.transform2(data3)
        label = self.labels[idx]
        time_mag = self.time_mag[idx]    
        
        
        return data2, data3, label, time_mag

# Example usage:
dir2 = '/mnt/nas05/data01/francesco/sdo_data/final_dataset/mag_24_jp2_1024/'

dir3 = '/mnt/nas05/data01/francesco/sdo_data/final_dataset/mag_flare_jp2_1024/'

df_lab = pd.read_csv("/mnt/nas05/data01/francesco/sdo_data/final_dataset/df_all_final_v2.csv")

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class CustomRotation:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        return TF.rotate(img, self.angle)

transform_hmi = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomVerticalFlip(p=1.0),  # This line adds a 90-degree rotation to the right
    transforms.Normalize(mean=(0.5), std=(0.5))
])


to_tensor = transforms.ToTensor()

dataset = PairedJP2Dataset(dir2, dir3, df_lab['Label'], df_lab['MAG'], transform_hmi)


from torch.utils.data import random_split

total_samples = len(dataset)
train_size = int(0.7 * total_samples)  # Using 80% for training as an example
val_size = total_samples - train_size


torch.manual_seed(42)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_data = DataLoader(train_dataset, batch_size=2,
                          shuffle=False,
                          pin_memory=True,# pin_memory set to True
                          num_workers=12,
                          prefetch_factor=4,
                          drop_last=False)

val_data = DataLoader(val_dataset, batch_size=2,
                          shuffle=False,
                          pin_memory=True,# pin_memory set to True
                          num_workers=12,
                          prefetch_factor=4,  # pin_memory set to True
                          drop_last=False)


print('Train loader and Valid loader are up!')

# Start Training

import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.run_name = "DDPM_Conditional"
args.epochs = 500
args.batch_size = 2
args.image_size = 512
args.device = "cuda"
args.lr = 3e-4

# from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torch import optim
import copy

import logging
import torch.nn as nn

# from utils import plot_images, save_images, get_data
from modules import PaletteModelV2, EMA
from diffusion import Diffusion_cond
from torch.utils.tensorboard import SummaryWriter

def setup_logging(run_name):
    """
    Setting up the folders for saving the model and the results

    """
    os.makedirs("models_512_DDPM_v2", exist_ok=True)
    os.makedirs("results_512_DDPM_v2", exist_ok=True)
    os.makedirs(os.path.join("models_512_DDPM_v2", run_name), exist_ok=True)
    os.makedirs(os.path.join("results_512_DDPM_v2", run_name), exist_ok=True)


setup_logging(args.run_name)
device = args.device
dataloader = train_data
dataloader_val = val_data
model = PaletteModelV2(c_in=2, c_out=1, num_classes=5,  image_size=int(64), true_img_size=64).to(device)
# ckpt = torch.load("./models_cloud_removal_1024_newnorm_CaII/DDPM_Conditional/ema_ckpt_cond.pt")
# model.load_state_dict(ckpt)

optimizer = optim.AdamW(model.parameters(), lr=args.lr)
mse = nn.MSELoss()
diffusion = Diffusion_cond(img_size=args.image_size, device=device)
logger = SummaryWriter(os.path.join("runs", args.run_name))
l = len(dataloader)
ema = EMA(0.995)
ema_model = copy.deepcopy(model).eval().requires_grad_(False)

wandb.init(project="sdo_img2img", entity="francescopio")

wandb.config = {"# Epochs" : 500,
                "Batch size" : 2,
                "Image size" : 512,
                "Device" : "cuda",
                "Lr" : 3e-4}

wandb.watch(model, log=None)

min_valid_loss = np.inf



from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

import gc  # Import Python's garbage collector


# Training loop
for epoch in range(args.epochs):
    logging.info(f"Starting epoch {epoch}:")
    pbar = tqdm(train_data)
    model.train()
    train_loss = 0.0
    # psnr_train = 0.0
    for i, (image_94, image_peak, label, time) in enumerate(pbar):
        img_24 = image_94.to(device).float()
        img_peak = image_peak.to(device).float()
        
        label = label
        time = time
        has_nan_24 = torch.isnan(img_24).any()
        has_nan_peak = torch.isnan(img_peak).any()
        
        if (has_nan_24.item() or has_nan_peak.item()) is True:
            print('The batch number is: {}'.format(i))
            del img_24
            del img_peak
            torch.cuda.empty_cache()
            gc.collect()
            continue
        
        labels = None
        t = diffusion.sample_timesteps(img_peak.shape[0]).to(device)
        x_t, noise = diffusion.noise_images(img_peak, t)
        
        
        with autocast():
            predicted_noise = model(x_t, img_24, labels, t)
            loss = mse(noise, predicted_noise)
        
        optimizer.zero_grad()
        
        if np.isnan(loss.item()):
            print('The batch number is: {}'.format(i))
            del img_24
            del img_peak
            del predicted_noise
            torch.cuda.empty_cache()
            gc.collect()
            continue
        else:
            peak_allocated = torch.cuda.max_memory_allocated(device=device)
            # print(f"Peak Allocated Memory: {peak_allocated / (1024 ** 3):.2f} GB")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.step_ema(ema_model, model)
            peak_cached = torch.cuda.max_memory_cached(device=device)
            # print(f"Peak Cached Memory: {peak_cached / (1024 ** 3):.2f} GB")
            
            # delete unnecessary variables to save memory
            torch.cuda.empty_cache()
            gc.collect()

            train_loss += loss.detach().item() * img_24.size(0)
            # psnr_train += psnr(predicted_noise, noise, torch.max(predicted_noise))
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * len(pbar) + i)
    
        # # Prefetch the next batch asynchronously (non-blocking)
        # if train_data.prefetch_factor > 0  and i < len(train_data) - 1:
        #     if i + 1 not in train_data.cpu_cache:
        #         prefetch_batch(train_data, i)
    
        # # Clear the cache after using the previous batch
        # if train_data.cpu_cache and i > 0:
        #     train_data.cpu_cache.pop(i - 1, None)

    # Clean up memory before validation
    torch.cuda.empty_cache()
    gc.collect()

    # Validation step
    valid_loss = 0.0
    # psnr_val = 0.0
    pbar_val = tqdm(val_data)
    model.eval()
    with torch.no_grad():
        for i, (image_94, image_peak, label, time) in enumerate(pbar_val):
            img_24 = image_94.to(device).float()
            img_peak = image_peak.to(device).float()
            label = label
            time = time
            
            has_nan_24 = torch.isnan(img_24).any()
            has_nan_peak = torch.isnan(img_peak).any()
            
            if (has_nan_24.item() or has_nan_peak.item()) is True:
                # print('The batch number is: {}'.format(i))
                del img_24
                del img_peak
                torch.cuda.empty_cache()
                gc.collect()
                continue
            
            labels = None
            t = diffusion.sample_timesteps(img_peak.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(img_peak, t)
            
            
            with autocast():
                predicted_noise = model(x_t, img_24, labels, t)
                loss = mse(noise, predicted_noise)
            
            if np.isnan(loss.item()):
                # print('The batch number is: {}'.format(i))
                del img_24
                del img_peak
                del predicted_noise
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                valid_loss += loss.detach().item() * img_24.size(0)
                # psnr_val += psnr(predicted_noise, noise, torch.max(predicted_noise))

        # Clean up memory after validation
        torch.cuda.empty_cache()
        gc.collect()
        
    # Logging and saving
    if epoch % 5 == 0:
        ema_sampled_images = diffusion.sample(ema_model, y=img_24[0].reshape(1, 1, 512, 512), labels=None, n=1)
        save_images(ema_sampled_images, os.path.join("results_512_DDPM_v2", args.run_name, f"{epoch}_ema_cond.png"))
        true_img = img_24[0].reshape(1, 512, 512).permute(1, 2, 0).cpu().numpy()
        gt_peak = img_peak[0].reshape(1, 512, 512).permute(1, 2, 0).cpu().numpy()
        ema_samp = ema_sampled_images[0].permute(1, 2, 0).cpu().numpy()
        # Create a figure with two subplots
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        # Plot the original image in the first subplot
        ax1.imshow(true_img, origin='lower')
        ax1.set_title('24 before flare')

        # Plot the EMA sampled image in the second subplot
        ax2.imshow(gt_peak, origin='lower')
        ax2.set_title('True flaring image')

        ax3.imshow(ema_samp, origin='lower')
        ax3.set_title('Predicted flaring image')
        
        # Add a big title in the middle of all subplots
        fig.suptitle('Time: {} \n Label: {}'.format(time[0], label[0]))
        # Adjust the spacing between subplots
        plt.tight_layout()

        # Show the plot
        plt.show()
        

    wandb.log({
        "Training Loss": train_loss / len(train_data),
        "Validation Loss": valid_loss / len(val_data),
        # "PSNR train value": psnr_train / len(train_data),
        # "PSNR valid value": psnr_val / len(val_data),
        'Sampled images': wandb.Image(plt)
    })

    plt.close()
    
    if min_valid_loss > valid_loss:
        logging.info(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
    
    # Saving State Dict
    torch.save(model.state_dict(), os.path.join("models_512_DDPM_v2", args.run_name, f"ckpt_test_cond.pt"))
    # torch.save(ema_model.state_dict(), os.path.join("models_512_DDPM_v2", args.run_name, f"ema_ckpt_cond.pt"))
    state = {
        'model_state': ema_model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }
    torch.save(state, os.path.join("models_512_DDPM_v2", args.run_name, "checkpoint.pt"))
    wandb.save('ema_model_epoch{}.pt'.format(epoch))
    wandb.save('model_epoch{}.pt'.format(epoch))