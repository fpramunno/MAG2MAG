# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 11:41:08 2024

@author: pio-r
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches


from scipy.ndimage import label
from matplotlib.patches import Circle
from tqdm import tqdm
from PIL import Image
from torch.utils.data import random_split
from sunpy.net import hek
from sunpy.net import attrs as a
from skimage.morphology import skeletonize
from scipy.ndimage import label

# My function
from util import reverse_scaling, compute_area, mask_outside_circle
from util import sun_earth_distance_in_meters, hpc_to_pixel, extract_and_format_datetime
from util import obtain_contour
from util import persistence_perd
from util import is_within_circle
from util import comput_jaccard_index
from util import obtain_contour_all

# This function will find distinct regions in the mask and sum the pixel values in these regions
def sum_masked_regions(masked_image, pix_area):
    # Label all the connected regions in the image
    labeled_array, num_features = label(masked_image)
    sums = []
    
    # Sum the pixel values for each region
    for region in range(1, num_features + 1):
        region_sum = np.sum(np.abs(masked_image[labeled_array == region]))
        num_pix = len(masked_image[labeled_array == region])
        total_area = pix_area * num_pix
        sums.append(region_sum * total_area)
    
    norm_sum = []
    for value in sums:
        norm_sum.append(value/sum(sums))
    
    return norm_sum

def filter_labeled_regions_and_sums(labeled_array, sums):
    # Get the indices of the regions that have sum less than 0.8
    sums_to_remove_indices = [i for i, sum in enumerate(sums) if sum < 0.8]

    # Check if all sums are less than 0.8
    if len(sums_to_remove_indices) == len(sums):
        # If all sums are less than 0.8, find the max sum and its index
        max_sum_index, max_sum = max(enumerate(sums), key=lambda x: x[1])
        # Keep only the region with the max sum
        for i, sum in enumerate(sums):
            if i != max_sum_index:
                labeled_array[labeled_array == i+1] = 0
        # Update the filtered sums to only include the max sum
        filtered_sums = [max_sum]
    else:
        # Set regions to zero in the labeled array for sums less than 0.8
        for region in sums_to_remove_indices:
            labeled_array[labeled_array == region+1] = 0
        # Filter sums list to remove values less than 0.8
        filtered_sums = [sum for i, sum in enumerate(sums) if i not in sums_to_remove_indices]

    return labeled_array, filtered_sums

transform = transforms.ToTensor()
norm = transforms.Normalize(mean=(0.5), std=(0.5))




for e in tqdm(range(0, 8)):
    
    directory = "Replace with your directory path containing your generated images"  # Replace with your directory path
    

    tensors = []
    filename_list = []
    
    
    for subdir, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.png'):
                image_path = os.path.join(subdir, filename)
                image = Image.open(image_path).convert('L')  # Convert to grayscale ('L' mode)
                tensor = norm(transform(image))
                filename_list.append(filename)
                tensors.append(tensor)
            
    
    
    transform_hmi = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.RandomVerticalFlip(p=1.0),  # This line adds a 90-degree rotation to the right
        transforms.Normalize(mean=(0.5), std=(0.5))
    ])
    
    df_lab = pd.read_csv("Directory containing the flaring list")
    
    df_lab['Label'] = [value[0] for value in df_lab['Label']]
    
    
    dir2 = 'Directory for the LoS magnetogram 24 hours before the flare peak'
    dir2_files = sorted([os.path.join(dir2, fname) for fname in os.listdir(dir2) if fname.endswith('.jp2')])
    
    total_samples = len(dir2_files)
    train_size = int(0.7 * total_samples)  # Using 70% for training as an example
    val_size = total_samples - train_size
    
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(dir2_files, [train_size, val_size])
    
    dir3 = 'Directory for the LoS magnetogram at the flare peak'
    dir3_files = sorted([os.path.join(dir3, fname) for fname in os.listdir(dir3) if fname.endswith('.jp2')])
    
    total_samples_flr = len(dir3_files)
    train_size_flr = int(0.7 * total_samples_flr)  # Using 70% for training as an example
    val_size_flr = total_samples_flr - train_size_flr
    
    
    torch.manual_seed(42)
    train_dataset_flr, val_dataset_flr = random_split(dir3_files, [train_size_flr, val_size_flr])
    
    val_times = []
    for value in dir2_files:
        val_times.append(value)
    
    val_times_flr = []
    for value in dir3_files:
        val_times_flr.append(value)
        
    times_24 = []
    
    for value in val_times:
        times_24.append(extract_and_format_datetime(value))
        
        
    df_val = df_lab.loc[df_lab['HMI'].isin(times_24)].reset_index(drop=True)
    df_val['Path_24'] = val_times
    df_val['Path_Flr'] = val_times_flr
        
    A = df_val.loc[df_val['Label'] == 'A']
    B = df_val.loc[df_val['Label'] == 'B']
    C = df_val.loc[df_val['Label'] == 'C']
    M = df_val.loc[df_val['Label'] == 'M']
    X = df_val.loc[df_val['Label'] == 'X']
    
    A_selected_rows = A.sample(n=50, random_state=42)
    B_selected_rows = B.sample(n=50, random_state=42)
    C_selected_rows = C.sample(n=50, random_state=42)
    M_selected_rows = M.sample(n=50, random_state=42)
    X_selected_rows = X.sample(n=len(X), random_state=42)
    
    list_24 = list(A_selected_rows['Path_24']) + list(B_selected_rows['Path_24']) + list(C_selected_rows['Path_24']) + list(M_selected_rows['Path_24']) + list(X_selected_rows['Path_24'])
    list_peak = list(A_selected_rows['Path_Flr']) + list(B_selected_rows['Path_Flr']) + list(C_selected_rows['Path_Flr']) + list(M_selected_rows['Path_Flr']) + list(X_selected_rows['Path_Flr'])
    
    
    def transform_string(original_string):
        # Define the part of the string to be removed
        part_to_remove = "_1024"
    
        # Replace the part with an empty string
        transformed_string = original_string.replace(part_to_remove, "")
    
        return transformed_string
    
    list_24_4k = [transform_string(value) for value in list_24]
    
    

    transform_hmi = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        # transforms.RandomVerticalFlip(p=1.0),  # This line adds a 90-degree rotation to the right
        transforms.Normalize(mean=(0.5), std=(0.5))
    ])
    
    original_size = (4096, 4096)  # Original image size
    new_size = (256, 256)        # New image size after resizing
    
    # Image scale in arcseconds per pixel
    scale_x = 0.6  # CDELT1
    scale_y = 0.6  # CDELT2
    
    # Reference point in pixel coordinates
    ref_pixel_x = 2048.5  # CRPIX1
    ref_pixel_y = 2048.5  # CRPIX2
    
    scale_img = 256 / 4096
    
    rsun_obs = 976.00842
    
    full_disk_tot = []
    full_disk_net = []
    full_disk_tot_samp = []
    full_disk_net_samp = []
    flux_diff_tot = []
    flux_diff_net = []
    flux_diff_tot_samp = []
    flux_diff_net_samp = []
    pv_full_disk_tot = []
    pv_full_disk_net = []
    pv_full_disk_tot_pers = []
    pv_full_disk_net_pers= []
    pv_flux_ev_fd_tot = []
    pv_flux_ev_fd_net = []
    pv_flux_ev_fd_tot_pers = []
    pv_flux_ev_fd_net_pers = []
    
    def create_dict():
        return {str(i): [] for i in range(len(list_24))}
    
    # Create the dictionaries using the function
    ar_tot = create_dict()
    ar_net = create_dict()
    ar_tot_samp = create_dict()
    ar_net_samp = create_dict()
    pv_ar_tot = create_dict()
    pv_ar_net = create_dict()
    pv_ar_tot_ev = create_dict()
    pv_ar_net_ev = create_dict()
    size_ar = create_dict()
    size_ar_positive = create_dict()
    size_ar_negative = create_dict()
    orientation = create_dict()
    distance_center = create_dict()
    pv_ar_tot_pers = create_dict()
    pv_ar_net_pers = create_dict()
    pv_orientation_samp = create_dict()
    pv_orientation_pers = create_dict()
    jacc_samp = create_dict()
    jacc_per = create_dict()
    
    for i in tqdm(range(len(list_24))):
        
        gen_data = (tensors[i].clamp(-1, 1) + 1) / 2 # to be in [-1, 1], the plus 1 and the division by 2 is to bring back values to [0, 1]
        gen_data = (gen_data * 255).type(torch.uint8) # to bring in valid pixel range
        
        img_peak = transform_hmi(Image.open(list_peak[i]))
    
        img_peak = (img_peak.clamp(-1, 1) + 1) / 2 # to be in [-1, 1], the plus 1 and the division by 2 is to bring back values to [0, 1]
        img_peak = (img_peak * 255).type(torch.uint8) # to bring in valid pixel range
        
        img_24 = transform_hmi(Image.open(list_24[i]))
    
        img_24 = (img_24.clamp(-1, 1) + 1) / 2 # to be in [-1, 1], the plus 1 and the division by 2 is to bring back values to [0, 1]
        img_24 = (img_24 * 255).type(torch.uint8) # to bring in valid pixel range
        
        # Go to Gauss values
        
        img_peak = reverse_scaling(img_peak)
        img_24 = reverse_scaling(img_24)
        gen_data = reverse_scaling(gen_data)
        
        
        # compute the radius
        
        radius_pixels = (rsun_obs / scale_x) * scale_img #- 4
        
        # mask the images
        
        center_pix = ref_pixel_x * scale_img
        
        mask_24, num_pix_true = mask_outside_circle(img_24, center_pix, center_pix, radius_pixels)
        mask_true, num_pix_true = mask_outside_circle(img_peak, center_pix, center_pix, radius_pixels)
        mask_samp, num_pix_true = mask_outside_circle(gen_data, center_pix, center_pix, radius_pixels)
        
        # Plot mask samples
        gt_24 = mask_24[0].reshape(1, 256, 256).permute(1, 2, 0).cpu().numpy()
        gt_peak = img_peak[0].reshape(1, 256, 256).permute(1, 2, 0).cpu().numpy()
        ema_samp = gen_data.permute(1, 2, 0).cpu().numpy()
        # # Create a figure with two subplots
        #TEST
        
        
        
        # Your existing code for generating the numpy array images
        gt_peak = img_peak[0].reshape(1, 256, 256).permute(1, 2, 0).cpu().numpy()
        ema_samp = gen_data.permute(1, 2, 0).cpu().numpy()
    
        
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
        # # Plot the original image in the first subplot
        # im1 = ax1.imshow(gt_peak)
        # ax1.set_title('True image')
        # cbar1 = fig.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the first image
    
        # # Plot the EMA sampled image in the second subplot
        # im2 = ax2.imshow(ema_samp)
        # ax2.set_title('1 day prediction')
        # cbar2 = fig.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the second image
    
        # # Adjust the spacing between subplots
        # fig.subplots_adjust(bottom=0.2, top=0.85, wspace=0.4)
    
        # # Show the plot
        # plt.show()
        # plt.close()
        
        ##### COMPUTE THE FULL DISK TOTAL MAG FLUX AND NET FLUX
        
        tstart, tend = extract_and_format_datetime(list_peak[i])
        
        distance_earth = sun_earth_distance_in_meters(tstart)
        
        pix_area = compute_area(scale_x, distance_earth) # in meters
        
        num_pix_true = 256 * 256
        
        total_area = pix_area * num_pix_true
        
        total_flux_24 = torch.sum(torch.abs(mask_24)).item() * total_area
        
        net_flux_24 = torch.sum(mask_24).item() * total_area
        
        total_unsigned_flux = torch.sum(torch.abs(mask_true)).item() * total_area
        
        total_net_flux = torch.sum(mask_true).item() * total_area
        
        print('Total Flux = {}'.format(total_unsigned_flux))
        print('Net Flux = {}'.format(total_net_flux))
        
        total_unsigned_flux_samp = torch.sum(torch.abs(mask_samp)).item() * total_area
        
        total_net_flux_samp = torch.sum(mask_samp).item() * total_area
        
        print('Total Flux Samp = {}'.format(total_unsigned_flux_samp))
        print('Net Flux Samp = {}'.format(total_net_flux_samp))
        
        perc_var_tot = ((total_unsigned_flux - total_unsigned_flux_samp) / total_unsigned_flux) * 100
        
        perc_var_net = ((total_net_flux - total_net_flux_samp) / total_net_flux) * 100
        
        # SAVE VALUES 
        
        flux_diff_tot.append(total_unsigned_flux - total_flux_24)
        flux_diff_net.append(total_net_flux - net_flux_24)
        full_disk_tot.append(total_unsigned_flux)
        full_disk_net.append(total_net_flux)
        flux_diff_tot_samp.append(total_unsigned_flux_samp - total_flux_24)
        flux_diff_net_samp.append(total_net_flux_samp - net_flux_24)
        full_disk_tot_samp.append(total_unsigned_flux_samp)
        full_disk_net_samp.append(total_net_flux_samp)
        pv_full_disk_tot.append(perc_var_tot)
        pv_full_disk_net.append(perc_var_net)
        pv_flux_ev_fd_tot.append((((total_unsigned_flux_samp - total_flux_24) - (total_unsigned_flux - total_flux_24)) / (total_unsigned_flux_samp - total_flux_24)) * 100)
        pv_flux_ev_fd_net.append((((total_net_flux_samp - net_flux_24) - (total_net_flux - net_flux_24)) / (total_net_flux_samp - net_flux_24)) * 100)
        
        ## Persistence Model
        
        original_tensor, persistence_pred = persistence_perd(list_24_4k[i])
        
        # mask_perst, num_pix_true = mask_outside_circle(persistence_pred, center_pix, center_pix, radius_pixels - 2)
        
        original_tensor = reverse_scaling(original_tensor)
        mask_perst = reverse_scaling(persistence_pred)
        
        original_24 = original_tensor.reshape(1, 256, 256).permute(1, 2, 0).cpu().numpy()
        persistence_24 = mask_perst.permute(1, 2, 0).cpu().numpy()
        persistence_24 = np.nan_to_num(persistence_24, nan=-250)
    
        mask_perst = torch.from_numpy(persistence_24).reshape(1, 256, 256)    
        
        # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))

        # # Plot the original image in the first subplot
        # im1 = ax1.imshow(gt_24, cmap='gray')
        # ax1.set_title('Input', fontsize=36)
        # ax1.set_xticks([])
        # ax1.set_yticks([])
        
        # # Plot the EMA sampled image in the second subplot
        # im2 = ax2.imshow(gt_peak, cmap='gray')
        # ax2.set_title('Target', fontsize=36)
        # ax2.set_xticks([])
        # ax2.set_yticks([])
        
        # # Plot the diffusion model image in the third subplot
        # im3 = ax3.imshow(ema_samp, cmap='gray')
        # ax3.set_title('Diffusion model', fontsize=36)
        # ax3.set_xticks([])
        # ax3.set_yticks([])
        
        # # Plot the persistence model image in the fourth subplot
        # im4 = ax4.imshow(persistence_24, cmap='gray')
        # ax4.set_title('Persistence model', fontsize=36)
        # ax4.set_xticks([])
        # ax4.set_yticks([])
        
        # # Adjust the spacing between subplots
        # fig.subplots_adjust(bottom=0.2, top=0.85, wspace=0.4)
        
        # # Show the plot
        # plt.show()
        # plt.close()
        
        # fig, (ax2, ax3, ax4) = plt.subplots(1, 3, figsize=(15, 5))
    
        # # Plot the original image in the first subplot
    
        # # Plot the EMA sampled image in the second subplot
        # im2 = ax2.imshow(gt_peak)
        # ax2.set_title('Target', fontsize=25)
        # cbar2 = fig.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the second image
        
        # im3 = ax3.imshow(ema_samp - gt_peak)
        # ax3.set_title('Diffusion model Difference', fontsize=25)
        # cbar2 = fig.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the second image
        
        # im4 = ax4.imshow(persistence_24 - gt_peak)
        # ax4.set_title('Persistence model Difference', fontsize=25)
        # cbar4 = fig.colorbar(im4, ax=ax4, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the second image
    
        # # Adjust the spacing between subplots
        # fig.subplots_adjust(bottom=0.2, top=0.85, wspace=0.4)
    
        # # Show the plot
        # plt.show()
        
        # Compute metrics with persistence model
        
        total_unsigned_flux_samp_per = torch.sum(torch.abs(mask_perst)).item() * total_area
        
        total_net_flux_samp_per = torch.sum(mask_perst).item() * total_area
        
        print('Total Flux Pers = {}'.format(total_unsigned_flux_samp_per))
        print('Net Flux Pers = {}'.format(total_net_flux_samp_per))
        
        
        perc_var_tot_per = ((total_unsigned_flux - total_unsigned_flux_samp_per) / total_unsigned_flux) * 100
        
        perc_var_net_per = ((total_net_flux - total_net_flux_samp_per) / total_net_flux) * 100
        
        pv_full_disk_tot_pers.append(perc_var_tot_per)
        pv_full_disk_net_pers.append(perc_var_net_per)
        pv_flux_ev_fd_tot_pers.append((((total_unsigned_flux_samp_per - total_flux_24) - (total_unsigned_flux - total_flux_24)) / (total_unsigned_flux_samp_per - total_flux_24)) * 100)
        pv_flux_ev_fd_net_pers.append((((total_net_flux_samp_per - net_flux_24) - (total_net_flux - net_flux_24)) / (total_net_flux_samp_per - net_flux_24)) * 100)
        
        ###### COMPUTE THE FULL DISK PER PIX #### IT IS THE SAME BECAUSE IT'S A LINEAR COMBINATION
        
        ###### COMPUTE THE ACTIVE REGION FLUX
        # 24 h INFO
        client_24 = hek.HEKClient()
        
        tstart24, tend24 = extract_and_format_datetime(list_24[i])
        
        # Query the HEK for all events in the time range
        events_24 = client_24.search(a.Time(tstart24, tend24), a.hek.AR)
    
        # Dictionary to store HPC coordinates for each NOAA AR number
        ar_coordinates_24 = {}
        ar_number_24 = []
        
        for event in events_24:
            noaa_number = event.get('ar_noaanum')
            if noaa_number is not None:  # Check if the AR number is not None
                hpc_coord = (event.get('hpc_x', 'N/A'), event.get('hpc_y', 'N/A'))
                ar_coordinates_24[noaa_number] = hpc_coord
                ar_number_24.append(noaa_number)
                if noaa_number in ar_number_24:
                    continue
                else:
                    ar_number_24.append(noaa_number)
                    
        ar_number_24 = list(set(ar_number_24))
        
        # PEAK INFO
        client_peak = hek.HEKClient()
        
        # Query the HEK for all events in the time range
        events_peak = client_peak.search(a.Time(tstart, tend), a.hek.AR)
    
        
        # Dictionary to store HPC coordinates for each NOAA AR number
        ar_coordinates = {}
        ar_number = []
        
        for event in events_peak:
            noaa_number = event.get('ar_noaanum')
            if noaa_number is not None:  # Check if the AR number is not None
                hpc_coord = (event.get('hpc_x', 'N/A'), event.get('hpc_y', 'N/A'))
                ar_coordinates[noaa_number] = hpc_coord
                if noaa_number in ar_number:
                    continue
                else:
                    ar_number.append(noaa_number)
    
        ar_number = list(set(ar_number))
            
        ar_fluxes = []
        
        indexes = [ar_number_24.index(item) for item in ar_number if item in ar_number_24]
        
        new_ar_number = [item for item in ar_number if item in ar_number_24]
        
        for j in range(len(new_ar_number)):
            
            # coordinate ar 24 h
            
            hpc_x_24 = ar_coordinates_24[ar_number_24[indexes[j]]][0]
            hpc_y_24 = ar_coordinates_24[ar_number_24[indexes[j]]][1]
            pixel_coordinates_24 = hpc_to_pixel(hpc_x_24, hpc_y_24, scale_x, scale_y, ref_pixel_x, ref_pixel_y, original_size, new_size)
            
            pixel_x_24 = pixel_coordinates_24[0]
            pixel_y_24 = pixel_coordinates_24[1]
            
            # coordinate ar peak
            
            hpc_x = ar_coordinates[new_ar_number[j]][0]
            hpc_y = ar_coordinates[new_ar_number[j]][1]
            pixel_coordinates = hpc_to_pixel(hpc_x, hpc_y, scale_x, scale_y, ref_pixel_x, ref_pixel_y, original_size, new_size)
            
            
            pixel_x = pixel_coordinates[0]
            pixel_y = pixel_coordinates[1]
            
            # Define the size of the box
            box_size_x = 50
            box_size_y = 40
            
            # Calculate the corners of the box 24h
            x_min_24 = int(pixel_x_24 - box_size_x / 2)
            x_max_24 = int(pixel_x_24 + box_size_x / 2)
            y_min_24 = int(pixel_y_24 - box_size_y / 2)
            y_max_24 = int(pixel_y_24 + box_size_y / 2)
            
            # Calculate the corners of the box
            x_min = int(pixel_x - box_size_x / 2)
            x_max = int(pixel_x + box_size_x / 2)
            y_min = int(pixel_y - box_size_y / 2)
            y_max = int(pixel_y + box_size_y / 2)

            # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))
        
            # # Plot the original image in the first subplot
            # im1 = ax1.imshow(gt_24)
            # ax1.set_title('Input Image', fontsize=25)
            
            # # Adding a box around the active region
            # rect_24 = patches.Rectangle((x_min_24, y_min_24), box_size_x, box_size_y, linewidth=2, edgecolor='r', facecolor='none')
            # rect = patches.Rectangle((x_min, y_min), box_size_x, box_size_y, linewidth=2, edgecolor='r', facecolor='none')
            # rect2 = patches.Rectangle((x_min, y_min), box_size_x, box_size_y, linewidth=2, edgecolor='r', facecolor='none')
            # rect3 = patches.Rectangle((x_min, y_min), box_size_x, box_size_y, linewidth=2, edgecolor='r', facecolor='none')
            # ax1.add_patch(rect_24)
            
            # # Annotate the center point
            # ax1.scatter(pixel_x_24, pixel_y_24, color='blue', s=10)  # Mark the center point
            
            # # Plot the EMA sampled image in the second subplot
            # im2 = ax2.imshow(gt_peak)
            # ax2.set_title('Target', fontsize=25)
            # ax2.add_patch(rect)
            # # Annotate the center point
            # ax2.scatter(pixel_x, pixel_y, color='blue', s=10)  # Mark the center point
            
            # im3 = ax3.imshow(ema_samp)
            # ax3.set_title('1 day prediction', fontsize=25)
            # ax3.add_patch(rect2)
            # # Annotate the center point
            # ax3.scatter(pixel_x, pixel_y, color='blue', s=10)  # Mark the center point
            
            # im4 = ax4.imshow(persistence_24)
            # ax4.set_title('1 day Persistence', fontsize=25)
            # ax4.add_patch(rect3)
            # # Annotate the center point
            # ax4.scatter(pixel_x, pixel_y, color='blue', s=10)  # Mark the center point
        
            # # Adjust the spacing between subplots
            # fig.subplots_adjust(bottom=0.2, top=0.85, wspace=0.4)
            
            # plt.tight_layout()
            # plt.close()
            # Show the plot
            # plt.savefig(f'./final_sampling/mag_physics/{i}/{j}_full.png')
            
            # plt.close()
        
            # Extract the ROI from both images
            roi_gt_24 = gt_24[y_min_24:y_max_24, x_min_24:x_max_24]
            roi_gt_peak = gt_peak[y_min:y_max, x_min:x_max]
            roi_ema_samp = ema_samp[y_min:y_max, x_min:x_max]
            roi_ema_per = persistence_24[y_min:y_max, x_min:x_max]
            
            total_area = pix_area * (box_size_x * box_size_y)
            
            total_flux_24_ar = np.sum(np.abs(roi_gt_24)) * total_area
            
            total_unsigned_flux_ar = np.sum(np.abs(roi_gt_peak)) * total_area
            
            total_net_24_ar = np.sum(roi_gt_24) * total_area
            
            total_net_flux_ar = np.sum(roi_gt_peak) * total_area
            
            print('Total Flux = {}'.format(total_unsigned_flux_ar))
            print('Net Flux = {}'.format(total_net_flux_ar))
            
            total_unsigned_flux_samp_ar = np.sum(np.abs(roi_ema_samp)).item() * total_area
            
            total_net_flux_samp_ar = np.sum(roi_ema_samp).item() * total_area
            
            print('Total Flux Samp = {}'.format(total_unsigned_flux_samp_ar))
            print('Net Flux Samp = {}'.format(total_net_flux_samp_ar))
            
            perc_tot_flux = ((total_unsigned_flux_ar - total_unsigned_flux_samp_ar) / total_unsigned_flux_ar) * 100
            
            perc_net_flux = ((total_net_flux_ar - total_net_flux_samp_ar) / total_net_flux_ar) * 100
            
            print('Perc var Tot flux = {}'.format(perc_tot_flux))
            print('Perc var Net flux = {}'.format(perc_net_flux))
            
            total_unsigned_flux_samp_ar_per = np.sum(np.abs(roi_ema_per)).item() * total_area
            
            total_net_flux_samp_ar_per = np.sum(roi_ema_per).item() * total_area
            
            print('Total Flux Pers = {}'.format(total_unsigned_flux_samp_ar_per))
            print('Net Flux Pers = {}'.format(total_net_flux_samp_ar_per))
            
            perc_tot_flux_per = ((total_unsigned_flux_ar - total_unsigned_flux_samp_ar_per) / total_unsigned_flux_ar) * 100
            
            perc_net_flux_per = ((total_net_flux_ar - total_net_flux_samp_ar_per) / total_net_flux_ar) * 100
            
            # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))
    
            # # Plot the original image in the first subplot
            # im1 = ax1.imshow(roi_gt_24)
            # ax1.set_title('Input', fontsize=36)
            # cbar1 = fig.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the first image
            # # Plot the EMA sampled image in the second subplot
            # im2 = ax2.imshow(roi_gt_peak)
            # ax2.set_title('Target', fontsize=36)
            # cbar2 = fig.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the first image
            # # Plot the EMA sampled image in the second subplot
            # im3 = ax3.imshow(roi_ema_samp)
            # ax3.set_title('1 day prediction', fontsize=36)
            # cbar3 = fig.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the first image
            
            # im4 = ax4.imshow(roi_ema_per)
            # ax4.set_title('1 day Persistence', fontsize=36)
            # cbar4 = fig.colorbar(im4, ax=ax4, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the first image
            
            # # Adjust the spacing between subplots
            # fig.subplots_adjust(bottom=0.2, top=0.85, wspace=0.4)
            
            # # plt.figtext(0.5, 0.25, f'Percentage variation Total Flux = {perc_tot_flux:.2f}%', ha='center', fontsize=36)
            # # plt.figtext(0.5, 0.20, f'Percentage variation Net Flux = {perc_net_flux:.2f}%', ha='center', fontsize=36)
    
            # plt.tight_layout()
            # plt.close()
            # Show the plot
            # plt.savefig(f'./final_sampling/mag_physics/{i}/{j}_crop.png')
            
            # fig, (ax2, ax3, ax4) = plt.subplots(1, 3, figsize=(15, 5))
    
        
            # # Plot the EMA sampled image in the second subplot
            # im2 = ax2.imshow(roi_gt_peak)
            # ax2.set_title('Target', fontsize=36)
            # cbar2 = fig.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the first image
            # # Plot the EMA sampled image in the second subplot
            # im3 = ax3.imshow(roi_ema_samp - roi_gt_peak)
            # ax3.set_title('1 day prediction', fontsize=36)
            # cbar3 = fig.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the first image
            
            # im4 = ax4.imshow(roi_ema_per - roi_gt_peak)
            # ax4.set_title('1 day Persistence', fontsize=36)
            # cbar4 = fig.colorbar(im4, ax=ax4, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the first image
            
            # # Adjust the spacing between subplots
            # fig.subplots_adjust(bottom=0.2, top=0.85, wspace=0.4)
            
            # plt.figtext(0.5, 0.25, f'Average difference DM = {np.mean(np.abs(roi_ema_samp) - np.abs(roi_gt_peak)):.2f}%', ha='center', fontsize=36)
            # plt.figtext(0.5, 0.20, f'Average difference PM = {np.mean(np.abs(roi_ema_per) - np.abs(roi_gt_peak)):.2f}%', ha='center', fontsize=36)
    
            # plt.tight_layout()
            
            # plt.close()
            
            # Varition from the 1 day 
            
            var_tot_flux = np.abs(total_unsigned_flux_ar - total_flux_24_ar)
            var_net_flux = np.abs(total_net_flux_ar - total_net_24_ar)
            
            var_tot_flux_samp = np.abs(total_unsigned_flux_samp_ar - total_flux_24_ar)
            var_net_flux_samp = np.abs(total_net_flux_samp_ar - total_net_24_ar)
            
            perc_var_ev_tot_ar = (var_tot_flux - var_tot_flux_samp) / var_tot_flux * 100
            perc_var_ev_net_ar = (var_net_flux - var_net_flux_samp) / var_net_flux * 100
            
            var_tot_flux_samp_per = np.abs(total_unsigned_flux_samp_ar_per - total_flux_24_ar)
            var_net_flux_samp_per = np.abs(total_net_flux_samp_ar_per - total_net_24_ar)
            
            perc_var_ev_tot_ar_per = (var_tot_flux - var_tot_flux_samp_per) / var_tot_flux * 100
            perc_var_ev_net_ar_per = (var_net_flux - var_net_flux_samp_per) / var_net_flux * 100
    
            
            if is_within_circle(tstart, hpc_x, hpc_y, degree=70):
            
                # Gaussian Filter
                
                fltr = 5
                
                width = 40
                height = 50
                
                blur_gt_24 = cv2.GaussianBlur(roi_gt_24, (fltr, fltr), sigmaX=0).reshape(width, height, 1)
                blur_gt_peak = cv2.GaussianBlur(roi_gt_peak, (fltr, fltr), sigmaX=0).reshape(width, height, 1)
                blur_ema_samp = cv2.GaussianBlur(roi_ema_samp, (fltr, fltr), sigmaX=0).reshape(width, height, 1)
                blur_ema_per = cv2.GaussianBlur(roi_ema_per, (fltr, fltr), sigmaX=0).reshape(width, height, 1)
        
                # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))
        
                # # Plot the original image in the first subplot
                # im1 = ax1.imshow(blur_gt_24)
                # ax1.set_title('Input', fontsize=36)
                # cbar1 = fig.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the first image
                # # Plot the EMA sampled image in the second subplot
                # im2 = ax2.imshow(blur_gt_peak)
                # ax2.set_title('Target', fontsize=36)
                # cbar2 = fig.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the first image
                # # Plot the EMA sampled image in the second subplot
                # im3 = ax3.imshow(blur_ema_samp)
                # ax3.set_title('1 day prediction', fontsize=36)
                # cbar3 = fig.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the first image
                
                # im4 = ax4.imshow(blur_ema_per)
                # ax4.set_title('1 day Persistence', fontsize=36)
                # cbar4 = fig.colorbar(im4, ax=ax4, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the first image
                
                # # Adjust the spacing between subplots
                # fig.subplots_adjust(bottom=0.2, top=0.85, wspace=0.4)
        
                # plt.tight_layout()    
                # plt.close()
        
                # Size of the active region
                
                contour_24 = obtain_contour_all(blur_gt_24)
                contour_peak = obtain_contour_all(blur_gt_peak)
                contour_samp = obtain_contour_all(blur_ema_samp)
                contour_per = obtain_contour_all(blur_ema_per)
                
                import cv2
                
                # List of contours and corresponding images
                contours = [contour_24, contour_peak, contour_samp, contour_per]
                images = [blur_gt_24, blur_gt_peak, blur_ema_samp, blur_ema_per]
                
                # Create subplot structure
                # fig, axs = plt.subplots(1, 4, figsize=(15, 5))
                
                # Titles for subplots
                titles = ['24 Hours', 'Target', 'Prediction', 'Persitence']
                
                ar_area = []
                # Iterate over each image and contour
                for image, cont, title in zip(images, contours, titles):
                    # Display the image
                    # ax.imshow(image, cmap='gray')
                    # ax.set_title(title, fontsize=36)
                
                    # Variable to store the total sum of values and total count of pixels inside all contours
                    total_sum = 0
                    total_pixel_count = 0
                    
                    for contour in cont:
                        # Create a mask for the contour
                        mask = np.zeros(image.shape, dtype=np.uint8)
                        cv2.drawContours(mask, [contour], -1, 255, -1)  # -1 in thickness fills the contour
                    
                        # Apply the mask to the image
                        masked_image = np.where(mask == 255, roi_gt_peak, 0)
                    
                        # Sum the values inside the contour
                        contour_sum = np.sum(masked_image)
                        total_sum += contour_sum
                    
                        # Count the number of pixels inside the contour
                        pixel_count = np.count_nonzero(mask)
                        total_pixel_count += pixel_count
                    
                        # Plot the contour on the second subplot
                        # ax.plot(contour[:, 0, 0], contour[:, 0, 1], color='green', linewidth=3)
                    
                    ar_area.append(total_pixel_count)
                    # ax.set_xlabel(f'Total pixels: {total_pixel_count}', fontsize=25)
                
                # Adjust layout
                # plt.tight_layout()
                # plt.show()
                # plt.close()
                
                
                thrs_24 = 30 
                thrs_gt = 30 
                thrs_peak = 30 
                thrs_pers = 30 
                
                contour_24_white = obtain_contour(blur_gt_24, thrs=thrs_24, white=True)
                contour_peak_white = obtain_contour(blur_gt_peak, thrs=thrs_gt, white=True)
                contour_samp_white = obtain_contour(blur_ema_samp, thrs=thrs_peak, white=True)
                contour_per_white = obtain_contour(blur_ema_per, thrs=thrs_pers, white=True)
                          
                contour_24_black = obtain_contour(blur_gt_24, thrs=thrs_24, white=False)
                contour_peak_black = obtain_contour(blur_gt_peak, thrs=thrs_gt, white=False)
                contour_samp_black = obtain_contour(blur_ema_samp, thrs=thrs_peak, white=False)
                contour_per_black = obtain_contour(blur_ema_per, thrs=thrs_pers, white=False)
                
                # List of contours and corresponding images for both white and black
                white_contours = [contour_24_white, contour_peak_white, contour_samp_white, contour_per_white]
                black_contours = [contour_24_black, contour_peak_black, contour_samp_black, contour_per_black]
                images = [blur_gt_24, blur_gt_peak, blur_ema_samp, blur_ema_per]
                
                # Create subplot structure
                # fig, axs = plt.subplots(2, 4, figsize=(20, 10))  # Adjusted for two rows for white and black contours
                
                # Titles for subplots
                titles = ['24 Hours', 'Target', 'Prediction', 'Persistence']
                
                white_areas = []
                black_areas = []
                white_mask_cont = []
                black_mask_cont = []
                # Iterate over each image and contour set
                for o in range(4):
                    image = images[o]
                    white_cont = white_contours[o]
                    black_cont = black_contours[o]
                
                    # Process white contours
                    white_mask = np.zeros(image.shape, dtype=np.uint8)
                    for contour in white_cont:
                        cv2.drawContours(white_mask, [contour], -1, 255, -1)
                    white_area = np.count_nonzero(white_mask)
                    white_mask_cont.append(white_mask)
                    white_areas.append(white_area)
                    
                    # Process black contours
                    black_mask = np.zeros(image.shape, dtype=np.uint8)
                    for contour in black_cont:
                        cv2.drawContours(black_mask, [contour], -1, 255, -1)
                    black_area = np.count_nonzero(black_mask)
                    black_mask_cont.append(black_mask)
                    black_areas.append(black_area)
                
                    # Display the image with white contours
                #     axs[0, o].imshow(image, cmap='gray')
                #     axs[0, o].set_title(titles[o], fontsize=36)
                #     for contour in white_cont:
                #         axs[0, o].plot(contour[:, 0, 0], contour[:, 0, 1], 'r', linewidth=2)  # Red for white contours
                #     axs[0, o].set_xlabel(f'White pixels: {white_area}', fontsize=25)
                    
                #     # Display the image with black contours
                #     axs[1, o].imshow(image, cmap='gray')
                #     for contour in black_cont:
                #         axs[1, o].plot(contour[:, 0, 0], contour[:, 0, 1], 'b', linewidth=2)  # Blue for black contours
                #     axs[1, o].set_xlabel(f'Black pixels: {black_area}', fontsize=25)
                
                # # Adjust layout
                # plt.tight_layout()
                # plt.show()
                # plt.close()
                
                contour_24_white = obtain_contour(blur_gt_24, thrs=thrs_24, white=True)
                contour_peak_white = obtain_contour(blur_gt_peak, thrs=thrs_gt, white=True)
                contour_samp_white = obtain_contour(blur_ema_samp, thrs=thrs_peak, white=True)
                contour_per_white = obtain_contour(blur_ema_per, thrs=thrs_pers, white=True)
                          
                contour_24_black = obtain_contour(blur_gt_24, thrs=thrs_24, white=False)
                contour_peak_black = obtain_contour(blur_gt_peak, thrs=thrs_gt, white=False)
                contour_samp_black = obtain_contour(blur_ema_samp, thrs=thrs_peak, white=False)
                contour_per_black = obtain_contour(blur_ema_per, thrs=thrs_pers, white=False)
                
                # Obtain Union and intersection for morphology determination
                
                int_pixels_samp, un_pixels_samp, jaccard_index_samp_white = comput_jaccard_index(contour_peak_white, contour_samp_white, blur_gt_peak)
                
                print(f"Number of pixels in the intersection: {int_pixels_samp}")
                print(f"Number of pixels in the union: {un_pixels_samp}")
                print(f"Jaccard index white: {jaccard_index_samp_white}")
                
                int_pixels_per, un_pixels_per, jaccard_index_per_white = comput_jaccard_index(contour_peak_white, contour_per_white, blur_gt_peak)
                
                print(f"Number of pixels in the intersection: {int_pixels_per}")
                print(f"Number of pixels in the union: {un_pixels_per}")
                print(f"Jaccard index white: {jaccard_index_per_white}")
                
                int_pixels_samp, un_pixels_samp, jaccard_index_samp_black = comput_jaccard_index(contour_peak_black, contour_samp_black, blur_gt_peak)
                
                print(f"Number of pixels in the intersection: {int_pixels_samp}")
                print(f"Number of pixels in the union: {un_pixels_samp}")
                print(f"Jaccard index black: {jaccard_index_samp_black}")
                
                int_pixels_per, un_pixels_per, jaccard_index_per_black = comput_jaccard_index(contour_peak_black, contour_per_black, blur_gt_peak)
                
                print(f"Number of pixels in the intersection: {int_pixels_per}")
                print(f"Number of pixels in the union: {un_pixels_per}")
                print(f"Jaccard index black: {jaccard_index_per_black}")
                
                jaccard_index_samp = np.mean([jaccard_index_samp_white, jaccard_index_samp_black])
                jaccard_index_per = np.mean([jaccard_index_per_white, jaccard_index_per_black])
                
                print(f'Jaccard index samp: {jaccard_index_samp}')
                print(f'Jaccard index pers: {jaccard_index_per}')
                
                # COMPUTE POLARITY INVERSION LINE
                
                kernel = np.ones((4,4), np.uint8)  # Example kernel size, adjust as necessary
                
                ropi = []
                
                # Create subplot structure
                # fig, axs = plt.subplots(2, 4, figsize=(20, 10))  # Adjusted for two rows for white and black contours
                
                # Titles for subplots
                titles = ['24 Hours', 'Target', 'Prediction', 'Persistence']
                
                # Iterate over each image and contour set
                for o in range(4):
                    image = images[o]
                    
                    # Dilate white and black masks
                    dilated_white_mask = cv2.dilate(white_mask_cont[o], kernel, iterations=1)
                    dilated_black_mask = cv2.dilate(black_mask_cont[o], kernel, iterations=1)
                    
                    # Find intersections (RoPI)
                    intersection_mask = cv2.bitwise_and(dilated_white_mask, dilated_black_mask)
                    
                    intersection_mask[intersection_mask == 255] = 1
    
                    
                    ropi.append(intersection_mask.reshape(40, 50, 1))
                    # Display the image with white contours
                #     axs[0, o].imshow(image, cmap='gray')
                #     axs[0, o].set_title(titles[o], fontsize=36)
                    
                #     # Display the image with black contours
                #     axs[1, o].imshow(intersection_mask, cmap='gray')
                
                # # Adjust layout
                # plt.tight_layout()
                # plt.show()
                # plt.close()
                
                # RoPI field strength magnitude filter
                
                ropi_filtered = [ropi[0]*blur_gt_24, ropi[1]*blur_gt_peak, ropi[2]*blur_ema_samp, ropi[3]*blur_ema_per]
                
                # fig, axs = plt.subplots(2, 4, figsize=(20, 10))  # Adjusted for two rows for white and black contours
                
                # Titles for subplots
                titles = ['24 Hours', 'Target', 'Prediction', 'Persistence']
                
                for o in range(4):
                    image = ropi_filtered[o]
                    
    
                    
                    ropi.append(intersection_mask.reshape(40, 50, 1))
                    # Display the image with white contours
                #     axs[0, o].imshow(image, cmap='gray')
                #     axs[0, o].set_title(titles[o], fontsize=36)
                    
                #     # Display the image with black contours
                #     axs[1, o].imshow(ropi[o], cmap='gray')
                    
                #     # Display the image with black contours
                # # Adjust layout
                # plt.tight_layout()
                # plt.show()
                # plt.close()
                
                # Calculate the sum of pixel values for each masked image's regions
                sums_per_image = [sum_masked_regions(image, pix_area) for image in ropi_filtered]
                
                # Magnetic field filtering
                
                labeled_array_list = [label(image)[0] for image in ropi_filtered]
                
                label_sum = [*zip(labeled_array_list, sums_per_image)]
                
                # Use the function on all image-label pairs
                new_label_sum = [filter_labeled_regions_and_sums(label, sums) if len(sums) >= 2 else (label, sums) for label, sums in label_sum]
                            
                lab = [value for value, _ in new_label_sum]
                # Plotting the masked images with the labels
                # fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # Adjust the size as needed
                # for ax, image, labeled_array in zip(axes, ropi_filtered, lab):
                #     ax.imshow(image, cmap='gray')
                #     # Overlay the labeled regions
                #     for region in range(1, labeled_array.max() + 1):
                #         # Find the coordinates of the region's center
                #         center = np.mean(np.argwhere(labeled_array == region), axis=0)
                #         ax.text(center[1], center[0], str(region), color='red', ha='center', va='center')
                #     ax.axis('off')
                
                # plt.tight_layout()
                # plt.show()
                # plt.close()
                
                pil = []
                lab = [value for value, _ in new_label_sum]
                images = [roi_gt_24, roi_gt_peak, roi_ema_samp, roi_ema_per]
                
                titles = ['Input', 'Ground Truth', 'Predicted', 'Persistence Model']
                
                # lab = lab[1:]
                # images = images[1:]
                
                # fig, axs = plt.subplots(1, 3, figsize=(20, 20))  # Set up a 2x2 grid of plots
                # axs = axs.flatten()  # Flatten the array to make indexing easier
                
                # for h in range(len(lab)):
                #     # Same processing as before
                #     skeleton = skeletonize(lab[h])
                #     pil.append(skeleton)
                #     labeled_skeleton, num_features = label(skeleton)
                #     final_PILs = np.zeros(labeled_skeleton.shape, dtype=bool)
                    
                #     for feature in range(1, num_features + 1):
                #         PIL_size = np.sum(labeled_skeleton == feature)
                #         if PIL_size >= 0:
                #             final_PILs[labeled_skeleton == feature] = True
                    
                #     skeleton_2d = np.squeeze(skeleton)
                #     y, x = np.where(skeleton_2d != 0)
                    
                #     # Plot on the current subplot
                #     axs[h].imshow(images[h], cmap='gray')
                #     axs[h].scatter(x, y, color='red')
                #     axs[h].set_title(f'{titles[h]}')
                #     axs[h].axis('off')  # Optionally remove the axis for a cleaner look
                
                # plt.tight_layout()  # Adjusts subplot params so that subplots fit in the figure area
                # plt.show()
                
                for h in range(len(lab)):
                    # Assuming ropi_topR is a binary image representing top R% of magnetic flux RoPIs
                    skeleton = skeletonize(lab[h])
                    pil.append(skeleton)
                    # Label the skeleton image
                    labeled_skeleton, num_features = label(skeleton)
                    # Initiate an empty array with same shape for final PIL result
                    final_PILs = np.zeros(labeled_skeleton.shape, dtype=bool)
                    
                    # Loop through each isolated PIL and check if its size is greater or equal than Lth
                    for feature in range(1, num_features + 1):
                        PIL_size = np.sum(labeled_skeleton == feature)
                        if PIL_size >= 0:
                            final_PILs[labeled_skeleton == feature] = True
                
                    skeleton_2d = np.squeeze(skeleton)
                    # Get coordinates where skeleton is equal to 1
                    y, x = np.where(skeleton_2d != 0)
                    # Plot original image and skeleton lines
                    # plt.figure(figsize=(10,10))
                    # plt.imshow(images[h], cmap='gray')
                    # plt.scatter(x, y, color='red')    # use scatter plot to draw red dots on skeleton
                    # plt.title(f'{titles[h]}')
                    # plt.show()
                #     # plt.close()
                
                pil_length = [np.count_nonzero(pil[0]), np.count_nonzero(pil[1]),
                              np.count_nonzero(pil[2]), np.count_nonzero(pil[3])]
                
                
                pv_ar_length_samp =  (((pil_length[1] - pil_length[2]))/pil_length[1]) * 100 if pil_length[1] != 0 else 0
                pv_ar_length_pers =  (((pil_length[1] - pil_length[3]))/pil_length[1]) * 100 if pil_length[1] != 0 else 0
                
                
            else:
                ar_area = -20000
                white_areas = -20000
                black_areas = -20000
                pil_length = []
                pv_ar_length_samp = -20000
                pv_ar_length_pers = -20000
                jaccard_index_samp = -20000
                jaccard_index_per = -20000
        
            
            ar_fluxes.append([total_unsigned_flux_ar, total_net_flux_ar, total_unsigned_flux_samp_ar,
                              total_net_flux_samp_ar, perc_tot_flux, perc_net_flux,
                              perc_var_ev_tot_ar, perc_var_ev_net_ar, ar_area, white_areas, black_areas,
                              pil_length[1:],
                              perc_tot_flux_per, perc_net_flux_per,
                              pv_ar_length_samp, pv_ar_length_pers, jaccard_index_samp, jaccard_index_per])
        
        for k in range(len(ar_fluxes)):
            ar_tot['{}'.format(i)].append(ar_fluxes[k][0])
            ar_net['{}'.format(i)].append(ar_fluxes[k][1])
            ar_tot_samp['{}'.format(i)].append(ar_fluxes[k][2])
            ar_net_samp['{}'.format(i)].append(ar_fluxes[k][3])
            pv_ar_tot['{}'.format(i)].append(ar_fluxes[k][4])
            pv_ar_net['{}'.format(i)].append(ar_fluxes[k][5])
            pv_ar_tot_ev['{}'.format(i)].append(ar_fluxes[k][6])
            pv_ar_net_ev['{}'.format(i)].append(ar_fluxes[k][7])
            size_ar['{}'.format(i)].append(ar_fluxes[k][8])
            size_ar_positive['{}'.format(i)].append(ar_fluxes[k][9])
            size_ar_negative['{}'.format(i)].append(ar_fluxes[k][10])
            orientation['{}'.format(i)].append(ar_fluxes[k][11])
            pv_ar_tot_pers['{}'.format(i)].append(ar_fluxes[k][12])
            pv_ar_net_pers['{}'.format(i)].append(ar_fluxes[k][13])
            pv_orientation_samp['{}'.format(i)].append(ar_fluxes[k][14])
            pv_orientation_pers['{}'.format(i)].append(ar_fluxes[k][15])
            jacc_samp['{}'.format(i)].append(ar_fluxes[k][16])
            jacc_per['{}'.format(i)].append(ar_fluxes[k][17])
            
    import pandas as pd
    
    df = pd.DataFrame(columns=['Flux FD TOT', 'Flux FD NET', 'Flux FD TOT SAMP', 'Flux FD NET SAMP',
                               'Perc Var FD TOT', 'Perc Var FD NET','Perc Var FD TOT PERS', 'Perc Var FD NET PERS',
                               'Evolution FD TOT Samp', 'Evolution FD NET Samp', 'Evolution FD TOT Pers', 'Evolution FD NET Pers',
                               'Flux AR TOT', 'Flux AR NET',
                               'Flux AR TOT SAMP', 'Flux AR NET SAMP', 'Perc Var AR TOT', 'Perc Var AR NET',
                               'Perc Var AR TOT PERS', 'Perc Var AR NET PERS', 'Size AR', 'Size Positive', 'Size Negative',
                               'PIL_length',
                               'Perc Var length Samp', 'Perc Var length Pers', 'Jaccard_samp', 'Jaccard_per'])
    
    df['Flux FD TOT'] = full_disk_tot
    df['Flux FD NET'] = full_disk_net
    df['Flux FD TOT SAMP'] = full_disk_tot_samp
    df['Flux FD NET SAMP'] = full_disk_net_samp
    df['Perc Var FD TOT'] = pv_full_disk_tot
    df['Perc Var FD NET'] = pv_full_disk_net
    df['Perc Var FD TOT PERS'] = pv_full_disk_tot_pers
    df['Perc Var FD NET PERS'] = pv_full_disk_net_pers
    df['Evolution FD TOT Samp'] = pv_flux_ev_fd_tot
    df['Evolution FD NET Samp'] = pv_flux_ev_fd_net
    df['Evolution FD TOT Pers'] = pv_flux_ev_fd_tot_pers
    df['Evolution FD NET Pers'] = pv_flux_ev_fd_net_pers
    df['Flux AR TOT'] = [(value) for key, value in dict(list(ar_tot.items())[:]).items()]# ar_tot[:5].items()]
    df['Flux AR NET'] = [(value) for key, value in dict(list(ar_net.items())[:]).items()]#  ar_net[:5].items()]
    df['Flux AR TOT SAMP'] = [(value) for key, value in dict(list(ar_tot_samp.items())[:]).items()]#  ar_tot_samp[:5].items()]
    df['Flux AR NET SAMP'] = [(value) for key, value in dict(list(ar_net_samp.items())[:]).items()]# ar_net_samp[:5].items()]
    df['Perc Var AR TOT'] = [(value) for key, value in dict(list(pv_ar_tot.items())[:]).items()]# pv_ar_tot[:5].items()]
    df['Perc Var AR NET'] = [(value) for key, value in dict(list(pv_ar_net.items())[:]).items()]# pv_ar_net[:5].items()]
    df['Size AR'] = [(value) for key, value in dict(list(size_ar.items())[:]).items()]# size_ar[:5].items()]
    df['Size Positive'] = [(value) for key, value in dict(list(size_ar_positive.items())[:]).items()]# size_ar[:5].items()]
    df['Size Negative'] = [(value) for key, value in dict(list(size_ar_negative.items())[:]).items()]# size_ar[:5].items()]
    df['PIL_length'] = [(value) for key, value in dict(list(orientation.items())[:]).items()]# orientation[:5].items()]
    df['Perc Var AR TOT PERS'] = [(value) for key, value in dict(list(pv_ar_tot_pers.items())[:]).items()]# pv_ar_tot[:5].items()]
    df['Perc Var AR NET PERS'] = [(value) for key, value in dict(list(pv_ar_net_pers.items())[:]).items()]# pv_ar_net[:5].items()]
    df['Perc Var length Samp'] = [(value) for key, value in dict(list(pv_orientation_samp.items())[:]).items()]
    df['Perc Var length Pers'] = [(value) for key, value in dict(list(pv_orientation_pers.items())[:]).items()]
    df['Jaccard_samp'] = [(value) for key, value in dict(list(jacc_samp.items())[:]).items()]
    df['Jaccard_per'] = [(value) for key, value in dict(list(jacc_per.items())[:]).items()]
    
    
    df.to_csv(f'Directory to save your final dataset')