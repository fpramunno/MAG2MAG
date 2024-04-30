# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 09:33:32 2023

@author: pio-r
"""

import re
from datetime import datetime, timedelta
import numpy as np
import torch
from astropy.coordinates import get_body, get_sun
from astropy.time import Time
import astropy.units as u
import math
import cv2
from sunpy.map.maputils import all_coordinates_from_map, coordinate_is_on_solar_disk
from sunpy.coordinates import Helioprojective, propagate_with_solar_surface
import sunpy.data.sample
import sunpy.map
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.time as T
import torch.nn.functional as F
from torchvision import transforms

flip = transforms.RandomVerticalFlip(p=1.0)

def compute_area(pix_res, distance):
    
    ang_rad = pix_res * (math.pi / 180 / 3600)
    lin_size = ang_rad * distance
    
    pix_area = lin_size ** 2
    
    return pix_area

def compute_flux(pix_area, num_pix, mag_gauss):
    tot_area = pix_area * num_pix
    
    return mag_gauss * tot_area

def reverse_scaling(image_tensor):
    """
    Reverse scales an image tensor from the range [0, 255] to [-250, 250].

    Parameters:
    image_tensor (torch.Tensor): A tensor representing the image, expected to be in the range [0, 255].

    Returns:
    torch.Tensor: The reverse scaled image tensor in the range [-250, 250].
    """

    # Constants for the original range
    original_min = -250.0
    original_max = 250.0

    # Constants for the new range
    new_min = 0.0
    new_max = 255.0

    # Reverse scaling formula
    reverse_scaled_tensor = ((image_tensor - new_min) * (original_max - original_min) / (new_max - new_min)) + original_min

    return reverse_scaled_tensor

def extract_and_format_datetime(input_string):
    # Extract the datetime part using a regular expression
    match = re.search(r'(\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2})', input_string)
    if not match:
        return "No datetime found in the string"

    datetime_str = match.group(1)

    # Replace underscores and double dashes with appropriate characters
    formatted_datetime_str = datetime_str.replace('_', '-').replace('--', 'T')
    
    parts = formatted_datetime_str.split('T')
    time_part = parts[1].replace('-', ':', 2)
    formatted_datetime_str = parts[0] + 'T' + time_part

    try:
        # Convert the string to a datetime object
        ts_start = datetime.strptime(formatted_datetime_str, '%Y-%m-%dT%H:%M:%S')
    except ValueError as e:
        return f"Error in datetime conversion: {e}"

    # Add 2 minutes to get the end time
    ts_end = ts_start + timedelta(minutes=2)

    return ts_start.strftime('%Y-%m-%dT%H:%M:%S'), ts_end.strftime('%Y-%m-%dT%H:%M:%S')


def mask_outside_circle(image_tensor, center_x, center_y, radius):
    """
    Masks out everything outside the specified circle in the image tensor.

    :param image_tensor: PyTorch tensor of the image.
    :param center_x: x-coordinate of the circle's center.
    :param center_y: y-coordinate of the circle's center.
    :param radius: Radius of the circle.
    :return: Masked image tensor.
    """
    # Get the dimensions of the image tensor
    height, width = image_tensor.shape[-2:]

    # Create a grid of x, y coordinates
    y, x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))

    # Calculate the distance of each pixel from the center
    distance_from_center = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # Create a mask where pixels within the radius are True
    mask = distance_from_center < radius

    # Apply the mask to the image tensor
    masked_tensor = torch.where(mask.unsqueeze(0), image_tensor, torch.tensor(-250))

    # Count the number of pixels inside the mask
    num_pixels_inside_mask = torch.sum(mask).item()

    return masked_tensor, num_pixels_inside_mask


def sun_earth_distance_in_meters(date):
    """
    Calculate the distance from the Earth to the Sun on a given date in meters.

    :param date: Date in 'YYYY-MM-DD' format.
    :return: Distance in meters.
    """
    time = Time(date)
    sun = get_sun(time)  # Sun's position
    earth = get_body('earth', time)  # Earth's position

    distance = (sun.cartesian - earth.cartesian).norm()
    return distance.to(u.m).value  # Convert to meters

def hpc_to_pixel(hpc_x, hpc_y, scale_x, scale_y, ref_pixel_x, ref_pixel_y, original_size, new_size):
    """
    Convert HPC coordinates to pixel coordinates, adjusted for image resizing.

    Parameters are the same as before, with the addition of:
    original_size (tuple): Original dimensions of the image (width, height).
    new_size (tuple): New dimensions of the image after resizing (width, height).
    """
    # Adjust the reference pixel based on the resizing
    scale_factor_x = new_size[0] / original_size[0]
    scale_factor_y = new_size[1] / original_size[1]
    adjusted_ref_pixel_x = ref_pixel_x * scale_factor_x
    adjusted_ref_pixel_y = ref_pixel_y * scale_factor_y

    # Adjust the scale if necessary
    adjusted_scale_x = scale_x / scale_factor_x
    adjusted_scale_y = scale_y / scale_factor_y

    # Convert coordinates
    pixel_x = adjusted_ref_pixel_x + (hpc_x / adjusted_scale_x)
    pixel_y = adjusted_ref_pixel_y - (hpc_y / adjusted_scale_y)  # Y-axis is usually inverted
    return pixel_x, pixel_y

def obtain_contour_all(image, thrs=30, min_contour_length=15, already_bin=None):
    # Binarize
    
    image = (np.abs(image) >= thrs).astype(int)
        
    # Convert binary image to uint8
    binary_map_uint8 = (image * 255).astype(np.uint8)
    
    # Find contours on the binary image
    contours, hierarchy = cv2.findContours(binary_map_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv2.findContours(binary_map_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    
    # Filter contours by length
    # filtered_contours = contours 
    filtered_contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) >= min_contour_length]
    
    return filtered_contours

def obtain_contour_canny(image, thrs=25, min_contour_length=10, white=True, already_bin=None, canny_threshold1=100, canny_threshold2=200):
    if already_bin is None:
        # Binarize
        if white:
            image = (image >= thrs).astype(int)
        else:
            image = (image <= -thrs).astype(int)
        
        # Convert binary image to uint8
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        # Assume the image is already a binary image if already_bin is not None
        image_uint8 = image
    
    # Apply Canny edge detection
    edges = cv2.Canny(image_uint8, canny_threshold1, canny_threshold2)
    
    # Find contours on the edges
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by length
    filtered_contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) >= min_contour_length]
    
    return filtered_contours

def obtain_contour(image, thrs=25, min_contour_length=10, white=True, already_bin=None):
    # Binarize
    if already_bin == None:
        if white:
            image = (image >= thrs).astype(int)
        else:
            image = (image <= -thrs).astype(int)
            
        # Convert binary image to uint8
        binary_map_uint8 = (image * 255).astype(np.uint8)
        
        # Find contours on the binary image
        contours, hierarchy = cv2.findContours(binary_map_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # contours, hierarchy = cv2.findContours(binary_map_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        
        # Filter contours by length
        # filtered_contours = contours 
        filtered_contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) >= min_contour_length]
        
        return filtered_contours
    
    else:
        # Convert binary image to uint8
        binary_map_uint8 = (image * 255).astype(np.uint8)
        
        # Find contours on the binary image
        contours, hierarchy = cv2.findContours(binary_map_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # contours, hierarchy = cv2.findContours(binary_map_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        
        # Filter contours by length
        # filtered_contours = contours 
        filtered_contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) >= min_contour_length]
        
        return filtered_contours
    
import cv2
import numpy as np

def auto_canny_threshold(image, sigma=0.33):
    # Compute the median of the single channel pixel intensities
    v = np.median(image)
    
    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    # edged = cv2.Canny(image, lower, upper)
    
    return lower, upper

def obtain_contour_with_canny(image, minVal=100, maxVal=200, min_contour_length=10):
    
    # Apply Canny edge detector
    edges = cv2.Canny(image.astype(np.uint8), minVal, maxVal)
    
    # Find contours on the edge detected image
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by length
    # filtered_contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) >= min_contour_length]
    
    return contours


def persistence_perd(FITS_data_path, img_size=256, day=1):
    
    map_hmi = sunpy.map.Map(FITS_data_path)

    hpc_coords = all_coordinates_from_map(map_hmi)

    mask = coordinate_is_on_solar_disk(hpc_coords)

    prep_hmi_data = map_hmi.data
    # prep_hmi_data = prep_hmi_data.astype(float)
    prep_hmi_data[~mask] = 0

    prep_hmi = sunpy.map.Map(prep_hmi_data, map_hmi.meta)

    prep_hmi = prep_hmi.rotate(order=3)
    
    out_time = prep_hmi.date + T.TimeDelta(day*u.day)

    out_frame = Helioprojective(observer='earth', obstime=out_time,

                                rsun=prep_hmi.coordinate_frame.rsun)


    out_center = SkyCoord(0*u.arcsec, 0*u.arcsec, frame=out_frame)

    header = sunpy.map.make_fitswcs_header(prep_hmi.data.shape,

                                           out_center,

                                           scale=u.Quantity(prep_hmi.scale))

    out_wcs = WCS(header)


    with propagate_with_solar_surface():

        out_warp = prep_hmi.reproject_to(out_wcs)


    # out_warp.peek()
    
    # Transform into tensor
    
    prep_hmi_data_tensor = torch.from_numpy(prep_hmi.data).float()

    prep_hmi_data_tensor = prep_hmi_data_tensor.reshape(1, 1, prep_hmi_data_tensor.shape[1], prep_hmi_data_tensor.shape[1])

    new_height, new_width = 256, 256  # example sizes, adjust as needed

    # Resize the tensor
    resized_tensor = F.interpolate(prep_hmi_data_tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)

    # Remove the batch dimension if you want to get back to (C, H, W)
    resized_tensor = resized_tensor.squeeze(0)

    # clipped_tensor = torch.clamp(resized_tensor, min=-250, max=250)
    
    out_warp_data_tensor = torch.from_numpy(out_warp.data).float()

    out_warp_data_tensor = out_warp_data_tensor.reshape(1, 1, out_warp_data_tensor.shape[1], out_warp_data_tensor.shape[1])

    new_height, new_width = 256, 256  # example sizes, adjust as needed

    # Resize the tensor
    resized_tensor_out_warp = F.interpolate(out_warp_data_tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)

    # Remove the batch dimension if you want to get back to (C, H, W)
    resized_tensor_out_warp = resized_tensor_out_warp.squeeze(0)

    # resized_tensor_out_warp = torch.clamp(resized_tensor_out_warp, min=-250, max=250)
    
    return flip(resized_tensor), flip(resized_tensor_out_warp)

from datetime import datetime, timedelta

def add_days_to_date(date_str, days):
    """
    Adds a specified number of days to a given date string.

    Parameters:
    date_str (str): The date string in the format 'YYYY-MM-DDTHH:MM:SS'.
    days (int): Number of days to add to the date.

    Returns:
    str: New date string in the same format after adding the specified days.
    """
    # Convert the string to a datetime object
    date_object = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")

    # Add the specified number of days
    new_date_object = date_object + timedelta(days=days)

    # Convert back to a string in the same format
    return new_date_object.strftime("%Y-%m-%dT%H:%M:%S")

import sunpy.coordinates.frames as frames
from sunpy.coordinates import get_earth

def is_within_circle(time, hpc_x, hpc_y, degree=70):
    
    hpc_x = hpc_x * u.arcsec
    hpc_y = hpc_y * u.arcsec
    
    # Create a SkyCoord object
    hpc_coord = SkyCoord(hpc_x, hpc_y, frame=frames.Helioprojective, obstime=Time(time), observer="earth")
    
    # Convert to Heliographic Stonyhurst
    hgs_coord = hpc_coord.transform_to(frames.HeliographicStonyhurst)
    
    # Extract the latitude from the HeliographicStonyhurst coordinates
    latitude_degrees = hgs_coord.lon.value
    
    is_inside = abs(latitude_degrees) <= degree
    
    return is_inside

import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
import sunpy.coordinates.frames as frames
from astropy.time import Time
from sunpy.map import Map


def plot_circle_on_sun(longitude_deg, obstime="2023-01-01T00:00:00"):
    # Constants
    solar_radius = 695700 * u.km  # Solar radius in kilometers
    num_points = 100  # Number of points to define the circle
    
    # Generate longitude points
    lat = np.linspace(-90, 90, num_points) * u.deg

    # Longitude is constant
    lon = np.full(num_points, longitude_deg) * u.deg
    
    # Convert Heliographic Stonyhurst coordinates to Helioprojective
    hgs_coords = SkyCoord(lon, lat, radius=solar_radius, frame=frames.HeliographicStonyhurst, obstime=Time(obstime))
    hpc_coords = hgs_coords.transform_to(frames.Helioprojective(observer="earth", obstime=Time(obstime)))
    
    # Plotting
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # Assuming the solar radius in arcseconds (approximate angular size of the Sun as seen from Earth)
    solar_radius_arcsec = 960  # Angular radius of the Sun in arcseconds

    # Plot the circle
    ax.plot(hpc_coords.Tx.value, hpc_coords.Ty.value, 'r-')  # Tx and Ty are in arcsec

    # Add the solar limb for reference, assuming center at (0,0)
    solar_limb = plt.Circle((0, 0), solar_radius_arcsec, color='b', fill=False)
    ax.add_artist(solar_limb)

    plt.xlim(-1100, 1100)
    plt.ylim(-1100, 1100)
    plt.xlabel('Solar X [arcsec]')
    plt.ylabel('Solar Y [arcsec]')
    plt.show()


def comput_jaccard_index(contour_1, contour_2, image_shape):
    
    mask_contour_image1 = np.zeros(image_shape.shape, dtype=np.uint8)
    mask_contour_image2 = np.zeros(image_shape.shape, dtype=np.uint8)
    
    # Draw the contours onto the respective masks
    cv2.drawContours(mask_contour_image1, contour_1, -1, 1, -1)
    cv2.drawContours(mask_contour_image2, contour_2, -1, 1, -1)
    
    # Now apply logical AND to get the intersection mask
    intersection_mask = np.logical_and(mask_contour_image1, mask_contour_image2)
    
    # Count the number of pixels in the intersection
    intersection_pixels = np.sum(intersection_mask)
    
    # Apply logical OR to get the union mask
    union_mask = np.logical_or(mask_contour_image1, mask_contour_image2)
    
    # Count the number of pixels in the union
    union_pixels = np.sum(union_mask)
    
    # If you want to calculate the Jaccard index (Intersection over Union)
    jaccard_index = intersection_pixels / union_pixels if union_pixels != 0 else 0
    
    return intersection_pixels, union_pixels, jaccard_index

import matplotlib.pyplot as plt

def plot_intersection_contour(white_contours, black_contours, roi_gt_peak):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))  # Single image for demonstration
    
    # Select the image and contours for demonstration
    image = roi_gt_peak  # For example, the first image
    white_cont = white_contours[1]
    white_cont_samp = white_contours[2]
    
    black_cont = black_contours[1]
    black_cont_samp = black_contours[2]
    
    # Initialize masks for white and black contours
    white_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    white_mask_samp = np.zeros(image.shape[:2], dtype=np.uint8)
    
    black_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    black_mask_samp = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Draw white contours
    for contour in white_cont:
        cv2.drawContours(white_mask, [contour], -1, 255, -1)
    
    # Draw black contours
    for contour in white_cont_samp:
        cv2.drawContours(white_mask_samp, [contour], -1, 255, -1)
    
    # Draw black contours
    for contour in black_cont:
        cv2.drawContours(black_mask, [contour], -1, 255, -1)
        
    # Draw black contours
    for contour in black_cont_samp:
        cv2.drawContours(black_mask_samp, [contour], -1, 255, -1)
    
    # Find intersection
    intersection_mask = cv2.bitwise_and(white_mask, white_mask_samp)
    
    intersection_mask_black = cv2.bitwise_and(black_mask, black_mask_samp)
    
    # Plotting
    ax.imshow(image, cmap='gray')
    for contour in white_cont:
        ax.plot(contour[:, 0, 0], contour[:, 0, 1], 'r', linewidth=2, label="GT" if not ax.get_legend() else "")  # Red for white contours, label only once
    for contour in white_cont_samp:
        ax.plot(contour[:, 0, 0], contour[:, 0, 1], 'b', linewidth=2, label="Predicted" if not ax.get_legend() else "") 
    for contour in black_cont:
        ax.plot(contour[:, 0, 0], contour[:, 0, 1], 'r', linewidth=2, label="GT" if not ax.get_legend() else "") 
    for contour in black_cont_samp:
        ax.plot(contour[:, 0, 0], contour[:, 0, 1], 'b', linewidth=2, label="Predicted" if not ax.get_legend() else "") 
    
    # Highlight intersection
    # For this, convert intersection_mask to coordinates
    ys, xs = np.where(intersection_mask == 255)
    ax.scatter(xs, ys, color='yellow', s=2)  # Use yellow to highlight intersection, adjust size as needed
    
    ys, xs = np.where(intersection_mask_black == 255)
    ax.scatter(xs, ys, color='yellow', s=2)  # Use yellow to highlight intersection, adjust size as needed
    
    
    # Adding a legend
    ax.legend(loc="upper right", fontsize=14)
    
    ax.set_title('Contour Intersection', fontsize=36)
    ax.set_xlabel(f'Image with White and Black Contours and Intersection', fontsize=25)
    
    # Hide axes
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()