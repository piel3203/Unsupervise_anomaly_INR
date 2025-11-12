import math
from typing import List

from skimage import measure

import numpy as np
import torch

from impl_recon.models import implicits
from impl_recon.utils import geometry_utils
from surface_distance import metrics


def calculate_curvature(label_image):
    """Calculate approximate mean curvature for a 3D binary label image."""
    # Ensure label image is binary (0 or 1 values)
    label_image = label_image.astype(bool)

    # Extract surface mesh using Marching Cubes
    verts, faces, _, _ = measure.marching_cubes(label_image, level=0.5)

    # Initialize an array to store curvature for each vertex
    curvature = np.zeros(len(verts))

    # Loop through faces to estimate curvature at each vertex
    for i, (v0, v1, v2) in enumerate(faces):
        # Get the vertices of the triangle face
        vec1 = verts[v1] - verts[v0]
        vec2 = verts[v2] - verts[v0]
        
        # Calculate normal vector for the face
        normal = np.cross(vec1, vec2)
        normal_length = np.linalg.norm(normal)
        
        # Normalize to avoid division by zero
        if normal_length != 0:
            normal /= normal_length

        # Accumulate curvature for each vertex
        curvature[v0] += normal_length
        curvature[v1] += normal_length
        curvature[v2] += normal_length

    # Average curvature across vertices
    mean_curvature = np.mean(curvature) if curvature.size else 0

    return mean_curvature

def calculate_3d_shape_properties(label_image):
    """Calculate approximate eccentricity and axis lengths for a 3D label image."""
    # Convert label_image to integer if not already
    label_image = label_image.astype(np.uint8)
    
    properties = measure.regionprops(label_image)
    
    eccentricities = []
    major_axes = []
    minor_axes = []

    for prop in properties:
        # Covariance matrix from which to calculate axis lengths
        covariance_matrix = prop.inertia_tensor
        
        # Get eigenvalues of the covariance matrix
        eigenvalues = np.linalg.eigvalsh(covariance_matrix)
        
        # Sort eigenvalues (largest to smallest for major, intermediate, minor axes)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Calculate axis lengths based on eigenvalues
        major_axis_length = 4 * np.sqrt(eigenvalues[0])
        minor_axis_length = 4 * np.sqrt(eigenvalues[-1])

        # Approximate eccentricity (ratio of minor to major axes)
        eccentricity = np.sqrt(1 - (eigenvalues[-1] / eigenvalues[0])) if eigenvalues[0] != 0 else 0

        major_axes.append(major_axis_length)
        minor_axes.append(minor_axis_length)
        eccentricities.append(eccentricity)

    return np.array(eccentricities), np.array(major_axes), np.array(minor_axes)

        
def volume_calc(prediction,spacing):
    vol=measure.regionprops_table(
        prediction.astype(int),
        properties=['area'],
    )# grâce à la librairie skimage.measure.regionprops on peut récupérer le nombre de voxels dans chaque label en one-hot
    #print("VOL:", vol)
    vol1=vol['area']*(spacing[0]*spacing[1]*spacing[2])*0.001 #calcul du volume: nb voxels x (spacing0 x spacing1 x spacing2)
    #print("VOL1:", vol1)
    #print(np.shape(vol1))
    return vol1


def sample_latents(latents: torch.Tensor, occ_net: implicits.AutoDecoder,
                   target_spatial_shape: torch.Tensor, target_spacings: torch.Tensor,
                   batch_size_coords: int = 64 ** 3) -> torch.Tensor:
    """Sample given batch of latent vectors at a given spatial resolution and spacing. The spatial
    resolution must be the same for all batch examples, therefore it doesn't contain the batch
    dimension. The spacings are individual per batch example.
    WARNING: this function assumes symmetric voxel sampling (without an offset)!
    """
    # Generate target coordinates
    batch_size_volumes = latents.shape[0]
    device = latents.device

    num_coords = int(torch.prod(target_spatial_shape).item())
    num_batches_coords = math.ceil(num_coords / batch_size_coords)
    labels_pred = torch.empty([batch_size_volumes, num_coords, 1, 1], dtype=torch.float32)
    for i in range(batch_size_volumes):
        image_size = target_spatial_shape * target_spacings[i]
        latent_volume_size = occ_net.image_size.detach().cpu()
        # noinspection PyTypeChecker
        if torch.any(image_size > latent_volume_size):
            print(
                f'Warning: sampling outside of latent volume: {image_size} > {latent_volume_size}.')
        coordinates = geometry_utils.generate_sampling_grid(target_spatial_shape, -1, 0.0,
                                                            image_size, device, 1, False)
        coordinates = coordinates.flatten(1, 3).unsqueeze(2).unsqueeze(2)  # [1, N, 1, 1, 3]

        for j in range(num_batches_coords):
            first_id = j * batch_size_coords
            last_id = min((j + 1) * batch_size_coords, num_coords)
            coordinates_curr = coordinates[:, first_id:last_id]
            # Since every volume is processed independently, make it look like batch size 1 here
            latents_curr = latents[i].unsqueeze(0)
            with torch.no_grad():
                labels_pred[i, first_id:last_id] = occ_net(
                    latents_curr, coordinates_curr).detach().cpu()[0]
    labels_pred = labels_pred.reshape(batch_size_volumes, *target_spatial_shape)
    return labels_pred  # [B, *ST]

'''
def eval_batch(labels_pred: np.ndarray, labels_gt: np.ndarray, spacings: np.ndarray,
               dices: List[float], asds: List[float], hd95s: List[float],
               max_distances: List[float], err_vol_cm_3_list: List[float], err_vol_percent_list: List[float], verbose: bool):
    """Evaluate a batch of volumetric masks."""
    if labels_pred.shape != labels_gt.shape:
        raise ValueError(f'Batch evaluation not possible: predicted shape {labels_pred.shape} '
                         f'is different from GT shape {labels_gt.shape}.')
    batch_size = labels_pred.shape[0]
    # Iterate through batch examples
    for j in range(batch_size):
        label_pred = labels_pred[j, 0]
        label_gt = labels_gt[j, 0]

        # Empty GT/prediction mess up metrics calculations...
        if label_gt.sum() == 0:
            print('Warning: empty GT occured!')
        if label_pred.sum() == 0:
            print('Warning: empty prediciton occured!')

        dice = metrics.compute_dice_coefficient(label_gt, label_pred)
        spacing = spacings[j]
        surf_distances = metrics.compute_surface_distances(label_gt, label_pred, spacing, True)
        avg_distance_gt_to_pred, avg_distance_pred_to_gt = \
            metrics.compute_average_surface_distance(surf_distances)
        asd = (avg_distance_gt_to_pred + avg_distance_pred_to_gt) / 2
        hausdorff = metrics.compute_robust_hausdorff(surf_distances, 100)
        hausdorff95 = metrics.compute_robust_hausdorff(surf_distances, 95)
        
        """
        print("shape of GT:", np.shape(label_gt))
        print(type(label_gt))
        print("shape of PRED:", np.shape(label_pred))
        
        
        num_classes = 1

        # Initialize the one-hot encoded array
        one_hot_label = np.zeros((num_classes, *label_pred.shape), dtype=np.float16)
        print(np.shape(one_hot_label))
        one_hot_gt = np.zeros((num_classes, *label_gt.shape), dtype=np.float16)
        print(np.shape(one_hot_gt))
        
                # Fill in the one-hot encoding
        for class_id in range(num_classes):
            one_hot_label[class_id] = (label_pred == class_id).astype(np.float16)
            one_hot_gt[class_id] = (label_gt == class_id).astype(np.float16)


        # Resulting one-hot array
        print(one_hot_label.shape) 
        print(one_hot_gt.shape)
        
        for i in range(1,len(one_hot_label)): 
            err_vol_cm_3=abs(volume_GT_all[i-1]-volume_pred_all[i-1])
            err_vol_percent=100*err_vol_cm_3/volume_GT_all[i-1]
        """
        volume_pred_all=volume_calc(label_pred,spacing)
        volume_GT_all=volume_calc(label_gt,spacing)
        
        #print('VOL of PRED:', volume_pred_all)
        #print(type(volume_pred_all))
        #print('VOL of GT:', volume_GT_all)
        
        err_vol_cm_3=abs(volume_GT_all[0]-volume_pred_all[0])
        err_vol_percent=100*err_vol_cm_3/volume_GT_all[0]

        
        # Calculate eccentricity
        ecc_pred = calculate_eccentricity(label_pred)
        ecc_gt = calculate_eccentricity(label_gt)

        # Store metrics
        dices.append(dice)
        asds.append(asd)
        hd95s.append(hausdorff95)
        max_distances.append(hausdorff)
        err_vol_cm_3_list.append(err_vol_cm_3)
        err_vol_percent_list.append(err_vol_percent)
        eccentricity_pred.append(np.mean(ecc_pred))  # Average eccentricity for the batch
        eccentricity_gt.append(np.mean(ecc_gt))      # Average eccentricity for the batch
        
        
        print(f"Volume error (cm³): {err_vol_cm_3}, Volume error (%): {err_vol_percent}")
        print(f"Eccentricity (pred): {eccentricity_pred[-1]}, Eccentricity (GT): {eccentricity_gt[-1]}")


        if verbose:
            print(f'Batch ASD: {np.mean(asds[-batch_size:]):.2f}, '
                  f'HSD: {np.mean(max_distances[-batch_size:]):.2f}, '
                  f'HSD95: {np.mean(hd95s[-batch_size:]):.2f}, '
                  f'DSC: {np.mean(dices[-batch_size:]):.2f}, '
                  f'err_vol_cm_3: {np.mean(err_vol_cm_3_list[-batch_size:]):.2f}, '
                  f'err_vol_percent: {np.mean(err_vol_percent_list[-batch_size:]):.2f}, '
                  f'Eccentricity (pred): {np.mean(eccentricity_pred[-batch_size:]):.2f}, '
                  f'Eccentricity (GT): {np.mean(eccentricity_gt[-batch_size:]):.2f}', flush=True)
'''
def eval_batch(labels_pred: np.ndarray, labels_gt: np.ndarray, spacings: np.ndarray,
               dices: List[float], asds: List[float], hd95s: List[float],
               max_distances: List[float], err_vol_cm_3_list: List[float], 
               err_vol_percent_list: List[float], eccentricity_pred: List[float], 
               eccentricity_gt: List[float], major_axis_pred: List[float],
               major_axis_gt: List[float], minor_axis_pred: List[float], 
               minor_axis_gt: List[float], curvature_pred: List[float],
               curvature_gt: List[float], verbose: bool):
    """Evaluate a batch of volumetric masks."""
    if labels_pred.shape != labels_gt.shape:
        raise ValueError(f'Batch evaluation not possible: predicted shape {labels_pred.shape} '
                         f'is different from GT shape {labels_gt.shape}.')
    batch_size = labels_pred.shape[0]

    for j in range(batch_size):
        label_pred = labels_pred[j, 0]
        label_gt = labels_gt[j, 0]

        if label_gt.sum() == 0:
            print('Warning: empty GT occurred!')
        if label_pred.sum() == 0:
            print('Warning: empty prediction occurred!')

        # Compute metrics
        dice = metrics.compute_dice_coefficient(label_gt, label_pred)
        spacing = spacings[j]
        surf_distances = metrics.compute_surface_distances(label_gt, label_pred, spacing, True)
        avg_distance_gt_to_pred, avg_distance_pred_to_gt = \
            metrics.compute_average_surface_distance(surf_distances)
        asd = (avg_distance_gt_to_pred + avg_distance_pred_to_gt) / 2
        hausdorff = metrics.compute_robust_hausdorff(surf_distances, 100)
        hausdorff95 = metrics.compute_robust_hausdorff(surf_distances, 95)

        # Calculate volumes
        volume_pred_all = volume_calc(label_pred, spacing)
        volume_GT_all = volume_calc(label_gt, spacing)
        
        err_vol_cm_3 = abs(volume_GT_all[0] - volume_pred_all[0])
        err_vol_percent = 100 * err_vol_cm_3 / volume_GT_all[0]

        # Calculate 3D shape properties for predictions and ground truth
        ecc_pred, major_pred, minor_pred = calculate_3d_shape_properties(label_pred)
        ecc_gt, major_gt, minor_gt = calculate_3d_shape_properties(label_gt)

        # Calculate curvature for predictions and ground truth
        curvature_value_pred = calculate_curvature(label_pred)
        curvature_value_gt = calculate_curvature(label_gt)

        # Store metrics
        dices.append(dice)
        asds.append(asd)
        hd95s.append(hausdorff95)
        max_distances.append(hausdorff)
        err_vol_cm_3_list.append(err_vol_cm_3)
        err_vol_percent_list.append(err_vol_percent)
        eccentricity_pred.append(np.mean(ecc_pred))
        eccentricity_gt.append(np.mean(ecc_gt))
        major_axis_pred.append(np.mean(major_pred))
        major_axis_gt.append(np.mean(major_gt))
        minor_axis_pred.append(np.mean(minor_pred))
        minor_axis_gt.append(np.mean(minor_gt))
        curvature_pred.append(np.mean(curvature_value_pred))
        curvature_gt.append(np.mean(curvature_value_gt))

        print(f"Volume error (cm³): {err_vol_cm_3}, Volume error (%): {err_vol_percent}")
        print(f"Eccentricity (pred): {eccentricity_pred[-1]}, Eccentricity (GT): {eccentricity_gt[-1]}")
        print(f"Curvature (pred): {curvature_pred[-1]}, Curvature (GT): {curvature_gt[-1]}")

    if verbose:
        print(f'Batch ASD: {np.mean(asds[-batch_size:]):.4f}, '
              f'HSD: {np.mean(max_distances[-batch_size:]):.4f}, '
              f'HSD95: {np.mean(hd95s[-batch_size:]):.4f}, '
              f'DSC: {np.mean(dices[-batch_size:]):.4f}, '
              f'err_vol_cm_3: {np.mean(err_vol_cm_3_list[-batch_size:]):.4f}, '
              f'err_vol_percent: {np.mean(err_vol_percent_list[-batch_size:]):.4f}, '
              f'Eccentricity (pred): {np.mean(eccentricity_pred[-batch_size:]):.4f}, '
              f'Eccentricity (GT): {np.mean(eccentricity_gt[-batch_size:]):.4f}, '
              f'Curvature (pred): {np.mean(curvature_pred[-batch_size:]):.4f}, '
              f'Curvature (GT): {np.mean(curvature_gt[-batch_size:]):.4f}', flush=True)
