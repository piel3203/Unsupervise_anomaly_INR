import time
from pathlib import Path
from typing import List

from pytorch3d.transforms import euler_angles_to_matrix
import numpy as np
import torch
import sys

import train
from impl_recon.models import implicits
from impl_recon.utils import config_io, data_generation, impl_utils, io_utils, patch_utils

import pandas as pd
import json
import re
import math
import os
import torch.nn.functional as F

def export_predictions_to_json(casenames, labels_pred_cpu, labels_gt_cpu, json_path_name):
    # Load existing data if the file exists
    if json_path_name.exists():
        try:
            with open(json_path_name, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to decode JSON file '{json_path_name}' at line {e.lineno}, column {e.colno}: {e.msg}")
            print(f"[INFO] You may want to fix or delete the corrupted file.")
            return  # or raise, or skip loading to overwrite
    else:
        data = {}

    # Add new entries for each case in the batch
    for idx, casename in enumerate(casenames):
        data[casename] = {
            "labels_pred_cpu": labels_pred_cpu[idx].tolist(),
            "labels_gt_cpu": labels_gt_cpu[idx].tolist()
        }

    # Write updated data back to the JSON file
    with open(json_path_name, 'w') as f:
        json.dump(data, f, indent=4)
'''
def export_predictions_to_json(casenames, labels_pred_cpu, labels_gt_cpu, json_path_name):
    # Load existing data if the file exists
    if json_path_name.exists():
        with open(json_path_name, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    # Add new entries for each case in the batch
    for idx, casename in enumerate(casenames):
        data[casename] = {
            "labels_pred_cpu": labels_pred_cpu[idx].tolist(),
            "labels_gt_cpu": labels_gt_cpu[idx].tolist()
        }

    # Write updated data back to the JSON file
    with open(json_path_name, 'w') as f:
        json.dump(data, f, indent=4)
   

def export_predictions_to_json(batch_index, case_ids, labels_pred, labels_gt, json_path_name):
    # Ensure case IDs are strings
    case_ids_str = [str(case_id) for case_id in case_ids]

    # Convert tensors to lists for JSON compatibility
    labels_pred_list = labels_pred.tolist()
    labels_gt_list = labels_gt.tolist()

    # Data structure for JSON
    data = {case_id: {'label_pred': pred, 'label_gt': gt}
            for case_id, pred, gt in zip(case_ids_str, labels_pred_list, labels_gt_list)}

    # Append to existing JSON file if it exists
    try:
        with open(json_path_name, 'r') as file:
            existing_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = {}

    existing_data.update(data)

    # Save data
    with open(json_path_name, 'w') as file:
        json.dump(existing_data, file, indent=4)

'''

def parse_and_split(folder_path, train_file='train_healthy_cases.txt', test_file='test_cases.txt', test_ratio=0.1):
    # Get unique subject names from filenames
    subjects = {}
    for file in os.listdir(folder_path):
        if file.endswith('.nii.gz'):
            subject_name = file.rsplit('_', 1)[0]  # Get subject name without .nii.gz and last part
            if subject_name not in subjects:
                subjects[subject_name] = []
            subjects[subject_name].append(file.rsplit('.nii.gz', 1)[0])  # Append name without extension

    # Convert subject names into a list and shuffle to randomize
    subject_list = list(subjects.keys())
    random.shuffle(subject_list)

    # Determine split index based on the test ratio
    split_index = max(1, int(len(subject_list) * test_ratio))
    test_subjects = subject_list[:split_index]
    train_subjects = subject_list[split_index:]

    # Write to text files, ensuring each subject's files stay together
    with open(train_file, 'w') as train_f, open(test_file, 'w') as test_f:
        for subject in train_subjects:
            train_f.write('\n'.join(subjects[subject]) + '\n')
        for subject in test_subjects:
            test_f.write('\n'.join(subjects[subject]) + '\n')


def export_batch(batch: dict, labels_pred: np.ndarray, spacings: np.ndarray,
                 task_type: config_io.TaskType, target_dir: Path):
    batch_casenames = batch['casenames']

    is_task_ad = task_type == config_io.TaskType.AD
    is_task_rn = task_type == config_io.TaskType.RN

    if is_task_ad:
        # Create a sparse volume with hi-res spacing/offset
        spacings_lr = batch['spacings_hr']
        offsets_lr = batch['offsets_hr']
        
        # Flatten labels to 1D
        label_values: torch.Tensor = batch['labels'].flatten(start_dim=1)
        
        # Calculate voxel indices
        voxel_ids = (batch['coords'] - offsets_lr) / spacings_lr
        #voxel_ids = torch.round(voxel_ids).to(torch.int64).flatten(start_dim=1, end_dim=3)
        voxel_ids = torch.round(voxel_ids.to(torch.float32)).to(torch.int64).flatten(start_dim=1, end_dim=3)

        
        # Create empty tensor with same shape as labels_hr
        labels_lr = torch.zeros_like(batch['labels_hr'])
        print("Shape of labels_lr:", labels_lr.shape)
        print("Max voxel indices for each dimension:", voxel_ids.max(dim=0))

        # Calculate valid voxel indices
        valid_voxel_ids = (voxel_ids[:, :, 0] < labels_lr.shape[1]) & \
                          (voxel_ids[:, :, 1] < labels_lr.shape[2]) & \
                          (voxel_ids[:, :, 2] < labels_lr.shape[3])
                          
        # Flatten voxel_ids and valid_voxel_ids for correct indexing
        voxel_ids_flat = voxel_ids.reshape(-1, 3)
        valid_voxel_ids_flat = valid_voxel_ids.flatten()

        # Apply the valid_voxel_ids mask to filter voxel_ids
        voxel_ids_valid = voxel_ids_flat[valid_voxel_ids_flat]
        label_values_valid = label_values.flatten()[valid_voxel_ids_flat]
        
        # Now correctly assign values to labels_lr
        labels_lr[0, voxel_ids_valid[:, 0], 
                     voxel_ids_valid[:, 1], 
                     voxel_ids_valid[:, 2]] = label_values_valid
        
        # Convert tensors to numpy arrays
        labels_lr = labels_lr.numpy()
        spacings_lr = spacings_lr.numpy()
        
        # Offsets for hi-res volume
        offsets_lr = np.zeros_like(spacings_lr)
    elif is_task_rn:
        labels_lr = batch['labels_lr'].squeeze(1).numpy()
        spacings_lr = batch['spacings'].numpy()
        offsets_lr = None
    else:
        raise ValueError(f'Unknown task type {task_type}.')

    # Save prediction results
    for i in range(labels_pred.shape[0]):
        casename = batch_casenames[i]
        spacing = spacings[i]
        target_file = target_dir / f'{casename}_pred.nii.gz'
        io_utils.save_nifti_file(labels_pred[i, 0].astype(np.uint8), np.diag([*spacing, 1]), target_file)
        
        label_lr = labels_lr[i]
        spacing_lr = spacings_lr[i]
        ijk_to_lps = np.diag([*spacing_lr, 1])
        
        if is_task_ad and offsets_lr is not None:
            ijk_to_lps[:3, 3] = offsets_lr[i]
        
        target_file_lr = target_dir / f'{casename}_lr.nii.gz'
        io_utils.save_nifti_file(label_lr.astype(np.uint8), ijk_to_lps, target_file_lr)



def main():
    params, eval_config_path = config_io.parse_config_eval()
    evaluate_predictions = params['evaluate_predictions']
    export_predictions = params['export_predictions']
    allow_overwriting = params['allow_overwriting']
    lat_reg_lambda = params['lat_reg_lambda']
    latent_num_iters = params['latent_num_iters']
    max_num_const_train_dsc = params['max_num_const_train_dsc']
    task_type = params['task_type']
    model_dir = params['model_basedir'] / params['model_name']
    target_basedir = params['output_basedir']
    target_dirname = params['model_name']
    sample_orthogonal_slices = params['sample_orthogonal_slices']
    biggest_size = params['biggest_size']
    selected_params=params['selected_params']
    missing_value=params['missing_value']
    which_selec_param=params['which_selec_param']
    opt_vect=params['opt_vect']
    opt_rot = params['opt_rot']
    angle_dir= params['angle_dir']
    print(biggest_size)
    print(type(biggest_size))
    
    
    if not evaluate_predictions and not export_predictions:
        raise ValueError('Neither evaluation nor export were requested.')

    params['crop_size'] = 0
    params['batch_size_val'] = 1
    latent_lr = 1e-2
    if task_type == config_io.TaskType.AD:
        df_selected=train.load_df_json(selected_params)
        if sample_orthogonal_slices:
            target_dirname += f'_{latent_num_iters}_eval_ortho'
        else:
            target_dirname += f'_{latent_num_iters}_eval_ax{params["slice_step_axis"]}' \
                              f'_x{params["slice_step_size"]}'
    elif task_type == config_io.TaskType.RN:
        target_dirname += f'_eval_ax{params["slice_step_axis"]}_x{params["slice_step_size"]}'
    else:
        raise ValueError(f'Unknown task type {task_type}.')
    target_dir = target_basedir / target_dirname
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    latent_dim = params['latent_dim']

    if export_predictions:
        if not target_basedir.exists():
            raise ValueError(f'Target base directory does not exist:\n{target_basedir}')
        if target_dir.exists():
            if not allow_overwriting:
                raise ValueError(f'Target directory exists and overwriting is forbidden:\n'
                                 f'{target_dir}')
            else:
                for filepath in target_dir.glob('*'):
                    filepath.unlink()
        else:
            target_dir.mkdir()

        print(f'Writing results to: {target_dir}')
        # Redirect stdout to file + stdout
        sys.stdout = io_utils.Logger(target_dir / 'log.txt', 'w')
        # Write eval config
        config_io.write_config(eval_config_path, target_dir)

    # Print some info
    print(f'Evaluating model \'{params["model_name"]}\'.')
    if 'sample_orthogonal_slices' in params and params['sample_orthogonal_slices']:
        print('Reconstructing from three orthogonal slices.')
    else:
        print(f'Reconstructing from slices with step size {params["slice_step_size"]} '
              f'along axis {params["slice_step_axis"]}.')

    ds_loader = data_generation.create_data_loader(params, data_generation.PhaseType.INF, True)
    # During inference we rely on image size stored in with the model (it's read later)
    net = train.create_model(params, torch.ones(3, dtype=torch.float32),)
    checkpoint = io_utils.load_latest_checkpoint(model_dir, 'checkpoint', 'pth', True)
    model_state = checkpoint[0]
    if 'net' not in model_state or 'latents_train' not in model_state:
        raise ValueError('Incompatible model state in stored checkpoint.')
    net_state = model_state['net']

    if 'latents_train' in model_state and model_state['latents_train'].numel() != 0:
        mean_sq_length = torch.mean(torch.sum(torch.square(model_state['latents_train']), dim=1))
        print(f'Train latents avg length squared: {mean_sq_length}')
    
    # Remove the 'module.' prefix
    new_state_dict = {}
    for k, v in net_state.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v

    #net.load_state_dict(net_state, strict=True)
    net.load_state_dict(new_state_dict, strict=True)
    spacing_scale=0.5
    biggest_size = [round(dim * spacing_scale, 2) for dim in biggest_size]
    print(biggest_size)
    net.image_size =torch.tensor(biggest_size, dtype=torch.float16)
    
    net = net.to(device)
    net.eval()

    # For AD, check that all volumes lie inside the training image_size
    if task_type == config_io.TaskType.AD:
        assert isinstance(ds_loader.dataset, data_generation.ImplicitDataset) or \
               isinstance(ds_loader.dataset, data_generation.OrthogonalSlices)
        assert isinstance(net, implicits.AutoDecoder)
        image_size_train = net.image_size.detach().cpu()
        image_size_curr = ds_loader.dataset.image_size
        # Allow some epsilon
        eps = 1e-2
        if torch.any(image_size_curr > image_size_train + eps):
            # This may not necessarily be a problem, but worth looking into if it happens
            raise ValueError(f'Max image size is larger than current model\'s: '
                             f'{image_size_curr} > {image_size_train} with epsilon {eps}.')
        num_examples = len(ds_loader.dataset)
        if opt_rot == True:
            rot_ang = torch.nn.Embedding(num_examples, 3).cuda()
            torch.nn.init.normal_(
                rot_ang.weight.data,
                0.0,
                (math.pi**2)/64,
            )
        else:
            rot_ang = torch.nn.Embedding(num_examples, 3).cuda()
            torch.nn.init.constant_(
                rot_ang.weight.data,
                0.0
            )
            
    all_dice_metrics: List[float] = []
    asds: List[float] = []
    hd95s: List[float] = []
    max_distances: List[float] = []
    err_vol_cm_3_list: List[float] = []
    err_vol_percent_list: List[float] = []
    eccentricity_pred_list: List[float]= []
    eccentricity_gt_list: List[float]= []
    major_axis_pred_list: List[float]= []
    major_axis_gt_list: List[float]= []
    minor_axis_pred_list: List[float]= []
    minor_axis_gt_list: List[float]= []
    curvature_pred_list: List[float]= []
    curvature_gt_list: List[float]= []


    for i, batch in enumerate(ds_loader):
        t0 = time.time()
        print(f'Batch {i + 1}/{len(ds_loader)}')
        labels_gt = batch['labels']
        # Target / GT spacings (are named differently for different tasks)
        spacings = batch['spacings'] if task_type != config_io.TaskType.AD else batch['spacings_hr']
        if task_type == config_io.TaskType.AD:
            labels_gt_sparse = batch['labels'].to(device)
            labels_gt = batch['labels_hr']  # no need to move to GPU
            if labels_gt_sparse.shape[0] != 1:
                raise ValueError(f'Only batch size 1 is supported for AD, instead got '
                                 f'{labels_gt_sparse.shape[0]}.')
            assert isinstance(net, implicits.AutoDecoder)
            print(f'Batch cases: {batch["casenames"]}')

            # Initialization scaling follows DeepSDF
            latents_batch = torch.nn.Parameter(
                torch.normal(0.0, 1e-4, [labels_gt_sparse.shape[0], latent_dim], device=device),
                requires_grad=True)
            
            
            
            #ADDED TO HANDLE CLINICAL INFOS
            if opt_vect==True:
                
                df_filtered = df_selected[df_selected["ID_subject"].isin(batch['casenames'])]
                vector_list = df_filtered.values.tolist()
                vector_trimmed = torch.tensor(vector_list[0][1:], dtype=torch.float32).unsqueeze(0)
                #print("vector_trimmed",vector_trimmed)


                if missing_value==True:
                    vector_trimmed[0, which_selec_param] = -1
                    #print('NEW vector_trimmed', vector_trimmed)

                latent_and_infos =  torch.cat((vector_trimmed.cuda(), latents_batch.cuda()), dim=1) 
            else:
                latent_and_infos=latents_batch
            #print("LATENTS_BATCH_AND_INFOS",latent_and_infos)
            coords = batch['coords'].to(device)
            batch_caseids=batch['caseids']
            '''
            original_shape = coords.shape
           
            coords_flattened = coords.view(1, -1, 3)  # shape: [1, 71*64*205, 3]
            
            if opt_rot:
                # Rotate coordinates using the sequence IDs and rotation angles
                #print("ROTATION MATRIX SHAPE",np.shape(euler_angles_to_matrix(rot_ang(batch['caseids'].cuda()), 'XYZ')))
                coords_flattened = coords_flattened.float()

                # Retrieve the rotation matrix for the given case ID
                rot_ang_tensor = rot_ang(batch['caseids'].long().cuda())
                train.save_rotation_angles(batch['caseids'], rot_ang_tensor, phase='train')
                xyz_rot = coords_flattened @ euler_angles_to_matrix(rot_ang_tensor, 'XYZ')  # shape: [1, 71*64*205, 3]
                xyz_rot = xyz_rot.view(original_shape)
                #print("xyz_rot SHAPE", xyz_rot.shape)
            else:
                # If no rotation, keep the original coordinates
                xyz_rot = coords
            '''
            if opt_rot == True:
                learning_rate_rot=params['learning_rate_rot']
                rot_matrix_optim=train.optimize_latents(
                    net, latent_and_infos, labels_gt_sparse, coords, latent_lr, lat_reg_lambda,
                    latent_num_iters, device, max_num_const_train_dsc, True, rot_ang, opt_rot, batch_caseids, learning_rate_rot)
            else :
                train.optimize_latents(
                    net, latent_and_infos, labels_gt_sparse, coords, latent_lr, lat_reg_lambda,
                    latent_num_iters, device, max_num_const_train_dsc, True, rot_ang, opt_rot, batch_caseids)
                
                
            print(f'L2^2(z): {torch.mean(torch.sum(torch.square(latent_and_infos), dim=1)):.4f}')
            print('z:',latent_and_infos)
            # Full resolution prediction
            #latent_and_infos =  torch.cat((vector_trimmed.cuda(), latents_batch.cuda()), dim=1) 
            # Assume labels_gt shape is [B, D, H, W]
            current_shape = torch.tensor(labels_gt.shape[1:])  # [D, H, W]
            target_spatial_shape = current_shape
            print (target_spatial_shape)
            
            if opt_rot:
                # Option 1: Add a margin (padding)
                #margin = 50  # adjust this number as needed
                #target_spatial_shape = target_spatial_shape + margin

                # Option 2 (alternative): Scale up the shape by a factor
                scale_factor = 1.5
                target_spatial_shape = (target_spatial_shape.float() * scale_factor).ceil().to(torch.int)
            diff = target_spatial_shape - current_shape  # amount to pad
            print(target_spatial_shape)
            
            '''
            if opt_rot == True:
                labels_pred = impl_utils.sample_latents_rot(rot_matrix_optim, latent_and_infos, net,
                                                        target_spatial_shape, spacings)
            else: 
                labels_pred = impl_utils.sample_latents( latent_and_infos, net,
                                                    target_spatial_shape, spacings)
            '''
                
            labels_pred = impl_utils.sample_latents( latent_and_infos, net,
                                                    target_spatial_shape, spacings)
            
            if opt_rot == True:
                pad = [
                    diff[2] // 2, diff[2] - diff[2] // 2,  # W
                    diff[1] // 2, diff[1] - diff[1] // 2,  # H
                    diff[0] // 2, diff[0] - diff[0] // 2   # D
                ]

                labels_gt_padded = F.pad(labels_gt, pad, mode='constant', value=0)
                print ("labels_gt_padded",labels_gt_padded)
                labels_gt = labels_gt_padded.unsqueeze(1)
            else:
                labels_gt = labels_gt.unsqueeze(1)
            
            
            # Add channels for consistency with ReconNet
            labels_pred = labels_pred.unsqueeze(1)
            #labels_gt = labels_gt.unsqueeze(1)
        elif task_type == config_io.TaskType.RN:
            labels_lr = batch['labels_lr'].to(device)
            with torch.no_grad():
                labels_pred = patch_utils.predict_tiled(labels_lr, net, params['rn_patch_size'])
        else:
            raise ValueError(f'Unknown task {task_type}.')
        print(f'Batch prediction time: {time.time() - t0:.1f}s')



  
        labels_pred = torch.sigmoid(labels_pred).gt(0.5)
        labels_gt = labels_gt.to(torch.bool)
        labels_pred_cpu = labels_pred.cpu().numpy()
        print("labels_pred_cpu SHAPE", np.shape(labels_pred_cpu))
        
        labels_gt_cpu = labels_gt.cpu().numpy()
        print("labels_gt_cpu SHAPE", np.shape(labels_gt_cpu))
        
        spacings_cpu = spacings.numpy()
        json_path_name= target_dir /'predictions.json'
        print(json_path_name)
        
        #if opt_rot==True:
        #    export_predictions_to_json(batch['casenames'], labels_pred_cpu, labels_gt_cpu, json_path_name)
        
        #export_predictions_to_json(batch['casenames'], labels_pred_cpu, labels_gt_cpu, json_path_name)

        if evaluate_predictions:
            impl_utils.eval_batch(labels_pred_cpu, labels_gt_cpu, spacings_cpu, all_dice_metrics,
                                  asds, hd95s, max_distances,err_vol_cm_3_list, err_vol_percent_list, eccentricity_pred_list, 
                                  eccentricity_gt_list, major_axis_pred_list, major_axis_gt_list, minor_axis_pred_list, 
                                  minor_axis_gt_list,curvature_pred_list, curvature_gt_list, True)

        
        if export_predictions:
            export_batch(batch, labels_pred_cpu, spacings_cpu, task_type, target_dir)
            '''
            target_dir_rot=target_dir /"rot"
            export_batch(batch, labels_pred_cpu_w_rot, spacings_cpu, task_type, target_dir_rot)
            '''

    print('\n')
    print('latent_and_infos :', latents_batch)

    if all_dice_metrics and asds and hd95s and max_distances:
        print(f'ASD: {np.mean(asds):.4f} +- {np.std(asds):.4f} in '
              f'[{np.min(asds):.4f}, {np.max(asds):.4f}]')
        print(f'HSD95: {np.mean(hd95s):.4f} +- {np.std(hd95s):.4f} in '
              f'[{np.min(hd95s):.4f}, {np.max(hd95s):.4f}]')
        print(f'HSD: {np.mean(max_distances):.4f} +- {np.std(max_distances):.4f} in '
              f'[{np.min(max_distances):.4f}, {np.max(max_distances):.4f}]')
        print(f'DSC: {np.mean(all_dice_metrics):.4f} +- {np.std(all_dice_metrics):.4f} in '
              f'[{np.min(all_dice_metrics):.4f}, {np.max(all_dice_metrics):.4f}]')
        print(f'err_vol_cm_3: {np.mean(err_vol_cm_3_list):.4f} +- {np.std(err_vol_cm_3_list):.4f} in '
              f'[{np.min(err_vol_cm_3_list):.4f}, {np.max(err_vol_cm_3_list):.4f}]')
        print(f'err_vol_cm_3: {np.mean(err_vol_percent_list):.4f} +- {np.std(err_vol_percent_list):.4f} in '
              f'[{np.min(err_vol_percent_list):.4f}, {np.max(err_vol_percent_list):.4f}]')




if __name__ == '__main__':
    main()
