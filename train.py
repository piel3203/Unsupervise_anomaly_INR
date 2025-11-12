import math
import sys
import time
from pathlib import Path
from typing import Optional
import numpy as np

import torch
from torch.optim import optimizer as opt
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.transforms import euler_angles_to_matrix

from impl_recon.models import implicits
from impl_recon.utils import config_io, data_generation, io_utils, nn_utils

import pandas as pd
import json
import re
import nibabel as nib
import os

import torch.nn.functional as F

def add_clinical(latents_batch, df_selected, batch_casenames):

    df_filtered = df_selected[df_selected["ID_subject"].isin(batch_casenames)]
    vector_list = df_filtered.values.tolist()
    vector_trimmed = torch.tensor(vector_list[0][1:], dtype=torch.float32).unsqueeze(0)
    '''
    print ("VECTOR", vector_list)
    print ("SHAPE OF VECTOR", np.shape(vector_list))
    print ("VECTOR TRIMMEd", vector_trimmed)
    print ("VECTOR TRIMMEd SHAPE", np.shape(vector_trimmed))
    print("latents_batch SHAPE", np.shape(latents_batch))
    '''
    latents_batch =  torch.cat((vector_trimmed.cuda(), latents_batch.cuda()), dim=1) 
    return latents_batch
    
def apply_rot(coords, rot_ang, batch_caseids, task='train_or_val'):
    original_shape = coords.shape
    #print("original_shape",original_shape)
    coords_flattened = coords.view(1, -1, 3)  # shape: [1, 71*64*205, 3]

    coords_flattened = coords_flattened.float()
    #print("coords_flattened SHAPE",np.shape(coords_flattened))
    if task=='infer': 
        rot_ang_tensor = F.embedding(batch_caseids.long().cuda(), rot_ang)
    else:
        rot_ang_tensor = F.embedding(batch_caseids.long().cuda(), rot_ang.weight)
    
    #print("rot_ang_tensor SHAPE AFTER",np.shape(rot_ang_tensor))

    # Retrieve the rotation matrix for the given case ID
    save_rotation_angles(batch_caseids, rot_ang_tensor, phase='test')

    # Apply rotation: Rotate the flattened coordinates by the rotation matrix
    rotation_matrix = euler_angles_to_matrix(rot_ang_tensor, 'XYZ')  # rotation matrix for each case

    xyz_rot = torch.matmul(coords_flattened, rotation_matrix)  # shape: [1, 71*64*205, 3]
    #print("xyz_rot SHAPE", np.shape(xyz_rot))
    # Restore original shape
    xyz_rot = xyz_rot.view(original_shape)
    
    return xyz_rot, rotation_matrix
        
        
def remove_rotation_from_pred(labels_pred, rotation_matrix, original_shape):

    # Compute the inverse rotation matrix (transpose of the rotation matrix)
    rotation_matrix_inv = rotation_matrix.transpose(1, 2)
    print("rotation_matrix_inv SHAPE", np.shape(rotation_matrix_inv))
    print("labels_pred SHAPE", np.shape(labels_pred))

    # Flatten the predictions to align with the coordinate transformations
    total_elements = labels_pred.numel()

    # Check if the total number of elements is divisible by 3
    if total_elements % 3 != 0:
        print("Total elements are not divisible by 3. Adjusting the shape.")
        # Compute the new size that is divisible by 3
        new_size = (total_elements // 3) * 3
        # Trim the tensor to the new size
        labels_pred_trimmed = labels_pred.view(-1)[:new_size]
        print("labels_pred_trimmed SHAPE", np.shape(labels_pred_trimmed))
        # Reshape to [1, -1, 3]
        labels_pred_flat = labels_pred_trimmed.view(1, -1, 3)
        
    else:
        # Directly reshape to [1, -1, 3]
        labels_pred_flat = labels_pred.view(1, -1, 3)
    
    print("labels_pred_flat SHAPE:", labels_pred_flat.shape)
    labels_pred_flat = labels_pred_flat.to(rotation_matrix.device)

    # Apply the inverse rotation to the predictions
    labels_pred_rot = torch.bmm(labels_pred_flat, rotation_matrix_inv)
    print("labels_pred_rot SHAPE:", labels_pred_rot.shape)
    
    # Check if reshaping back works
    if labels_pred_rot.numel() == total_elements:
        try:
            labels_pred = labels_pred_rot.view(original_shape)
            print("labels_pred SHAPE retrieve original shape", np.shape(labels_pred))
        except RuntimeError as e:
            print(f"Error reshaping labels_pred_rot: {e}")
            print(f"Original shape: {original_shape}, New shape: {labels_pred_rot.shape}")
    else:
        #print(f"Mismatch in the number of elements after rotation. Expected {total_elements}, but got {labels_pred_rot.numel()}.")
        # Padding approach: add padding to the rotated tensor to match the original shape
        pad_size = total_elements - labels_pred_rot.numel()
        if pad_size > 0:
            padding = torch.zeros((1, pad_size), device=labels_pred.device)
            print("labels_pred_rot before padding", np.shape(labels_pred_rot))
            labels_pred_rot = torch.cat((labels_pred_rot.view(1, -1), padding), dim=1)
            print("labels_pred_rot after padding", np.shape(labels_pred_rot))

        try:
            labels_pred = labels_pred_rot.view(original_shape)
            print("labels_pred RESHAPED", np.shape(labels_pred))
        except Exception as e:
            print(f"Error after padding and reshaping: {e}")
            print(f"Final size: {labels_pred_rot.shape}, Target shape: {original_shape}")
            
    return labels_pred


'''
def load_df_json(selected_params, json_path='./data/clinical_data.json'):
    df = pd.read_json(json_path, orient='index')
    df_selected = df[selected_params]
    #df_selected["ID_subject"] = df_selected["ID_subject"].apply(modify_id)
    df_selected.loc[:, "ID_subject"] = df_selected["ID_subject"].apply(modify_id)
    return df_selected
'''
def load_df_json(selected_params, json_path='./data/clinical_data.json'):
    df = pd.read_json(json_path, orient='index')

    # Harmoniser la colonne d'identifiant
    if "ID_subject" not in df.columns and "ID_patient" in df.columns:
        df["ID_subject"] = df["ID_patient"]
    elif "ID_subject" in df.columns and "ID_patient" in df.columns:
        # Priorité à ID_subject, mais remplir les vides avec ID_patient
        df["ID_subject"].fillna(df["ID_patient"], inplace=True)
    elif "ID_subject" not in df.columns and "ID_patient" not in df.columns:
        raise ValueError("Ni 'ID_subject' ni 'ID_patient' n'existent dans le JSON.")

    # Filtrer selon les paramètres demandés (en s'assurant qu'ID_subject est inclus)
    if "ID_subject" not in selected_params:
        selected_params = ["ID_subject"] + selected_params

    df_selected = df[selected_params].copy()

    # Appliquer modify_id seulement aux IDs de type 'sujet***'
    mask_sujet = df_selected["ID_subject"].astype(str).str.startswith("sujet")
    df_selected.loc[mask_sujet, "ID_subject"] = (
        df_selected.loc[mask_sujet, "ID_subject"]
        .apply(modify_id)
    )

    return df_selected
    
def modify_id(subject_id):
    pattern = r"^(sujet\d{1,2})_(pad|fort|faible)_3DUS$"
    match = re.match(pattern, subject_id)
    if match:
        return f"{match.group(1)}_RF_{match.group(2)}_3DUS"  # Add '_RF'
    return subject_id  # Return unchanged if no match


def save_rotation_angles(batch_caseids, rot_ang_tensor, phase, file_path="./rotation_angles/rotation_angles"):
    # Crée le dossier s'il n'existe pas
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Nom de fichier basé uniquement sur la phase
    filename = f"{file_path}_{phase}.txt"

    # Ouvre le fichier en mode ajout (append)
    with open(filename, 'a') as f:
        for caseid, angle in zip(batch_caseids, rot_ang_tensor):
            f.write(f"CaseID: {caseid}, Rotation Angles: {angle.cpu().detach().numpy()}\n")


def create_model(params: dict, image_size: Optional[torch.Tensor]) -> torch.nn.Module:
    task_type = params['task_type']

    net: torch.nn.Module
    if task_type == config_io.TaskType.AD:
        if image_size is None:
            raise ValueError('Image size is required for AD model creation.')
        latent_dim = params['latent_dim']
        op_num_layers = params['op_num_layers']
        op_coord_layers = params['op_coord_layers']
        selected_params=params['selected_params']
        additional_vector_size = len(selected_params)-1 #the number of param we want to include in the vector minus the ID_subject
        #print('additional_vector_size', additional_vector_size)
        net = implicits.AutoDecoder(latent_dim, len(image_size), image_size,
                                    op_num_layers, op_coord_layers, additional_vector_size=additional_vector_size)
    elif task_type == config_io.TaskType.RN:
        net = implicits.ReconNet()
    else:
        raise ValueError(f'Unknown task type {task_type}.')
    return net


def create_loss() -> torch.nn.Module:
    return nn_utils.BCEWithDiceLoss('mean', 1.0)


def train_one_epoch(task_type: config_io.TaskType, ds_loader: data.DataLoader, net: torch.nn.Module,
                    latents: torch.nn.Parameter, lat_reg_lambda: float,
                    optimizer: opt.Optimizer, criterion: torch.nn.Module, metric: torch.nn.Module,
                    device: torch.device, epoch: int, num_epochs_target: int,
                    global_step: torch.Tensor, log_epoch_count: int,
                    logger: Optional[SummaryWriter], verbose: bool, opt_rot: bool, rot_ang, df_selected, opt_vect: bool):
    loss_running = 0.0
    num_losses = 0
    metric_running = 0.0
    num_metrics = 0
    lat_reg = None
    t0 = time.time()
    net.train()
    
    for batch in ds_loader:
        labels = batch['labels'].to(device)
        #print("LABEL SHAPE", np.shape(labels))
        #print("Batch keys:", batch.keys())

        optimizer.zero_grad()

        if task_type == config_io.TaskType.AD:
            latents_batch = latents[batch['caseids']].to(device)
            #print('CASENAME', batch['casenames'])
            #print(batch.keys())  # Check available keys
            #print(batch)
            # Get the coords tensor and print the shape
            coords = batch['coords'].to(device)
            batch_caseids=batch['caseids'] 
            if opt_rot==True:
                coords, rotation_matrix= apply_rot(coords, rot_ang, batch_caseids)
            batch_casenames= batch['casenames']
            if opt_vect==True:
                latents_batch= add_clinical(latents_batch, df_selected, batch_casenames)

                       
            #print("latents_batch SHAPE", np.shape(latents_batch))
            #print("DATAFRAME df_selected", df_selected)
            # Pass latents and rotated coords through the network
            labels_pred = net(latents_batch, coords)

            # Regularization for latents
            lat_reg = torch.mean(torch.sum(torch.square(latents_batch), dim=1))

        elif task_type == config_io.TaskType.RN:
            labels_lr = batch['labels_lr'].to(device)
            labels_pred = net(labels_lr)
        else:
            raise ValueError(f'Unknown task type {task_type}.')
        #print("labels_pred SHAPE", np.shape(labels_pred))
        #print ("labels GT SHAPE", np.shape(labels))
        '''
        if opt_rot==True:
            original_label_shape = labels_pred.shape
            labels_pred =remove_rotation_from_pred(labels_pred, rotation_matrix, original_label_shape)
            '''

          
        # Calculate loss
        loss = criterion(labels_pred, labels)

        if lat_reg is not None and lat_reg_lambda > 0:
            # Gradually build up for the first 100 epochs (follows DeepSDF)
            loss += min(1.0, epoch / 100) * lat_reg_lambda * lat_reg

        loss.backward()
        optimizer.step()
        
       

        # Update running totals for loss and metrics
        loss_running += loss.item()
        num_losses += 1
        metric_running += metric(labels_pred, labels).item()
        num_metrics += batch['labels'].shape[0]
        global_step += 1

    # Logging every log_epoch_count epochs
    if epoch % log_epoch_count == 0:
        loss_avg = loss_running / num_losses
        metric_avg = metric_running / num_metrics
        num_epochs_trained = epoch + 1

        if logger is not None:
            logger.add_scalar('loss', loss_avg, global_step=num_epochs_trained)
            logger.add_scalar('metric/train', metric_avg, global_step=num_epochs_trained)
            if lat_reg is not None:
                logger.add_scalar('lat_norm2/train', lat_reg.item(), global_step=num_epochs_trained)

        if verbose:
            epoch_duration = time.time() - t0

            print(f'[{num_epochs_trained}/{num_epochs_target}] '
                  f'Avg loss: {loss_avg:.4f}; '
                  f'metric: {metric_avg:.3f}; '
                  f'global step nb. {global_step} '
                  f'({epoch_duration:.1f}s)')



def optimize_latents(net: implicits.AutoDecoder,
                     latent_and_infos: torch.Tensor, labels: torch.Tensor, coords: torch.Tensor,
                     lr: float, lat_reg_lambda: float, num_iters: int,
                     device: torch.device,
                     max_num_const_train_dsc: int,
                     verbose: bool, rot_ang, opt_rot, batch_caseids, learning_rate_rot: float = None) -> None:
    """Optimize latent vectors for a single example."""
    
    latent_and_infos = torch.nn.Parameter(latent_and_infos)
    rot_ang = rot_ang.weight
    rot_ang = torch.nn.Parameter(rot_ang)  # Rotation angles as parameter
    criterion = create_loss().to(device)

    # Choose optimizer based on whether rotation is being optimized
    if learning_rate_rot is not None:
        optimizer_val = torch.optim.Adam([
            {'params': latent_and_infos, 'lr': lr},
            {'params': rot_ang, 'lr': learning_rate_rot}
        ])
    else:
        if opt_rot:
            raise ValueError("learning_rate_rot must be specified when opt_rot is True.")
        optimizer_val = torch.optim.Adam([latent_and_infos], lr=lr)


    eval_every_x_steps = 10
    print_every_x_evals = 10
    prev_train_dsc = 0.0
    num_const_train_dsc = 0

    net.eval()

    t0 = time.time()
    for i in range(num_iters):
        #print("batch_caseids",batch_caseids)
        if opt_rot==True:
            coords1, rotation_matrix= apply_rot(coords, rot_ang, batch_caseids,task='infer')
            #print("DIFFERENCE COORDS et COORDS ROT:", coords1-coords)
        else: 
            coords1=coords

        # Forward pass with rotated coordinates
        labels_pred = net(latent_and_infos, coords1)
        #rint("ROTATION MATRIX", rotation_matrix)
        '''
        if opt_rot==True:
            original_label_shape = labels_pred.shape
            labels_pred = remove_rotation_from_pred(labels_pred, rotation_matrix, original_label_shape)
            '''
           
            
        # Loss calculation
        loss = criterion(labels_pred, labels)
        if lat_reg_lambda > 0:
            lat_reg = torch.mean(torch.sum(torch.square(latent_and_infos), dim=1))
            # Gradually build up regularization for the first 100 iterations (follows DeepSDF)
            loss += min(1.0, i / 100) * lat_reg_lambda * lat_reg

        optimizer_val.zero_grad()
        loss.backward(retain_graph=True)

        #loss.backward()
        optimizer_val.step()
        

        # Evaluation
        if (i + 1) % eval_every_x_steps == 0:
            dsc = nn_utils.dice_coeff(torch.sigmoid(labels_pred), labels, 0.5).item()
            if verbose and round((i + 1) / eval_every_x_steps) % print_every_x_evals == 0:
                print(f'Step {i + 1:04d}/{num_iters:04d}: loss {loss.item():.4f} DSC {dsc:.3f} '
                      f'L2^2(z): {torch.mean(torch.sum(torch.square(latent_and_infos), dim=1)):.2f} '
                      f'({time.time() - t0:.1f}s)')
                t0 = time.time()

            # Early stopping based on DSC
            
            if round(dsc, 3) == round(prev_train_dsc, 3):
                num_const_train_dsc += 1
            else:
                num_const_train_dsc = 0

            if num_const_train_dsc == max_num_const_train_dsc:
                print(f'Reached stopping criterion after {i + 1} steps. Optimization has converged.')
                break

            prev_train_dsc = dsc
        if i == num_iters - 1 and opt_rot==True:
            print(f"Reached the last epoch ({i + 1}/{num_iters}).")
            return rotation_matrix

            
            

def validate(task_type: config_io.TaskType, ds_loader: data.DataLoader, net: torch.nn.Module,
             latents: torch.nn.Parameter, metric: torch.nn.Module, device: torch.device, epoch: int,
             logger: Optional[SummaryWriter], verbose: bool, opt_rot: bool, rot_ang, df_selected, opt_vect:bool):
    metric_running = 0.0
    num_metrics = 0

    t0 = time.time()
    net.eval()
    with torch.no_grad():
        for batch in ds_loader:
            labels = batch['labels'].to(device)
            if task_type == config_io.TaskType.AD:
                latents_batch = latents[batch['caseids']].to(device)
                coords = batch['coords'].to(device)
                batch_caseids=batch['caseids']
                
                if opt_rot==True:
                    coords, rotation_matrix= apply_rot(coords, rot_ang, batch_caseids)
                batch_casenames= batch['casenames']
                if opt_vect==True:
                    latents_batch= add_clinical(latents_batch, df_selected, batch_casenames)


                #print("VAL latents_batch SHAPE", np.shape(latents_batch))
                #print("DATAFRAME df_selected", df_selected)
                # Pass latents and rotated coords through the network
                labels_pred = net(latents_batch, coords)

                # Pass latents and rotated coords through the network
                #labels_pred = net(latents_batch, coords)
            elif task_type == config_io.TaskType.RN:
                labels_lr = batch['labels_lr'].to(device)
                labels_pred = net(labels_lr)
            else:
                raise ValueError(f'Unknown task type {task_type}.')
            '''    
            if opt_rot:
                original_label_shape = labels_pred.shape
                labels_pred = remove_rotation_from_pred(labels_pred, rotation_matrix, original_label_shape)
                '''


            metric_running += metric(labels_pred, labels).item()
            num_metrics += batch['labels'].shape[0]

        metric_avg = metric_running / num_metrics

    if logger is not None:
        logger.add_scalar('metric/val', metric_avg, global_step=(epoch + 1))

    if verbose:
        t1 = time.time()
        val_duration = t1 - t0
        print(f'[val] metric {metric_avg:.3f} ({val_duration:.1f}s)')


def main():
    params, config_filepath = config_io.parse_config_train()
    model_basedir: Path = params['model_basedir']
    model_dir = model_basedir / params['model_name'] if params['model_name'] is not None else None
    task_type = params['task_type']
    learning_rate = params['learning_rate'] * params['batch_size_train']
    lat_reg_lambda = params['lat_reg_lambda']
    num_epochs_target = params['num_epochs']
    log_epoch_count = params['log_epoch_count']
    checkpoint_epoch_count = params['checkpoint_epoch_count']
    max_num_checkpoints = params['max_num_checkpoints']
    opt_rot = params['opt_rot']
    selected_params=params['selected_params']
    opt_vect=params['opt_vect']
    print("OPT ROT", opt_rot)
    
    writer: Optional[SummaryWriter]
    checkpoint_writer: Optional[io_utils.RollingCheckpointWriter]
    if model_dir is not None:
        if model_dir.exists():
            raise ValueError('Model directory already exists. Exiting to prevent accidental '
                             f'overwriting.\n{model_dir}')
        model_dir.mkdir()
        # Write the parameters to the model folder
        config_io.write_config(config_filepath, model_dir)

        # Redirect stdout to file + stdout
        sys.stdout = io_utils.Logger(model_dir / 'log.txt', 'a')
        writer = SummaryWriter(log_dir=str(model_dir))
        checkpoint_writer = io_utils.RollingCheckpointWriter(model_dir, 'checkpoint',
                                                             max_num_checkpoints, 'pth')
    else:
        print('Warning: no model name provided; not writing anything to the file system.')
        writer = None
        checkpoint_writer = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    if not torch.cuda.is_available():
        print('Warning: no GPU available; training on CPU.')

    ds_loader_train = data_generation.create_data_loader(params, data_generation.PhaseType.TRAIN,
                                                         True)
    ds_loader_val = data_generation.create_data_loader(params, data_generation.PhaseType.VAL, True)
    image_size = ds_loader_train.dataset.image_size \
        if isinstance(ds_loader_train.dataset, data_generation.ImplicitDataset) else None

    if not ds_loader_train:
        raise ValueError(f'Number of training examples is smaller than the batch size.')
    
    net = create_model(params, image_size)
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs for training.')
        net = torch.nn.DataParallel(net)  # experimental
    net = net.to(device)
    print(net)

    if task_type == config_io.TaskType.AD:
        lr_lats = params['learning_rate_lat']
        latent_dim = params['latent_dim']
        num_examples_train = len(ds_loader_train.dataset)  # type: ignore[arg-type]
        df_selected=load_df_json(selected_params)
        
        # Initialization scaling follows DeepSDF
        latents_train = torch.nn.Parameter(
            torch.normal(0.0, 1 / math.sqrt(latent_dim), [num_examples_train, latent_dim],
                         device=device))
        
        if opt_rot:
            learning_rate_rot=params['learning_rate_rot']
            rot_ang = torch.nn.Embedding(num_examples_train, 3).cuda()
            torch.nn.init.normal_(
                rot_ang.weight.data,
                0.0,
                (math.pi**2)/64,
            )
            
            optimizer = torch.optim.Adam([
            {'params': net.parameters(), 'lr': learning_rate},
            {'params': latents_train, 'lr': lr_lats},
            {'params': rot_ang.parameters(), 'lr': learning_rate_rot} 
            ])
        else:
            rot_ang = torch.nn.Embedding(num_examples_train, 3).cuda()
            torch.nn.init.constant_(
                rot_ang.weight.data,
                0.0
            )
            optimizer = torch.optim.Adam([
            {'params': net.parameters(), 'lr': learning_rate},
            {'params': latents_train, 'lr': lr_lats} 
            ])
        
    else:
        latents_train = torch.nn.Parameter(torch.empty(0))
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    criterion = create_loss().to(device)
    metric = nn_utils.DiceLoss(0.5, 'sum', True).to(device)

    # This is a tensor so that it is mutable within other functions
    global_step = torch.tensor(0, dtype=torch.int64)
    num_epochs_trained = 0

    for epoch in range(num_epochs_trained, num_epochs_target):
        torch.autograd.set_detect_anomaly(True)
        train_one_epoch(task_type, ds_loader_train, net, latents_train, lat_reg_lambda, optimizer,
                        criterion, metric, device, epoch, num_epochs_target, global_step,
                        log_epoch_count, writer, True, opt_rot, rot_ang, df_selected, opt_vect)
        if epoch % log_epoch_count == 0:
            validate(task_type, ds_loader_val, net, latents_train, metric, device, epoch, writer,
                     True, opt_rot, rot_ang, df_selected, opt_vect)

        if checkpoint_writer is not None and epoch % checkpoint_epoch_count == 0:
            checkpoint_writer.write_rolling_checkpoint(
                {'net': net.state_dict(), 'latents_train': latents_train},
                optimizer.state_dict(), int(global_step.item()), epoch + 1)

    if checkpoint_writer is not None:
        checkpoint_writer.write_rolling_checkpoint(
            {'net': net.state_dict(), 'latents_train': latents_train},
            optimizer.state_dict(), int(global_step.item()), num_epochs_target)


if __name__ == '__main__':
    main()