from typing import List

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt

import time

OMEGA = 10

class SirenActivation(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return torch.sin(OMEGA * input)

    @staticmethod
    def sine_init(m):  
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / OMEGA, np.sqrt(6 / num_input) / OMEGA)
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.zero_()  # Fix: Set bias to 0
    
    @staticmethod
    def first_layer_sine_init(m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-1 / num_input, 1 / num_input)
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.zero_()  # Fix: Set bias to 0

# BUILDING THE NETWORK
class OccupancyPredictor(nn.Module):
    def __init__(self, latent_dim: int, spatial_dim: int, num_layers: int,
                 layers_with_coords: List[int], use_sine: bool = True):
        
        #Defines a fully connected layer followed by an activation fct
        #Uses SIREN init if use_sine=True otherwise it applies ReLU
        def block(num_ch_in: int, num_ch_out: int, first_layer: bool = False):
            linear = nn.Linear(num_ch_in, num_ch_out)
            if use_sine:
                if first_layer:
                    SirenActivation.first_layer_sine_init(linear)
                else:
                    SirenActivation.sine_init(linear)
                activation = SirenActivation()
            else:
                activation = nn.ReLU(True)
            return nn.Sequential(linear, activation)
        
        super().__init__()
        #Store which layers will receive extra spatial coords
        self.layers_with_coords = layers_with_coords
        self.use_sine = use_sine

        #Define inputs dims for each layer
        #Some layers also take spatial coordinates, so their input size is latent_dim + spatial_dim
        in_channels = [latent_dim] * num_layers
        channels_with_coords = latent_dim + spatial_dim
        for lyr_id in self.layers_with_coords:
            in_channels[lyr_id] = channels_with_coords

        #Creates residual layers using the block function.
        self.res_layers = nn.ModuleList(
            [block(in_channels[i], latent_dim, first_layer=(i == 0)) for i in range(num_layers - 1)]
        )
        #The final layer produces a single scalar output (occupancy value)
        #If using SIREN, it also applies sine_init
        self.last_layer = nn.Linear(in_channels[-1], 1)
        if use_sine:
            SirenActivation.sine_init(self.last_layer)

    def forward(self, closest_latents: torch.Tensor, local_coords: torch.Tensor) -> torch.Tensor: #Takes latent features and spatial coordinates
        #iterate through layers 
        #Appends spatial coords when needed 
        #Uses redidual connections (if no coords are appenned)
        features = closest_latents
        for i, layer in enumerate(self.res_layers):
            append_coords = i in self.layers_with_coords
            if append_coords:
                features = torch.cat([features, local_coords], dim=-1)
            out = layer(features)
            features = out if append_coords else features + out
        # Passes through final linear layer and returns occupancy prediction
        features = self.last_layer(features)
        return features

class AutoDecoder(nn.Module):
    def __init__(self, lat_dim: int, spatial_dim: int, image_size: torch.Tensor,
                 occnet_num_layers: int, occnet_layers_with_coords: List[int], use_sine: bool = True):
        super().__init__()
        #Stores image size and center coordinates (for normalization)
        self.register_buffer('image_size', image_size)
        latent_coords = image_size / 2
        self.register_buffer('latent_coords', latent_coords)
        #Creates an occupancy predictor using the given parameters
        self.occp_pred = OccupancyPredictor(lat_dim, spatial_dim, occnet_num_layers,
                                            occnet_layers_with_coords, use_sine=use_sine)
        
    #Takes latent features and spatial coordinates
    def forward(self, latents: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        local_coords = coordinates - self.latent_coords #converts coordinates to a local reference frame
        #Expands latent vectors to match the spatial dimensions of input coordinates
        latents = latents.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        latents = latents.expand(-1, coordinates.shape[1], coordinates.shape[2],
                                 coordinates.shape[3], -1)
        #Passes through OccupancyPredictor and removes extra dimensions
        predictions = self.occp_pred(latents, local_coords)
        predictions = predictions.squeeze(4)
        return predictions



class ReconNet(torch.nn.Module):
    """Implementation taken from https://github.com/FedeTure/ReconNet"""
    def __init__(self):
        def consecutive_conv(in_channels: int, out_channels: int,
                             do_half_intermediate: bool = False):
            interm_channels = int(out_channels / 2) if do_half_intermediate else in_channels
            return torch.nn.Sequential(torch.nn.Conv3d(in_channels, interm_channels, 3, padding=1),
                                       torch.nn.BatchNorm3d(interm_channels),
                                       torch.nn.ReLU(inplace=True),
                                       torch.nn.Conv3d(interm_channels, out_channels, 3, padding=1),
                                       torch.nn.BatchNorm3d(out_channels),
                                       torch.nn.ReLU(inplace=True))

        def consecutive_conv_up(in_channels: int, out_channels: int):
            return torch.nn.Sequential(torch.nn.Conv3d(in_channels, out_channels, 3, padding=1),
                                       torch.nn.BatchNorm3d(out_channels),
                                       torch.nn.ReLU(inplace=True),
                                       torch.nn.Conv3d(out_channels, out_channels, 3, padding=1),
                                       torch.nn.BatchNorm3d(out_channels),
                                       torch.nn.ReLU(inplace=True))

        super().__init__()

        num_channels = 32
        self.conv_initial = consecutive_conv(1, num_channels, True)

        self.conv_rest_x_64 = consecutive_conv(num_channels, num_channels * 2)
        self.conv_rest_x_32 = consecutive_conv(num_channels * 2, num_channels * 4)
        self.conv_rest_x_16 = consecutive_conv(num_channels * 4, num_channels * 8)

        self.conv_rest_u_32 = consecutive_conv_up(num_channels * 8 + num_channels * 4,
                                                  num_channels * 4)
        self.conv_rest_u_64 = consecutive_conv_up(num_channels * 4 + num_channels * 2,
                                                  num_channels * 2)
        self.conv_rest_u_128 = consecutive_conv_up(num_channels * 2 + num_channels,
                                                   num_channels)

        # noinspection PyTypeChecker
        self.conv_final = torch.nn.Conv3d(num_channels, 1, 3, padding=1)

        self.contract = torch.nn.MaxPool3d(2, stride=2)
        self.expand = torch.nn.Upsample(scale_factor=2)

    def forward(self, x):
        x_128 = self.conv_initial(x)  # conv_initial 1->16->32
        x_64 = self.contract(x_128)
        x_64 = self.conv_rest_x_64(x_64)  # rest 32->32->64
        x_32 = self.contract(x_64)
        x_32 = self.conv_rest_x_32(x_32)  # rest 64->64->128
        x_16 = self.contract(x_32)
        x_16 = self.conv_rest_x_16(x_16)  # rest 128->128->256

        u_32 = self.expand(x_16)
        u_32 = self.conv_rest_u_32(torch.cat((x_32, u_32), 1))  # rest 256+128-> 128 -> 128
        u_64 = self.expand(u_32)
        u_64 = self.conv_rest_u_64(torch.cat((x_64, u_64), 1))  # rest 128+64-> 64 -> 64
        u_128 = self.expand(u_64)
        u_128 = self.conv_rest_u_128(torch.cat((x_128, u_128), 1))  # rest 64+32-> 32 -> 32
        u_128 = self.conv_final(u_128)

        return u_128
