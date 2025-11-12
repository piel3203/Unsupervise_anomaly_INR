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

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))#omega0*(Wx+b)
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class OccupancyPredictor(nn.Module):
    def __init__(self, latent_dim: int, spatial_dim: int, num_layers: int,
                 layers_with_coords: List[int], use_sine: bool = False, first_omega_0: float = 30):
        
        def block(num_ch_in: int, num_ch_out: int, is_first: bool = True):
            if use_sine:
                if is_first==True:
                    return SineLayer(num_ch_in, num_ch_out, is_first=True, omega_0=first_omega_0 if is_first else 30)
                else:
                    return SineLayer(num_ch_in, num_ch_out, is_first=False, omega_0=first_omega_0 if is_first else 30)
            else:
                return nn.Sequential(
                    nn.Linear(num_ch_in, num_ch_out),
                    nn.ReLU(True)
                )

        super().__init__()

        self.layers_with_coords = layers_with_coords
        in_channels = [latent_dim] * num_layers
        channels_with_coords = latent_dim + spatial_dim
        for lyr_id in self.layers_with_coords:
            in_channels[lyr_id] = channels_with_coords
        
        self.res_layers = nn.ModuleList(
            [block(in_channels[i], latent_dim, is_first=(i == 0)) for i in range(num_layers - 1)]
        )
        
        self.last_layer = nn.Linear(in_channels[-1], 1)

    def forward(self, closest_latents: torch.Tensor, local_coords: torch.Tensor) -> torch.Tensor:
        features = closest_latents

        for i, layer in enumerate(self.res_layers):
            append_coords = i in self.layers_with_coords
            if append_coords:
                features = torch.cat([features, local_coords], dim=-1)

            out = layer(features)
            features = out if append_coords else features + out

        features = self.last_layer(features)
        return features


class AutoDecoder(torch.nn.Module):
    """Encoder-free implicit shape prediction."""
    def __init__(self, lat_dim: int, spatial_dim: int, image_size: torch.Tensor,
                 occnet_num_layers: int, occnet_layers_with_coords: List[int]):
        """
        Implicit AutoDecoder.

        :param lat_dim: Dimensionality of the latent vector.
        :param spatial_dim: Number of spatial dimentions.
        :param image_size: Physical size of the image volume.
        :param occnet_num_layers: Number of linear layers in the implicit occupancy MLP.
        :param occnet_layers_with_coords: Layers with concatenated coordinates in occupancy MLP.
        """
        super().__init__()
        self.image_size: torch.Tensor
        self.register_buffer('image_size', image_size)
        # Global coordinates of the latent vector
        latent_coords: torch.Tensor = image_size / 2  # noqa
        self.latent_coords: torch.Tensor
        self.register_buffer('latent_coords', latent_coords)
        self.occp_pred: torch.nn.Module
        self.occp_pred = OccupancyPredictor(lat_dim, spatial_dim, occnet_num_layers,
                                            occnet_layers_with_coords,use_sine=True)

    def forward(self, latents: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """Latents have shape [B, Z]. Coordinates are in the global coord. system with shape
        [B, *ST, 3].
        """
        # Compute local coordinates w.r.t. the latent vector position.
        local_coords = coordinates - self.latent_coords
        latents = latents.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        latents = latents.expand(-1, coordinates.shape[1], coordinates.shape[2],
                                 coordinates.shape[3], -1)
        predictions = self.occp_pred(latents, local_coords)
        # Remove the feature dim (MLP, channels-last)
        predictions = predictions.squeeze(4)
        return predictions  # [B, *ST]


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
