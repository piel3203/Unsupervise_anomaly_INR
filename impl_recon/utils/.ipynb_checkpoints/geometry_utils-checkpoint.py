from typing import Optional, Union

import numpy as np
import torch

'''
def is_matrix_scaling_and_transform(m: np.ndarray) -> bool:
    """Check if the given matrix is 4 x 4, has only scaling and translation components, and scaling
    is positive."""
    is_affine = m.shape == (4, 4)
    # Check if the 3 x 3 component is a diagonal matrix
    linear_comp = m[:3, :3]
    is_lin_comp_diag = np.all(linear_comp == np.diag(np.diagonal(linear_comp)))
    # Check if last row has no projection
    is_last_row_valid = np.all(m[3] == np.array([0, 0, 0, 1], m.dtype))
    # Check if scaling is positive
    is_scaling_positive = np.all(np.diag(m) > 0)

    is_matrix_ok = is_affine and bool(is_lin_comp_diag) and bool(is_last_row_valid) and \
        bool(is_scaling_positive)
    return is_matrix_ok
'''
def is_matrix_scaling_and_transform(m: np.ndarray) -> bool:
    """Check if the given matrix is 4 x 4, has only scaling and translation components, and scaling
    is positive."""
    tol = 1e-2
    is_affine = m.shape == (4, 4)
    
    # Extract the 3x3 linear component and check if it's diagonal within a tolerance
    linear_comp = m[:3, :3]
    is_lin_comp_diag = np.all(np.abs(linear_comp - np.diag(np.diagonal(linear_comp))) < tol)
    
    # Check if last row is [0, 0, 0, 1]
    is_last_row_valid = np.all(m[3] == np.array([0, 0, 0, 1], dtype=m.dtype))
    
    # Check if the scaling factors (diagonal elements of the 3x3 component) are positive
    scaling_factors = np.diagonal(linear_comp)
    is_scaling_positive = np.all(scaling_factors > 0)
    
    # Print diagnostic information if any condition fails
    if not is_affine:
        print(f"Matrix is not 4x4:\n{m}")
    if not is_lin_comp_diag:
        print(f"Matrix has non-diagonal linear components:\n{m}")
    if not is_last_row_valid:
        print(f"Last row is not valid:\n{m}")
    if not is_scaling_positive:
        print(f"Scaling factors are not positive:\n{scaling_factors}")
    
    # Return True only if all checks pass
    is_matrix_ok = is_affine and is_lin_comp_diag and is_last_row_valid and is_scaling_positive
    return is_matrix_ok

def generate_sampling_grid(spatial_shape: torch.Tensor, coord_axis: int = -1,
                           cmin: Union[float, torch.Tensor] = -1.0,
                           cmax: Union[float, torch.Tensor] = 1.0,
                           device: Optional[torch.device] = None, batch_size: int = 1,
                           reverse_coord_order: bool = False) -> torch.Tensor:
    """Create coordinates of a grid. Batch axis is added after the coordinate axis at the desired
    position. Coordinates are normalized between cmin and cmax. Sampling strategy corresponds to
    align_corners=False in e.g. torch.nn.funcional.interpolate(...).
    Reversing coordinates is required when the returned grid will be used as grid in the
    grid_sample(..) function.
    """
    spacing = (cmax - cmin) / spatial_shape
    start = cmin + (spacing / 2)
    # The last term is just to ensure that we get spatial_shape number of samples in the arange(..)
    end = start + spatial_shape * spacing - spacing / 2
    if device is None:
        device = torch.device('cpu')
    # Compute the coordinates on each spatial axis
    individual_coords = [torch.arange(start=curr_start, end=curr_end, step=curr_step,
                                      dtype=torch.float32, device=device)
                         for curr_start, curr_end, curr_step in zip(start, end, spacing)]
    coords = torch.meshgrid(individual_coords, indexing='ij')
    if reverse_coord_order:
        coords = tuple(reversed(coords))
    grid = torch.stack(coords, coord_axis)
    grid = grid.expand(batch_size, *grid.shape)
    return grid
