import trimesh
import numpy as np
import torch


def create_batch_cuboid_occupancy_grid_torch(
    cuboid_dims, poses, workspace_bounds, voxel_size, matrix_from_quat, device="cpu"
):
    """
    Creates a 3D occupancy grid from cuboid dimensions and a batch of poses, using PyTorch.

    Args:
        cuboid_dims: Torch tensor (3,) representing (length, width, height) of the cuboid.
        poses: Torch tensor (batch_size, 7) representing [x, y, z, w, qx, qy, qz] poses.
        workspace_bounds: Tuple of (min_x, max_x, min_y, max_y, min_z, max_z).
        voxel_size: Float, size of each voxel in meters.
        device: Torch device to use (e.g., "cpu", "cuda").

    Returns:
        A 3D Torch tensor representing the occupancy grid.
    """

    length, width, height = cuboid_dims.to(device)
    min_x, max_x, min_y, max_y, min_z, max_z = workspace_bounds
    grid_dims = (
        int((max_x - min_x) / voxel_size),
        int((max_y - min_y) / voxel_size),
        int((max_z - min_z) / voxel_size),
    )

    bs = poses.shape[0]
    occupancy_grid = torch.zeros((bs, *grid_dims), dtype=torch.uint8, device=device)
    batch_size = poses.shape[0]

    # Calculate cuboid corner points in its local frame.
    corner_points = torch.tensor(
        [
            [-length / 2, -width / 2, -height / 2],
            [length / 2, -width / 2, -height / 2],
            [length / 2, width / 2, -height / 2],
            [-length / 2, width / 2, -height / 2],
            [-length / 2, -width / 2, height / 2],
            [length / 2, -width / 2, height / 2],
            [length / 2, width / 2, height / 2],
            [-length / 2, width / 2, height / 2],
        ],
        device=device,
    )

    # Determine bounding box of the cuboid.
    min_bounds = torch.min(corner_points, dim=0).values
    max_bounds = torch.max(corner_points, dim=0).values

    # Calculate voxel indices within the cuboid's bounding box.
    x_range = torch.arange(
        min_bounds[0], max_bounds[0] + voxel_size, voxel_size, device=device
    )
    y_range = torch.arange(
        min_bounds[1], max_bounds[1] + voxel_size, voxel_size, device=device
    )
    z_range = torch.arange(
        min_bounds[2], max_bounds[2] + voxel_size, voxel_size, device=device
    )

    voxel_indices = torch.stack(
        torch.meshgrid(x_range, y_range, z_range, indexing="ij"), dim=-1
    ).view(-1, 3)

    # Convert voxel indices to homogeneous coordinates.
    voxel_centers = torch.cat(
        [voxel_indices, torch.ones((voxel_indices.shape[0], 1), device=device)], dim=-1
    )  # (N_voxels, 4)

    # Convert poses to transformation matrices.
    translations = poses[:, :3]
    quaternions = poses[:, 3:]

    rotation_matrices = matrix_from_quat(quaternions)

    transformation_matrices = (
        torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    )
    transformation_matrices[:, :3, :3] = rotation_matrices
    transformation_matrices[:, :3, 3] = translations

    # Apply batched transformations.
    transformed_voxel_centers = torch.einsum(
        "bij,xj->bxi", transformation_matrices, voxel_centers
    )  # (batch_size, N_voxels, 4)

    # Convert to grid coordinates.
    grid_x = ((transformed_voxel_centers[:, :, 0] - min_x) / voxel_size).long()
    grid_y = ((transformed_voxel_centers[:, :, 1] - min_y) / voxel_size).long()
    grid_z = ((transformed_voxel_centers[:, :, 2] - min_z) / voxel_size).long()

    # Filter out of bounds voxels.
    valid_indices = (grid_x >= 0) \
        & (grid_x < grid_dims[0]) \
        & (grid_y >= 0) \
        & (grid_y < grid_dims[1]) \
        & (grid_z >= 0) \
        & (grid_z < grid_dims[2])

    all_points = torch.stack([grid_x, grid_y, grid_z], dim=-1)
    # all_points shape: (batch_size, N_voxels_of_cuboid, 3)


    # Use the mask to filter valid points
    valid_points = all_points[valid_indices]  # Shape: (num_valid_points, 3)
    batch_indices = torch.arange(bs).repeat_interleave(valid_indices.sum(dim=1))  # Batch indices for valid points

    # Set the valid points in the voxel grid to 1.0
    occupancy_grid[batch_indices, valid_points[:, 0], valid_points[:, 1], valid_points[:, 2]] = 1.0
    
    return occupancy_grid

def generate_pts_on_cuboid(cuboid_dims, n_pts):
    """
    Generates points on the surface of a cuboid.

    Args:
        cuboid_dims: Torch tensor (n_cuboids, 3) representing (length, width, height) of the cuboids.
        n_pts: Number of points to generate per cuboid.

    Returns:
        Torch tensor (n_cuboids, n_pts, 3) representing points on the cuboid surfaces.
    """
    
    n_cuboids = cuboid_dims.shape[0]

    # Compute face areas
    face_areas = torch.stack([
        cuboid_dims[:, 1] * cuboid_dims[:, 2],  # x+ and x- faces
        cuboid_dims[:, 1] * cuboid_dims[:, 2],
        cuboid_dims[:, 0] * cuboid_dims[:, 2],  # y+ and y- faces
        cuboid_dims[:, 0] * cuboid_dims[:, 2],
        cuboid_dims[:, 0] * cuboid_dims[:, 1],  # z+ and z- faces
        cuboid_dims[:, 0] * cuboid_dims[:, 1],
    ], dim=1)  # Shape: (n_cuboids, 6)

    # Normalize face areas to get sampling probabilities
    face_probs = face_areas / face_areas.sum(dim=1, keepdim=True)
    face_probs = torch.clamp(face_probs, min=1e-8)  # Avoid zero probabilities

    # Assign points to faces based on probabilities
    face_indices = torch.multinomial(face_probs, n_pts, replacement=True)  # Shape: (n_cuboids, n_pts)

    final_points = torch.zeros((n_cuboids, n_pts, 3), device=cuboid_dims.device)

    for i in range(6):
        face_mask = (face_indices == i).unsqueeze(2).expand(-1, -1, 3)  # Shape: (n_cuboids, n_pts, 3)
        
        # Generate random points on the selected face
        if i == 0 or i == 1:  # x+ and x- faces
            u = torch.rand((n_cuboids, n_pts), device=cuboid_dims.device) * cuboid_dims[:, 1].unsqueeze(1) - cuboid_dims[:, 1].unsqueeze(1) / 2
            v = torch.rand((n_cuboids, n_pts), device=cuboid_dims.device) * cuboid_dims[:, 2].unsqueeze(1) - cuboid_dims[:, 2].unsqueeze(1) / 2
            
            final_points[face_mask] = torch.stack([
                torch.ones_like(u) * (cuboid_dims[:, 0].unsqueeze(1) / 2 * (-1 if i == 0 else 1)), # Corrected
                u,
                v,
            ], dim=2)[face_mask]

        elif i == 2 or i == 3:  # y+ and y- faces
            u = torch.rand((n_cuboids, n_pts), device=cuboid_dims.device) * cuboid_dims[:, 0].unsqueeze(1) - cuboid_dims[:, 0].unsqueeze(1) / 2
            v = torch.rand((n_cuboids, n_pts), device=cuboid_dims.device) * cuboid_dims[:, 2].unsqueeze(1) - cuboid_dims[:, 2].unsqueeze(1) / 2
            
            final_points[face_mask] = torch.stack([
                u,
                torch.ones_like(u) * (cuboid_dims[:, 1].unsqueeze(1) / 2 * (-1 if i == 2 else 1)), # Corrected
                v,
            ], dim=2)[face_mask]

        else:  # z+ and z- faces
            u = torch.rand((n_cuboids, n_pts), device=cuboid_dims.device) * cuboid_dims[:, 0].unsqueeze(1) - cuboid_dims[:, 0].unsqueeze(1) / 2
            v = torch.rand((n_cuboids, n_pts), device=cuboid_dims.device) * cuboid_dims[:, 1].unsqueeze(1) - cuboid_dims[:, 1].unsqueeze(1) / 2
            
            final_points[face_mask] = torch.stack([
                u,
                v,
                torch.ones_like(u) * (cuboid_dims[:, 2].unsqueeze(1) / 2 * (-1 if i == 4 else 1)), # Corrected
            ], dim=2)[face_mask]

    return final_points

def generate_batched_cuboid_point_clouds(cuboid_dims, poses, matrix_from_quat, num_points=4000, device="cpu", noise_std=0.01,
                                         workspace_bounds=None):
    """
    Generates batched point clouds from cuboid dimensions and batched poses.

    Args:
        cuboid_dims: Torch tensor (3,) representing (length, width, height) of the cuboid.
        poses: Torch tensor (batch_size, 7) representing [x, y, z, w, qx, qy, qz] poses.
        num_points: Number of points to sample from the cuboid.
        device: Torch device to use (e.g., "cpu", "cuda").
        workspace_bounds: Optional tuple of (min_x, max_x, min_y, max_y, min_z, max_z).

    Returns:
        Torch tensor (batch_size, num_points, 3) representing batched point clouds.
    """

    # length, width, height = cuboid_dims.to(device)
    if cuboid_dims.dim() == 1:  
        length, width, height = cuboid_dims
        shape_tensor = torch.tensor([length, width, height], device=device).unsqueeze(0).repeat(poses.shape[0], 1)
    else:
        assert cuboid_dims.shape[0] == poses.shape[0]
        shape_tensor = cuboid_dims
        
    # volume sampling
    # points = torch.rand((poses.shape[0], num_points, 3), device=device) * shape_tensor.unsqueeze(1) - shape_tensor.unsqueeze(1) / 2

    # face sampling
    points = generate_pts_on_cuboid(shape_tensor, num_points)
    points += torch.randn_like(points) * noise_std

    batch_size = poses.shape[0]
    # Generate random points within the cuboid's bounds.
    # points = torch.rand((num_points, 3), device=device) * torch.tensor([length, width, height], device=device) - torch.tensor([length, width, height], device=device) / 2

    translations = poses[:, :3]
    quaternions = poses[:, 3:]

    rotation_matrices = matrix_from_quat(quaternions)

    transformation_matrices = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    transformation_matrices[:, :3, :3] = rotation_matrices
    transformation_matrices[:, :3, 3] = translations

    homogeneous_points = torch.cat([points, torch.ones((batch_size, num_points, 1), device=device)], dim=-1)

    transformed_points = torch.einsum('bij,bkj->bik', transformation_matrices, homogeneous_points)[:, :3, :]
    transformed_points = transformed_points.permute(0, 2, 1) # (batch_size, num_points, 3)  


    # If workspace bounds are provided, filter points within the bounds
    # if workspace_bounds:
    #     x_min, x_max, y_min, y_max, z_min, z_max = workspace_bounds
    #     transformed_points


    return transformed_points