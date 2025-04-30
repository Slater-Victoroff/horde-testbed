import numpy as np
from enemies import ENEMY_REGISTRY
import time

def generate_fields(player_state, enemies_state, grid_size, ground_size):
    start_time = time.time()
    H, W = grid_size
    scalar_field = np.zeros((H, W), dtype=np.float32)
    vector_field = np.zeros((H, W, 2), dtype=np.float32)

    extent = ground_size / 2.0

    # Build world grid
    x = np.linspace(-extent, extent, W)
    z = np.linspace(-extent, extent, H)
    gz, gx = np.meshgrid(z, x, indexing="ij")
    grid = np.stack([gx, gz], axis=-1)  # (W, H, 2)

    for kind, kernel in ENEMY_REGISTRY.items():
        positions = enemies_state.position_array(kind)
        velocities = enemies_state.velocity_array(kind)
        accelerations = enemies_state.acceleration_array(kind)

        scalar_kernels, vector_kernels = kernel.compute(player_state, positions, velocities, accelerations, grid)
        scalar_field += scalar_kernels
        vector_field += vector_kernels

    end_time = time.time()
    print(f"generate_fields runtime: {end_time - start_time:.4f} seconds")
    return scalar_field, vector_field

def generate_fields_from_enemies(enemies, kinds, grid_size, ground_size):
    start_time = time.time()
    H, W = grid_size
    scalar_field = np.zeros((H, W), dtype=np.float32)
    vector_field = np.zeros((H, W, 2), dtype=np.float32)

    extent = ground_size / 2.0

    # Build world grid
    x = np.linspace(-extent, extent, W)
    z = np.linspace(-extent, extent, H)
    gz, gx = np.meshgrid(z, x, indexing="ij")
    grid = np.stack([gx, gz], axis=-1)  # (W, H, 2)

    enemies = np.array(enemies)
    positions = enemies[:, :2]
    velocities = enemies[:, 2:]

    kinds = np.array(kinds)
    unique_kinds = np.unique(kinds)

    for kind in unique_kinds:
        if kind not in ENEMY_REGISTRY:
            print(f"Warning: Enemy kind '{kind}' not found in ENEMY_REGISTRY.")
            continue

        kernel = ENEMY_REGISTRY[kind]

        mask = kinds == kind
        pos_subset = positions[mask]
        vel_subset = velocities[mask]

        for pos, vel in zip(pos_subset, vel_subset):
            # Compute the scalar and vector fields for this enemy directly
            scalar_kernel, vector_kernel = kernel.compute(pos, grid, vel)

            # Add the computed fields to the global fields
            scalar_field += scalar_kernel
            vector_field += vector_kernel

    end_time = time.time()
    print(f"generate_fields_from_enemies runtime: {end_time - start_time:.4f} seconds")
    return scalar_field, vector_field
