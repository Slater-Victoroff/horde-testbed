from typing import Optional

import numpy as np

SCALAR_KERNELS = {}
VECTOR_KERNELS = {}


class EnemyKernelDefinition:
    def __init__(
        self,
        targeting: str,
        attacks: list,
        radius: float = 1.0,
        scale: float = 1.0,
        baseSpeed: float = 0.05,
        maxSpeed: float = 0.2,
        acceleration: float = 0.01,
        rotationSpeed: float = 0.1,
        material: Optional[str] = None,
        params: Optional[dict] = None,
    ):
        self.targeting = targeting  # Targeting behavior (e.g., "simple_position")
        self.attacks = attacks  # List of attack behaviors (e.g., ["intersect"])
        self.radius = radius
        self.scale = scale
        self.baseSpeed = baseSpeed
        self.maxSpeed = maxSpeed
        self.acceleration = acceleration
        self.rotationSpeed = rotationSpeed
        self.material = material
        self.params = params or {}

        # Resolve targeting and attack kernel functions
        self.vector_fn = VECTOR_KERNELS.get(self.targeting)
        if not self.vector_fn:
            raise ValueError(f"Vector kernel '{self.targeting}' not found.")

        self.scalar_fns = [SCALAR_KERNELS.get(attack) for attack in self.attacks]
        if not all(self.scalar_fns):
            missing = [attack for attack, fn in zip(self.attacks, self.scalar_fns) if fn is None]
            raise ValueError(f"Scalar kernels not found for: {missing}")

    def compute(self, player_state, positions, velocities, accelerations, grid):
        """
        Compute the scalar and vector fields for multiple enemies.
        """
        # Compute vector field (targeting behavior)
        vector_field = self.vector_fn(self, player_state, positions, velocities, accelerations, grid)

        # Compute scalar field (attack behaviors)
        scalar_field = np.zeros(grid.shape[:-1], dtype=np.float32)
        for scalar_fn in self.scalar_fns:
            scalar_field += scalar_fn(self, player_state, positions, velocities, accelerations, grid)

        return scalar_field, vector_field

    def to_dict(self):
        """Convert the enemy kernel definition to a dictionary for JSON serialization."""
        return {
            "targeting": self.targeting,
            "attacks": self.attacks,
            "radius": self.radius,
            "scale": self.scale,
            "baseSpeed": self.baseSpeed,
            "maxSpeed": self.maxSpeed,
            "acceleration": self.acceleration,
            "rotationSpeed": self.rotationSpeed,
            "material": self.material,
            "params": self.params,
        }




# Interface for scalar and vector kernel functions
def register_scalar_kernel(name):
    """Decorator to register a scalar kernel function."""
    def decorator(func):
        SCALAR_KERNELS[name] = func
        return func
    return decorator


def register_vector_kernel(name):
    """Decorator to register a vector kernel function."""
    def decorator(func):
        VECTOR_KERNELS[name] = func
        return func
    return decorator


@register_scalar_kernel("intersect")
def circular_scalar_kernel(kernel, player_state, positions, velocities, accelerations, grid):
    """
    Compute the scalar field (e.g., damage) for multiple enemies.
    kernel: The EnemyKernelDefinition instance
    positions: Array of enemy positions (N, 2)
    grid: Grid of points in the world
    """
    scalar_field = np.zeros(grid.shape[:-1], dtype=np.float32)

    for pos in positions:
        # Extract parameters from the kernel
        scale = kernel.scale
        radius = kernel.radius

        # Expand the bounding box slightly to include grid points just outside the circle
        x_min, x_max = pos[0] - radius - 1, pos[0] + radius + 1
        z_min, z_max = pos[1] - radius - 1, pos[1] + radius + 1

        # Find the indices of grid points within the expanded bounding box
        mask = (grid[..., 0] >= x_min) & (grid[..., 0] <= x_max) & \
               (grid[..., 1] >= z_min) & (grid[..., 1] <= z_max)
        relevant_grid = grid[mask]

        # Compute distances only for relevant grid points
        dx = relevant_grid[..., 0] - pos[0]
        dz = relevant_grid[..., 1] - pos[1]
        dist_sq = dx**2 + dz**2

        # Compute the proportion of the circle intersecting each relevant grid cell
        dist = np.sqrt(dist_sq)
        overlap = np.clip(radius - dist, 0, radius) / radius  # Proportional overlap (linear falloff)

        # Damage is scaled by the overlap proportion
        damage = scale * overlap

        # Update the scalar field
        scalar_field[mask] += damage

    return scalar_field

@register_vector_kernel("simple_position")
def vector_pursuit_kernel(kernel, player_state, positions, velocities, accelerations, grid):
    """
    Compute the vector field (e.g., movement direction) for multiple enemies.
    kernel: The EnemyKernelDefinition instance
    positions: Array of enemy positions (N, 2)
    velocities: Array of enemy velocities (N, 2)
    grid: Grid of points in the world
    """
    vector_field = np.zeros(grid.shape, dtype=np.float32)
    print(f"Grid shape: {grid.shape}")

    for pos, vel in zip(positions, velocities):
        d = grid - pos  # Relative positions
        dist_sq = np.sum(d**2, axis=-1)

        # Only compute vectors within the circle
        within_circle = dist_sq <= kernel.radius**2

        # Direction is constant (normalized velocity)
        v_unit = vel / (np.linalg.norm(vel) + 1e-6)

        # Scale by speed within the circle
        weight = np.zeros_like(dist_sq)
        weight[within_circle] = kernel.scale * kernel.baseSpeed
        vector_field[within_circle] += weight[within_circle][..., None] * v_unit

    print(f"Vector Field shape: {vector_field.shape}")
    return vector_field
