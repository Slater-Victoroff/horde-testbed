import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from scipy.spatial import Delaunay

from asset_rep import MeshData
from data_conversion import load_fbx_to_meshdata, export_meshdata_list


class VertNet(torch.nn.Module):
    def __init__(self, latent_dim:int = 8, hidden_dim:int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
        )
        self.pos_head = torch.nn.Linear(hidden_dim, 3)
        self.color_head = torch.nn.Linear(hidden_dim, 3)
        self.normal_head = torch.nn.Linear(hidden_dim, 3)
        
        self.edge_predictor = torch.nn.Sequential(
            torch.nn.Linear(latent_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid()
        )

def sample_surface_points(mesh: MeshData, num_samples: int = 500) -> torch.Tensor:
    idxs = torch.linspace(0, mesh.polyvert_attrs.shape[0]-1, num_samples).long()
    xyz = mesh.positions[mesh.polyvert_attrs[idxs, 0]]
    if mesh.uvs is None:
        raise ValueError("Mesh does not have UV coordinates to sample from.")
    uv = mesh.uvs[mesh.polyvert_attrs[idxs, 1]]
    new_mesh = MeshData(
        name=mesh.name + "_sampled",
        positions = xyz,
        uvs = uv,
        mapped_texture=mesh.mapped_texture
    )
    return new_mesh


def bake_colors_from_texture(mesh: MeshData) -> MeshData:
    # Sample the texture image at the given UVs, producing rgb values
    # Assumes texture is (C, H, W), uv is (N, 2) in [0, 1]
    N = mesh.polyvert_attrs.shape[0]
    H, W = mesh.mapped_texture.image.shape[1:]
    uvs = mesh.uvs[mesh.polyvert_attrs[:, 1]]
    x = (uvs[:, 0] * (W - 1)).long()
    y = (uvs[:, 1] * (H - 1)).long()
    rgb = mesh.mapped_texture.image[x, y, :]
    return MeshData(
        name=mesh.name + "_baked_colors",
        positions = mesh.positions,
        colors = rgb,
    )  # No textures or uvs in the output, just colors baked from texture


def run_delaunay_baseline(mesh: MeshData):
    positions = mesh.positions[mesh.polyvert_attrs[:, 0]]
    points = positions.detach().cpu().numpy()
    tetra = Delaunay(points)
    unique_tris = []
    for simplex in tetra.simplices:
        tetra_tris = [
            sorted(simplex[[0, 1, 2]]),
            sorted(simplex[[0, 1, 3]]), 
            sorted(simplex[[0, 2, 3]]),
            sorted(simplex[[1, 2, 3]])
        ]
        unique_tris.extend(tetra_tris)
    unique_tris = np.unique(np.array(unique_tris), axis=0)
    print(f"Found {len(unique_tris)} unique triangles from Delaunay triangulation.")
    print(f"unique triangles: {unique_tris[:5]}...")  # Print first 5 for brevity
    mesh.triangles = torch.tensor(unique_tris, dtype=torch.long, device=mesh.positions.device)


if __name__ == "__main__":
    # Load the mesh from an FBX file
    meshes = load_fbx_to_meshdata("static/meshes/Horse.fbx")
    
    updated_meshes = []
    for mesh in meshes:
        # Sample points on the surface of the mesh
        sampled_mesh = sample_surface_points(mesh, mesh.positions.shape[0])
        colored_mesh = bake_colors_from_texture(sampled_mesh)
        run_delaunay_baseline(colored_mesh)
        
        updated_meshes.append(colored_mesh)
    
    # Export the modified mesh to OBJ format
    export_meshdata_list(updated_meshes, "static/meshes/Horse_sampled.fbx")
