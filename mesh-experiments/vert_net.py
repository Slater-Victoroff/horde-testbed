import os
import json
from collections import defaultdict

import tracemalloc
import imageio.v3 as iio
from PIL import Image
import numpy as np
from scipy.spatial import Delaunay
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision.io import write_png
from torchvision.transforms.functional import to_pil_image

from pytorch3d.renderer import (
    TexturesVertex,
    TexturesUV,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    look_at_view_transform,
    OrthographicCameras,
)

from pytorch3d.structures import Meshes
from pytorch3d.ops import interpolate_face_attributes
from piq import SSIMLoss

import lpips
from soap import SOAP
from asset_rep import MeshData
from data_conversion import load_fbx_to_meshdata, export_meshdata_list
import cProfile
import pstats
import io



def generate_polyvert_grid(res=32, device="cuda"):
    """
    Creates a grid of triangles (2 per square cell), each with its own independent vertices.
    Returns:
        polyverts: (N_tris * 3, 2) - each triangle has 3 unique vertices in UV space
        triangles: (N_tris, 3) - indices into polyverts (just [0,1,2], [3,4,5], ...)
    """
    ys, xs = torch.meshgrid(
        torch.linspace(0, 1, res, device=device),
        torch.linspace(0, 1, res, device=device),
        indexing='ij'
    )
    verts_grid = torch.stack([xs, ys], dim=-1)  # (res, res, 2)

    polyverts = []
    for i in range(res - 1):
        for j in range(res - 1):
            # Get corners of the cell
            v00 = verts_grid[i, j]
            v01 = verts_grid[i, j + 1]
            v10 = verts_grid[i + 1, j]
            v11 = verts_grid[i + 1, j + 1]

            # Triangle 1: v00, v10, v01
            polyverts.extend([
                v00.clone(),
                v10.clone(),
                v01.clone(),
            ])
            # Triangle 2: v01, v10, v11
            polyverts.extend([
                v01.clone(),
                v10.clone(),
                v11.clone()
            ])

    polyverts = torch.stack(polyverts, dim=0)  # (N * 3, 2)
    print(f"Polyvert stats: {polyverts.shape[0]} verts, {polyverts.shape[1]} dims")
    print(f"Polyvert range: {polyverts.min().item()} to {polyverts.max().item()}")
    num_tris = polyverts.shape[0] // 3
    triangles = torch.arange(num_tris * 3, device=device).reshape(num_tris, 3)

    return polyverts, triangles


def generate_delaunay(n_points=1024, device="cuda", seed=42):
    """
    Generates a scattered 2D triangulation in UV space using Delaunay.

    Returns:
        polyverts: (N_tris * 3, 2)
        triangles: (N_tris, 3)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate jittered grid in [0,1]^2
    grid_res = int(np.sqrt(n_points))
    xs = torch.linspace(0, 1, grid_res)
    ys = torch.linspace(0, 1, grid_res)
    base = torch.stack(torch.meshgrid(xs, ys, indexing='ij'), dim=-1).reshape(-1, 2)
    jitter = (torch.rand_like(base) - 0.5) * (1.0 / grid_res)
    points = base + jitter
    points = points.clamp(0.0, 1.0)

    # Delaunay triangulation using scipy
    tri = Delaunay(points.cpu().numpy())
    faces = torch.from_numpy(tri.simplices).long().to(device)
    verts = points.to(device)

    def compute_knn_graph(k=16):
        barycenters = verts[faces].mean(dim=1)  # (N_tris, 2)
        dists = torch.cdist(barycenters, barycenters)  # (N_tris, N_tris)
        knn_idxs = dists.topk(k=k+1, largest=False).indices[:, 1:]  # (N_tris, k)
        return knn_idxs

    return verts, faces, compute_knn_graph()


def pos_encoding(x, num_frequencies=6):
    freqs = 2 ** torch.arange(num_frequencies, device=x.device) * torch.pi
    x = x.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(x), torch.cos(x)], dim=-1).flatten(-2)


class NeighborAttention(nn.Module):
    def __init__(self, dim, neighbors):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )
        self.register_buffer("neighbors", neighbors)  # (N_tris, k)

    def forward(self, latents):
        N, K = self.neighbors.shape
        neighbors = latents[self.neighbors]  # (N, K, D)
        central = latents.unsqueeze(1).expand(-1, K, -1)  # (N, K, D)
        pair = torch.cat([central, neighbors], dim=-1)  # (N, K, 2D)
        weights = F.softmax(self.score(pair).squeeze(-1), dim=-1)  # (N, K)
        return (weights.unsqueeze(-1) * neighbors).sum(dim=1)  # (N, D)


class TextureNet(torch.nn.Module):

    def __init__(self, source_tex, res=64, latent_dim=16, hidden_dim=64):
        super().__init__()
        self.shape = source_tex.shape
        device = source_tex.device

        # polyverts, triangles = generate_polyvert_grid(res=res, device=device)
        verts, triangles, neighbors = generate_delaunay(n_points=res**2, device=device)
        self.register_buffer("base_uvs", verts * 2 - 1)  # Scale to [-1, 1] range
        self.register_buffer("faces", triangles)
        self.register_buffer("neighbors", neighbors)  # (N_tris, k)
        triangle_latents = torch.randn(triangles.shape[0], latent_dim, device=device)
        triangle_latents = F.normalize(triangle_latents, dim=-1)
        self.triangle_latents = torch.nn.Parameter(triangle_latents)
        self.tri_attention = NeighborAttention(dim=latent_dim, neighbors=neighbors)

        n_freqs = 6
        pos_encoding_dim = 2 * n_freqs * 2

        self.pos_head = torch.nn.Sequential(
            torch.nn.Linear(pos_encoding_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
        )

        self.tri_head = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()  # Scale and shift for modulation
        )

        self.delta_trunk = torch.nn.Sequential(
            torch.nn.Linear(pos_encoding_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
            torch.nn.LayerNorm(hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2),  # Delta for position adjustment
            torch.nn.Tanh()  # Output in [-1, 1] range
        )

        self.color_trunk = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
            torch.nn.LayerNorm(hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 3),  # RGB output
            torch.nn.Sigmoid()  # Normalize colors to [0, 1]
        )

        # Xavier initialization for all Linear layers
        def xavier_init_sequential(seq):
            for m in seq.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)

        xavier_init_sequential(self.pos_head)
        xavier_init_sequential(self.tri_head)
        xavier_init_sequential(self.delta_trunk)
        xavier_init_sequential(self.color_trunk)

    def forward(self):
        encoded_uvs = pos_encoding(self.base_uvs)  # (V, 2 * num_frequencies * 2)
        pos_latents = self.pos_head(encoded_uvs)

        attn_latents = self.tri_attention(self.triangle_latents)
        triangle_latents = self.tri_head(attn_latents)  # (T, hidden_dim)
        # triangle_latents = 0.8 * triangle_latents + 0.2 * triangle_latents[self.neighbors].mean(dim=1)
        tri_latents_expanded = triangle_latents.repeat_interleave(3, dim=0)  # (num_tris*3, hidden_dim)

        deltas = self.delta_trunk(encoded_uvs)  # (V, 2)
        updated_uvs = self.base_uvs + deltas * 0.05  # (V, 2)
        updated_base = torch.cat([pos_latents, tri_latents_expanded, updated_uvs], dim=-1)  # (V, hidden_dim*2 + 2)
        colors = self.color_trunk(updated_base)  # (V, 3)
        return updated_uvs, colors

    def render(self, updated_uvs: torch.Tensor = None, colors: torch.Tensor = None):
        if updated_uvs is None or colors is None:
            updated_uvs, colors = self.forward()
        verts_3d = torch.cat([updated_uvs, torch.zeros_like(updated_uvs[:, :1])], dim=-1)  # (V, 3)
        print(f"self faces shape: {self.faces.shape}, verts_3d shape: {verts_3d.shape}, colors shape: {colors.shape}")
        mesh = Meshes(
            verts=[verts_3d],
            faces=[self.faces],
            textures=TexturesVertex(verts_features=[colors])
        )

        renderer = make_uvspace_renderer(self.shape[0], self.shape[1], device=updated_uvs.device)
        rendered = renderer(mesh)
        return rendered[0]
    
    def export_json(self, outfilename="tex_output.json"):
        """
        Exports the current mesh with updated UVs and colors to an OBJ file.
        """
        updated_uvs, colors = self.forward()
        triangles = self.faces.cpu().numpy()
        output = {
            "vertices": updated_uvs.cpu().numpy().tolist(),
            "faces": triangles.tolist(),
            "colors": colors.cpu().numpy().tolist()
        }
        with open(outfilename, 'w') as f:
            json.dump(output, f, indent=4)


class TextureNet2(TextureNet):

    def __init__(self, source_tex, res=64, latent_dim=128, latent_channels = 4, hidden_dim=64):
        super().__init__(source_tex, res=res, latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.shape = source_tex.shape
        device = source_tex.device

        verts, triangles, _ = generate_delaunay(n_points=res**2, device=device)
        self.register_buffer("base_uvs", verts * 2 - 1)  # Scale to [-1, 1] range
        self.register_buffer("faces", triangles)
        self.latent = nn.Parameter(torch.randn(latent_channels, latent_dim, latent_dim))

        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_dim * 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU()
        )

        self.delta_map = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 2, 1),   # (2, H, W)
            nn.Tanh()
        )
        self.delta_scale = 1e-4
        self.color_map = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 3, 1),    # (3, H, W)
            nn.Sigmoid()
        )
    
    def increment_delta_scale(self, increment):
        print("#" * 50)
        print(f"Incrementing delta scale by {increment:.6f}, current scale: {self.delta_scale:.6f}")
        self.delta_scale += increment

    def forward(self):
        grid = self.base_uvs.view(1, -1, 1, 2).flip(-1)  # (1, N, 1, 2), (u,v) → (x,y)

        feat = self.decoder(self.latent.unsqueeze(0))  # (1, C, H, W)
        delta = F.grid_sample(self.delta_map(feat), grid, mode='bilinear', align_corners=True)
        delta = delta.squeeze(-1).squeeze(0).transpose(0, 1)
        print(f"Delta shape: {delta.shape}, UVs shape: {self.base_uvs.shape}")
        new_uvs = self.base_uvs + delta * self.delta_scale  # (N, 2)

        new_uv_grid = new_uvs.view(1, -1, 1, 2).flip(-1)  # (1, N, 1, 2)
        color = F.grid_sample(self.color_map(feat), new_uv_grid, mode='bilinear', align_corners=True)

        color = color.squeeze(-1).squeeze(0).transpose(0, 1)
        print(f"Color shape: {color.shape}, UVs shape: {new_uvs.shape}")
        return new_uvs, color


class DeltaScaler:
    def __init__(self, init_scale, max_scale, grow_speed):
        self.scale = init_scale
        self.max_scale = max_scale
        self.grow_speed = grow_speed

    def update(self, loss_improvement):
        if loss_improvement < 1e-4:
            self.scale = min(self.scale + self.grow_speed, self.max_scale)

    def __call__(self):
        return self.scale


class UnlitShader(torch.nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device

    def forward(self, fragments, meshes, **kwargs):
        # Get vertex colors from mesh textures
        faces = meshes.faces_packed()                   # (F, 3)
        verts_colors = meshes.textures.verts_features_packed()  # (V, 3)
        face_colors = verts_colors[faces]               # (F, 3, 3)

        # Sample colors using barycentric coords
        pixel_colors = interpolate_face_attributes(
            fragments.pix_to_face,                      # (N, H, W, K)
            fragments.bary_coords,                      # (N, H, W, K, 3)
            face_colors                                 # (F, 3, 3)
        )

        # Return the first sample (K=1 assumed)
        return pixel_colors[..., 0, :]


def make_uvspace_renderer(H, W, device="cuda"):
    cameras = OrthographicCameras(
        R=torch.eye(3).unsqueeze(0).to(device),
        T=torch.tensor([[0, 0, 1.0]], device=device),
        device=device
    )

    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=UnlitShader(device=device)
    )
    return renderer



def meshdata_to_pytorch3d_mesh(mesh_data: MeshData):
    verts = mesh_data.get_split_attribute("positions").float()
    faces = mesh_data.triangles
    if mesh_data.uvs is not None:
        uvs = mesh_data.get_split_attribute("uvs")
        uvs = uvs % 1.0
        tex = mesh_data.mapped_texture.image[:, :, :3].float()
        textures = TexturesUV(maps=[tex], faces_uvs=[faces], verts_uvs=[uvs])
    else:
        # print(f"No UVs found in mesh data: {mesh_data.name}, using vertex colors.")
        cols = mesh_data.get_split_attribute("colors").float()[:, :3]
        textures = TexturesVertex(verts_features=[cols])  # (1, V, 3)
    return Meshes(
            verts=[verts],
            faces=[faces],
            textures=textures
        )


def snap_uvs_to_base(uvs, base_uvs, n_digits):
    factor = 10 ** n_digits
    uvs_q = torch.round(uvs * factor) / factor
    base_q = torch.round(base_uvs * factor) / factor

    matches = (uvs_q.unsqueeze(1) == base_q.unsqueeze(0)).all(dim=-1)  # (uvs_q.shape[0], base_q.shape[0])
    matched_any = matches.any(dim=1)
    matched_idx = matches.float().argmax(dim=1)

    uvs_out = uvs.clone()
    uvs_out[matched_any] = base_uvs[matched_idx[matched_any]]
    return uvs_out


def dedupe_uvs_quantized_fixed(uvs: torch.Tensor, n_digits):
    factor = 10 ** n_digits
    quantized = torch.round(uvs * factor) / factor

    unique_uvs, inverse_indices = torch.unique(quantized, dim=0, return_inverse=True)
    return unique_uvs, inverse_indices


def map_learned_texture_to_mesh(
    base_mesh: MeshData,
    learned_texture,
    outfilename: str = "output_mesh.fbx",
):
    """
    Maps the learned texture (from TextureNet or JSON) onto the base_mesh and exports as FBX/OBJ.
    Args:
        base_mesh: MeshData object to receive the new colors.
        learned_texture: Either a TextureNet object or a path to a JSON file containing 'vertices', 'faces', 'colors'.
        outfilename: Output filename for the exported mesh.
    """
    # Load learned texture data
    if isinstance(learned_texture, str):
        # Assume JSON file
        with open(learned_texture, "r") as f:
            data = json.load(f)
        uvs = torch.tensor(data["vertices"], dtype=torch.float32, device=base_mesh.uvs.device)
        colors = torch.tensor(data["colors"], dtype=torch.float32, device=base_mesh.uvs.device)
        triangles = torch.tensor(data["faces"], dtype=torch.long, device=base_mesh.uvs.device)
    elif hasattr(learned_texture, "forward"):
        # Assume TextureNet object
        uvs, colors = learned_texture.forward()
        triangles = learned_texture.faces
        # Move to CPU and detach for safety
        uvs = uvs.detach()
        colors = colors.detach()
        triangles = triangles.detach()
    else:
        raise ValueError("learned_texture must be a TextureNet object or a JSON filename.")

    uvs = ((uvs + 1.0) / 2.0) % 1.0
    modded_uvs = base_mesh.uvs.clone() % 1.0

    uvs = snap_uvs_to_base(uvs, modded_uvs, n_digits=2)
    uvs, inverse_indices = dedupe_uvs_quantized_fixed(uvs, n_digits=3)
    triangles = inverse_indices[triangles]

    membership = get_triangle_membership(uvs, base_mesh)
    analyze_learned_triangle_coherence(triangles, membership)
    for tri in triangles:
        tri_members = [membership[uv_idx]._indices()[0] for uv_idx in tri]
        # print(f"Triangle {tri.tolist()} members: {tri_members}")


def analyze_learned_triangle_coherence(triangles, membership):
    # membership: [num_uvs, num_source_tris]
    tri_match_stats = {"all_matched_same": 0, "all_matched_diff": 0, "partial_or_none": 0}

    for tri in triangles:
        # Get set of source triangle indices for each vertex
        tri_members = [membership[uv_idx]._indices()[0] for uv_idx in tri]
        if any(len(m) == 0 for m in tri_members):
            tri_match_stats["partial_or_none"] += 1
            continue

        # Intersect or union check
        if set(tri_members[0].tolist()) == set(tri_members[1].tolist()) == set(tri_members[2].tolist()):
            tri_match_stats["all_matched_same"] += 1
        else:
            tri_match_stats["all_matched_diff"] += 1

    print("Learned triangle match coherence:")
    for k, v in tri_match_stats.items():
        print(f"{k}: {v} / {len(triangles)}")


def get_triangle_membership(polyverts: torch.Tensor, source_mesh: MeshData):
    source_uvs = source_mesh.get_split_attribute("uvs")
    polyverts = polyverts.to(source_uvs.device)

    row_indices = []
    col_indices = []

    for tri_idx, face in enumerate(source_mesh.triangles):
        tri_uvs = source_uvs[face]  # (3, 2)
        v0, v1, v2 = tri_uvs
        tri_min = tri_uvs.min(dim=0).values
        tri_max = tri_uvs.max(dim=0).values

        # Bounding box filter (fast)
        
        in_box = ((polyverts >= tri_min) & (polyverts <= tri_max)).all(dim=-1)
        candidates = polyverts[in_box]
        candidate_idxs = torch.nonzero(in_box).squeeze(-1)

        if candidates.shape[0] == 0:
            continue

        # Compute barycentric coords for all candidates (vectorized)
        b = batch_uv_barycentric_coords(candidates, v0, v1, v2)
        in_tri = (b >= -1e-4) & (b <= 1.0 + 1e-4)
        mask = in_tri.all(dim=1)
        matched = candidate_idxs[mask]

        row_indices.extend(matched.tolist())
        col_indices.extend([tri_idx] * len(matched))

    values = torch.ones(len(row_indices), dtype=torch.bool, device=polyverts.device)
    indices = torch.tensor([row_indices, col_indices], dtype=torch.long, device=polyverts.device)
    return torch.sparse_coo_tensor(indices, values, size=(polyverts.shape[0], source_mesh.triangles.shape[0]))


def batch_uv_barycentric_coords(uvs, v0, v1, v2):
    # All uvs: (N, 2)
    v0v1 = v1 - v0  # (2,)
    v0v2 = v2 - v0  # (2,)
    denom = (v0v1[0] * v0v2[1] - v0v1[1] * v0v2[0]).clamp(min=1e-8)

    u = ((v1[0] - uvs[:, 0]) * (v2[1] - uvs[:, 1]) - (v1[1] - uvs[:, 1]) * (v2[0] - uvs[:, 0])) / denom
    v = ((v2[0] - uvs[:, 0]) * (v0[1] - uvs[:, 1]) - (v2[1] - uvs[:, 1]) * (v0[0] - uvs[:, 0])) / denom
    w = 1.0 - u - v
    return torch.stack([w, u, v], dim=-1)


def triangle_aspect_loss(uvs, faces):
    v0, v1, v2 = uvs[faces[:, 0]], uvs[faces[:, 1]], uvs[faces[:, 2]]
    e0 = (v1 - v0).norm(dim=1)
    e1 = (v2 - v1).norm(dim=1)
    e2 = (v0 - v2).norm(dim=1)
    max_edge = torch.stack([e0, e1, e2], dim=1).max(dim=1).values
    min_edge = torch.stack([e0, e1, e2], dim=1).min(dim=1).values
    return ((max_edge / (min_edge + 1e-6)) - 1).clamp(min=0).mean()


def uv_area_loss(uvs, faces):
    v0, v1, v2 = uvs[faces[:, 0]], uvs[faces[:, 1]], uvs[faces[:, 2]]
    areas = 0.5 * torch.abs((v1 - v0)[:, 0] * (v2 - v0)[:, 1] - (v1 - v0)[:, 1] * (v2 - v0)[:, 0])
    return torch.var(areas)


def train_latent_texture(
    source_tex: torch.Tensor,
    epochs: int,
    save_every: int,
    save_dir: str,
    **kwargs,
):
    device = source_tex.device
    model = TextureNet2(source_tex, **kwargs).to(device)
    model.train()

    opt = SOAP(model.parameters(), lr=1e-3, weight_decay=0.0)
    # opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)
    output = None
    last_loss = float('inf')
    flat_epochs = 0
    ssim_loss = SSIMLoss().to(device)
    for epoch in range(epochs):
        opt.zero_grad()
        updated_uvs, colors = model.forward()
        output = model.render(updated_uvs, colors)
        geom_loss = triangle_aspect_loss(updated_uvs, model.faces) + uv_area_loss(updated_uvs, model.faces)
        print(f"Geometry loss: {geom_loss.item():.6f}")
        # output and source_tex should be (C, H, W) and in [0, 1] float
        output_img = output[:, :, :3].permute(2, 0, 1).clamp(0, 1).unsqueeze(0)  # Add batch dimension
        source_img = source_tex[:, :, :3].permute(2, 0, 1).clamp(0, 1).unsqueeze(0)  # Add batch dimension
        image_loss = ssim_loss(output_img, source_img)

        # color_loss = lpips_loss(output_rgb, source_tex_rgb).mean()
        loss = (geom_loss + image_loss) / 2
        if torch.abs(last_loss - loss) < 1e-5:
            model.increment_delta_scale(1e-4)
            flat_epochs += 1
            if flat_epochs > 10:
                model.increment_delta_scale(1e-3)
                flat_epochs = 0
        else:
            flat_epochs = 0
        last_loss = loss
        loss.backward()
        # Compute and print grad norms for each submodule
        print(f"Latent grad norm: {model.latent.grad.norm().item():.6f}")
        grad_norms = {}
        for name, module in [
            # ("latent", model.latent),
            ("decoder", model.decoder),
            ("delta_map", model.delta_map),
            ("color_map", model.color_map),
        ]:
            norm = 0.0
            for p in module.parameters():
                if p.grad is not None:
                    norm += p.grad.norm().item() ** 2
            grad_norms[name] = norm ** 0.5
        for k, v in grad_norms.items():
            print(f"{k} grad norm: {v:.6f}")
        opt.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

        # Write output image to disk after training
        if epoch % save_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                if output is not None:
                    out_img = (output[:, :, :3].detach().cpu().clamp(0, 1) * 255).to(torch.uint8)
                    out_img = out_img.permute(2, 0, 1)  # (C, H, W)
                    os.makedirs(f"debug_renders/{save_dir}", exist_ok=True)
                    write_png(out_img, f"debug_renders/{save_dir}/texture_output_{epoch}.png")
                    model.export_json(f"debug_renders/{save_dir}/texture_output_{epoch}.json")
    return model


if __name__ == "__main__":
    # Load the mesh from an FBX file
    meshes = load_fbx_to_meshdata("static/meshes/Horse.fbx")

    for mesh in meshes:
        model = train_latent_texture(
            mesh.mapped_texture.image.float().detach(),
            epochs=2500,
            save_every=25,
            save_dir="scratch_space",
            hidden_dim=128,

        )

        # map_learned_texture_to_mesh(
        #     mesh,
        #     "debug_renders/scratch_space/texture_output_2000.json",
        #     outfilename=f"debug_renders/scratch_space/learned_texture_{mesh.name}.fbx"
        # )

    # updated_meshes = []
    # for mesh in meshes:
    #     pr = cProfile.Profile()
    #     pr.enable()
    #     # Only run for 1 epoch to profile
    #     train_latent_texture(mesh.mapped_texture.image.float(), epochs=3, num_tris=5000, latent_dim=16)
    #     pr.disable()
    #     s = io.StringIO()
    #     ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    #     ps.print_stats(30)  # Show top 30 slowest functions
    #     print(s.getvalue())
