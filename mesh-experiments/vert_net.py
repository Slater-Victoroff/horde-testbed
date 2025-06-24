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
from asset_rep import MeshData, TextureData
from data_conversion import load_fbx_to_meshdata, export_meshdata_list
import cProfile
import pstats
import io
import glob


def generate_vert_grid(res=32, device="cuda"):
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

    verts = []
    triangles = []
    for i in range(res - 1):
        for j in range(res - 1):
            # Get corners of the cell
            current_idx = len(verts)
            v00 = verts_grid[i, j]
            v01 = verts_grid[i, j + 1]
            v10 = verts_grid[i + 1, j]
            v11 = verts_grid[i + 1, j + 1]
            verts.extend([v00, v10, v01, v11])
            triangles.append([current_idx, current_idx + 1, current_idx + 2])  # Triangle 1
            triangles.append([current_idx + 2, current_idx + 1, current_idx + 3])  # Triangle 2

    verts = torch.stack(verts, dim=0)  # (N * 3, 2)
    triangles = torch.tensor(triangles, dtype=torch.long, device=device)  # (N, 3)

    return verts, triangles


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


class TextureNet(torch.nn.Module):

    def __init__(self, source_tex, res=64, latent_dim=16, hidden_dim=64):
        super().__init__()
        self.source_tex = source_tex
        self.shape = source_tex.shape

    def render(self, updated_uvs: torch.Tensor = None, colors: torch.Tensor = None):
        if updated_uvs is None or colors is None:
            updated_uvs, colors = self.forward()
        verts_3d = torch.cat([updated_uvs, torch.zeros_like(updated_uvs[:, :1])], dim=-1)  # (V, 3)
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
        # Apply gamma correction (sRGB, gamma=2.2)
        colors = colors.clamp(0, 1).pow(2.2)
        triangles = self.faces.cpu().numpy()
        output = {
            "vertices": updated_uvs.cpu().numpy().tolist(),
            "faces": triangles.tolist(),
            "colors": colors.cpu().numpy().tolist()
        }
        with open(outfilename, 'w') as f:
            json.dump(output, f, indent=4)


class TextureNet2(TextureNet):

    def __init__(self, source_tex, res=128, latent_dim=128, latent_channels = 4, hidden_dim=64):
        super().__init__(source_tex, res=res, latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.shape = source_tex.shape
        device = source_tex.device
        self.source_tex = source_tex

        # verts, triangles, _ = generate_delaunay(n_points=res**2, device=device)
        verts, triangles = generate_vert_grid(res=res, device=device)
        self.register_buffer("base_uvs", verts * 2 - 1)  # Scale to [-1, 1] range
        self.register_buffer("faces", triangles)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=7, stride=2, padding=3),   # (1024x1024)
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),  # (512x512)
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1),  # (256x256)
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 4, latent_channels, kernel_size=3, stride=2, padding=1),  # (128x128)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((latent_dim, latent_dim)),  # force final shape (latent_dim, latent_dim)
            nn.Tanh()
        )

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

        # Permute source_tex to (1, 3, H, W)
        source_tex = self.source_tex[..., :3].permute(2, 0, 1).unsqueeze(0)

        latent = self.encoder(source_tex)  # (1, latent_channels, H, W)
        feat = self.decoder(latent)  # (1, C, H, W)
        delta = F.grid_sample(self.delta_map(feat), grid, mode='bilinear', align_corners=True)
        delta = delta.squeeze(-1).squeeze(0).transpose(0, 1)
        new_uvs = self.base_uvs + delta * self.delta_scale  # (N, 2)

        new_uv_grid = new_uvs.view(1, -1, 1, 2).flip(-1)  # (1, N, 1, 2)
        color = F.grid_sample(self.color_map(feat), new_uv_grid, mode='bilinear', align_corners=True)

        color = color.squeeze(-1).squeeze(0).transpose(0, 1)
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
        

class SliceColorShader(torch.nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device

    def forward(self, fragments, meshes, **kwargs):
        faces = meshes.faces_packed()  # (F, 3)
        verts_colors = meshes.textures.verts_features_packed()  # (V, 3)
        face_colors = verts_colors[faces]  # (F, 3, 3)

        face_idx = fragments.pix_to_face[..., 0]  # (N, H, W)
        bary = fragments.bary_coords[..., 0, :]   # (N, H, W, 3)

        # Get index of max barycentric coordinate → closest vertex in triangle
        max_idx = bary.argmax(dim=-1)  # (N, H, W)

        # For valid pixels
        mask = face_idx >= 0
        output = torch.zeros_like(bary[..., :3])  # (N, H, W, 3)
        output[mask] = face_colors[face_idx[mask], max_idx[mask]]

        return output


class FlatColorShader(torch.nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device

    def forward(self, fragments, meshes, **kwargs):
        faces = meshes.faces_packed()  # (F, 3)
        verts_colors = meshes.textures.verts_features_packed()  # (V, 3)
        face_colors = verts_colors[faces]  # (F, 3, 3)
        avg_face_colors = face_colors.mean(dim=1)  # (F, 3)

        # Use only first face per pixel (K=1 assumed)
        face_idx = fragments.pix_to_face[..., 0]  # (N, H, W)
        mask = face_idx >= 0
        pixel_colors = torch.zeros_like(face_idx, dtype=torch.float32).unsqueeze(-1).repeat(1, 1, 1, 3)

        valid_idx = face_idx[mask]
        pixel_colors[mask] = avg_face_colors[valid_idx]

        return pixel_colors


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

    ssim_loss = SSIMLoss().to(device)
    for epoch in range(epochs):
        opt.zero_grad()

        updated_uvs, colors = model.forward()
        output = model.render(updated_uvs, colors)
        geom_loss = triangle_aspect_loss(updated_uvs, model.faces) + uv_area_loss(updated_uvs, model.faces)
        # output and source_tex should be (C, H, W) and in [0, 1] float
        output_img = output[:, :, :3].permute(2, 0, 1).clamp(0, 1).unsqueeze(0)  # Add batch dimension
        source_img = source_tex[:, :, :3].permute(2, 0, 1).clamp(0, 1).unsqueeze(0)  # Add batch dimension
        image_loss = ssim_loss(output_img, source_img)

        loss = geom_loss * 0.5 + image_loss * 0.5

        loss.backward()
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

    texture_files = glob.glob("static/*.tga")

    for tex_path in texture_files:
        texture = TextureData.load(tex_path)
        tex_name = os.path.splitext(os.path.basename(tex_path))[0]
        train_latent_texture(
            texture.image.float().detach(),
            epochs=1500,
            save_every=25,
            save_dir=f"scratch_space_{tex_name}",
        )
