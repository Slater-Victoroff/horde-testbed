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
from torch.nn.utils import spectral_norm
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

    return verts, faces


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
        self.source_tex = source_tex
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
        print(f"Delta shape: {delta.shape}, UVs shape: {self.base_uvs.shape}")
        new_uvs = self.base_uvs + delta * self.delta_scale  # (N, 2)

        new_uv_grid = new_uvs.view(1, -1, 1, 2).flip(-1)  # (1, N, 1, 2)
        color = F.grid_sample(self.color_map(feat), new_uv_grid, mode='bilinear', align_corners=True)

        color = color.squeeze(-1).squeeze(0).transpose(0, 1)
        print(f"Color shape: {color.shape}, UVs shape: {new_uvs.shape}")
        return new_uvs, color


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out + self.skip(x))


class ResidualMLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc1 = spectral_norm(nn.Linear(in_dim, hidden_dim))
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = spectral_norm(nn.Linear(hidden_dim, in_dim))
        self.ln2 = nn.LayerNorm(in_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.activation(self.ln1(self.fc1(x)))
        x = self.activation(self.ln2(self.fc2(x)))
        return x + residual


class TextureNet3(TextureNet):
    def __init__(self, source_tex, n_points, latent_dim=128, hidden_dim=128, latent_channels=4):
        super().__init__(source_tex)
        self.n_points = n_points
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((latent_dim, latent_dim)),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU()
        )

        # A deeper and wider point_head with more nonlinearities and context
        self.point_backbone = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim * 2),
            ResidualBlock(hidden_dim * 2, hidden_dim * 4, dilation=2),
            ResidualBlock(hidden_dim * 4, hidden_dim * 2),
            ResidualBlock(hidden_dim * 2, hidden_dim)
        )

        self.pos_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            ResidualMLPBlock(hidden_dim, hidden_dim * 4),
            ResidualMLPBlock(hidden_dim, hidden_dim * 4),
            nn.Linear(hidden_dim, n_points * 2),
            nn.Tanh()
        )

        self.color_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            ResidualMLPBlock(hidden_dim, hidden_dim * 2),
            ResidualMLPBlock(hidden_dim, hidden_dim * 2),
            nn.Linear(hidden_dim, n_points * 3),
            nn.Sigmoid()
        )

        def improved_init_sequential(seq, final_out_dim=None, epsilon=1e-4):
            for m in seq.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    if final_out_dim is not None and getattr(m, 'out_features', None) == final_out_dim:
                        # Final point_mlp Linear layer
                        u = torch.empty(final_out_dim).uniform_(-1 + epsilon, 1 - epsilon)
                        pre_tanh = 0.5 * (torch.log1p(u) - torch.log1p(-u))  # atanh(x)
                        with torch.no_grad():
                            nn.init.kaiming_uniform_(m.weight, a=0.1)  # Small nonzero weights
                            if m.bias is not None:
                                m.bias.copy_(pre_tanh)
                    else:
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

        improved_init_sequential(self.encoder)
        improved_init_sequential(self.decoder)
        improved_init_sequential(self.point_backbone)
        improved_init_sequential(self.pos_mlp, final_out_dim=n_points * 2)

        for m in self.color_mlp.modules():
            if isinstance(m, nn.Linear) and m.out_features == self.n_points * 3:
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, torch.log(torch.tensor(1.0 / 0.5 - 1.0)))



    def forward(self):
        source_tex = self.source_tex[..., :3].permute(2, 0, 1).unsqueeze(0)

        latent = self.encoder(source_tex)
        feat = self.decoder(latent)

        # Predict point positions
        x = self.point_backbone(feat)                  # (1, hidden_dim, H, W)
        point_feat = x.mean(dim=[2, 3])               # (1, C)

        point_logits = self.pos_mlp(point_feat)     # nn.Linear(C, n_points * 2)
        uvs = point_logits.view(self.n_points, 2)

        # Predict per-point colors
        color_logits = self.color_mlp(point_feat)  # (1, 3 * n_points, H, W)
        colors = color_logits.view(self.n_points, 3)

        uvs_np = uvs.detach().cpu().numpy()
        tri = Delaunay(uvs_np)
        self.faces = torch.tensor(tri.simplices, dtype=torch.long, device=uvs.device)

        return uvs, colors


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.dim_head = hidden_dim // n_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q, kv):
        B, N, C = q.shape  # (batch, points, hidden_dim)
        _, M, _ = kv.shape

        q = self.q_proj(q).view(B, N, self.n_heads, self.dim_head).transpose(1, 2)  # (B, h, N, d)
        k = self.k_proj(kv).view(B, M, self.n_heads, self.dim_head).transpose(1, 2)  # (B, h, M, d)
        v = self.v_proj(kv).view(B, M, self.n_heads, self.dim_head).transpose(1, 2)  # (B, h, M, d)

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        out = out.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)


class TextureNet4(TextureNet):
    def __init__(self, source_tex, n_points, latent_dim=64, hidden_dim=64, n_heads=2, n_attn_layers=2):
        super().__init__(source_tex)
        self.n_points = n_points

        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.decoder_proj = nn.Linear(latent_dim, hidden_dim)
        self.token_emb = nn.Parameter(torch.randn(n_points, hidden_dim))

        self.attn_layers = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, n_heads) for _ in range(n_attn_layers)]
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 5),  # 2 for UV, 3 for color
        )

    def forward(self):
        x = self.source_tex[..., :3].permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        feat = self.encoder(x)  # (1, C, H', W')
        B, C, H, W = feat.shape
        decoder_tokens = feat.flatten(2).transpose(1, 2)  # (1, HW, C)
        decoder_tokens = self.decoder_proj(decoder_tokens)  # (1, HW, hidden_dim)

        point_tokens = self.token_emb.unsqueeze(0).expand(B, -1, -1)  # (1, N, hidden_dim)

        for attn in self.attn_layers:
            point_tokens = attn(point_tokens, decoder_tokens)  # (1, N, hidden_dim)

        enriched = point_tokens
        out = self.mlp_head(enriched).squeeze(0)  # (N, 5)
        uvs = torch.tanh(out[:, :2])  # (N, 2)
        colors = torch.sigmoid(out[:, 2:])  # (N, 3)

        print(f"UVs shape: {uvs.shape}, Colors shape: {colors.shape}")
        uvs_np = uvs.detach().cpu().numpy()
        tri = Delaunay(uvs_np)
        print(f"Generated {len(tri.simplices)} triangles from Delaunay triangulation.")
        self.faces = torch.tensor(tri.simplices, dtype=torch.long, device=uvs.device)

        return uvs, colors


def set_requires_grad(modules, flag: bool):
    if not isinstance(modules, (list, tuple)):
        modules = [modules]
    for m in modules:
        for p in m.parameters():
            p.requires_grad = flag


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


def make_uvspace_renderer(H, W, device="cuda", shader=UnlitShader(device="cuda")):
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
        shader=shader
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
    eps = 1e-6  # Small epsilon to avoid division by zero
    v0, v1, v2 = uvs[faces[:, 0]], uvs[faces[:, 1]], uvs[faces[:, 2]]
    e0 = (v1 - v0).norm(dim=1)
    e1 = (v2 - v1).norm(dim=1)
    e2 = (v0 - v2).norm(dim=1)
    max_edge = torch.stack([e0, e1, e2], dim=1).max(dim=1).values
    min_edge = torch.stack([e0, e1, e2], dim=1).min(dim=1).values
    aspect_ratio = (max_edge + eps) / (min_edge + eps)
    return torch.log(aspect_ratio).clamp(min=0).mean()


def uv_area_loss(uvs, faces):
    v0, v1, v2 = uvs[faces[:, 0]], uvs[faces[:, 1]], uvs[faces[:, 2]]
    areas = 0.5 * torch.abs((v1 - v0)[:, 0] * (v2 - v0)[:, 1] - (v1 - v0)[:, 1] * (v2 - v0)[:, 0])
    return torch.var(areas / (areas.mean() + 1e-6))


def uv_spread_loss(uvs, sigma=0.05):
    # Gaussian repulsion loss - points repel each other
    dists = torch.cdist(uvs, uvs)  # (N, N)
    
    # Remove diagonal
    eye_mask = torch.eye(uvs.shape[0], dtype=torch.bool, device=uvs.device)
    dists = dists.masked_fill(eye_mask, float('inf'))
    
    # Gaussian repulsion: higher energy when points are close
    repulsion = torch.exp(-dists.pow(2) / (2 * sigma**2))
    return repulsion.mean()  # Minimize total repulsion energy


def training_schedule(epoch):
    if epoch < 150:
        # Phase 1: Color warm-up (freeze everything but color)
        freeze_encoder = False
        freeze_decoder = False
        freeze_points = False
        freeze_colors = True

    elif 150 <= epoch < 250:
        # Phase 2: Geometry warm-up (freeze color, learn points)
        freeze_encoder = True
        freeze_decoder = False
        freeze_points = True
        freeze_colors = False

    elif 250 <= epoch < 500:
        # Phase 3: Joint refinement on fixed encoder/decoder
        freeze_encoder = True
        freeze_decoder = True
        freeze_points = False
        freeze_colors = False

    elif 500 <= epoch < 1000:
        # Phase 4: Decoder unfreezing, refine representation
        freeze_encoder = False
        freeze_decoder = False
        freeze_points = False
        freeze_colors = False

    else:
        # Phase 5: Final joint fine-tuning
        freeze_encoder = False
        freeze_decoder = False
        freeze_points = False
        freeze_colors = False
    return freeze_encoder, freeze_decoder, freeze_points, freeze_colors


def schedule_loss(epoch, model, updated_uvs, output, source_tex, ssim_loss):
    uvd_loss = uv_spread_loss(updated_uvs)
    uva_loss = uv_area_loss(updated_uvs, model.faces)
    ta_loss = triangle_aspect_loss(updated_uvs, model.faces)
    print(f"Epoch {epoch}: uvd_loss={uvd_loss:.6f}, uva_loss={uva_loss:.6f}, ta_loss={ta_loss:.6f}")

    output_img = output[:, :, :3].permute(2, 0, 1).clamp(0, 1).unsqueeze(0)  # Add batch dimension
    source_img = source_tex[:, :, :3].permute(2, 0, 1).clamp(0, 1).unsqueeze(0)  # Add batch dimension
    l1_loss = F.l1_loss(output_img, source_img)
    ssim = ssim_loss(output_img, source_img)
    
    # Define per-term schedules as dicts: {start: weight, ...}
    # Each term's weight is linearly interpolated between keys
    term_schedules = {
        "uvd_loss": {
            0: 1.0,
            100: 1.0,
            300: 0.5,
            500: 0.25,
            1000: 0.1
        },
        "uva_loss": {
            0: 0.0,
            100: 0.5,
            300: 1.0,
            600: 0.5,
        },
        "ta_loss": {
            0: 0.0,
            100: 0.5,
            300: 1.0,
            600: 0.5,
            1000: 0.25
        },
        "l1_loss": {
            0: 0.0,
            100: 0.0,
            250: 0.5,
            500: 1.0,
            1000: 0.25
        },
        "ssim_loss": {
            0: 0.0,
            100: 0.0,
            250: 0.1,
            500: 0.5,
            1000: 1.0
        }
    }

    def interp_schedule(schedule, epoch):
        keys = sorted(schedule.keys())
        for i, k in enumerate(keys):
            if epoch < k:
                if i == 0:
                    return schedule[keys[0]]
                k0, k1 = keys[i-1], k
                v0, v1 = schedule[k0], schedule[k1]
                t = (epoch - k0) / (k1 - k0)
                return v0 * (1 - t) + v1 * t
        return schedule[keys[-1]]

    weights = {k: interp_schedule(v, epoch) for k, v in term_schedules.items()}

    loss = (
        weights["uvd_loss"] * uvd_loss +
        weights["uva_loss"] * uva_loss +
        weights["ta_loss"] * ta_loss
    )
    if weights["l1_loss"] > 0:
        loss = loss + weights["l1_loss"] * l1_loss
    if weights["ssim_loss"] > 0:
        loss = loss + weights["ssim_loss"] * ssim
    return loss


def train_latent_texture(
    source_tex: torch.Tensor,
    epochs: int,
    save_every: int,
    save_dir: str,
    **kwargs,
):
    device = source_tex.device
    model = TextureNet3(source_tex, **kwargs).to(device)
    model.train()

    opt = SOAP(model.parameters())
    # opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)
    output = None

    ssim_loss = SSIMLoss().to(device)
    for epoch in range(epochs):
        opt.zero_grad()
        # freeze_encoder, freeze_decoder, freeze_points, freeze_colors = training_schedule(epoch)

        # model.update_freeze_states(
        #     freeze_points=freeze_points,
        #     freeze_colors=freeze_colors,
        #     freeze_encoder=freeze_encoder,
        #     freeze_decoder=freeze_decoder
        # )

        updated_uvs, colors = model.forward()
        print(f"updated_uvs requires_grad: {updated_uvs.requires_grad}, colors requires_grad: {colors.requires_grad}")
        output = model.render(updated_uvs, colors)
        # loss = uv_spread_loss(updated_uvs)

        loss = schedule_loss(epoch, model, updated_uvs, output, source_tex, ssim_loss)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.point_backbone.parameters(), max_norm=5.0)
        # torch.nn.utils.clip_grad_norm_(model.pos_mlp.parameters(), max_norm=5.0)
        # torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=5.0)
        # torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=5.0)

        # Compute and print grad norms for each submodule
        # print(f"Latent grad norm: {model.latent.grad.norm().item():.6f}")
        grad_norms = {}
        for name, module in [
            ("encoder", model.encoder),
            ("decoder", model.decoder),
            ("point_backbone", model.point_backbone),
            ("pos_mlp", model.pos_mlp),
            ("color_mlp", model.color_mlp),
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
    # meshes = load_fbx_to_meshdata("static/meshes/Horse.fbx")

    # texture_files = [
    #     f for f in os.listdir("static")
    #     if f.lower().endswith((".tga", ".png"))
    # ]
    # texture_files = [os.path.join("static", f) for f in texture_files]
    # print(f"Found texture files: {texture_files}")

    texture_files = {
        "static/T_Horse_Body_M_D_WhS.tga": 10000,
        "static/T_Horse_Saddle_M_D_Bk.tga": 5000,
        "static/Bows_1A1_Bows_1A1_AlbedoTransparency.png": 10000,
        "static/Arrows_1A1_Arrows_1A1_AlbedoTransparency.png": 10000,
        "static/gradient test low_Shadow_Test_A_AlbedoTransparency.png": 10000,
        "static/Hammers_1A1_Hammers_1A1_AlbedoTransparency.png": 10000,
        "static/Hammers_1A1_Hammers_1B1_AlbedoTransparency.png": 10000,
        "static/Magics_1A1_Magics_1A1_AlbedoTransparency.png": 10000,
        "static/KoraDrveta_Unutra3.tga": 10000,
        "static/mat_ArmyHelmet01_Albedo.png": 5000,
        "static/mat_BaseballHat01_Albedo (1).png": 2500,
        "static/mat_ExecutionerHat01_Albedo.png": 2500,
        "static/mat_KettleHat01_Albedo.png": 2500,
        "static/mat_PopeHat01_Albedo (1).png": 1000,
        "static/Misc_1A1_Misc_1A1_AlbedoTransparency.png": 10000,
        "static/Presek3.tga": 10000,
        "static/Shields_1A3_Shields_1A1_AlbedoTransparency.png": 10000,
        "static/Staves_1A1_Staves_1A1_AlbedoTransparency.png": 10000,
        "static/Swords_A_Swords_1B1_AlbedoTransparency.png": 10000,
        "static/Swords_A_Swords_1C1_AlbedoTransparency.png": 10000,
        "static/T_Cart_C.png": 10000,
        "static/T_ForksKnives_01_AlbedoTransparency.tga": 10000,
    }

    for tex_path, n_points in texture_files.items():
        texture = TextureData.load(tex_path)
        tex_name = os.path.splitext(os.path.basename(tex_path))[0]
        train_latent_texture(
            texture.image.float().detach(),
            epochs=2500,
            save_every=25,
            save_dir=f"scratch_space_{tex_name}",
            n_points=n_points,
        )

    # for mesh in meshes:
    #     model = train_latent_texture(
    #         mesh.mapped_texture.image.float().detach(),
    #         epochs=2500,
    #         save_every=25,
    #         save_dir="scratch_space",
    #         hidden_dim=128,

    #     )

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
