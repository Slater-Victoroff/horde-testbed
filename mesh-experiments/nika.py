"""
Code for Neural Interpolated Kernel Array (NIKA) experiments.
"""
import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch3d
from pytorch3d.renderer import (
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    FoVPerspectiveCameras,
    PointLights,
    SoftPhongShader
)

from torchvision.transforms.functional import to_pil_image
from pytorch3d.transforms import matrix_to_quaternion
from pytorch3d.renderer import look_at_view_transform, TexturesVertex
from pytorch3d.structures import Meshes
from torchvision.utils import save_image, make_grid
from piq import SSIMLoss

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from soap import SOAP
from asset_rep import MeshData
from data_conversion import load_fbx_to_meshdata, meshdata_to_pytorch3d_mesh
import copy


class NikaNet(torch.nn.Module):
    """
    Neural Interpolated Kernel Array (NIKA).
    """
    def __init__(self, latent_dim, output_dim=4, hidden_dim=64, pos_dims=64):
        super().__init__()
        # Define the layers and parameters of the NIKA network here

        self.trunk_1 = torch.nn.Linear(latent_dim, hidden_dim)

        self.pos_film = torch.nn.Sequential(
            torch.nn.Linear(pos_dims, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
        )

        self.trunk_2 = torch.nn.Linear(hidden_dim + latent_dim, hidden_dim)

        self.cam_film = torch.nn.Linear(3 + 4 + 1, hidden_dim * 2)  # 3 for position, 4 for quaternion, 1 for fov

        self.to_color = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + latent_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.Sigmoid()  # Ensure output is in [0, 1] range
        )

    def get_frag_shader(self, cameras, lights, img_size):
        return NeuralFragmentShader(
            self.trunk_1,
            self.pos_film,
            self.trunk_2,
            self.cam_film,
            self.to_color,
            cameras = cameras,
            lights = lights,
            img_size = img_size
        ).to(cameras.device)


class NeuralFragmentShader(torch.nn.Module):
    """
    Shader that uses a neural network (MLP) to compute per-pixel color.
    """
    def __init__(self, trunk_1, pos_film, trunk_2, cam_film, to_color, cameras, lights, img_size):
        super().__init__()
        self.latent_dim = trunk_1.in_features
        self.trunk_1 = trunk_1
        self.pos_dims = pos_film[0].in_features
        self.pos_film = pos_film
        self.trunk_2 = trunk_2
        self.cam_film = cam_film
        self.to_color = to_color
        self.cameras = cameras
        self.img_size = img_size
        self.pix_id = camera_idx_flat = torch.arange(
            len(cameras), device=cameras.device
        ).repeat_interleave(img_size * img_size).view(-1, 1)

    def forward(self, fragments, meshes, **kwargs):
        verts_world = meshes.verts_packed()
        faces = meshes.faces_packed()
        face_verts = verts_world[faces]  # (F, 3, 3)
        pixel_positions = pytorch3d.renderer.mesh.shading.interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, face_verts
        )

        verts_latents = meshes.textures.verts_features_packed()  # (V, 4)
        face_latents = verts_latents[faces]  # (F, 3, 4)
        pixel_latents = pytorch3d.renderer.mesh.shading.interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, face_latents
        )

        pixel_positions_flat = pixel_positions.view(-1, 3)
        pixel_latents_flat = pixel_latents.view(-1, self.latent_dim)  # (N*H*W, latent_dim)

        trunk_embedding = self.trunk_1(pixel_latents_flat)  # (N*H*W, hidden_dim)

        pos_encoded = compute_targeted_encodings(
            pixel_positions_flat,
            target_dim=self.pos_dims,
            scheme='sinusoidal',
        )

        pos_features = self.pos_film(pos_encoded)  # (N*H*W, hidden_dim)
        pos_gamma, pos_beta = torch.chunk(pos_features, 2, dim=-1)

        trunk_embedding = (trunk_embedding * pos_gamma) + pos_beta  # (N*H*W, hidden_dim)
        activated_embeddings = F.gelu(trunk_embedding)  # (N*H*W, hidden_dim)

        trunk_2_input = torch.cat([activated_embeddings, pixel_latents_flat], dim=-1)  # (N*H*W, hidden_dim + latent_dim)
        trunk_embedding = self.trunk_2(trunk_2_input)  # (N*H*W, hidden_dim)

        cam_positions = self.cameras.get_camera_center().view(-1, 3)  # (N, 3)
        cam_quaternions = matrix_to_quaternion(self.cameras.R)
        cam_fovs = self.cameras.fov.view(-1, 1) 

        cam_features = torch.cat([cam_positions, cam_quaternions, cam_fovs], dim=-1)  # (N, 3 + 4 + 1)
        cam_film_params = self.cam_film(cam_features)  # (N, hidden_dim *

        cam_gamma, cam_beta = torch.chunk(cam_film_params, 2, dim=-1)

        pix_id_flat = self.pix_id.view(-1)  # (N*H*W,)
        gamma_per_pixel = torch.index_select(cam_gamma, dim=0, index=pix_id_flat)  # (N*H*W, hidden_dim)
        beta_per_pixel  = torch.index_select(cam_beta,  dim=0, index=pix_id_flat)  # (N*H*W, hidden_dim)
        modulated_features = (trunk_embedding * gamma_per_pixel) + beta_per_pixel
        activated_features = F.gelu(modulated_features)  # (N*H*W, hidden_dim)

        final_input = torch.cat([activated_features, pixel_latents_flat], dim=-1)  # (N*H*W, hidden_dim + latent_dim)
        colors_flat = self.to_color(final_input)
        colors = colors_flat.view(fragments.pix_to_face.shape[:3] + (4,))
        return colors


def compute_spiral_encoding(x, num_harmonics):
    out = []
    for i in range(1, num_harmonics + 1):
        out += [torch.sin(i * x) / i, torch.cos(i * x) / i]
    return torch.cat(out, dim=-1)


def compute_sinusoidal_encoding(coords, num_harmonics):
    encodings = []
    for i in range(1, num_harmonics + 1):
        encodings += [torch.sin(i * coords), torch.cos(i * coords)]
    return torch.cat(encodings, dim=-1)


def compute_linear_encoding(x, target_dim):
    return x.repeat(1, target_dim // x.shape[-1])[:, :target_dim]


def compute_polynomial_encoding(x, max_degree):
    out = []
    for i in range(1, max_degree + 1):
        out.append(x ** i)
    return torch.cat(out, dim=-1)


def compute_gaussian_encoding(x, target_dim, std=10.0, seed=42):
    generator = torch.Generator(device=x.device).manual_seed(seed)
    B = torch.randn(x.shape[1], target_dim // 2, generator=generator, device=x.device) * std
    x_proj = 2 * torch.pi * x @ B  # [B, F]
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def compute_targeted_encodings(x, target_dim, scheme="spiral", norm_2pi=True, include_norm=False, include_raw=False, seed=42):
    _, N = x.shape
    encodings = []

    if include_raw:
        encodings.append(x)

    if norm_2pi:
        x = x * 2 * torch.pi
        if include_norm:
            encodings.append(x)

    if scheme in ["spiral", "sinusoidal"]:
        num_harmonics = ((target_dim - (N if include_raw else 0)) // 2) + 1
        encoding_fn = {
            "spiral": compute_spiral_encoding,
            "sinusoidal": compute_sinusoidal_encoding,
        }[scheme]
        encodings.append(encoding_fn(x, num_harmonics=num_harmonics))
    elif scheme == "gaussian":
        encodings.append(compute_gaussian_encoding(x, target_dim, seed=seed))
    elif scheme == "linear":
        encodings.append(compute_linear_encoding(x, target_dim))
    elif scheme == "polynomial":
        deg = target_dim // x.shape[-1]
        encodings.append(compute_polynomial_encoding(x, deg))
    elif scheme is None:
        encodings.append(torch.zeros(x.shape[0], target_dim, device=x.device))
    else:
        raise ValueError(f"Unknown encoding scheme: {scheme}")

    return torch.cat(encodings, dim=-1)[:, :target_dim]


class LatentMesh(torch.nn.Module):

    def __init__(self, source_mesh: MeshData, latent_dim: int, use_polyverts=False):
        super().__init__()
        self.source_mesh = source_mesh
        if use_polyverts:
            self.verts = source_mesh.get_split_attribute("positions").float()
            self.faces = source_mesh.triangles
        else:
            verts = source_mesh.get_split_attribute("positions").float()
            unique_positions, inverse_indices = torch.unique(verts, dim=0, return_inverse=True)

            remapped_faces = inverse_indices[source_mesh.triangles]
            self.verts = unique_positions
            self.faces = remapped_faces

        self.vertex_latents = torch.randn(self.verts.shape[0], latent_dim, device=self.verts.device)

    def get_mesh(self):
        textures = TexturesVertex(verts_features=[self.vertex_latents])
        return Meshes(
            verts = [self.verts],
            faces = [self.faces],
            textures = textures,
        )


class GNNEncoder(nn.Module):
    """
    Simple GNN to encode vertex positions into latent embeddings.
    """
    def __init__(self, in_dim=3, latent_dim=8, hidden_dim=32, num_layers=3):
        super().__init__()
        layers = []
        layers.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            layers.append(GCNConv(hidden_dim, hidden_dim))
        layers.append(GCNConv(hidden_dim, latent_dim))
        self.layers = nn.ModuleList(layers)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        """
        x: (V, in_dim) vertex positions
        edge_index: (2, E) edge connectivity
        """
        for layer in self.layers[:-1]:
            x = self.activation(layer(x, edge_index))
        x = self.layers[-1](x, edge_index)
        return x  # (V, latent_dim)


def fibonacci_cameras(
    n_cameras=8,
    dist=1.5,
    fov=45.0,
    tilt_elev=0.0,
    device="cuda"
):
    """
    Generate cameras evenly distributed on a sphere using Fibonacci lattice.
    
    Args:
        n_cameras (int): Number of cameras to generate.
        dist (float): Distance of cameras from origin.
        fov (float): Field of view in degrees.
        tilt_elev (float): Additional tilt to elevate all cameras (degrees).
        device (str): Torch device.

    Returns:
        FoVPerspectiveCameras: Batched cameras on sphere.
    """
    indices = torch.arange(0, n_cameras, dtype=torch.float32, device=device) + 0.5
    phi = torch.acos(1 - 2 * indices / n_cameras)  # polar angle
    theta = torch.pi * (1 + 5**0.5) * indices      # azimuthal angle

    # Convert spherical coordinates to Cartesian
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)

    # Apply optional tilt to elevate cameras
    elev = torch.asin(z) * (180.0 / torch.pi) + tilt_elev
    azim = torch.atan2(x, y) * (180.0 / torch.pi)

    # Create camera transforms
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)

    # Uniform FoV per camera
    fovs = torch.full((n_cameras,), fov, device=device)

    cameras = FoVPerspectiveCameras(R=R, T=T, fov=fovs, device=device)
    return cameras


def sample_random_cameras(
    n_cameras=32,
    dist_range=(1.5, 3.0),
    elev_range=(-60, 60),
    azim_range=(0, 360),
    fov_range=(30, 90),
    device="cuda"
):
    dist = torch.empty(n_cameras, device=device).uniform_(*dist_range)
    elev = torch.empty(n_cameras, device=device).uniform_(*elev_range)
    azim = torch.empty(n_cameras, device=device).uniform_(*azim_range)
    fovs = torch.empty(n_cameras, device=device).uniform_(*fov_range)

    # Compute rotation and translation
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)

    # Create camera batch
    cameras = FoVPerspectiveCameras(R=R, T=T, fov=fovs, device=device)
    return cameras


def select_coverage_cameras(
    n_cameras,
    base_mesh,
    latent_mesh,
    model,
    lights,
    num_candidates=128,
    image_test_size=128,
    weight_by_error=False
):
    """
    Greedily select cameras that maximize mesh surface coverage.
    """
    device = base_mesh.device

    # Create detached copies of base_mesh, latent_mesh, and model for candidate camera evaluation
    latent_mesh_detached = LatentMesh(latent_mesh.source_mesh, latent_mesh.vertex_latents.shape[1]).to(device)
    latent_mesh_detached.verts = latent_mesh.verts.detach()
    latent_mesh_detached.faces = latent_mesh.faces.detach()
    latent_mesh_detached.vertex_latents = latent_mesh.vertex_latents.detach()

    # Detach model weights
    model_detached = NikaNet(latent_mesh.vertex_latents.shape[1]).to(device)
    model_detached.load_state_dict(model.state_dict())
    for param in model_detached.parameters():
        param.requires_grad_(False)
    model_detached.eval()

    with torch.no_grad():
        candidate_cameras = sample_random_cameras(n_cameras=num_candidates, device=device)

        # Render visibility maps for each camera
        visibility_maps = []
        for cam in candidate_cameras:
            rasterizer = make_rasterizer(cam, image_size=image_test_size, device=device)
            fragments = rasterizer(base_mesh.extend(1))
            visible_faces = fragments.pix_to_face[0].unique()
            visible_faces = visible_faces[visible_faces >= 0]  # Remove -1 (no face)
            visibility_maps.append(set(visible_faces.cpu().tolist()))

        # Optionally compute errors per camera
        if weight_by_error:
            base_images = render_base_mesh(base_mesh, candidate_cameras, lights, image_test_size)
            neural_images = render_neural_mesh(candidate_cameras, lights, latent_mesh, model, image_test_size)
            per_camera_errors = compute_ssim_loss(neural_images, base_images, reduction="none").cpu()

    # Greedy selection
    covered_faces = set()
    selected_indices = []
    for _ in range(n_cameras):
        best_idx = None
        best_gain = -1
        for i, faces in enumerate(visibility_maps):
            if i in selected_indices:
                continue  # Already selected
            new_faces = faces - covered_faces
            gain = len(new_faces)
            if weight_by_error:
                gain *= per_camera_errors[i].item()  # Weighted by camera error
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx is None:
            break  # No more useful cameras
        selected_indices.append(best_idx)
        covered_faces |= visibility_maps[best_idx]

    # Build final FoVPerspectiveCameras batch
    selected_cameras = FoVPerspectiveCameras(
        R=candidate_cameras.R[selected_indices],
        T=candidate_cameras.T[selected_indices],
        fov=candidate_cameras.fov[selected_indices],
        device=device
    )
    return selected_cameras


def make_lights(lighting_mode: str, light_location: tuple = (2.0, 2.0, -2.0)):
    """
    Create lights for the renderer based on the lighting mode.
    
    lighting_mode: "unlit", "phong", or "hard"
    light_location: tuple of (x, y, z) coordinates for the light position
    """
    if lighting_mode == "phong":
        lights = PointLights(device="cuda", location=[light_location])
    elif lighting_mode == "hard":
        lights = PointLights(device="cuda", location=[light_location])
    elif lighting_mode == "unlit":
        lights = PointLights(
            device="cuda",
            ambient_color=((1.0, 1.0, 1.0),),
            diffuse_color=((0.0, 0.0, 0.0),),
            specular_color=((0.0, 0.0, 0.0),)
        )
    else:
        raise ValueError(f"Unknown lighting mode: {lighting_mode}")
    
    return lights


def make_shader(
    cameras,
    lights,
    shader_type="unlit", 
    device="cuda"
):
    if shader_type == "phong":
        shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)
    elif shader_type == "hard":
        shader = HardPhongShader(device=device, cameras=cameras, lights=lights)
    elif shader_type == "unlit":
        shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)
    else:
        raise ValueError(f"Unknown shader_type: {shader_type}")
    return shader


def make_rasterizer(
    cameras,
    image_size=256,
    blur_radius=0.0,
    faces_per_pixel=1,
    device="cuda"
):
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel,
        cull_backfaces=False,
    )
    
    return MeshRasterizer(cameras=cameras, raster_settings=raster_settings)


def print_grad_norms(model, latent_mesh):
    grad_norms = {}
    for name, module in [
        ("trunk_1", model.trunk_1),
        ("pos_film", model.pos_film),
        ("trunk_2", model.trunk_2),
        ("cam_film", model.cam_film),
        ("to_color", model.to_color),
    ]:
        norm = 0.0
        for p in module.parameters():
            if p.grad is not None:
                norm += p.grad.norm().item() ** 2
        grad_norms[name] = norm ** 0.5
    for k, v in grad_norms.items():
        print(f"{k} grad norm: {v:.6f}")


def render_base_mesh(base_mesh, camera_batch, lights, img_size):
    rasterizer = make_rasterizer(camera_batch, image_size=img_size, device="cuda")
    base_shader = make_shader(camera_batch, lights, "unlit")
    base_renderer = MeshRenderer(rasterizer=rasterizer, shader=base_shader)

    base_mesh_batch = base_mesh.extend(camera_batch.R.shape[0])
    return base_renderer(base_mesh_batch)


def render_neural_mesh(camera_batch, lights, latent_mesh, model, img_size):
    rasterizer = make_rasterizer(camera_batch, image_size=img_size, device="cuda")
    neural_shader = model.get_frag_shader(camera_batch, lights, img_size)
    neural_renderer = MeshRenderer(rasterizer=rasterizer, shader=neural_shader)
    neural_mesh = latent_mesh.get_mesh()
    neural_mesh_batch = neural_mesh.extend(camera_batch.R.shape[0])
    return neural_renderer(neural_mesh_batch)


def compute_ssim_loss(images, base_images, reduction="none", use_laplacian=True):
    device = images.device
    ssim_loss = SSIMLoss(reduction=reduction).to(device)

    # Permute to (N, C, H, W) and use only RGB channels
    images_rgb = images[..., :3].permute(0, 3, 1, 2)
    base_images_rgb = base_images[..., :3].permute(0, 3, 1, 2)

    # Define 3x3 Laplacian kernel
    if use_laplacian:
        laplacian_kernel = torch.tensor(
            [[0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]],
            dtype=torch.float32, device=device
        ).unsqueeze(0).unsqueeze(0)
        laplacian_kernel = laplacian_kernel.repeat(images_rgb.shape[1], 1, 1, 1)  # For RGB

        # Apply Laplacian filter to both images
        images_rgb = F.conv2d(images_rgb, laplacian_kernel, padding=1, groups=images_rgb.shape[1])
        base_images_rgb = F.conv2d(base_images_rgb, laplacian_kernel, padding=1, groups=base_images_rgb.shape[1])

        images_rgb = images_rgb.abs()
        base_images_rgb = base_images_rgb.abs()

        # Normalize to [0, 1]
        images_rgb = (images_rgb - images_rgb.min()) / (images_rgb.max() - images_rgb.min() + 1e-8)
        base_images_rgb = (base_images_rgb - base_images_rgb.min()) / (base_images_rgb.max() - base_images_rgb.min() + 1e-8)

    return ssim_loss(images_rgb, base_images_rgb)


def train_latent_mesh(source_mesh: MeshData, model: torch.nn.Module, latent_mesh: LatentMesh):
    lights = make_lights("unlit")

    base_mesh = meshdata_to_pytorch3d_mesh(source_mesh)
    
    params = list(model.parameters()) + list(latent_mesh.parameters())
    opt = SOAP(params, lr=1e-3)
    ssim_loss = SSIMLoss().to("cuda")

    model.train()
    
    image_size = 256
    n_cameras = 16

    # fib_cameras = fibonacci_cameras(n_cameras=max_cameras)
    cameras = fibonacci_cameras(n_cameras, dist=2.5)
    # cameras = select_coverage_cameras(
    #     n_cameras,
    #     base_mesh,
    #     latent_mesh,
    #     model,
    #     lights,
    #     weight_by_error=False
    # )

    for epoch in range(10000):
        if epoch % 25 == 0 and epoch > 0:
            cameras = select_coverage_cameras(
                n_cameras,
                base_mesh,
                latent_mesh,
                model,
                lights,
                weight_by_error=False
            )

        # if epoch == 5000:
        #     n_cameras = 8
        #     image_size = 512
        # if epoch == 9000:
        #     n_cameras = 4
        #     image_size = 1024

        # if epoch % 500 == 0 and epoch > 0:
        #     print("Reselecting cameras after 500 epochs...")
        #     cameras = select_coverage_cameras(
        #         n_cameras,
        #         base_mesh,
        #         latent_mesh,
        #         model,
        #         lights,
        #         weight_by_error=True,
        #     )

        base_images = render_base_mesh(base_mesh, cameras, lights, image_size)
        images = render_neural_mesh(cameras, lights, latent_mesh, model, image_size)

        base_loss = compute_ssim_loss(images, base_images, reduction="mean", use_laplacian=False)
        laplacian_loss = compute_ssim_loss(images, base_images, reduction="mean", use_laplacian=True)
        color_loss = F.l1_loss(images[..., :3], base_images[..., :3], reduction="mean")

        # all_loss = base_loss + laplacian_loss + color_loss
        # mean_loss = all_loss.mean()
        mean_loss = base_loss
        opt.zero_grad()
        mean_loss.backward()

        opt.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {mean_loss.item():.6f}, Base Loss: {base_loss.mean().item():.6f}, "
                  f"Laplacian Loss: {laplacian_loss.mean().item():.6f}, Color Loss: {color_loss.item():.6f}")
            print_grad_norms(model, latent_mesh)
            os.makedirs("debug_renders", exist_ok=True)
            os.makedirs(f"debug_renders/{source_mesh.name}_epoch{epoch}", exist_ok=True)

            # Save the first image of each batch as PNG using torchvision
            grid_base = make_grid(base_images[..., :3].detach().cpu().permute(0, 3, 1, 2), nrow=8)
            grid_learned = make_grid(images[..., :3].detach().cpu().permute(0, 3, 1, 2), nrow=8)
            save_image(grid_base, f"debug_renders/{source_mesh.name}_epoch{epoch}/base_grid.png")
            save_image(grid_learned, f"debug_renders/{source_mesh.name}_epoch{epoch}/learned_grid.png")

if __name__ == "__main__":
    # Load the mesh data
    mesh_data = load_fbx_to_meshdata("static/Moon.fbx")
    latent_dim = 12
    for mesh in mesh_data:
        model = NikaNet(latent_dim=latent_dim).to("cuda")
        latent_mesh = LatentMesh(mesh, latent_dim=latent_dim).to("cuda")
        print(f"Training latent mesh for: {mesh.name}")
        train_latent_mesh(mesh, model, latent_mesh)
