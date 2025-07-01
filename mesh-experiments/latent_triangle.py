import torch
import torch.nn as nn
import torch.nn.functional as F

from asset_rep import TextureData
from soap import SOAP
from PIL import Image
from scipy.spatial import Delaunay
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.ops import interpolate_face_attributes

from vert_net import make_uvspace_renderer

class LatentPointEncoder(torch.nn.Module):
    def __init__(self, image_channels=4, n_points=10000, latent_dim=12):
        super().__init__()
        self.n_points = n_points
        self.backbone = nn.Sequential(
            nn.Conv2d(image_channels, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.head = nn.Sequential(
            nn.Linear(128, n_points * (2 + 1 + latent_dim)),
            nn.Sigmoid()
        )
    
    def forward(self, image):
        B = image.shape[0]
        x = self.backbone(image).view(B, -1)
        out = self.head(x).view(B, self.n_points, -1)
        return out[..., :2], out[..., 2], out[..., 3:]  # uv, importance, latent


class ColorDecoder(nn.Module):
    def __init__(self, latent_dim=12, hidden_dim=64, output_dim=4):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # Output RGBA
        )
    
    def forward(self, latent):
        return self.backbone(latent)


def uv_range_loss(uv):
    # uv: [B, N, 2] in [0, 1]
    min_uv = uv.min(dim=1).values  # [B, 2]
    max_uv = uv.max(dim=1).values  # [B, 2]
    coverage = max_uv - min_uv     # [B, 2]
    loss = ((1.0 - coverage)**2).mean()  # Want range to be 1.0 in both dims
    return loss


def gmean_sparsity_loss(importances):
    # importances: [B, N]
    # We want the geometric mean of importances to be close to 0.0
    eps = 1e-6
    log_importances = torch.log(importances + eps)  # Avoid log(0)
    gmean = torch.exp(log_importances.mean(dim=1))  # Geometric mean across points
    return gmean.mean()  # Want gmean to be close to 0.0


def predictibility_loss(uv, importance, latent, temp):
    dist = torch.cdist(uv, uv, p=2)  # [N, N]
    weights = torch.exp(-temp * dist**2)  # [B, N, N]

    # Ensure weights are not self-referential
    eye = torch.eye(weights.shape[1], device=weights.device).unsqueeze(0)  # [1, N, N]
    weights = weights * (1.0 - eye)

    weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-8)  # Normalize weights

    pred = weights @ latent  # [B, N, D]
    err = ((latent - pred)**2).sum(dim=2)
    weight = 1.0 - importance.unsqueeze(2)  # [B, N, 1]
    loss_predict = (weight * err).mean(dim=1)  # [B]

    return loss_predict


def vert_recon(source_tex, encoder, decoder, H=512, W=512, thresh=None, train=False, pass_latents=False):

    if train is False:
        encoder.eval()
        decoder.eval()

    def inner_recon():
        uvs, importances, latents = encoder(source_tex)
        print(f"Importance stats: min={importances.min().item()}, max={importances.max().item()}, mean={importances.mean().item()}")
        if thresh is not None:
            threshold_mask = importances > thresh
            num_selected = threshold_mask.sum().item()
            print(f"Thresholding: {num_selected} points above threshold {thresh}")
            if num_selected < 1000:
                # Pick top-100 importances
                topk = min(1000, importances.shape[1])
                topk_indices = torch.topk(importances, topk, dim=1).indices.squeeze(0)
                threshold_mask = torch.zeros_like(importances, dtype=torch.bool)
                threshold_mask[0, topk_indices] = True
                print(f"Selected top-{topk} points instead.")
            uvs = uvs[threshold_mask]
            latents = latents[threshold_mask]

        if pass_latents:
            vert_data = latents.squeeze(0)
        else:
            vert_data = decoder(latents).squeeze(0)  # colors

        uvs = uvs.squeeze(0)  # (V, 2)

        # Convert UVs from [0, 1] to [-1, 1] for rendering
        uvs = uvs * 2.0 - 1.0
        z = torch.zeros(uvs.shape[0], 1, device=uvs.device, dtype=uvs.dtype)
        verts_3d = torch.cat([uvs, z], dim=-1)  # (V, 3)

        tri = Delaunay(uvs.detach().cpu().numpy())
        faces = torch.tensor(tri.simplices, dtype=torch.long, device=uvs.device)

        mesh = Meshes(
            verts=[verts_3d],
            faces=[faces],
            textures=TexturesVertex(verts_features=[vert_data])
        )

        if pass_latents:
            renderer = make_uvspace_renderer(H, W, device=uvs.device, shader=DecoderShader(decoder))
        else:
            renderer = make_uvspace_renderer(H, W, device=uvs.device)
        return renderer(mesh)

    if not train:
        with torch.no_grad():
            rendered = inner_recon()
    else:
        rendered = inner_recon()

    return rendered[0]


class DecoderShader(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, fragments, meshes, **kwargs):
        # Get latent features per vertex
        verts_latent = meshes.textures.verts_features_packed()  # [V, D]
        faces = meshes.faces_packed()                           # [F, 3]
        face_latents = verts_latent[faces]                      # [F, 3, D]

        # Interpolate per-pixel latent using barycentric coords
        bary_coords = fragments.bary_coords                     # [N, H, W, K, 3]
        pix_to_face = fragments.pix_to_face                     # [N, H, W, K]

        pixel_latent = interpolate_face_attributes(
            pix_to_face, bary_coords, face_latents              # [N, H, W, K, D]
        )

        # Apply MLP shader to predicted latent
        shaded_color = self.decoder(pixel_latent)                   # [N, H, W, K, 4]

        # Return first sample per pixel (K=1)
        return shaded_color[..., 0, :]
        

def debug_image(model, source_tex):
    model.eval()
    with torch.no_grad():
        output = model(source_tex)
        output = (output * 255).clamp(0, 255).to(torch.uint8)
        return output


def pixel_train_latent(encoder, decoder, source_tex, epochs=10000):
    print(f"Source texture shape: {source_tex.shape}")
    source_tex = source_tex.unsqueeze(0).permute(0, 3, 1, 2).to(device)
    encoder.train()
    decoder.train()
    opt = SOAP(list(encoder.parameters()) + list(decoder.parameters()), lr=0.01, weight_decay=1e-4)

    for epoch in range(epochs):
        opt.zero_grad()
        uv, importance, latent = encoder(source_tex)
        struct_loss = predictibility_loss(uv, importance, latent, temp=0.05).mean()
        sparsity_loss = gmean_sparsity_loss(importance)
        struct_loss = struct_loss * 0.75 + sparsity_loss * 0.25

        spread_loss = uv_range_loss(uv)
        color = decoder(latent)

        uv_grid = uv.clone() * 2.0 - 1.0  # [1, 10000, 2]
        uv_grid = uv_grid.view(1, 10000, 1, 2)  # [1, 10000, 1, 2]
        target_colors = F.grid_sample(source_tex, uv_grid, mode='bilinear', align_corners=True)  # [1, 4, 10000, 1]
        target_colors = target_colors.squeeze(3).permute(0, 2, 1)  # [1, 10000, 4]

        pixel_loss = F.l1_loss(color, target_colors)

        render = vert_recon(source_tex, encoder, decoder, H=2048, W=2048, thresh=0.5, train=True, pass_latents=True)

        render_reshaped = render.permute(2, 0, 1).unsqueeze(0)
        image_loss = F.l1_loss(render_reshaped, source_tex)
        loss = image_loss
        loss.backward()
        opt.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            image = vert_recon(source_tex, encoder, decoder, thresh=0.5)
            image = (image * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            im = Image.fromarray(image)
            im.save(f"neural_shader/output_epoch_{epoch}.png")

            # Save render_reshaped as image
            render_img = render_reshaped.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            render_img = (render_img * 255).clip(0, 255).astype('uint8')
            render_im = Image.fromarray(render_img)
            render_im.save(f"neural_shader/render_reshaped_epoch_{epoch}.png")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test = TextureData.load("static/T_Horse_Body_M_D_WhS.tga")
    encoder = LatentPointEncoder(image_channels=4, n_points=10000, latent_dim=16).to(device)
    decoder = ColorDecoder(latent_dim=16, hidden_dim=64).to(device)

    pixel_train_latent(
        encoder,
        decoder,
        test.image.float(),
        epochs=1000,
    )
