import torch
import torch.nn as nn

from utils import SineLayer, Tanh01, compute_positional_encodings, kernel_expand

class VFXSpiralNetDecoder(nn.Module):
    def __init__(
            self,
            latent_dim=4,
            pos_channels=6,
            control_channels=2,
            output_channels=4,
            hidden_dim=64,
            prefilm_dims=16,
        ):
        super().__init__()
        self.pos_channels = pos_channels
        input_dim = latent_dim + pos_channels

        self.control_embed = nn.Sequential(
            nn.Linear(control_channels, prefilm_dims),
            nn.ReLU(),
        )
        self.time_embed = nn.Sequential(
            nn.Linear(6, prefilm_dims),  # time is a scalar (normalized frame index)
            nn.ReLU(),
        )

        self.pos_embed = nn.Sequential(
            nn.Linear(12, prefilm_dims),  # Don't use positional encodings in Film layer
            nn.ReLU(),
        )

        self.film = nn.Linear(3 * prefilm_dims, hidden_dim * 2)
        
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_channels),
            nn.Sigmoid()
        ])
    
    def forward(self, latent, raw_pos, control, return_hidden_layer=None):
        H, W, C = latent.shape
        latent_flat = latent.view(-1, C)  # Flatten latent to [H*W, C]
        linear_indices = raw_pos[:, 1] * W + raw_pos[:, 0]
        indexed_latent = latent_flat[linear_indices]
        pos_enc = compute_positional_encodings(raw_pos, H, W, self.pos_channels)
        main_input = torch.concat([indexed_latent, pos_enc], dim=1)  # [B, LATENT_IMAGE_CHANNELS + POS_CHANNELS]


        control_feat = self.control_embed(control)
        x = pos_enc[:, 0:1]  # normalized x in [0, 1]
        y = pos_enc[:, 1:2]  # normalized y in [0, 1]

        # Scale to [0, 2π]
        x = x * 2 * torch.pi
        y = y * 2 * torch.pi

        # Spiral-style position embedding
        spiral_pos = torch.cat([
            torch.sin(x), torch.cos(x),
            torch.sin(2 * x), torch.cos(2 * x),
            torch.sin(3 * x), torch.cos(3 * x),
            torch.sin(y), torch.cos(y),
            torch.sin(2 * y), torch.cos(2 * y),
            torch.sin(3 * y), torch.cos(3 * y)
        ], dim=-1)  # [B, 12]

        pos_feat = self.pos_embed(spiral_pos)
        t = control[:, 0:1] * 2 * torch.pi  # map to [0, 2π]

        # Spiral embedding: sin/cos for base cycle, sin/cos of harmonic for texture
        spiral_time = torch.cat([
            torch.sin(t), torch.cos(t),
            torch.sin(2 * t), torch.cos(2 * t),
            torch.sin(3 * t), torch.cos(3 * t)
        ], dim=-1)  # [B, 6]

        time_feat = self.time_embed(spiral_time)

        film_input = torch.cat([control_feat, pos_feat, time_feat], dim=-1)  # [B, 3 * pref_dim]

        outputs = main_input
        for i, layer in enumerate(self.layers):
            if i == 1:  # First layer, apply film
                gamma, beta = self.film(film_input).chunk(2, dim=-1)
                outputs = layer((gamma * outputs) + beta)
            else:    
                outputs = layer(outputs)
            if return_hidden_layer is not None and i == return_hidden_layer:
                return outputs
        return outputs


class VFXNetContextDecoder(nn.Module):
    def __init__(
            self,
            latent_dim=4,
            pos_channels=6,
            control_channels=2,
            output_channels=4,
            hidden_dim=64,
            prefilm_dims=16,
        ):
        super().__init__()
        self.pos_channels = pos_channels
        input_dim = latent_dim + pos_channels

        self.control_embed = nn.Sequential(
            nn.Linear(control_channels, prefilm_dims),
            nn.ReLU(),
        )
        self.time_embed = nn.Sequential(
            nn.Linear(1, prefilm_dims),  # time is a scalar (normalized frame index)
            nn.ReLU(),
        )
        self.pos_embed = nn.Sequential(
            nn.Linear(2, prefilm_dims),  # Don't use positional encodings in Film layer
            nn.ReLU(),
        )

        self.film = nn.Linear(3 * prefilm_dims, hidden_dim * 2)
        
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_channels),
            nn.Sigmoid()
        ])
    
    def forward(self, latent, raw_pos, control, return_hidden_layer=None):
        H, W, C = latent.shape
        latent_flat = latent.view(-1, C)  # Flatten latent to [H*W, C]
        linear_indices = raw_pos[:, 1] * W + raw_pos[:, 0]
        indexed_latent = latent_flat[linear_indices]
        pos_enc = compute_positional_encodings(raw_pos, H, W, self.pos_channels)
        x = torch.concat([indexed_latent, pos_enc], dim=1)  # [B, LATENT_IMAGE_CHANNELS + POS_CHANNELS]


        control_feat = self.control_embed(control)
        pos_feat = self.pos_embed(pos_enc[:, 0:2])  # First two channels are x and y
        time_feat = self.time_embed(control[:, 0:1])  # Assuming time is the first control channel
        film_input = torch.cat([control_feat, pos_feat, time_feat], dim=-1)  # [B, 3 * pref_dim]

        outputs = x
        for i, layer in enumerate(self.layers):
            if i == 1:  # First layer, apply film
                gamma, beta = self.film(film_input).chunk(2, dim=-1)
                outputs = layer((gamma * outputs) + beta)
            else:    
                outputs = layer(outputs)
            if return_hidden_layer is not None and i == return_hidden_layer:
                return outputs
        return outputs


class VFXNetContextSirenDecoder(VFXNetContextDecoder):
    def __init__(
        self,
        latent_dim=4,
        pos_channels=6,
        control_channels=2,
        output_channels=4,
        hidden_dim=64,
        prefilm_dims=16,
    ):
        super().__init__(latent_dim, pos_channels, control_channels, output_channels, hidden_dim, prefilm_dims)
        self.layers = nn.ModuleList([
            SineLayer(self.input_dim, self.first_hidden_dim, is_first=True),
            SineLayer(self.first_hidden_dim, 2 * self.first_hidden_dim),
            SineLayer(2 * self.first_hidden_dim, self.first_hidden_dim),
            nn.Linear(self.first_hidden_dim, output_channels),
            Tanh01()
        ])


class VFXNetPixelDecoder(nn.Module):
    def __init__(
            self,
            latent_dim=4,
            pos_channels=6,
            control_channels=2,
            output_channels=4,
            hidden_dim=64,
        ):
        super().__init__()
        self.pos_channels = pos_channels
        self.film = nn.Linear(control_channels, hidden_dim * 2)
        self.layers = nn.ModuleList([
            nn.Linear(latent_dim + pos_channels + control_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_channels),
            nn.Sigmoid()
        ])

    def forward(self, latent, raw_pos, control, return_hidden_layer=None):
        H, W, C = latent.shape
        latent_flat = latent.view(-1, C)
        linear_indices = raw_pos[:, 1] * W + raw_pos[:, 0]
        indexed_latent = latent_flat[linear_indices]
        pos_enc = compute_positional_encodings(raw_pos, H, W, self.pos_channels)
        x = torch.cat([indexed_latent, pos_enc, control], dim=1)

        outputs = x
        for i, layer in enumerate(self.layers):
            if i == 1:  # First layer, apply film
                gamma, beta = self.film(control).chunk(2, dim=-1)
                outputs = layer((gamma * outputs) + beta)
            else:    
                outputs = layer(outputs)
            if return_hidden_layer is not None and i == return_hidden_layer:
                return outputs
        return outputs


class VFXNetSirenDecoder(VFXNetPixelDecoder):
    def __init__(
        self,
        latent_dim=4,
        pos_channels=6,
        control_channels=2,
        output_channels=4,
        hidden_dim=64,
    ):
        super().__init__(latent_dim, pos_channels, control_channels, output_channels, hidden_dim)
        self.layers = nn.ModuleList([
            SineLayer(latent_dim + pos_channels + control_channels, hidden_dim, is_first=True),
            SineLayer(hidden_dim, 2 * hidden_dim),
            SineLayer(2 * hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_channels),
            Tanh01()
        ])


class VFXNetPatchDecoder(nn.Module):
    def __init__(
        self,
        latent_dim=4,
        pos_channels=6,
        control_channels=2,
        output_channels=4,
        hidden_dim=64,
        kernel_size=3,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.pos_channels = pos_channels
        input_channels = latent_dim * (kernel_size ** 2) + pos_channels * (kernel_size ** 2) + control_channels * (kernel_size ** 2)

        self.layers = nn.ModuleList([
            nn.Linear(input_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_channels),
            nn.Sigmoid()
        ])
    
    def _forward(self, x, return_hidden_layer=None):
        output = x
        for i, layer in enumerate(self.layers):
            output = layer(output)
            if return_hidden_layer is not None and i == return_hidden_layer:
                return output
        return output
    
    def forward(self, latent, raw_pos, control):
        """
        latent:   [B, H, W, latent_dim]
        pos_enc:  [B, H, W, pos_dim]
        control:  [B, control_dim]
        """
        H, W, C = latent.shape
        expanded_pos = kernel_expand(raw_pos, H, W, kernel_size=self.kernel_size)

        latent_flat = latent.view(-1, latent.shape[-1])
        linear_indices = expanded_pos[..., 1] * W + expanded_pos[..., 0]
        latent_values = latent_flat[linear_indices]
        pos_enc = compute_positional_encodings(expanded_pos, H, W, self.pos_channels)

        control_expanded = control.unsqueeze(1).expand(-1, self.kernel_size ** 2, -1)

        patches = torch.cat([latent_values, pos_enc, control_expanded], dim=-1)
        unrolled = patches.view(patches.shape[0], -1)  # Unroll to [B, kernel**2 * Something]
        return self._forward(unrolled)
