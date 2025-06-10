import torch
import torch.nn as nn

from encoding_utils import SineLayer, Tanh01, kernel_expand, compute_targeted_encodings

class VFXSpiralNetDecoder(nn.Module):
    def __init__(self, **kwargs):
        defaults = {
            "latent_dim": 4,
            "trunk_pos_channels": 0,
            "trunk_pos_scheme": "sinusoidal",
            "trunk_pos_include_raw": True,
            "trunk_pos_include_norm": True,
            "trunk_time_channels": 0,
            "trunk_time_scheme": "sinusoidal",
            "trunk_time_include_raw": True,
            "trunk_time_include_norm": True,
            "film_time_channels": 8,
            "film_time_scheme": "spiral",
            "film_time_include_raw": True,
            "film_time_include_norm": True,
            "film_pos_channels": 16,
            "film_pos_scheme": "spiral",
            "film_pos_include_raw": True,
            "film_pos_include_norm": True,
            "output_channels": 4,
            "hidden_dim": 64,
            "prefilm_dims": 32,
            "apply_film": [1],
        }
        defaults.update(kwargs)
        for key, value in defaults.items():
            setattr(self, key, value)

        super().__init__()
        input_dim = self.latent_dim + self.trunk_pos_channels + self.trunk_time_channels

        if self.prefilm_dims > 0:
            if self.film_time_channels > 0:
                self.time_embed = nn.Sequential(
                    nn.Linear(self.film_time_channels, self.prefilm_dims),
                    nn.ReLU(),
                )

            if self.film_pos_channels > 0:
                self.pos_embed = nn.Sequential(
                    nn.Linear(self.film_pos_channels, self.prefilm_dims),
                    nn.ReLU(),
                )

            if self.film_time_channels > 0 or self.film_pos_channels > 0:
                prefilm_input_dim = (self.film_time_channels > 0) * self.prefilm_dims + (self.film_pos_channels > 0) * self.prefilm_dims
                self.film = nn.Linear(prefilm_input_dim, self.hidden_dim * 2)
        else:
            if self.film_time_channels > 0 or self.film_pos_channels > 0:
                prefilm_input_dim = self.film_time_channels + self.film_pos_channels
                self.film = nn.Linear(prefilm_input_dim, self.hidden_dim * 2)

        self.layers = nn.ModuleList([
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_channels),
            nn.Sigmoid()
        ])

    def forward(self, raw_pos, time, latent=None, return_hidden_layer=None):
        if self.latent_dim > 0:
            print("The world has ended")
            x = raw_pos[:, 0:1]
            y = raw_pos[:, 1:2]
            B, H, W, C = latent.shape
            latent_flat = latent.view(B, -1, C)
            flat_idx = y * W + x  # [B]
            indexed_latent = latent_flat[torch.arange(B), flat_idx]

        main_input = []
        if self.latent_dim > 0:
            main_input.append(indexed_latent)
        
        if self.trunk_pos_channels > 0:
            pos_enc = compute_targeted_encodings(
                raw_pos,
                self.trunk_pos_channels,
                scheme=self.trunk_pos_scheme,
                include_raw=self.trunk_pos_include_raw,
                norm_2pi=True,
                include_norm=self.trunk_pos_include_norm,
            )
            main_input.append(pos_enc)

        if self.trunk_time_channels > 0:
            trunk_time = compute_targeted_encodings(
                time,
                self.trunk_time_channels,
                scheme=self.trunk_time_scheme,
                include_raw=self.trunk_time_include_raw,
                norm_2pi=True,
                include_norm=self.trunk_time_include_norm,
            )
            main_input.append(trunk_time)

        main_input = torch.cat(main_input, dim=1)

        film_input = []
        if self.film_pos_channels > 0:
            film_pos = compute_targeted_encodings(
                raw_pos,
                self.film_pos_channels,
                scheme=self.film_pos_scheme,
                include_raw=self.film_pos_include_raw,
                norm_2pi=True,
                include_norm=self.film_pos_include_norm,
            )
            if self.prefilm_dims > 0:
                film_pos = self.pos_embed(film_pos)
            film_input.append(film_pos)

        if self.film_time_channels > 0:
            film_time = compute_targeted_encodings(
                time,
                self.film_time_channels,
                scheme=self.film_time_scheme,
                include_raw=self.film_time_include_raw,
                norm_2pi=True,
                include_norm=self.film_time_include_norm,
            )
            if self.prefilm_dims > 0:
                film_time = self.time_embed(film_time)
            film_input.append(film_time)

        if film_input:
            film_input = torch.cat(film_input, dim=-1)
        else:
            film_input = None

        outputs = main_input
        for i, layer in enumerate(self.layers):
            if i in self.apply_film and film_input is not None:
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
        x_coords = torch.clamp(raw_pos[..., 0:1], 0, W - 1) / W
        y_coords = torch.clamp(raw_pos[..., 1:2], 0, H - 1) / H

        pos_enc = compute_targeted_encodings(
            torch.cat(x_coords, y_coords, dim=1),
            self.pos_channels,
            True,
            scheme="sinusoidal"
        )
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
        x_coords = torch.clamp(raw_pos[..., 0:1], 0, W - 1) / W
        y_coords = torch.clamp(raw_pos[..., 1:2], 0, H - 1) / H

        pos_enc = compute_targeted_encodings(
            torch.cat(x_coords, y_coords, dim=1),
            self.pos_channels,
            True,
            scheme="sinusoidal"
        )
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
        x_coords = torch.clamp(raw_pos[..., 0:1], 0, W - 1) / W
        y_coords = torch.clamp(raw_pos[..., 1:2], 0, H - 1) / H

        pos_enc = compute_targeted_encodings(
            torch.cat(x_coords, y_coords, dim=1),
            self.pos_channels,
            True,
            scheme="sinusoidal"
        )

        control_expanded = control.unsqueeze(1).expand(-1, self.kernel_size ** 2, -1)

        patches = torch.cat([latent_values, pos_enc, control_expanded], dim=-1)
        unrolled = patches.view(patches.shape[0], -1)  # Unroll to [B, kernel**2 * Something]
        return self._forward(unrolled)
