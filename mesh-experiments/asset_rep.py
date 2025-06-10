import os
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from difflib import SequenceMatcher

from PIL import Image
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch.nn import functional as F
import open3d as o3d


@dataclass
class TextureData:
    name: str                       # e.g. "T_Horse_Body_M_D_WhS.tga"
    image: torch.Tensor             # H×W×4 float64 in [0,1], stored on GPU
    width: int
    height: int

    @classmethod
    def load(cls, filepath: str):
        from PIL import Image
        name = os.path.basename(filepath)
        img = Image.open(filepath)
        
        # Ensure the image is in RGBA format (add alpha channel if needed)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        
        arr = np.array(img, dtype=np.float64)
        if arr.max() < 20:
            raise ValueError(f"Image {name} appears to be already normalized or in an unexpected format (max value: {arr.max()}).")
        arr = arr / 255.0
        tensor = torch.tensor(arr, dtype=torch.float64).to('cuda' if torch.cuda.is_available() else 'cpu')
        return cls(name=name, image=tensor, width=tensor.shape[1], height=tensor.shape[0])

class MTLLib:
    def __init__(self, mtl_path: str, valid_types: List[str] = [".tga", ".png"]):
        self.mtl_path = mtl_path
        self.name = os.path.basename(mtl_path)
        self.valid_types = valid_types
        self.materials = self._parse_mtl(mtl_path)
        print(self.name)

    def load_texture_array(self) -> List[TextureData]:
        textures = []
        index = 0
        for mat_name, mat_props in self.materials.items():
            if "mat_path" in mat_props and mat_props["mtl_index"] == index:
                tex_path = mat_props["mat_path"]
                if os.path.exists(tex_path):
                    img = TextureData.load(tex_path)
                    textures.append(img.image)
                    index += 1
                else:
                    print(f"Warning: Texture '{tex_path}' for material '{mat_name}' not found.")
        return textures

    def _parse_mtl(self, file_path):
        """
        Parse a .mtl file and return a dictionary of materials and their properties.
        """
        materials = {}
        current_material = None

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):  # Skip comments and empty lines
                    continue

                tokens = line.split()
                key = tokens[0]
                mtl_index = 0

                if key == 'newmtl':  # Start a new material
                    current_material = tokens[1]
                    materials[current_material] = {"mtl_index": mtl_index}
                    mtl_index += 1
                elif current_material is not None:
                    if key in {'Ka', 'Kd', 'Ks', 'Ke'}:  # RGB values
                        materials[current_material][key] = list(map(float, tokens[1:4]))
                    elif key == 'Ns':  # Specular exponent
                        materials[current_material][key] = float(tokens[1])
                    elif key == 'Ni':  # Index of refraction
                        materials[current_material][key] = float(tokens[1])
                    elif key == 'd':  # Transparency
                        materials[current_material][key] = float(tokens[1])
                    elif key == 'illum':  # Illumination model
                        materials[current_material][key] = int(tokens[1])
                    elif key.startswith('map_'):  # Texture maps
                        materials[current_material][key] = tokens[1]
                    else:
                        # Handle unknown keys
                        materials[current_material][key] = tokens[1:]

            def _find_actual_textures(tex_dir: str) -> List[str]:
                return [
                    os.path.join(tex_dir, fn)
                    for fn in os.listdir(tex_dir)
                    if any(fn.lower().endswith(ext) for ext in self.valid_types)
                ]
        local_textures = _find_actual_textures(os.path.dirname(file_path))
        mtl_names = list(material["map_Kd"] for material in materials.values() if "map_Kd" in material)
        mapping = self._match_textures(mtl_names, local_textures)
        for material in materials:
            if "map_Kd" in materials[material]:
                tex_name = materials[material]["map_Kd"]
                if tex_name in mapping:
                    materials[material]["mat_path"] = mapping[tex_name]
                else:
                    print(f"Warning: Texture '{tex_name}' for material '{material}' not found in local textures.")
            else:
                print(f"Warning: Material '{material}' does not have a diffuse texture (map_Kd).")
        return materials
        
    def _match_textures(self, mtl_names: List[str], actual_texs: List[str], min_ratio: float = 0.3) -> Dict[str, str]:
        # compute all pairwise ratios
        scores = []
        for m in mtl_names:
            m_base = os.path.splitext(m)[0]
            for a in actual_texs:
                a_base = os.path.splitext(a)[0]
                ratio = SequenceMatcher(None, m_base, a_base).ratio()
                scores.append((ratio, m, a))
        # sort descending
        scores.sort(key=lambda x: x[0], reverse=True)

        mapping = {}
        used_m = set()
        used_a = set()

        for ratio, m, a in scores:
            if m in used_m or a in used_a:
                continue
            if ratio > min_ratio:
                mapping[m] = a
                used_m.add(m)
                used_a.add(a)

        return mapping


@dataclass
class MeshData:
    name: str
    # By vertex
    positions: torch.FloatTensor  # (N,3) vertex positions
    polyvert_attrs: Optional[torch.LongTensor] = None # (M, 1-4) index into positions, then uvs, normals, colors
    triangles: Optional[torch.LongTensor] = None  # (F,3) triangle indices into polyverts

    bone_names: Optional[List[str]] = None # List of bone names. bone_indices will map to this list.
    bone_indices: Optional[torch.LongTensor] = None  # (N, 4) per-vertex bone indices
    bone_weights: Optional[torch.FloatTensor] = None  # (N, 4) per-vertex bone weights

    # By polygon-vertex
    uvs: Optional[torch.FloatTensor] = None  # (M,2) UV coordinates
    normals: Optional[torch.FloatTensor] = None  # (M,3) vertex normals
    colors: Optional[torch.FloatTensor] = None  # (M,4) vertex colors

    base_texture_name: Optional[str] = ""
    mapped_texture: Optional[TextureData] = None  # Mapped texture data if available

    def __post_init__(self):
        if self.polyvert_attrs is None:
            # Create polyvert_attrs with only position indices
            self.polyvert_attrs = torch.zeros((self.positions.shape[0], 4), dtype=torch.long, device=self.positions.device)
            pos_length = self.positions.shape[0]
            if self.positions is not None:
                self.polyvert_attrs[:, 0] = torch.arange(pos_length, dtype=torch.long, device=self.positions.device)
            if self.uvs is not None:
                if self.uvs.shape[0] != pos_length:
                    raise ValueError(f"UVs length {self.uvs.shape[0]} does not match positions length {pos_length}. Please provide polyvert_attrs manually to resolve.")
                self.polyvert_attrs[:, 1] = torch.arange(pos_length, dtype=torch.long, device=self.positions.device)
            if self.normals is not None:
                if self.normals.shape[0] != pos_length:
                    raise ValueError(f"Normals length {self.normals.shape[0]} does not match positions length {pos_length}. Please provide polyvert_attrs manually to resolve.")
                self.polyvert_attrs[:, 2] = torch.arange(pos_length, dtype=torch.long, device=self.positions.device)
            if self.colors is not None:
                if self.colors.shape[0] != pos_length:
                    raise ValueError(f"Colors length {self.colors.shape[0]} does not match positions length {pos_length}. Please provide polyvert_attrs manually to resolve.")
                self.polyvert_attrs[:, 3] = torch.arange(pos_length, dtype=torch.long, device=self.positions.device)
        if self.colors is not None:
            print(f"Initial colors shape: {self.colors.shape}")
            if self.colors.shape[1] == 3:
                # If colors are RGB, convert to RGBA by adding alpha channel
                self.colors = torch.cat([self.colors, torch.ones((self.colors.shape[0], 1), device=self.colors.device)], dim=1)
                print(f"Self colors shape: {self.colors.shape}")
            max_val = self.colors.max()
            if max_val > 150 and max_val <= 255:  # Arbitrary threshold for 8-bit color
                self.colors = self.colors / 255.0
            elif max_val > 1.0 and max_val <= 150:
                raise ValueError("Colors appear to be in an unexpected range (e.g., EXR values).")
        if self.uvs is None:
            print("Warning: UVs are not provided. Setting the 1st index of polyvert_attrs to -1.")
            if self.polyvert_attrs is not None:
                self.polyvert_attrs[:, 1] = -1
        if self.normals is None:
            print("Warning: Normals are not provided. Setting the 2nd index of polyvert_attrs to -1.")
            if self.polyvert_attrs is not None:
                self.polyvert_attrs[:, 2] = -1
        if self.colors is None:
            print("Warning: Colors are not provided. Setting the 3rd index of polyvert_attrs to -1.")
            if self.polyvert_attrs is not None:
                self.polyvert_attrs[:, 3] = -1
        # self.dedupe_attrs()
        self.validate()
        self.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    def dedupe_attrs(self):
        """
        Remove duplicate attributes from positions, uvs, normals, and colors.
        Updates polyvert_attrs to point to unique entries.
        """
        unique_positions, pos_indices = torch.unique(self.positions, dim=0, return_inverse=True)
        self.positions = unique_positions
        self.polyvert_attrs[:, 0] = pos_indices[self.polyvert_attrs[:, 0]]

        if self.uvs is not None:
            unique_uvs, uv_indices = torch.unique(self.uvs, dim=0, return_inverse=True)
            self.uvs = unique_uvs
            self.polyvert_attrs[:, 1] = uv_indices[self.polyvert_attrs[:, 1]]

        if self.normals is not None:
            unique_normals, normal_indices = torch.unique(self.normals, dim=0, return_inverse=True)
            self.normals = unique_normals
            self.polyvert_attrs[:, 2] = normal_indices[self.polyvert_attrs[:, 2]]

        if self.colors is not None:
            unique_colors, color_indices = torch.unique(self.colors, dim=0, return_inverse=True)
            self.colors = unique_colors
            self.polyvert_attrs[:, 3] = color_indices[self.polyvert_attrs[:, 3]]
        print(f"Deduplicated mesh '{self.name}': {self.positions.shape[0]} positions, "
              f"{self.uvs.shape[0] if self.uvs is not None else 0} uvs, "
              f"{self.normals.shape[0] if self.normals is not None else 0} normals, "
              f"{self.colors.shape[0] if self.colors is not None else 0} colors.")

    def validate(self):
        print(f"Number of normals: {self.normals.shape[0] if self.normals is not None else 0}")
        print(f"number of referenced normals: {self.polyvert_attrs[:, 2].max().item() if self.normals is not None else -1}")
        N_pos = self.positions.shape[0]

        if self.bone_indices is not None:
            assert self.bone_indices.shape == (N_pos, 4), \
                f"bone_indices shape {self.bone_indices.shape} must match (N, 4), N={N_pos}"
        if self.bone_weights is not None:
            assert self.bone_weights.shape == (N_pos, 4), \
                f"bone_weights shape {self.bone_weights.shape} must match (N, 4), N={N_pos}"

        if self.bone_names is not None:
            max_bone_index = self.bone_indices.max().item()
            assert max_bone_index < len(self.bone_names), \
                f"bone_indices contain {max_bone_index}, but bone_names has only {len(self.bone_names)} entries"
        if self.uvs is not None:
            assert self.uvs.shape[1] == 2, f"UVs must be (M, 2), got {self.uvs.shape}"
        if self.normals is not None:
            assert self.normals.shape[1] == 3, f"Normals must be (M, 3), got {self.normals.shape}"
        if self.colors is not None:
            assert self.colors.shape[1] == 4, f"Colors must be (M, 4), got {self.colors.shape}"

        if self.polyvert_attrs is not None:
            assert self.polyvert_attrs.shape[1] == 4, f"polyvert_attrs must be (M, 4), got {self.polyvert_attrs.shape}"
            if self.uvs is not None:
                assert self.polyvert_attrs[:, 1].max().item() < self.uvs.shape[0], \
                f"Second index in polyvert_attrs must be within range of uvs (0 to {self.uvs.shape[0] - 1})"
            else:
                assert (self.polyvert_attrs[:, 1] == -1).all(), \
                "Second index in polyvert_attrs must be -1 when uvs are not provided"

            if self.normals is not None:
                assert self.polyvert_attrs[:, 2].max().item() < self.normals.shape[0], \
                f"Third index in polyvert_attrs must be within range of normals (0 to {self.normals.shape[0] - 1})"
            else:
                assert (self.polyvert_attrs[:, 2] == -1).all(), \
                "Third index in polyvert_attrs must be -1 when normals are not provided"

            if self.colors is not None:
                assert self.polyvert_attrs[:, 3].max().item() < self.colors.shape[0], \
                f"Fourth index in polyvert_attrs must be within range of colors (0 to {self.colors.shape[0] - 1})"
            else:
                assert (self.polyvert_attrs[:, 3] == -1).all(), \
                "Fourth index in polyvert_attrs must be -1 when colors are not provided"

        if self.triangles is not None:
            assert self.triangles.shape[1] == 3, f"triangles must be (F, 3), got {self.triangles.shape}"
            assert self.triangles.max().item() < self.polyvert_attrs.shape[0], \
                f"Triangle index out of range for polygon-verts"
        print(f"MeshData '{self.name}' validated successfully: {N_pos} vertices")

    def merge_positions(self, pi: int, pj: int, v_opt: torch.Tensor, k_weights: int = 4, validate:bool = False) -> int:
        device = self.positions.device
        v_opt = v_opt.to(device)
        v_i = self.positions[pi]
        v_j = self.positions[pj]

        # Compute relative weighting based on proximity
        d_i = (v_opt - v_i).norm()
        d_j = (v_opt - v_j).norm()
        w_i = d_j / (d_i + d_j + 1e-8)
        w_j = 1.0 - w_i
        
        bone_ids = torch.cat([self.bone_indices[pi], self.bone_indices[pj]])  # [8]
        bone_wts = torch.cat([
            self.bone_weights[pi] * w_i,
            self.bone_weights[pj] * w_j
        ])  # [8]

        # Merge duplicate bone ids
        unique_ids, inverse = torch.unique(bone_ids, return_inverse=True)
        summed_wts = torch.zeros_like(unique_ids, dtype=torch.float64, device=device)
        summed_wts.scatter_add_(0, inverse, bone_wts)

        # Top-k and normalize
        top_wts, top_idx = torch.topk(summed_wts, k=min(k_weights, summed_wts.numel()))
        top_ids = unique_ids[top_idx]
        top_wts = top_wts / top_wts.sum().clamp(min=1e-8)

        # Pad if fewer than k
        if top_ids.shape[0] < k_weights:
            pad_len = k_weights - top_ids.shape[0]
            top_ids = torch.cat([top_ids, torch.zeros(pad_len, dtype=top_ids.dtype, device=device)])
            top_wts = torch.cat([top_wts, torch.zeros(pad_len, dtype=top_wts.dtype, device=device)])
        self.positions = torch.cat([self.positions, v_opt.unsqueeze(0)], dim=0)
        self.bone_indices = torch.cat([self.bone_indices, top_ids.unsqueeze(0)], dim=0)
        self.bone_weights = torch.cat([self.bone_weights, top_wts.unsqueeze(0)], dim=0)
        if validate:
            self.validate()
        return self.positions.shape[0] - 1
    
    def remove_degens(self):
        pos_per_corner = self.polyvert_attrs[self.triangles, 0]
        deg_mask = (
            (pos_per_corner[:, 0] == pos_per_corner[:, 1]) |
            (pos_per_corner[:, 1] == pos_per_corner[:, 2]) |
            (pos_per_corner[:, 2] == pos_per_corner[:, 0])
        )
        # print(f"Removing {deg_mask.sum().item()} degenerate triangles from mesh '{self.name}'")
        self.triangles = self.triangles[~deg_mask]
    
    def get_degens(self, attr: int = 0):
        """
        Returns a list of degenerate triangles based on the specified attribute index.
        A triangle is considered degenerate if any two vertices share the same attribute value.
        """
        attr_per_corner = self.polyvert_attrs[self.triangles, attr]
        return (
            (attr_per_corner[:, 0] == attr_per_corner[:, 1]) |
            (attr_per_corner[:, 1] == attr_per_corner[:, 2]) |
            (attr_per_corner[:, 2] == attr_per_corner[:, 0])
        )
    def remove_orphans(self):
        """
        Remove vertices, UVs, normals, and colors that are not referenced by any triangle.
        This is useful after merging or removing triangles.
        """
        print(f"Starting normals for mesh '{self.name}': {self.normals.shape[0] if self.normals is not None else 0} normals")
        print(f"Starting normal indices for mesh '{self.name}': {self.polyvert_attrs[:, 2].max().item() if self.normals is not None else -1}")
        # Find used indices for each attribute
        used_polyverts = self.triangles.view(-1).unique()
        print(f"Used polyverts: {used_polyverts.shape[0]} unique polygon-vertex indices")

        used_positions = self.polyvert_attrs[used_polyverts, 0].view(-1).unique()
        mask_positions = torch.zeros(self.positions.shape[0], dtype=torch.bool, device=self.positions.device)
        mask_positions[used_positions] = True
        self.positions = self.positions[mask_positions]
        self.bone_indices = self.bone_indices[mask_positions]
        self.bone_weights = self.bone_weights[mask_positions]
        self.polyvert_attrs[:, 0] = torch.searchsorted(used_positions.contiguous(), self.polyvert_attrs[:, 0].contiguous())

        if self.uvs is not None:
            used_uvs = self.polyvert_attrs[used_polyverts, 1].view(-1).unique()
            used_uvs = used_uvs.sort().values  # ensure ascending order

            mask_uvs = torch.zeros(self.uvs.shape[0], dtype=torch.bool, device=self.uvs.device)
            mask_uvs[used_uvs] = True
            self.uvs = self.uvs[mask_uvs]

            device   = self.uvs.device
            full_map = torch.full((self.uvs.shape[0] + mask_uvs.sum().item(),), -1,
                    dtype=torch.long, device=device)
            old_count = (~mask_uvs).sum().item() + mask_uvs.sum().item()
            full_map = torch.full((old_count,), -1, dtype=torch.long, device=device)
            full_map[used_uvs] = torch.arange(used_uvs.numel(), device=device, dtype=torch.long)

            P = self.polyvert_attrs.shape[0]
            new_uv_idxs = torch.full((P,), -1, dtype=torch.long, device=device)

            old_idxs = self.polyvert_attrs[:, 1]          # shape (P,)
            mask_pv_used = torch.zeros(P, dtype=torch.bool, device=device)
            mask_pv_used[used_polyverts] = True

            idxs_to_remap = old_idxs[mask_pv_used]         # maybe contains values not in used_uvs?

            new_uv_idxs[mask_pv_used] = full_map[idxs_to_remap]
            self.polyvert_attrs[:, 1] = new_uv_idxs

            print(f"Ending UVs for mesh '{self.name}': {self.uvs.shape[0]} UVs")
            print(f"Ending UV indices for mesh '{self.name}': {self.polyvert_attrs[:, 1].max().item()}")

        if self.normals is not None:
            used_normals = self.polyvert_attrs[used_polyverts, 2].view(-1).unique()
            used_normals = used_normals.sort().values  # ensure ascending order

            mask_normals = torch.zeros(self.normals.shape[0], dtype=torch.bool, device=self.normals.device)
            mask_normals[used_normals] = True
            self.normals = self.normals[mask_normals]

            device   = self.normals.device
            full_map = torch.full((self.normals.shape[0] + mask_normals.sum().item(),), -1,
                                dtype=torch.long, device=device)
            old_count = (~mask_normals).sum().item() + mask_normals.sum().item()
            full_map = torch.full((old_count,), -1, dtype=torch.long, device=device)
            full_map[used_normals] = torch.arange(used_normals.numel(), device=device, dtype=torch.long)

            P = self.polyvert_attrs.shape[0]
            new_norm_idxs = torch.full((P,), -1, dtype=torch.long, device=device)

            old_idxs = self.polyvert_attrs[:, 2]          # shape (P,)
            mask_pv_used = torch.zeros(P, dtype=torch.bool, device=device)
            mask_pv_used[used_polyverts] = True

            idxs_to_remap = old_idxs[mask_pv_used]         # maybe contains values not in used_normals?

            new_norm_idxs[mask_pv_used] = full_map[idxs_to_remap]
            self.polyvert_attrs[:, 2] = new_norm_idxs

            print(f"Ending normals for mesh '{self.name}': {self.normals.shape[0]} normals")
            print(f"Ending normal indices for mesh '{self.name}': {self.polyvert_attrs[:, 2].max().item()}")


        if self.colors is not None:
            used_colors = self.polyvert_attrs[used_polyverts, 3].view(-1).unique()
            mask_colors = torch.zeros(self.colors.shape[0], dtype=torch.bool, device=self.colors.device)
            mask_colors[used_colors] = True
            self.colors = self.colors[mask_colors]
            self.polyvert_attrs[:, 3] = torch.where(
                self.polyvert_attrs[:, 3] != -1,
                torch.searchsorted(used_colors.contiguous(), self.polyvert_attrs[:, 3].contiguous()),
                -1,
            )
        
        print(f"Ending normals for mesh '{self.name}': {self.normals.shape[0] if self.normals is not None else 0} normals")
        print(f"Ending normal indices for mesh '{self.name}': {self.polyvert_attrs[:, 2].max().item() if self.normals is not None else -1}")
        

    def to(self, device: torch.device):
        """Move all tensors to `device`."""
        self.positions = self.positions.to(device)
        optional_attributes = [
            "polyvert_attrs", "triangles", "bone_indices", "bone_weights",
            "uvs", "normals", "colors"
        ]
        for attr in optional_attributes:
            tensor = getattr(self, attr, None)
            if tensor is not None:
                setattr(self, attr, tensor.to(device))
        return self

    def load_texture(self, texture_path: str):
        if not os.path.exists(texture_path):
            raise FileNotFoundError(f"Texture file '{texture_path}' does not exist.")
        
        self.mapped_texture = TextureData.load(texture_path)


def blue_noise_2d_multiscale(
    grad_norm: torch.Tensor,
    radii: torch.Tensor,
    thresholds: torch.Tensor,
    n_points: int = 100,
    device='cuda'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate blue noise points in 2D space based on gradient magnitude.

    Args:
    grad_norm (torch.Tensor): Gradient magnitude (H, W).
    radii (torch.Tensor): Radii for multi-scale sampling.
    thresholds (torch.Tensor): Thresholds for gradient magnitude.
    n_points (int): Maximum number of points to sample.
    device (str): Device to perform computation on.

    Returns:
    Tuple[torch.Tensor, torch.Tensor]: UV coordinates (N, 2) and importance values (N).
    """
    H, W = grad_norm.shape
    all_xys = []
    all_importances = []
    mask_accum = torch.zeros((H, W), dtype=torch.bool, device=device)
    for r, thresh in zip(radii, thresholds):
        mask = (grad_norm >= thresh) & (~mask_accum)
        if mask.sum() == 0:
            raise ValueError("No valid points found in the gradient magnitude mask.")
        # Create random field in mask
        rand_vals = torch.rand((H, W), device=device) * (grad_norm + 1e-3)
        rand_vals[~mask] = -1e6  # force to negative for pooling
        kernel_size = int(r) * 2 + 1
        pool = F.max_pool2d(rand_vals.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        keep_mask = (rand_vals == pool.squeeze()) & mask
        ys, xs = torch.where(keep_mask)
        xys = torch.stack([xs.float() / W, 1 - ys.float() / H], dim=1)
        importances = grad_norm[ys, xs]
        all_xys.append(xys)
        all_importances.append(importances)
        mask_accum |= keep_mask  # don't double-sample pixels
    all_xys = torch.cat(all_xys, dim=0)
    all_importances = torch.cat(all_importances, dim=0)
    return all_xys, all_importances


def _compute_sobels(texture: TextureData) -> torch.Tensor:
    arr = texture.image.detach().clone()
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr_gray = arr.mean(dim=-1, keepdim=True)  # (H, W, 1)
    arr_gray = arr_gray.permute(2, 0, 1).unsqueeze(0)  # (1, 1, H, W)
    sobel_kernel_x = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], dtype=arr_gray.dtype, device=arr_gray.device)
    sobel_kernel_y = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], dtype=arr_gray.dtype, device=arr_gray.device)
    print(f"types of inputs: arr_gray={type(arr_gray)}, sobel_kernel_x={type(sobel_kernel_x)}, sobel_kernel_y={type(sobel_kernel_y)}")
    grad_x = F.conv2d(arr_gray, sobel_kernel_x, padding=1)
    grad_y = F.conv2d(arr_gray, sobel_kernel_y, padding=1)
    grad = torch.sqrt(grad_x**2 + grad_y**2).squeeze()
    grad_norm = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
    return grad_norm  # (H, W)


def _gaussian_blur(img:torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    # Simple 2D Gaussian blur (channels last)
    size = int(2 * round(3 * sigma) + 1)
    if size % 2 == 0: size += 1
    kernel_1d = torch.arange(-(size//2), size//2+1, device=img.device).float()
    kernel_1d = torch.exp(-0.5 * (kernel_1d / sigma) ** 2)
    kernel_1d /= kernel_1d.sum()
    kernel_2d = kernel_1d[:,None] * kernel_1d[None,:]
    kernel_2d = kernel_2d.expand(img.shape[-3], 1, size, size)  # (C, 1, K, K)
    img = img.unsqueeze(0) if img.dim() == 3 else img  # Add batch dim if needed
    kernel_2d = kernel_2d.to(dtype=torch.float64)  # Ensure kernel is float64
    img = img.to(dtype=torch.float64)  # Ensure input image is float64
    blurred = F.conv2d(img, kernel_2d, padding=size//2, groups=img.shape[1])
    return blurred.squeeze(0)  # Remove batch dim


def compute_dog(texture: TextureData, sigma1:float = 1.0, sigma2:float = 3.0):
    arr = texture.image.detach().clone()
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr_gray = arr.mean(dim=-1, keepdim=True).permute(2, 0, 1).unsqueeze(0)  # (1, 1, H, W)
    blurred1 = _gaussian_blur(arr_gray, sigma1)
    blurred2 = _gaussian_blur(arr_gray, sigma2)
    dog = (blurred1 - blurred2).abs().squeeze()
    dog_norm = (dog - dog.min()) / (dog.max() - dog.min() + 1e-8)
    return dog_norm  # (H, W)


def uvs_to_3d(mesh: 'MeshData', uv_points: torch.Tensor) -> torch.Tensor:
    tri_uv_idxs = mesh.polyvert_attrs[mesh.triangles][:, :, 1]

    tri_uvs = mesh.uvs[tri_uv_idxs]                      # (F,3,2)
    mod_uvs = tri_uvs % 1.0
    A, B, C = mod_uvs[:, 0, :], mod_uvs[:, 1, :], mod_uvs[:, 2, :]  # (F,2)

    e0 = B - A  # (F,2)
    e1 = C - A  # (F,2)
    e2 = uv_points.unsqueeze(1) - A.unsqueeze(0)  # (N, F, 2)
    d00 = (e0 * e0).sum(dim=1)  # (F,)
    d01 = (e0 * e1).sum(dim=1)  # (F,)
    d11 = (e1 * e1).sum(dim=1)  # (F,)
    d20 = (e2 * e0).sum(dim=2)  # (N, F)
    d21 = (e2 * e1).sum(dim=2)  # (N, F)
    denom = d00 * d11 - d01 * d01 + 1e-12  # (F,)

    v = (d11.unsqueeze(0) * d20 - d01.unsqueeze(0) * d21) / denom.unsqueeze(0)  # (N, F)
    w = (d00.unsqueeze(0) * d21 - d01.unsqueeze(0) * d20) / denom.unsqueeze(0)  # (N, F)
    u = 1.0 - v - w  # (N, F)

    inside = (u > 0) & (v > 0) & (w > 0) & (u < 1) & (v < 1) & (w < 1)
    idx_uv, idx_tri = torch.where(inside)
    tri_xyzs = mesh.positions[mesh.polyvert_attrs[mesh.triangles][:, :, 0]]  # (F,3,3)
    return (
        u[idx_uv, idx_tri][:, None] * tri_xyzs[idx_tri, 0, :] +
        v[idx_uv, idx_tri][:, None] * tri_xyzs[idx_tri, 1, :] +
        w[idx_uv, idx_tri][:, None] * tri_xyzs[idx_tri, 2, :]
    )


def _compute_curvature(mesh: 'MeshData') -> torch.Tensor:
    N = mesh.positions.shape[0]
    V = mesh.positions
    triangles = mesh.triangles  # (F, 3)

    # Get all per-triangle corner angles
    verts = V[mesh.polyvert_attrs[triangles, 0]]  # (F, 3, 3)
    A = verts[:, 0, :]
    B = verts[:, 1, :]
    C = verts[:, 2, :]
    BA = B - A
    CA = C - A
    CB = C - B
    AB = A - B
    AC = A - C
    BC = B - C

    def angle(u, v):
        # returns angle at vertex between vectors u, v, safe for degens
        u = u / (u.norm(dim=1, keepdim=True) + 1e-10)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-10)
        cos_angle = (u * v).sum(dim=1).clamp(-1.0, 1.0)
        return torch.acos(cos_angle)

    angles_A = angle(BA, CA)
    angles_B = angle(AB, CB)
    angles_C = angle(AC, BC)

    # For each vertex, sum angles at all corners where it's present
    angles_per_vertex = torch.zeros(N, device=V.device, dtype=torch.float64)
    idx_A = mesh.polyvert_attrs[triangles[:, 0], 0]
    idx_B = mesh.polyvert_attrs[triangles[:, 1], 0]
    idx_C = mesh.polyvert_attrs[triangles[:, 2], 0]
    angles_per_vertex.index_add_(0, idx_A, angles_A)
    angles_per_vertex.index_add_(0, idx_B, angles_B)
    angles_per_vertex.index_add_(0, idx_C, angles_C)

    curvature = 2 * torch.pi - angles_per_vertex
    return curvature.abs()


def sample_curvature(mesh: 'MeshData', top_k: int = 10) -> torch.Tensor:
    curvature = _compute_curvature(mesh)
    threshold = torch.quantile(curvature, 1 - (top_k / 100))
    selected_indices = torch.where(curvature >= threshold)[0]
    return mesh.positions[selected_indices], curvature[selected_indices].unsqueeze(1)


def triangulate_mesh(point_cloud: torch.Tensor, depth: int = 6) -> 'MeshData':
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud.cpu().numpy())
    points = np.asarray(pcd.points)
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)  # k=1 is self, k=2 is nearest neighbor
    mean_spacing, stdev_spacing = dists[:, 1].mean(), dists[:, 1].std()
    print(f"Mean spacing: {mean_spacing}, Stddev spacing: {stdev_spacing}")
    radii = [mean_spacing * f for f in [stdev_spacing, 2 * stdev_spacing, 4 * stdev_spacing]]
    print(f"Using radii for ball pivoting: {radii}")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)[0]
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #     pcd, o3d.utility.DoubleVector(radii)
    # )
    triangles = torch.tensor(np.asarray(mesh.triangles), dtype=torch.long, device=point_cloud.device)
    print(f"Triangulated mesh with {triangles.shape} triangles from point cloud of shape {point_cloud.shape}.")
    positions = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float64, device=point_cloud.device)
    normals = torch.tensor(np.asarray(mesh.vertex_normals), dtype=torch.float64, device=point_cloud.device)
    print(f"Mesh has {positions.shape} vertices and {normals.shape} normals.")
    print(f"Triangle sample: {triangles[:5]}")
    assert positions.shape[0] == normals.shape[0]
    polyvert_attrs = torch.full((positions.shape[0], 4), -1, dtype=torch.long, device=positions.device)

    polyvert_attrs[:, 0] = torch.arange(positions.shape[0], device=positions.device).repeat_interleave(3)[:polyvert_attrs.shape[0]]
    polyvert_attrs[:, 2] = torch.arange(normals.shape[0], device=positions.device).repeat_interleave(3)[:polyvert_attrs.shape[0]]

    new_mesh = MeshData(
        name="triangulated_mesh",
        positions=positions,
        normals=normals,
        polyvert_attrs=polyvert_attrs,
        triangles=triangles
    )
    return new_mesh

def optimize_mesh(mesh: 'MeshData') -> 'MeshData':
    n_tex = 100  # ends up as more
    n_curv = 25
    sobels = _compute_sobels(mesh.mapped_texture)

    dogs = compute_dog(mesh.mapped_texture, sigma1=1.0, sigma2=3.0)
    # Compute radii as a percentage of the texture dimensions
    texture_width, texture_height = mesh.mapped_texture.width, mesh.mapped_texture.height
    min_dim = min(texture_width, texture_height)
    num_radii = 7
    min_pct, max_pct = 0.01, 0.25
    radii = np.geomspace(min_pct, max_pct, num=num_radii)  # [0.01, 0.017, ..., 0.25]
    radii = torch.tensor(radii, device=sobels.device) * min_dim
    radii = radii.round().to(torch.int64)  # Convert to integer pixel radii

    # Compute thresholds based on standard deviations of the sobel gradients
    quantiles = torch.tensor([0.995, 0.99, 0.97, 0.95, 0.92, 0.88, 0.80], dtype=sobels.dtype, device=sobels.device)
    thresholds = torch.quantile(sobels, quantiles)

    print(f"Using radii: {radii.tolist()} and thresholds: {thresholds.tolist()} for blue noise sampling.")

    uvs, _ = blue_noise_2d_multiscale(
        sobels,
        radii=radii,
        thresholds=thresholds,
        n_points=n_tex / 2,
    )
    uv_samples = uvs_to_3d(mesh, uvs)
    print(f"Sampled {uv_samples.shape} points from texture '{mesh.mapped_texture.name}' for optimization.")
    curvature_samples, _ = sample_curvature(mesh, top_k=n_curv)
    print(f"Sampled {curvature_samples.shape} points from mesh '{mesh.name}' based on curvature.")
    all_samples = torch.cat([uv_samples, curvature_samples], dim=0)
    return MeshData(
        name=f"pointcloud_optimized",
        positions=all_samples,
    )
    # return triangulate_mesh(all_samples)
