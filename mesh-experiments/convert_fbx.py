import subprocess
import tempfile
import os
import re

import numpy as np
from PIL import Image
import torch
import trimesh
from pytorch3d.io import load_obj
from difflib import SequenceMatcher

from process_lod import compute_geometry_quadrics, compute_color_quadrics
from blender_scripts import convert_fbx_to_obj, list_mesh_details

def load_mesh(obj_path: str):
    verts, faces_idx, aux = load_obj(obj_path)
    faces     = faces_idx.verts_idx        # (F,3)
    uvs       = aux.verts_uvs              # (N_uv,2)
    uv_faces  = faces_idx.textures_idx    # (F,3)
    return verts, faces, uvs, uv_faces


def optimize_mesh(obj_path: str, base_dir:str = "static/meshes/", target_faces: int = 1000):
    """
    Load an OBJ mesh, compute LODs, and save them.
    """
    verts, faces, uvs, uv_faces = load_mesh(os.path.join(base_dir, obj_path))
    print(f"Loaded mesh: {verts.shape[0]} vertices, {faces.shape[0]} faces")

    # Compute quadrics for decimation
    quadrics = compute_geometry_quadrics(verts, faces)
    print(f"Computed quadrics for {faces.shape[0]} faces")
    print(f"Quadrics shape: {quadrics[0]}")


class MeshProcessor:
    def __init__(self, obj_name: str, base_dir: str = "static/meshes/"):
        self.obj_path = os.path.join(base_dir, obj_name)
        self.base_dir = base_dir
        self.verts, self.faces, self.uvs, self.uv_faces = load_mesh(self.obj_path)
        print(f"Loaded mesh: {self.verts.shape[0]} vertices, {self.faces.shape[0]} faces")
        self.mapping = load_materials(obj_name)
        print(f"Found {len(self.mapping)} materials in MTL file.")

        loaded = trimesh.load(self.obj_path, process=False)
        if isinstance(loaded, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
        else:
            mesh = loaded
        fm = mesh.visual.face_materials
        if fm is None:
            print("Warning: no per-face materials found; defaulting to 0")
            fm = np.zeros(len(mesh.faces), dtype=np.int32)
        self.face_mats = np.array(fm, dtype=np.int32)

        # if scene.visual.face_materials is not None:
        #     self.face_mats = np.array([
        #         material_to_index.get(mat, 0)  # Default to 0 if material is not found
        #         for mat in scene.visual.face_materials
        #     ], dtype=np.int32)
        # else:
        #     print("Warning: No materials found. Assigning default material ID 0 to all faces.")
        #     self.face_mats = np.zeros(len(self.faces), dtype=np.int32)

        print(f"Loaded {len(self.face_mats)} face materials from scene.")

        self.tex_images = {}
        for mtl_name, (actual_name, _) in self.mapping.items():
            img = Image.open(os.path.join(self.base_dir, actual_name)).convert("RGB")
            self.tex_images[mtl_name] = np.array(img, dtype=np.float32)/255.0
        print(f"Loaded {len(self.tex_images)} textures into memory.")

    def _compute_quadrics(self, col_weight: float = 0.5):
        """
        Compute the combined quadrics for geometry and color.
        """
        Q_geom = compute_geometry_quadrics(self.verts, self.faces)
        vcols = self.sample_vertex_colors()
        Q_col = compute_color_quadrics(self.verts, self.faces, vcols)
        Q_all = Q_geom + col_weight * Q_col
        return Q_all

    def compute_edge_costs(self, Q_all: torch.Tensor):
        device = self.verts.device

        F = self.faces                       # (F,3)
        edges = torch.cat([
            F[:, [0,1]],
            F[:, [1,2]],
            F[:, [2,0]],
        ], dim=0)                           # (3F,2)
        edges_u, _ = torch.sort(edges, dim=1)
        # Unique edges
        edges_u    = torch.unique(edges_u, dim=0)  # (E,2)

        Qe = Q_all[edges_u[:,0]] + Q_all[edges_u[:,1]]  # (E,4,4)

        A = Qe[:, :3, :3]                 # (E,3,3)
        b = -Qe[:, :3, 3]                 # (E,3)

        pinvA = torch.linalg.pinv(A)      # Moore–Penrose pseudoinverse
        # v_opt = pinvA @ b  → (E,3)
        v_opt = torch.einsum('eij,ej->ei', pinvA, b)

        ones = torch.ones(edges_u.shape[0], 1, device=device)
        v4   = torch.cat([v_opt, ones], dim=1)   # (E,4)
        costs = torch.einsum('bi,bij,bj->b', v4, Qe, v4)

        return edges_u, costs

    def optimize(self, target_faces: int = 1000, col_weight: float = 0.5):
        Q_all = self._compute_quadrics(col_weight)
        edges_u, costs = self.compute_edge_costs(Q_all)
        print(f"Computed edge costs for {edges_u.shape[0]} edges.")
        

    def sample_uv_colors(self, uv_coords: np.ndarray, mat_ids: np.ndarray):
        """
        uv_coords: (N,2) numpy array of (u,v) in [0,1]
        mat_ids:   (N,) numpy array of ints indexing into self.mapping order
        returns (N,3) numpy RGB
        """
        out = np.zeros((len(uv_coords), 3), dtype=np.float64)
        for i,(u,v) in enumerate(uv_coords):
            mtl_name = list(self.mapping.keys())[ mat_ids[i] ]
            img = self.tex_images.get(mtl_name)
            if img is not None:
                H,W,_ = img.shape
                px = min(int(u*(W-1)), W-1)
                py = min(int((1-v)*(H-1)), H-1)
                out[i] = img[py,px]
            else:
                out[i] = 1.0
        return out
    
    def sample_vertex_colors(self) -> torch.FloatTensor:
        """
        returns per-vertex (V,3) torch.FloatTensor of RGB in [0,1],
        by sampling each face's 3 UVs & averaging into vertices.
        """
        device = self.verts.device
        V = self.verts.shape[0]
        vcols  = torch.zeros((V,3), device=device)
        counts = torch.zeros((V,),   device=device)

        face_mats = torch.from_numpy(self.face_mats).long().to(device)

        for f_idx in range(self.faces.shape[0]):
            mtl_name = list(self.mapping.keys())[ face_mats[f_idx].item() ]
            img = self.tex_images.get(mtl_name)
            if img is None:
                continue
            H,W,_ = img.shape
            for c in range(3):
                vidx = self.faces[f_idx, c].item()
                uvidx = self.uv_faces[f_idx, c].item()
                u,v = self.uvs[uvidx].tolist()
                px = min(int(u*(W-1)), W-1)
                py = min(int((1-v)*(H-1)), H-1)
                color = torch.tensor(img[py,px], device=device)
                vcols[vidx]  += color
                counts[vidx] += 1

        mask = counts>0
        vcols[mask] /= counts[mask].unsqueeze(1)
        vcols[~mask] = 1.0
        return vcols


if __name__ == "__main__":
    list_mesh_details("static/meshes/Horse.fbx")
    # convert_fbx_to_obj(
    #     "static/meshes/Horse.fbx",
    #     "static/meshes/Horse.obj"
    # )

    # horse = MeshProcessor("Horse.obj")
    # horse.optimize(target_faces=1000)
