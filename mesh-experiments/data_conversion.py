import os
from typing import List, Dict
from difflib import SequenceMatcher

import torch
import trimesh
import numpy as np
import fbx
from fbx import FbxManager, FbxImporter, FbxScene, FbxMesh
from PIL import Image
from pytorch3d.renderer import TexturesVertex, TexturesUV
from pytorch3d.structures import Meshes

from asset_rep import MeshData


def load_fbx_to_meshdata(fbx_path: str, normalize: bool = True) -> List[MeshData]:
    manager = FbxManager.Create()
    importer = FbxImporter.Create(manager, "")
    scene = FbxScene.Create(manager, "Scene")

    importer.Initialize(fbx_path, -1, manager.GetIOSettings())
    importer.Import(scene)
    importer.Destroy()

    meshes = []
    fbx_obj = scene.GetRootNode()

    for i in range(fbx_obj.GetChildCount()):
        node = fbx_obj.GetChild(i)
        if type(node.GetNodeAttribute()) == fbx.FbxMesh:
            print(f"Found mesh node: {node.GetName()}")
            name = node.GetName()
            mesh = node.GetMesh()
            meshes.append(_extract_data(node))
    base_textures = [mesh.base_texture_name for mesh in meshes if mesh.base_texture_name]

    mapped_textures = match_textures(
        source_names=base_textures,
        local_dir=os.path.dirname(fbx_path),
        min_ratio=0.3
    )

    for i, mesh in enumerate(meshes):
        mesh.load_texture(mapped_textures[i]) if mesh.base_texture_name else None
        print("texture size:", mesh.mapped_texture.height, mesh.mapped_texture.width)
    print(f"Loaded {len(meshes)} meshes from {fbx_path}")

    return meshes


def meshdata_to_pytorch3d_mesh(mesh_data: MeshData, texture_map:str = "uv") -> Meshes:
    """
    texture_map can be "neural", "uv", or "color"
    """
    verts = mesh_data.get_split_attribute("positions").float()
    faces = mesh_data.triangles

    if mesh_data.uvs is not None and getattr(mesh_data, "mapped_texture", None) is not None and mesh_data.mapped_texture.image is not None:
        uvs = mesh_data.get_split_attribute("uvs")
        uvs = uvs % 1.0
        tex = mesh_data.mapped_texture.image[:, :, :3].float()
        textures = TexturesUV(maps=[tex], faces_uvs=[faces], verts_uvs=[uvs])
    elif mesh_data.colors is not None:
        cols = mesh_data.get_split_attribute("colors").float()[:, :3]
        textures = TexturesVertex(verts_features=[cols])  # (1, V, 3)
    return Meshes(
        verts=[verts],
        faces=[faces],
        textures=textures
    )


def _build_polyvert_attrs(faces, uv_faces, norm_faces, color_faces):
    """
    Combines per-face indices into a single polyvert_attrs tensor (M, 4)
    """
    num_faces = faces.shape[0]
    polyverts = []

    for i in range(num_faces):
        for j in range(3):
            pos_idx = faces[i, j].item()
            uv_idx = uv_faces[i, j].item() if uv_faces is not None else -1
            norm_idx = norm_faces[i, j].item() if norm_faces is not None else -1
            color_idx = color_faces[i, j].item() if color_faces is not None else -1
            polyverts.append([pos_idx, uv_idx, norm_idx, color_idx])

    return torch.tensor(polyverts, dtype=torch.long)


def _extract_data(node):
    mesh = node.GetMesh()
    name = node.GetName()
    if not mesh:
        raise ValueError(f"Node '{name}' does not contain a valid FbxMesh.")
    faces = []
    positions = torch.empty((mesh.GetPolygonVertexCount(), 3), dtype=torch.float32)
    polyverts = torch.full((mesh.GetPolygonVertexCount(), 4), -1, dtype=torch.long)  # (pos_idx, uv_idx, norm_idx, color_idx)
    extract_uvs = False
    extract_normals = False
    extract_colors = False

    if mesh.GetElementUVCount() > 0:
        print("UVs found in mesh, extracting data...")
        if mesh.GetElementUVCount() > 1:
            raise ValueError("Multiple UV sets not supported in this function.")
        uv_element = mesh.GetElementUV(0)
        uv_set_name = uv_element.GetName()
        print(f"Using UV set: {uv_set_name}")
        extract_uvs = True
        uvs = torch.empty((mesh.GetPolygonVertexCount(), 2), dtype=torch.float32)
    if mesh.GetElementNormalCount() > 0:
        print("Normals found in mesh, extracting data...")
        if mesh.GetElementNormalCount() > 1:
            raise ValueError("Multiple normal sets not supported in this function.")
        extract_normals = True
        norms = torch.empty((mesh.GetPolygonVertexCount(), 3), dtype=torch.float32)
    if mesh.GetElementVertexColorCount() > 0:
        print("Vertex colors found in mesh, extracting data...")
        extract_colors = True
        colors = torch.empty((mesh.GetPolygonVertexCount(), 3), dtype=torch.float32)
    polyvert_id = 0
    for i in range(mesh.GetPolygonCount()):
        polygon_size = mesh.GetPolygonSize(i)

        for k in range(1, polygon_size - 1):  # triangulate for n-gons
            faces.append([polyvert_id, polyvert_id + k, polyvert_id + k + 1])

        for j in range(polygon_size):
            ctrl_point_idx = mesh.GetPolygonVertex(i, j)
            position = mesh.GetControlPointAt(ctrl_point_idx)
            positions[polyvert_id] = torch.tensor([position[0], position[1], position[2]], dtype=torch.float32)

            if extract_uvs:
                uv_store = fbx.FbxVector2()
                mesh.GetPolygonVertexUV(i, j, uv_set_name, uv_store)  # Assuming first UV set
                uvs[polyvert_id] = torch.tensor([uv_store[0], uv_store[1]], dtype=torch.float32)
            if extract_normals:
                normal_store = fbx.FbxVector4()
                mesh.GetPolygonVertexNormal(i, j, normal_store)  # Assuming first normal set
                norms[polyvert_id] = torch.tensor([normal_store[0], normal_store[1], normal_store[2]], dtype=torch.float32)
            # if extract_colors:
                # print("Vertex colors extraction not implemented in this function.")
            polyvert_id += 1
    
    print(f"Starting shapes - Faces: {len(faces)}, Positions: {positions.shape}, UVs: {uvs.shape if extract_uvs else 'N/A'}, Normals: {norms.shape if extract_normals else 'N/A'}")
    # Deduplicate positions using torch.unique (along rows)
    unique_positions, inverse_indices = torch.unique(positions, dim=0, return_inverse=True)
    polyverts[:, 0] = inverse_indices
    print(f"Unique positions: {unique_positions.shape[0]}")
    if extract_uvs:
        unique_uvs, uv_inverse_indices = torch.unique(uvs, dim=0, return_inverse=True)
        print(f"Unique UVs: {unique_uvs.shape[0]}")
        polyverts[:, 1] = uv_inverse_indices
    if extract_normals:
        unique_normals, norm_inverse_indices = torch.unique(norms, dim=0, return_inverse=True)
        print(f"Unique normals: {unique_normals.shape[0]}")
        polyverts[:, 2] = norm_inverse_indices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    faces = torch.tensor(faces, dtype=torch.long, device=device)

    texture = _get_base_texture(mesh)

    return MeshData(
        name=name,
        positions=unique_positions.to(device),
        uvs=unique_uvs.to(device) if extract_uvs else None,
        normals=unique_normals.to(device) if extract_normals else None,
        colors=None,  # Vertex colors not implemented in this function
        polyvert_attrs=polyverts.to(device),
        triangles=faces,
        base_texture_name=texture,
        normalized=False  # Normalization flag
    )


def _get_fbx_normal_data(mesh: FbxMesh):
    print("Extracting normals from mesh...")
    if mesh.GetElementNormalCount() != 1:
        raise ValueError(f"Expected 1 normal element, got {mesh.GetElementNormalCount()}")

    normal_element = mesh.GetElementNormal(0)

    if normal_element.GetMappingMode() != normal_element.EMappingMode.eByPolygonVertex:
        raise ValueError(f"Unsupported normal mapping mode: {normal_element.GetMappingMode()}")

    ref_mode = normal_element.GetReferenceMode()
    poly_normals = []

    for i in range(mesh.GetPolygonVertexCount()):
        if ref_mode == normal_element.EReferenceMode.eDirect:
            normal = normal_element.GetDirectArray().GetAt(i)
        elif ref_mode == normal_element.EReferenceMode.eIndexToDirect:
            idx = normal_element.GetIndexArray().GetAt(i)
            normal = normal_element.GetDirectArray().GetAt(idx)
        else:
            raise ValueError(f"Unsupported normal reference mode: {ref_mode}")
        poly_normals.append((normal[0], normal[1], normal[2]))

    # Deduplicate normals and build remapping
    unique_normals, inverse_indices = np.unique(poly_normals, axis=0, return_inverse=True)
    norm_faces = inverse_indices.reshape((-1, 3))  # assuming triangulated input

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return (
        torch.tensor(unique_normals, dtype=torch.float32, device=device),
        torch.tensor(norm_faces, dtype=torch.long, device=device)
    )


def _get_fbx_color_data(mesh: FbxMesh):
    print("Extracting vertex colors from mesh...")
    if mesh.GetElementVertexColorCount() != 1:
        raise ValueError(f"Expected 1 vertex color element, got {mesh.GetElementVertexColorCount()}")

    color_element = mesh.GetElementVertexColor(0)

    if color_element.GetMappingMode() != color_element.EMappingMode.eByPolygonVertex:
        raise ValueError(f"Unsupported vertex color mapping mode: {color_element.GetMappingMode()}")

    ref_mode = color_element.GetReferenceMode()
    poly_colors = []

    for i in range(mesh.GetPolygonVertexCount()):
        if ref_mode == color_element.EReferenceMode.eDirect:
            color = color_element.GetDirectArray().GetAt(i)
        elif ref_mode == color_element.EReferenceMode.eIndexToDirect:
            idx = color_element.GetIndexArray().GetAt(i)
            color = color_element.GetDirectArray().GetAt(idx)
        else:
            raise ValueError(f"Unsupported vertex color reference mode: {ref_mode}")
        poly_colors.append((color.mRed, color.mGreen, color.mBlue, color.mAlpha))

    # Deduplicate colors and build remapping
    unique_colors, inverse_indices = np.unique(poly_colors, axis=0, return_inverse=True)
    color_faces = inverse_indices.reshape((-1, 3))  # assuming triangulated input

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return (
        torch.tensor(unique_colors, dtype=torch.float32, device=device),
        torch.tensor(color_faces, dtype=torch.long, device=device)
    )


def _get_base_texture(mesh: FbxMesh):
    print(f"Mesh layers: {mesh.GetLayerCount()}")
    node = mesh.GetNode()
    texture_path = None
    if node.GetMaterialCount() == 1:
        material = node.GetMaterial(0)
 
        for slot in ("Diffuse", "DiffuseColor", "BaseColor", "Albedo", "diffuse", "albedo", "Color"):
            prop = material.FindProperty(slot)
            if not prop.IsValid():
                continue
            if prop.GetSrcObjectCount() > 0 and prop.GetSrcObject(0).GetFileName():
                texture_path = prop.GetSrcObject(0).GetFileName()
            else:
                texture_path = "Diffuse.png"  # Default fallback if no texture found
            
    return texture_path


def _find_node_by_name(root_node: fbx.FbxNode, name: str) -> fbx.FbxNode:
    """
    Recursively search under root_node for a child (or descendant) whose GetName() == name.
    Returns the first match or None if not found.
    """
    if root_node.GetName() == name:
        return root_node

    for i in range(root_node.GetChildCount()):
        found = _find_node_by_name(root_node.GetChild(i), name)
        if found:
            return found

    return None


def match_textures(source_names: List[str], local_dir: List[str], min_ratio: float = 0.3) -> Dict[str, str]:

    def _find_local_textures(local_dir: str, valid_types: List[str] = None) -> List[str]:
        return [
            os.path.join(local_dir, fn)
            for fn in os.listdir(local_dir)
            if any(fn.lower().endswith(ext) for ext in valid_types)
        ]

    source_extensions = {os.path.splitext(name)[1].lower() for name in source_names}
    local_names = _find_local_textures(local_dir, valid_types=source_extensions)
    print(f"Found {len(local_names)} local textures in '{local_dir}' with extensions {source_extensions}")
    print(f"Source textures: {source_names}")

    scores = []
    for m in source_names:
        m_base = os.path.basename(os.path.splitext(os.path.normpath(m.replace("\\", "/")))[0])
        print(f"Matching source: {m_base}")
        for a in local_names:
            a_base = os.path.basename(os.path.splitext(os.path.normpath(a))[0])
            print(f"  Against local: {a_base}")
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
    print(mapping)
    return list(mapping.values())


def write_meshdata_list_to_fbx(
    original_fbx_path: str,
    new_fbx_path: str,
    meshdata_list: List["MeshData"],
    export_ascii: bool = False
):
    """
    1) Opens original_fbx_path into an FbxScene.
    2) Finds every FbxMesh node whose name matches mesh_data.name.
    3) Replaces that node's control points, UVs, normals, and polygon layout 
       with the data in mesh_data.
    4) Exports the entire scene to new_fbx_path, preserving animations, skins, cameras, etc.
    """

    manager = FbxManager.Create()
    importer = FbxImporter.Create(manager, "")
    scene = FbxScene.Create(manager, "Scene")
    if not importer.Initialize(original_fbx_path, -1, manager.GetIOSettings()):
        raise RuntimeError(f"Failed to open '{original_fbx_path}' for import.")
    importer.Import(scene)
    importer.Destroy()

    # 2) Build a lookup by mesh name
    mdict = {md.name: md for md in meshdata_list}

    # 3) Traverse all nodes recursively, looking for FbxMesh attributes
    def recurse_replace(node: fbx.FbxNode, depth=0):
        for i in range(node.GetChildCount()):
            child = node.GetChild(i)
            attr = child.GetNodeAttribute()
            if attr and type(attr) == fbx.FbxMesh:
                fbx_mesh = child.GetMesh()
                node_name = child.GetName()
                if node_name in mdict:
                    print("  " * (depth + 1) + f"Found mesh: {node_name}")
                    _replace_fbxmesh_with_meshdata(fbx_mesh, mdict[node_name], scene.GetRootNode(), manager)
            recurse_replace(child, depth + 1)

    recurse_replace(scene.GetRootNode())
    # 4) Export the modified scene to new_fbx_path
    exporter = fbx.FbxExporter.Create(manager, "")
    num_formats = manager.GetIOPluginRegistry().GetWriterFormatCount()
    ascii_format = -1
    for i in range(num_formats):
        desc = manager.GetIOPluginRegistry().GetWriterFormatDescription(i)
        print(f"Format {i}: {desc}")
        if "ascii" in desc.lower():
            ascii_format = i
            break

    if export_ascii and ascii_format == -1:
        raise RuntimeError("No ASCII FBX format found in SDK.")

    # Use ascii_format if flag is enabled, otherwise default format
    export_format = ascii_format if export_ascii else -1

    if not exporter.Initialize(new_fbx_path, export_format, manager.GetIOSettings()):
        raise RuntimeError(f"Failed to open '{new_fbx_path}' for export.")
    exporter.Export(scene)
    exporter.Destroy()
    manager.Destroy()
    print(f"Wrote updated FBX to '{new_fbx_path}' using {'ASCII' if export_ascii else 'default binary'} format.")


def meshdata_to_glb(
    mesh_data: "MeshData",
    out_glb_path: str
):
    if mesh_data.normalized:
        positions = mesh_data.get_split_attribute("positions") * mesh_data.ranges + mesh_data.mins
    else:
        positions = mesh_data.get_split_attribute("positions")
    mesh = trimesh.Trimesh(
        vertices=positions.cpu().numpy(),
        faces=mesh_data.triangles.cpu().numpy(),
        process=False
    )
    verts_np = positions.cpu().numpy()
    uvs_np = mesh_data.get_split_attribute("uvs").cpu().numpy()
    faces_np = mesh_data.triangles.cpu().numpy()

    uvs = mesh_data.get_split_attribute("uvs").cpu().numpy()
    img = mesh_data.mapped_texture.image.cpu().numpy()

        # Sanity checks
    print(f"[GLB] verts: {verts_np.shape}")
    print(f"[GLB] uvs: {uvs_np.shape}")
    print(f"[GLB] faces: {faces_np.shape}")
    print(f"[GLB] faces.min(): {faces_np.min()}  faces.max(): {faces_np.max()}")

    if verts_np.shape[0] != uvs_np.shape[0]:
        print("[WARN] Vertex count does not match UV count. This will break UV mapping.")
    if faces_np.max() >= verts_np.shape[0]:
        print("[ERROR] Face index out of bounds for vertex array!")
    if np.any(np.isnan(uvs_np)) or np.any(np.isinf(uvs_np)):
        print("[ERROR] UVs contain NaN or Inf values.")
    if np.any((uvs_np < -1e-3) | (uvs_np > 1 + 1e-3)):
        print("[WARN] UVs fall outside expected [0, 1] range.")

    img8 = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img
    img_pil = Image.fromarray(img8)
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=uvs,
        image=img_pil
    )

    glb_bytes = mesh.export(file_type='glb')
    with open(out_glb_path, 'wb') as f:
        f.write(glb_bytes)


def export_meshdata_list(meshdata_list, out_fbx_path, export_ascii=False):
    manager = fbx.FbxManager.Create()
    scene = fbx.FbxScene.Create(manager, "ExportScene")
    root = scene.GetRootNode()

    for mesh in meshdata_list:
        fbx_mesh = fbx.FbxMesh.Create(scene, mesh.name)
        node = fbx.FbxNode.Create(scene, mesh.name)
        node.SetNodeAttribute(fbx_mesh)
        root.AddChild(node)

        positions = mesh.positions
        if mesh.normalized:
            positions = mesh.positions * mesh.ranges + mesh.mins
        positions = positions.cpu().float()  # Ensure positions are on CPU and float
        # Positions
        fbx_mesh.InitControlPoints(positions.shape[0])
        for idx in range(positions.shape[0]):
            x, y, z = positions[idx].tolist()
            fbx_mesh.SetControlPointAt(fbx.FbxVector4(x, y, z), idx)

        if mesh.colors is not None:
            if fbx_mesh.GetElementVertexColorCount() > 0:
                fbx_mesh.RemoveElementVertexColor(fbx_mesh.GetElementVertexColor(0))
            if mesh.colors.shape[0] != positions.shape[0]:
                raise ValueError("Number of vertex colors does not match number of positions.")
            if fbx_mesh.GetLayerCount() == 0:
                fbx_mesh.CreateLayer()
            
            vc_layer = fbx.FbxLayerElementVertexColor.Create(fbx_mesh, "VertexColor")
            vc_layer.SetMappingMode(fbx.FbxLayerElement.EMappingMode.eByControlPoint)
            vc_layer.SetReferenceMode(fbx.FbxLayerElement.EReferenceMode.eDirect)

            for idx in range(positions.shape[0]):
                r, g, b, a = mesh.colors[idx].tolist()
                vc_layer.GetDirectArray().Add(fbx.FbxColor(r, g, b, a))
            fbx_mesh.GetLayer(0).SetVertexColors(vc_layer)
        # Faces (triangles) Technically optional
        if mesh.triangles is not None:
            triangles = mesh.triangles.cpu().tolist()
            poly_attrs = mesh.polyvert_attrs.cpu().tolist()
            for tri in triangles:
                fbx_mesh.BeginPolygon()
                for pv_idx in tri:
                    pos_idx, _, _, _ = poly_attrs[pv_idx]
                    fbx_mesh.AddPolygon(pos_idx)
                fbx_mesh.EndPolygon()

    # Export!
    exporter = fbx.FbxExporter.Create(manager, "")
    num_formats = manager.GetIOPluginRegistry().GetWriterFormatCount()
    ascii_format = -1
    for i in range(num_formats):
        desc = manager.GetIOPluginRegistry().GetWriterFormatDescription(i)
        if "ascii" in desc.lower():
            ascii_format = i
            break

    export_format = ascii_format if export_ascii else -1
    if not exporter.Initialize(out_fbx_path, export_format, manager.GetIOSettings()):
        raise RuntimeError(f"Failed to open '{out_fbx_path}' for export.")
    exporter.Export(scene)
    exporter.Destroy()
    manager.Destroy()
    print(f"Exported clean FBX to '{out_fbx_path}' as {'ASCII' if export_ascii else 'binary'}.")


if __name__ == "__main__":
    from asset_rep import optimize_mesh
    test = load_fbx_to_meshdata("static/meshes/Horse.fbx")
    new_meshes = []
    for mesh in test:
        meshdata_to_glb(mesh, f"static/meshes/{mesh.name}.glb")
