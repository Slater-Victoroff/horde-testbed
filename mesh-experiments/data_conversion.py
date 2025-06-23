import os
from typing import List, Dict
from difflib import SequenceMatcher

import torch
import trimesh
import numpy as np
import fbx
from fbx import FbxManager, FbxImporter, FbxScene, FbxMesh
from PIL import Image

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
            positions = _get_fbx_positions(mesh)
            faces = _get_position_faces(mesh)
            uvs, uv_faces = _get_fbx_uv_data(mesh) if mesh.GetElementUVCount() > 0 else (None, None)
            normals, norm_faces = _get_fbx_normal_data(mesh) if  mesh.GetElementNormalCount() > 0 else (None, None)
            colors, color_faces = _get_fbx_color_data(mesh) if mesh.GetElementVertexColorCount() > 0 else (None, None)

            texture = _get_base_texture(mesh)
            mesh_data = MeshData(
                name=name,
                positions=positions,
                uvs = uvs,
                normals=normals,
                colors=colors,
                polyvert_attrs= _build_polyvert_attrs(
                    faces=faces,
                    uv_faces=uv_faces,
                    norm_faces=norm_faces,
                    color_faces=color_faces
                ),
                triangles=_build_triangle_indices(faces),
                base_texture_name=texture,
                normalized=normalize
            )
            meshes.append(mesh_data)
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

def _build_triangle_indices(faces):
    """
    Faces are (F, 3). After flattening into polyverts, triangle indices become [0,1,2], [3,4,5], ...
    """
    num_faces = faces.shape[0]
    return torch.arange(num_faces * 3, dtype=torch.long).reshape(-1, 3)


def _get_fbx_positions(mesh: FbxMesh):
    verts = []
    for i in range(mesh.GetControlPointsCount()):
        cp = mesh.GetControlPoints()[i]
        verts.append([cp[0], cp[1], cp[2]])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.tensor(verts, dtype=torch.float64, device=device)


def _get_position_faces(mesh: FbxMesh):
    faces = []
    for i in range(mesh.GetPolygonCount()):
        face = []
        for j in range(mesh.GetPolygonSize(i)):
            idx = mesh.GetPolygonVertex(i, j)
            face.append(idx)
        faces.append(face)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.tensor(faces, dtype=torch.long, device=device)


def _get_fbx_uv_data(mesh: FbxMesh):
    print("Extracting UVs from mesh...")
    if mesh.GetElementUVCount() != 1:
        raise ValueError(f"Expected 1 UV element, got {mesh.GetElementUVCount()}")

    uv_element = mesh.GetElementUV(0)

    if uv_element.GetMappingMode() != uv_element.EMappingMode.eByPolygonVertex:
        raise ValueError(f"Unsupported UV mapping mode: {uv_element.GetMappingMode()}")

    ref_mode = uv_element.GetReferenceMode()
    poly_uvs = []

    for i in range(mesh.GetPolygonVertexCount()):
        if ref_mode == uv_element.EReferenceMode.eDirect:
            uv = uv_element.GetDirectArray().GetAt(i)
        elif ref_mode == uv_element.EReferenceMode.eIndexToDirect:
            idx = uv_element.GetIndexArray().GetAt(i)
            uv = uv_element.GetDirectArray().GetAt(idx)
        else:
            raise ValueError(f"Unsupported UV reference mode: {ref_mode}")
        poly_uvs.append((uv[0], uv[1]))

    # Deduplicate UVs and build remapping
    unique_uvs, inverse_indices = np.unique(poly_uvs, axis=0, return_inverse=True)
    uv_faces = inverse_indices.reshape((-1, 3))  # assuming triangulated input

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return (
        torch.tensor(unique_uvs, dtype=torch.float32, device=device),
        torch.tensor(uv_faces, dtype=torch.long, device=device)
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
    node = mesh.GetNode()
    texture_path = None
    if node.GetMaterialCount() == 1:
        material = node.GetMaterial(0)
        for slot in (fbx.FbxSurfaceMaterial.sDiffuse,
                     "BaseColor", "DiffuseColor"):
            prop = material.FindProperty(slot)
            if not prop.IsValid():
                continue
            if prop.GetSrcObjectCount() == 1:
                texture_path = prop.GetSrcObject(0).GetFileName()
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

def _replace_fbxmesh_with_meshdata(
    fbx_mesh: fbx.FbxMesh,
    mesh_data: "MeshData",
    scene_root: fbx.FbxNode,
    manager: fbx.FbxManager,
):
    """
    Overwrite fbx_mesh's control points, UVs, normals, and polygon layout 
    so that it matches mesh_data exactly. Any skinning/animation on fbx_mesh
    remains attached, but internal geometry is replaced.
    """
    
    # --- A) Replace control points (positions) ---
    num_positions = mesh_data.positions.shape[0]
    fbx_mesh.SetControlPointCount(num_positions)
    for idx in range(num_positions):
        x, y, z = mesh_data.positions[idx].tolist()
        fbx_mesh.SetControlPointAt(fbx.FbxVector4(x, y, z), idx)

    # --- B) Remove all existing polygons from the FBX mesh ---
    while fbx_mesh.GetPolygonCount() > 0:
        fbx_mesh.RemovePolygon(fbx_mesh.GetPolygonCount() - 1)
    print(f"Replaced {num_positions} control points in mesh '{fbx_mesh.GetName()}'.")

    # --- C) Prepare UV & Normal layers ---

    # Ensure at least one layer exists
    if fbx_mesh.GetLayerCount() == 0:
        fbx_mesh.CreateLayer()
    layer = fbx_mesh.GetLayer(0)
    layer.SetSmoothing(None)

    # UV layer (we use the “diffuse” channel as default)
    uv_elem = layer.GetUVs()
    if uv_elem:
        print(f"Removing existing UV layer for mesh '{fbx_mesh.GetName()}'.")
        layer.SetUVs(None)
    print(f"Creating new UV layer for mesh '{fbx_mesh.GetName()}'.")
    uv_elem = fbx.FbxLayerElementUV.Create(fbx_mesh, "")
    uv_elem.SetMappingMode(fbx.FbxLayerElementUV.EMappingMode.eByPolygonVertex)
    uv_elem.SetReferenceMode(fbx.FbxLayerElementUV.EReferenceMode.eIndexToDirect)
    layer.SetUVs(uv_elem)

    # Normal layer
    norm_elem = layer.GetNormals()
    if norm_elem:
        print(f"Removing existing normals layer for mesh '{fbx_mesh.GetName()}'.")
        layer.SetNormals(None)
    print(f"Adding normals layer to mesh '{fbx_mesh.GetName()}'.")
    norm_elem = fbx.FbxLayerElementNormal.Create(fbx_mesh, "")
    norm_elem.SetMappingMode(fbx.FbxLayerElementNormal.EMappingMode.eByPolygonVertex)
    norm_elem.SetReferenceMode(fbx.FbxLayerElementNormal.EReferenceMode.eIndexToDirect)
    layer.SetNormals(norm_elem)

    # Clear any existing direct/index arrays
    uv_direct = uv_elem.GetDirectArray()
    uv_index = uv_elem.GetIndexArray()
    norm_direct = norm_elem.GetDirectArray()
    norm_index = norm_elem.GetIndexArray()

    uv_direct.Clear()
    uv_direct.SetCount(mesh_data.uvs.shape[0])  # Set size to match mesh_data.uvs
    uv_index.Clear()
    uv_index.SetCount(mesh_data.triangles.shape[0] * 3)  # Set size to match triangles
    norm_direct.Clear()
    norm_direct.SetCount(mesh_data.normals.shape[0])  # Set size to match mesh_data.normals
    norm_index.Clear()
    norm_index.SetCount(mesh_data.triangles.shape[0] * 3)  # Set size to match triangles

    triangles = mesh_data.triangles.cpu().tolist()         # list of [pv0, pv1, pv2]
    poly_attrs = mesh_data.polyvert_attrs.cpu().tolist()   # each row = [pos, uv, norm, color?]

    for tri in triangles:
        fbx_mesh.BeginPolygon()
        for pv_idx in tri:
            pos_idx, uv_idx, norm_idx, *_ = poly_attrs[pv_idx]

            # 1) Assign control point index
            fbx_mesh.AddPolygon(pos_idx)

            # 2) Append UV
            u, v = mesh_data.uvs[uv_idx].tolist()
            uv_direct.SetAt(uv_idx, fbx.FbxVector2(u, v))
            uv_index.SetAt(pv_idx, uv_idx)

            # 3) Append normal
            nx, ny, nz = mesh_data.normals[norm_idx].tolist()
            norm_direct.SetAt(norm_idx, fbx.FbxVector4(nx, ny, nz, 0.0))
            norm_index.SetAt(pv_idx, norm_idx)

        fbx_mesh.EndPolygon()
    fbx_mesh.BuildMeshEdgeArray()


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
