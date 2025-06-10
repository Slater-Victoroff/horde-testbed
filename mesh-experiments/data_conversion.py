import os
from typing import List, Dict
from difflib import SequenceMatcher

import torch
import numpy as np
import fbx
from fbx import FbxManager, FbxImporter, FbxScene, FbxMesh

from asset_rep import MeshData


def load_fbx_to_meshdata(fbx_path: str):
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
            bone_indices, bone_weights, unique_bone_names = _get_bone_data(mesh)
            uvs = _get_fbx_uvs(mesh)
            normals = _get_fbx_normals(mesh)

            faces = _get_position_faces(mesh)
            uv_faces = _get_uv_faces(mesh)
            norm_faces = _get_normal_faces(mesh)
            texture = _get_base_texture(mesh)
            mesh_data = MeshData(
                name=name,
                positions=positions,
                bone_names=unique_bone_names,
                bone_indices=bone_indices,
                bone_weights=bone_weights,
                uvs = uvs,
                normals=normals,
                colors=None,
                polyvert_attrs= _build_polyvert_attrs(
                    faces=faces,
                    uv_faces=uv_faces,
                    norm_faces=norm_faces,
                    color_faces=None
                ),
                triangles=_build_triangle_indices(faces),
                base_texture_name=texture,
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


def _get_bone_data(mesh: FbxMesh, max_influences: int = 4):
    bone_names = []
    bone_weights = []
    if mesh.GetDeformerCount() == 1:
        deformer = mesh.GetDeformer(0, fbx.FbxDeformer.EDeformerType.eSkin)
        for i in range(mesh.GetControlPointsCount()):
            indices = []
            weights = []
            for j in range(deformer.GetClusterCount()):
                cluster = deformer.GetCluster(j)
                for index, weight in zip(cluster.GetControlPointIndices(), cluster.GetControlPointWeights()):
                    if index == i:
                        indices.append(cluster.GetLink().GetName())
                        weights.append(weight)
                        
            bone_names.append(indices[:max_influences])
            bone_weights.append(weights[:max_influences])

    # Process bone names to get unique names and replace with indices
    unique_bone_names = list(set(name for names in bone_names for name in names))
    bone_indices = [
        [unique_bone_names.index(name) for name in names]
        for names in bone_names
    ]

    # Convert to tensors
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Pad bone_indices and bone_weights to ensure consistent length (max_influences)
    padded_bone_indices = [indices + [-1] * (max_influences - len(indices)) for indices in bone_indices]
    padded_bone_weights = [weights + [0.0] * (max_influences - len(weights)) for weights in bone_weights]

    bone_indices_tensor = torch.tensor(padded_bone_indices, dtype=torch.long, device=device)
    bone_weights_tensor = torch.tensor(padded_bone_weights, dtype=torch.float64, device=device)

    return bone_indices_tensor, bone_weights_tensor, unique_bone_names


def _get_fbx_uvs(mesh: FbxMesh):
    uvs = []
    if mesh.GetElementUVCount() == 1:
        uv_element = mesh.GetElementUV(0)
        if uv_element.GetMappingMode() == uv_element.EMappingMode.eByPolygonVertex:
            if uv_element.GetReferenceMode() == uv_element.EReferenceMode.eDirect:
                for i in range(mesh.GetPolygonVertexCount()):
                    uv = uv_element.GetDirectArray().GetAt(i)
                    uvs.append([uv[0], uv[1]])
            elif uv_element.GetReferenceMode() == uv_element.EReferenceMode.eIndexToDirect:
                print(f"$" * 50)
                for i in range(mesh.GetPolygonVertexCount()):
                    idx = uv_element.GetIndexArray().GetAt(i)
                    uv = uv_element.GetDirectArray().GetAt(idx)
                    uvs.append([uv[0], uv[1]])
            else:
                raise ValueError(f"Unsupported UV reference mode: {uv_element.GetReferenceMode()}")
        else:
            raise ValueError(f"Unsupported UV mapping mode: {uv_element.GetMappingMode()}")
    else:
        raise ValueError(f"Unsupported number of UV elements: {mesh.GetElementUVCount()}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.tensor(uvs, dtype=torch.float64, device=device)


def _get_uv_faces(mesh: FbxMesh):
    """
    Build an (F,3) tensor of UV‐indices per face, handling both eDirect and eIndexToDirect.
    Assumes MappingMode = eByPolygonVertex.
    """
    if mesh.GetElementUVCount() != 1:
        raise ValueError(f"Unsupported number of UV elements: {mesh.GetElementUVCount()}")

    uv_element = mesh.GetElementUV(0)

    # Must be “by polygon‐vertex” to match how we flattened faces.
    if uv_element.GetMappingMode() != uv_element.EMappingMode.eByPolygonVertex:
        raise ValueError(f"Unsupported UV mapping mode: {uv_element.GetMappingMode()}")

    ref_mode = uv_element.GetReferenceMode()
    poly_count = mesh.GetPolygonCount()
    uv_faces = []

    # A running counter over “polygon‐vertices” (0..(F*3 − 1)).
    polyvert_counter = 0

    for i in range(poly_count):
        face_uvs = []
        for j in range(mesh.GetPolygonSize(i)):
            if ref_mode == uv_element.EReferenceMode.eDirect:
                # In eDirect, DirectArray holds one UV per poly‐vertex in sequence.
                idx = polyvert_counter
                polyvert_counter += 1
            else:
                # eIndexToDirect: look up index in IndexArray, then later
                # you’ll use that index to read DirectArray when reconstructing UV coords.
                idx = uv_element.GetIndexArray().GetAt(polyvert_counter)
                polyvert_counter += 1

            face_uvs.append(idx)
        uv_faces.append(face_uvs)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.tensor(uv_faces, dtype=torch.long, device=device)


def _get_fbx_normals(mesh: FbxMesh):
    normals = []
    if mesh.GetElementNormalCount() == 1:
        normal_element = mesh.GetElementNormal(0)
        if normal_element.GetMappingMode() == normal_element.EMappingMode.eByPolygonVertex:
            if normal_element.GetReferenceMode() == normal_element.EReferenceMode.eDirect:
                for i in range(mesh.GetPolygonVertexCount()):
                    normal = normal_element.GetDirectArray().GetAt(i)
                    normals.append([normal[0], normal[1], normal[2]])
            elif normal_element.GetReferenceMode() == normal_element.EReferenceMode.eIndexToDirect:
                print("@" * 50)
                for i in range(mesh.GetPolygonVertexCount()):
                    idx = normal_element.GetIndexArray().GetAt(i)
                    normal = normal_element.GetDirectArray().GetAt(idx)
                    normals.append([normal[0], normal[1], normal[2]])
            else:
                raise ValueError(f"Unsupported normal reference mode: {normal_element.GetReferenceMode()}")
        else:
            raise ValueError(f"Unsupported normal mapping mode: {normal_element.GetMappingMode()}")
    else:
        raise ValueError(f"Unsupported number of normal elements: {mesh.GetElementNormalCount()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.tensor(normals, dtype=torch.float64, device=device)


def _get_normal_faces(mesh: FbxMesh):
    """
    Build an (F,3) tensor of normal‐indices per face, matching _get_fbx_normals().
    Works for MappingMode=eByPolygonVertex with either eDirect or eIndexToDirect.
    """
    if mesh.GetElementNormalCount() != 1:
        raise ValueError(f"Unsupported number of normal elements: {mesh.GetElementNormalCount()}")

    normal_element = mesh.GetElementNormal(0)

    # Must be ByPolygonVertex
    if normal_element.GetMappingMode() != normal_element.EMappingMode.eByPolygonVertex:
        raise ValueError(f"Unsupported normal mapping mode: {normal_element.GetMappingMode()}")

    ref_mode = normal_element.GetReferenceMode()
    poly_count = mesh.GetPolygonCount()
    norm_faces = []

    # Keep a running counter for "direct" mode:
    polyvert_counter = 0

    for i in range(poly_count):
        face_normals = []
        for j in range(mesh.GetPolygonSize(i)):
            if ref_mode == normal_element.EReferenceMode.eDirect:
                # In eDirect, DirectArray holds one normal per poly-vert in order.
                idx = polyvert_counter
                polyvert_counter += 1
            else:
                # eIndexToDirect: look up the index in IndexArray first
                # (IndexArray is length = polygonVertexCount)
                idx = normal_element.GetIndexArray().GetAt(polyvert_counter)
                polyvert_counter += 1

            face_normals.append(idx)
        norm_faces.append(face_normals)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.tensor(norm_faces, dtype=torch.long, device=device)


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


def _rebuild_skin_from_meshdata(
    fbx_mesh: fbx.FbxMesh,
    mesh_data: "MeshData",
    scene_root: fbx.FbxNode,
    manager: fbx.FbxManager
):
    """
    Completely discards any existing FbxSkin on fbx_mesh and builds a new one 
    using mesh_data.bone_names, bone_indices, and bone_weights.

    - fbx_mesh: the FbxMesh to re‐skin
    - mesh_data.bone_names: List[str], the bone names (matches FBX node names)
    - mesh_data.bone_indices: LongTensor[N,4], per‐vertex bone indices
    - mesh_data.bone_weights: FloatTensor[N,4], per‐vertex bone weights
    - scene_root: typically scene.GetRootNode(), used to find bone nodes
    """
    # 1) If fbx_mesh already has a FbxSkin deformer, remove it.
    num_deformers = fbx_mesh.GetDeformerCount()
    print(f"Removing {num_deformers} existing FbxSkin deformers from mesh '{fbx_mesh.GetName()}'.")
    for i in reversed(range(num_deformers)):
        deformer = fbx_mesh.GetDeformer(i)
        for cidx in reversed(range(deformer.GetClusterCount())):
            cluster = deformer.GetCluster(cidx)
            deformer.RemoveCluster(cluster)
        fbx_mesh.RemoveDeformer(i)

    # 2) Create a new FbxSkin and attach to fbx_mesh
    new_skin = fbx.FbxSkin.Create(manager, "")
    fbx_mesh.AddDeformer(new_skin)

    # 3) Build a map from bone_name -> FbxNode (skeleton). If any bone not found, warn.
    bone_node_map = {}
    for bone_name in mesh_data.bone_names:
        bone_node = _find_node_by_name(scene_root, bone_name)
        if bone_node.GetParent() is None:
            print(f"[Fixing] Adding bone node '{bone_node.GetName()}' to scene root")
            scene_root.AddChild(bone_node)
        if not bone_node:
            print(f"[Warning] Could not find bone node '{bone_name}' in FBX scene.")
            continue
        bone_node_map[bone_name] = bone_node

    # 4) For each bone, create one cluster and add it to the skin.
    #    We’ll store clusters in a list so we can add control‐point indices afterward.
    clusters = []
    for bone_idx, bone_name in enumerate(mesh_data.bone_names):
        bone_node = bone_node_map.get(bone_name)

        cluster = fbx.FbxCluster.Create(manager, "")
        cluster.SetLink(bone_node)
        cluster.SetTransformMatrix(fbx_mesh.GetNode().EvaluateGlobalTransform())
        cluster.SetTransformLinkMatrix(bone_node.EvaluateGlobalTransform())
        cluster.SetLinkMode(fbx.FbxCluster.ELinkMode.eNormalize)

        new_skin.AddCluster(cluster)
        clusters.append((bone_idx, cluster))

    # 5) Now assign control-point indices + weights to each cluster:
    # Assume all control points already exist; we're adding bone weights and indices to them.
    num_points = mesh_data.positions.shape[0]
    bone_indices = mesh_data.bone_indices.cpu().tolist()  # list of [i0, i1, i2, i3]
    bone_weights = mesh_data.bone_weights.cpu().tolist()  # list of [w0, w1, w2, w3]

    # Build a quick lookup: bone_idx -> cluster
    boneidx_to_cluster = {bi: c for (bi, c) in clusters}

    for cp_idx in range(num_points):
        idx_list = bone_indices[cp_idx]   # e.g. [2, 5, 7, 0]
        w_list   = bone_weights[cp_idx]   # e.g. [0.72, 0.28, 0.0, 0.0]

        for slot in range(len(idx_list)):
            bidx = idx_list[slot]
            wgt  = w_list[slot]
            if wgt > 0.0:  # Only process weights greater than 0
                cluster = boneidx_to_cluster.get(bidx)
                if cluster:
                    cluster.AddControlPointIndex(cp_idx, wgt)


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
    _rebuild_skin_from_meshdata(fbx_mesh, mesh_data, scene_root, manager)


def export_meshdata_list(meshdata_list, out_fbx_path, export_ascii=False):
    manager = fbx.FbxManager.Create()
    scene = fbx.FbxScene.Create(manager, "ExportScene")
    root = scene.GetRootNode()

    # 1. Create all unique bones as FbxNodes (flat under root for now)
    bone_nodes = {}
    for mesh in meshdata_list:
        if mesh.bone_names is not None:
            for bone_name in mesh.bone_names:
                if bone_name not in bone_nodes:
                    node = fbx.FbxNode.Create(scene, bone_name)
                    # Optional: node.SetNodeAttribute(...) to make it a limb node, etc.
                    root.AddChild(node)
                    bone_nodes[bone_name] = node

    # 2. For each mesh, build FbxMesh, FbxNode, add to root
    for mesh in meshdata_list:
        fbx_mesh = fbx.FbxMesh.Create(scene, mesh.name)
        node = fbx.FbxNode.Create(scene, mesh.name)
        node.SetNodeAttribute(fbx_mesh)
        root.AddChild(node)

        # Positions
        fbx_mesh.InitControlPoints(mesh.positions.shape[0])
        for idx in range(mesh.positions.shape[0]):
            x, y, z = mesh.positions[idx].tolist()
            fbx_mesh.SetControlPointAt(fbx.FbxVector4(x, y, z), idx)

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

        if mesh.bone_names is not None:
            skin = fbx.FbxSkin.Create(manager, "")
            fbx_mesh.AddDeformer(skin)
            clusters = []
            for bone_idx, bone_name in enumerate(mesh.bone_names):
                bone_node = bone_nodes[bone_name]
                cluster = fbx.FbxCluster.Create(manager, "")
                cluster.SetLink(bone_node)
                cluster.SetLinkMode(fbx.FbxCluster.ELinkMode.eNormalize)
                skin.AddCluster(cluster)
                clusters.append(cluster)

            # Add skin weights (same as before)
            bone_indices = mesh.bone_indices.cpu().tolist()
            bone_weights = mesh.bone_weights.cpu().tolist()
            for cp_idx in range(mesh.positions.shape[0]):
                idx_list = bone_indices[cp_idx]
                w_list = bone_weights[cp_idx]
                for slot in range(len(idx_list)):
                    bidx = idx_list[slot]
                    wgt = w_list[slot]
                    if wgt > 0.0 and 0 <= bidx < len(clusters):
                        clusters[bidx].AddControlPointIndex(cp_idx, wgt)

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
        new_meshes.append(optimize_mesh(mesh))
    export_meshdata_list(new_meshes, "static/meshes/Horse_optimized.fbx", export_ascii=False)
