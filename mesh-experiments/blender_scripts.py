import tempfile
import os
import subprocess

def list_mesh_details(input_file: str):
    """
    List details of all mesh objects in the Blender scene after loading a file.
    """
    script = f"""
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=r'{input_file}')

for obj in bpy.data.objects:
    if obj.type == 'MESH':
        print(f"Mesh: {{obj.name}}")
        meshes = obj.data
        print("  Vertices:", len(meshes.vertices))
        print("  Faces:",    len(meshes.polygons))
        print("  Vertex Groups (bones):")
        for vg in obj.vertex_groups:
            print("    ", vg.name)
        # show weights on the first few verts
        for v in meshes.vertices[:5]:
            groups = {{obj.vertex_groups[g.group].name: g.weight
                      for g in v.groups}}
            print(f"    vert {{v.index:3d}} weights:", groups)

# List the armature bones
for obj in bpy.data.objects:
    if obj.type == 'ARMATURE':
        print(f"Armature: {{obj.name}}")
        for b in obj.data.bones:
            print("    Bone:", b.name)
"""
    run_blender_script(script)

def convert_fbx_to_obj(input_fbx: str, output_obj: str, use_uvs: bool = True, use_materials: bool = True, use_triangles: bool = True, axis_up: str = 'Z', axis_forward: str = '-Y'):
    """
    Convert an FBX into an OBJ (with UVs) by invoking Blender headless.
    """
    script = f"""
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=r'{input_fbx}')
bpy.ops.export_scene.obj(
    filepath=r'{output_obj}',
    use_uvs={str(use_uvs)},
    use_materials={str(use_materials)},
    use_triangles={str(use_triangles)},
    axis_up='{axis_up}',
    axis_forward='{axis_forward}'
)
"""
    run_blender_script(script)


def load_fbx_to_meshdata(fbx_path: str, max_influences: int = 4):
    # 1) Prepare a temp .npz path
    npz_fd, npz_path = tempfile.mkstemp(suffix=".npz")
    os.close(npz_fd)

    # 2) Build the Blender script that dumps verts/faces/uvs/normals/joints into .npz
    blender_script = f'''
import bpy, numpy as np

# ---- parameters ----
input_fbx      = r"{fbx_path}"
output_npz     = r"{npz_path}"
max_influences = {max_influences}

# 1) fresh scene + import FBX
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=input_fbx,
                         use_custom_normals=True)

# 2) pick first mesh + its armature
mesh_obj = next(o for o in bpy.context.scene.objects if o.type=="MESH")
arm_obj  = next(o for o in bpy.context.scene.objects if o.type=="ARMATURE")
mesh = mesh_obj.data
arm  = arm_obj.data

# 3) geometry arrays
verts      = np.array([v.co[:]     for v in mesh.vertices], dtype=np.float32)
faces      = np.array([p.vertices[:] for p in mesh.polygons], dtype=np.int32)

# 4) UVs (per-loop → flatten)
uv_layer   = mesh_obj.data.uv_layers.active.data
uvs, uv_fs = [], []
for poly in mesh.polygons:
    idxs = []
    for li in poly.loop_indices:
        uvs.append(uv_layer[li].uv[:])
        idxs.append(len(uvs)-1)
    uv_fs.append(idxs)
uvs      = np.array(uvs,      dtype=np.float32)
uv_fs    = np.array(uv_fs,    dtype=np.int32)

# 5) normals (per-vertex)
normals  = np.array([v.normal[:] for v in mesh.vertices], dtype=np.float32)
norm_fs  = faces.copy()

# 6) skin data
V = len(mesh.vertices)
joint_ids     = np.zeros((V, max_influences), dtype=np.int32)
joint_weights = np.zeros((V, max_influences), dtype=np.float32)
bone_index    = {{b.name:i for i,b in enumerate(arm.bones)}}

for v in mesh.vertices:
    infls = []
    for g in v.groups:
        name = mesh_obj.vertex_groups[g.group].name
        if name in bone_index:
            infls.append((bone_index[name], g.weight))
    infls = sorted(infls, key=lambda x:-x[1])[:max_influences]
    total = sum(w for _,w in infls) or 1.0
    for slot,(bidx,w) in enumerate(infls):
        joint_ids[v.index, slot]     = bidx
        joint_weights[v.index,slot]  = w/total

# 7) save everything
np.savez(output_npz,
         verts=verts, faces=faces,
         uvs=uvs, uv_faces=uv_fs,
         normals=normals, norm_faces=norm_fs,
         joint_ids=joint_ids, joint_weights=joint_weights)
print("Export complete:", output_npz)
'''
    # 3) Run the Blender script
    run_blender_script(blender_script)

    # 4) Load the .npz file and return the data
    import numpy as np
    data = np.load(npz_path)
    os.remove(npz_path)  # Clean up the temporary file
    return data


def run_blender_script(script: str):
    """
    Helper function to run a Blender script in headless mode.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".py")
    os.close(fd)
    with open(tmp_path, "w") as f:
        f.write(script)

    try:
        subprocess.check_call([
            "blender",
            "--background",
            "--python", tmp_path
        ])
    finally:
        os.remove(tmp_path)
