# -----------------------------------------------------------------------------
# Adapted from Text2Tex (https://github.com/daveredrum/Text2Tex)
# Copyright (c) 2023-2024 Dave Zhenyu Chen et al.
# Licensed under CC BY-NC-SA 3.0.
#
# Modifications Copyright (c) 2025 Aliev Mishan.
# This file has been modified from the original version.
# -----------------------------------------------------------------------------

import sys
import shutil
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import os
import objaverse
import subprocess
import xatlas
import trimesh

from PIL import Image
from pathlib import Path
from tqdm import tqdm

# torch
import torch

from torchvision import transforms

# pytorch3d
from pytorch3d.io import (
    load_obj,
    save_obj
)

# customized
sys.path.append(".")
from parameterize_mesh import parameterize_mesh

SUBSET = "./objaverse_eval/assets/objaverse_subset.txt"
DATA_DIR = "./objaverse_eval/"
GLB_DIR = "./objaverse_eval/objaverse_data/glbs"
OBJ_DIR = "./objaverse_eval/objaverse_data/obj"
BLENDER_EXE = "./objaverse_eval/blender-3.3.21-linux-x64/blender"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(GLB_DIR, exist_ok=True)
os.makedirs(OBJ_DIR, exist_ok=True)

# Setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)
else:
    DEVICE = torch.device("cpu")

os.makedirs(DATA_DIR, exist_ok=True)

def get_objaverse_subset():
    with open(SUBSET) as f:
        ids = [l.rstrip() for l in f.readlines()]

    return ids

def adjust_uv_map(faces, aux, device=DEVICE):
    """
        adjust UV map to be compatiable with multiple textures.
        UVs for different materials will be decomposed and placed horizontally

        +-----+-----+-----+--
        |  1  |  2  |  3  |
        +-----+-----+-----+--

    """

    textures_ids = faces.textures_idx
    materials_idx = faces.materials_idx
    verts_uvs = aux.verts_uvs

    num_materials = torch.unique(materials_idx).shape[0]

    try:
        new_verts_uvs = verts_uvs.clone()

        # HACK map verts_uvs to 0 and 1
        new_verts_uvs[new_verts_uvs != 1] %= 1

        for material_id in range(num_materials):
            # apply offsets to horizontal axis
            faces_ids = textures_ids[materials_idx == material_id].unique()
            new_verts_uvs[faces_ids, 0] += material_id

        new_verts_uvs[:, 0] /= num_materials
        new_faces_uvs = faces.textures_idx
    except AttributeError:
        new_verts_uvs = torch.tensor([
            [0, 0],
            [0, 1],
            [1, 0]
        ]).to(device)
        
        num_faces = faces.verts_idx.shape[0]
        new_faces_uvs = torch.tensor([[0, 1, 2]]).to(device).long()
        new_faces_uvs = new_faces_uvs.repeat(num_faces, 1)

    return new_verts_uvs, new_faces_uvs

def load_and_adjust_mesh(mesh_path, device=DEVICE):
    verts, faces, aux = load_obj(mesh_path, device=device)

    dummy_texture = Image.open("./objaverse_eval/assets/white.png").convert("RGB").resize((512, 512))

    # collapse textures of multiple materials to one texture map
    new_verts_uvs, new_faces_uvs = adjust_uv_map(faces, aux)

    return verts, faces, new_verts_uvs, new_faces_uvs, dummy_texture

def collapse_objects(input_path, output_path, device=DEVICE, inplace=False):
    verts, faces, new_verts_uvs, new_faces_uvs, dummy_texture = load_and_adjust_mesh(input_path)
    output_path = input_path if inplace else output_path
    os.makedirs(output_path.parent, exist_ok=True)

    texture_map = transforms.ToTensor()(dummy_texture).to(device)
    texture_map = texture_map.permute(1, 2, 0)

    save_obj(
        str(output_path),
        verts=verts,
        faces=faces.verts_idx,
        decimal_places=5,
        verts_uvs=new_verts_uvs,
        faces_uvs=new_faces_uvs,
        texture_map=texture_map
    )

def remove_tails(mtl_path):
    with open(mtl_path) as f:
        mtl_data = [l.rstrip() for l in f.readlines()]

    with open(mtl_path, "w") as f:
        for l in mtl_data:
            if "map_Bump" not in l and "map_Kd" not in l:
                f.write(l+'\n')

if __name__ == "__main__":

    blender_cmd = str(BLENDER_EXE)
    
    objaverse_subset = get_objaverse_subset()
    subset_uids = [name.split("_")[-1] for name in objaverse_subset]
    uid_to_name = dict(zip(subset_uids, objaverse_subset))

    TEMP_DOWNLOAD_DIR = Path(DATA_DIR) / "temp_objaverse_cache"
    objaverse._VERSIONED_PATH = str(TEMP_DOWNLOAD_DIR)

    print("=> downloading objects...")
    objects = objaverse.load_objects(subset_uids)
    
    print("=> processing...")
    for uid, original_path_str in tqdm(objects.items()):

        original_path = Path(original_path_str)
        base_name = uid_to_name.get(uid, uid).replace(" ", "_")
        new_filename = f"{base_name}.glb"

        target_glb_path = Path(GLB_DIR) / new_filename
        shutil.move(str(original_path), str(target_glb_path))

        cmd = [
            str(BLENDER_EXE), # NOTE please make sure you installed blender
            "--background", "--factory-startup", "--python", "objaverse_eval/objaverse_data_utils/blender_process_glb.py", "--",
            str(target_glb_path),
            OBJ_DIR
        ]
        _ = subprocess.call(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

        expected_output_dir = Path(OBJ_DIR) / target_glb_path.stem
        obj_path = expected_output_dir / "mesh.obj"
        mtl_path = expected_output_dir / "mesh.mtl"

        remove_tails(mtl_path)
        collapse_objects(obj_path, obj_path, DEVICE, inplace=True)
        parameterize_mesh(obj_path, obj_path) # xatlas produces great UVs

    shutil.rmtree(TEMP_DOWNLOAD_DIR)
    print("=> done!")