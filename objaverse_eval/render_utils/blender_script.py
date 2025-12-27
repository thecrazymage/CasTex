"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

Example usage:
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 12 \
        --camera_dist 1.2

Here, input_model_paths.json is a json file containing a list of paths to .glb.
"""

import os
import sys
import math
import time
import random
import argparse
import urllib.request
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set, Tuple

import bpy
from mathutils import Vector


IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default="./views")
parser.add_argument(
    "--engine", type=str, default="BLENDER_EEVEE", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--camera_dist", type=float, default=1.5)
parser.add_argument("--trajectory", type=str, default="frames")
parser.add_argument("--num_images", type=int, default=60)
parser.add_argument("--env_map_path", type=str, default='objaverse_eval/assets/studio_small_06_2k.hdr')
parser.add_argument("--env_map_strength", type=float, default=1.0)
parser.add_argument("--device", type=str, default="OPTIX", choices=["CUDA", "OPTIX", "CPU"])

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

context = bpy.context
scene = context.scene
render = scene.render

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 512
render.resolution_y = 512
render.resolution_percentage = 100

if args.device == "CPU":
    scene.cycles.device = "CPU"
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
else:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = args.device
    prefs.get_devices()
    
    for d in prefs.devices:
        d.use = (d.type == args.device.upper())

    if not any(d.use for d in prefs.devices):
        raise RuntimeError(f"No {args.device} devices found")

    scene.cycles.device = "GPU"
    scene.cycles.denoiser = 'OPTIX' if args.device.upper() == 'OPTIX' else 'OPENIMAGEDENOISE'

scene.cycles.samples = 32
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.use_persistent_data = True
scene.render.film_transparent = True

def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )


def add_lighting() -> None:
    # delete the default light
    bpy.data.objects["Light"].select_set(True)
    bpy.ops.object.delete()
    nodeTree = scene.world.node_tree
    nodes = nodeTree.nodes
    nodes.clear()

    env = nodes.new(type='ShaderNodeTexEnvironment')
    env.image = bpy.data.images.load(args.env_map_path)
    output = nodes.new(type='ShaderNodeOutputWorld')

    bg = nodes.new(type='ShaderNodeBackground')
    bg.inputs['Strength'].default_value = args.env_map_strength
    links = nodeTree.links
    links.new(env.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], output.inputs["Surface"])


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True)
    else:
        import_function(filepath=object_path)

    if file_extension == 'obj':
        mesh_to_render = bpy.context.selected_objects[0]
        material = mesh_to_render.data.materials[0]
        dirname = os.path.dirname(os.path.realpath(object_path))
        # roughness
        texture_path = os.path.join(dirname, 'texture_r.png')
        if os.path.exists(texture_path):
            print('Importning roughness map...')
            load_texture(material, texture_path, 'Roughness')
        else:
            print('Unable to locate roughness map...')
        # metallic
        texture_path = os.path.join(dirname, 'texture_m.png')
        if os.path.exists(texture_path):
            print('Importning metallic map...')
            load_texture(material, texture_path, 'Metallic')
        else:
            print('Unable to locate metallic map...')
        # specular 
        texture_path = os.path.join(dirname, 'texture_ks.png')
        if os.path.exists(texture_path):
            print('Importning specular map...')
            load_texture(material, texture_path, 'Specular')
        else:
            print('Unable to locate specular map...')
        # TODO
        texture_path = os.path.join(dirname, 'texture_n.png')
        if os.path.exists(texture_path):
            print('Importning normal map...')
            load_texture(material, texture_path, 'Normal')
        else:
            print('Unable to locate normal map...')
        # TODO
        mesh_to_render.data.materials[0] = material


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def shade_smooth():
    for obj in scene_meshes():
        for poly in obj.data.polygons:
            poly.use_smooth = True

def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

def setup_camera():
    cam = scene.objects["Camera"]
    cam.location = (0, 1.2, 0)
    cam.data.lens = 35
    cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint

def fid_trajectory():
    phis = (
        [0.5 * math.pi * (0.5 + i) for i in range(4)]
        + [0.25 * math.pi * i for i in range(8)]
        + [0.25 * math.pi * i for i in range(8)]
    )
    thetas = 4 * [0,] + 8 * [math.pi / 6] + 8 * [math.pi / 3]
    for phi, theta in zip(phis, thetas):
        point = (
            args.camera_dist * math.cos(theta) * math.cos(phi),  # x
            args.camera_dist * math.cos(theta) * math.sin(phi),  # y
            args.camera_dist * math.sin(theta),                  # z
        )
        yield point

def video_trajectory():
    for i in range(args.num_images):
        t = 2 * math.pi * (i / args.num_images)
        point = (
            args.camera_dist * math.cos(t) * (1 - (0.25 * math.sin(t)) ** 2) ** 0.5,
            args.camera_dist * math.sin(t) * (1 - (0.25 * math.sin(t)) ** 2) ** 0.5,
            args.camera_dist * 0.25 * math.sin(t),
        )
        yield point

def save_images(object_file: str) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    reset_scene()
    # load the object
    load_object(object_file)
    shade_smooth()
    parent_dir_name = os.path.basename(os.path.dirname(object_file))
    if parent_dir_name.startswith("model_stage_"):
        object_uid = os.path.basename(os.path.dirname(os.path.dirname(object_file)))
    else:
        object_uid = os.path.splitext(os.path.basename(object_file))[0]
    normalize_scene()
    add_lighting()
    cam, cam_constraint = setup_camera()
    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    if args.trajectory == 'frames':
        trajectory = fid_trajectory
    elif args.trajectory == 'video':
        trajectory = video_trajectory
    for i, point in enumerate(trajectory()):
        # set the camera position
        cam.location = point
        # render the image
        scene.render.filepath = os.path.join(args.output_dir, args.trajectory, object_uid, f"{i:03d}.png")
        bpy.ops.render.render(write_still=True)


def load_texture(material, texture_path, input_name):
    bsdf = material.node_tree.nodes['Principled BSDF']
    texture = material.node_tree.nodes.new('ShaderNodeTexImage')
    texture.image = bpy.data.images.load(texture_path)
    # Trying to compensate for gamma-correction
    texture.image.colorspace_settings.name = 'Non-Color'
    if input_name == 'Normal':
        normal_map_node = material.node_tree.nodes.new('ShaderNodeNormalMap')
        normal_map_node.inputs['Strength'].default_value = 1.0
        texture.image.colorspace_settings.name = 'Non-Color'
        material.node_tree.links.new(
            normal_map_node.inputs['Color'],
            texture.outputs['Color']
        )
        material.node_tree.links.new(
            bsdf.inputs[input_name],
            normal_map_node.outputs['Normal']
        )
    else:
        material.node_tree.links.new(
            bsdf.inputs[input_name],
            texture.outputs['Color']
        )


if __name__ == "__main__":
    try:
        start_i = time.time()
        local_path = args.object_path
        save_images(local_path)
        end_i = time.time()
        print("Finished", local_path, "in", end_i - start_i, "seconds")
    except Exception as e:
        print("Failed to render", args.object_path)
        print(e)
