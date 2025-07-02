import bpy
import sys
import os
import subprocess
import pandas as pd
import numpy as np
import yaml


# Vector fallback (for testing outside Blender)
try:
    from mathutils import Vector
except ImportError:
    class Vector:
        def __init__(self, values):
            self.values = values
        def __repr__(self):
            return f"Vector({self.values})"

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Load config
with open(os.path.join(script_dir, "batch_config.yaml"), "r") as f:
    config = yaml.safe_load(f)

sys.path.append(config["site_packages_path"])
blender_executable = config["blender_executable"]
excel_path = config["excel_path"]

df = pd.read_excel(excel_path)

# Mapping dictionaries
cubesat_file_map = config["cubesat_file_map"]
sun_energy_map = config["sun_energy_map"]
sun_angle_map = config["sun_angle_map"]
sat_movement_map = config["sat_movement_map"]
camera_distance_map = {int(k): Vector(v) for k, v in config["camera_distance_map"].items()}
fov_map = config["fov_map"]
noise_map = config["noise_map"]
background_image_path_map = config["background_image_path_map"]

frame_start, frame_end, fps = config["frame_start"], config["frame_end"], config["fps"]
angular_velocities = tuple(config["angular_velocities"])

for i in range(0, len(df)):
    row = df.iloc[i]

    cubesat_file = cubesat_file_map[str(row["cubesat_file"])]
    sun_energy = sun_energy_map[str(row["sun_energy"])]
    sun_angle = tuple(sun_angle_map[str(row["sun_angle"])])
    sat_movement = sat_movement_map[str(row["sat_movement"])]
    camera_distance = camera_distance_map[int(row["camera_distance"])]
    fov = fov_map[str(row["fov"])]
    noise = noise_map[str(row["noise"])]
    background_image_path = background_image_path_map[str(row["background_image_path"])]
    output_path = os.path.join(config["output"],row["output_path"])

    if noise == "Salt":
        row2 = df.iloc[i-3]
        input_path = row2["output_path"]
    elif noise == "Gaussian":
        row2 = df.iloc[i-6]
        input_path = row2["output_path"]
    else:
        input_path = "None"

    subprocess.run([
        blender_executable, "--background", "--python-expr",
        f'''
import bpy
import sys
sys.path.append("{config["site_packages_path"]}")
sys.path.append("{script_dir}")
import blender_functions as functions
from math import tan, pi
from mathutils import Vector

functions.clear_scene_except_world()
functions.setup_sunlight({sun_angle}, {sun_energy})

cubesat = functions.import_object("{cubesat_file}", "Cubesat")
if cubesat:
    cubesat.location = Vector((0, 0, 680))
    cubesat.scale = (0.001, 0.001, 0.001)
    for material in cubesat.data.materials:
        if material:
            material.blend_method = material.shadow_method = 'OPAQUE'

functions.setup_camera(cubesat, {camera_distance}, {fov})
camera = bpy.data.objects.get("Camera")
functions.animate_camera_motion(camera, cubesat, {frame_start}, {frame_end}, {fps})

x_distance = {camera_distance}[0]
size = 2 * (tan(({fov} / 2) * pi / 180)) * (x_distance + 5) + 2.5

if "{background_image_path}":
    functions.setup_background_plane_2D("{background_image_path}", camera, cubesat, size=size, distance=5)
else:
    functions.setup_deep_space_background()

scene = bpy.context.scene
scene.frame_start = {frame_start}
scene.frame_end = {frame_end}
scene.render.resolution_x = scene.render.resolution_y = 1024
scene.render.resolution_percentage = 100
scene.render.fps = {fps}
scene.render.filepath = "{output_path}"
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.ffmpeg.constant_rate_factor = 'HIGH'
scene.render.ffmpeg.ffmpeg_preset = 'GOOD'

functions.random_cubesat_movement(cubesat, {frame_start}, {frame_end}, {fps}, {angular_velocities})

if "{noise}" == "None":
    bpy.ops.render.render(animation=True)
    print(f"Rendered animation saved to: {output_path}")
else:
    functions.corrupt_video("{input_path}", "{output_path}", noise="{noise}")
    print(f"Corrupted video with {noise} noise saved to: {output_path}")

bpy.ops.wm.quit_blender()
'''
    ])

    print(f"Iteration {i + 1} complete. Blender restarted.\n")

print("Simulation ended")
