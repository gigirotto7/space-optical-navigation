"""
blender_main.py

This script sets up the Blender scene by clearing existing objects, adding lighting,
importing a CubeSat object, setting up the camera and background, animating the scene,
rendering the animation, and finally applying a noise effect to the resulting video.
"""
import bpy
import numpy as np
import os
import sys
import yaml
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

config_path = os.path.join(script_dir, "input_config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
sys.path.append(config.get("site_packages_path", ""))

import blender_functions as functions
from mathutils import Vector, Matrix, Quaternion, Euler


# Resolve paths relative to repo root
def resolve_path(rel_path):
    return os.path.join(script_dir, rel_path) if rel_path else None

sun_angle = tuple(config["sun_angle"])
sun_energy = config["sun_energy"]
sat_movement_mode = config["sat_movement_mode"]
sat_file = resolve_path(config.get("sat_file"))
column_map = {
    "time": "time",
    "position": ["pos_x", "pos_y", "pos_z"],
    "orientation": ["rot_x", "rot_y", "rot_z"]
}

camera_distance = Vector(config["camera_distance"])
fov = config["fov"]
noise = config["noise"]

cubesat_file = resolve_path(config["cubesat_file"])
background_image_path = resolve_path(config["background_image_path"])
output_video_path = resolve_path(config["output_video_path"])
corrupted_video_path = resolve_path(config["corrupted_video_path"])

frame_start = config["frame_start"]
frame_end = config["frame_end"]
fps = config["fps"]


## Scene setup

# clear existing scene objects 
functions.clear_scene_except_world()

# set up sunlight
functions.setup_sunlight(sun_angle, sun_energy)

# import the CubeSat object
cubesat = functions.import_object(cubesat_file, "Cubesat")
if cubesat:
    # adjust position and scale as needed
    cubesat.location = Vector((0, 0, 680))
    cubesat.scale = (0.001, 0.001, 0.001) # [mm] (.STEP) to [m] in .blend
    for material in cubesat.data.materials:
        if material:
            material.blend_method = material.shadow_method = 'OPAQUE'

# set up the camera focusing on the CubeSat
camera = functions.setup_camera(cubesat, camera_distance, fov)

# optional - add animation to the camera also
functions.animate_camera_motion(camera, cubesat, frame_start, frame_end, fps)

x_distance = camera_distance[0]
size = 2 * (np.tan((fov / 2) * np.pi / 180)) * (x_distance + 5) + 2.5

if background_image_path:
    functions.setup_background_plane_2D(background_image_path, camera, cubesat, size=size, distance=5)
else:
    functions.setup_deep_space_background()

# configure scene render settings
scene = bpy.context.scene
scene.frame_start = frame_start
scene.frame_end = frame_end
scene.render.resolution_x = scene.render.resolution_y = 1024
scene.render.resolution_percentage = 100
scene.render.fps = fps

# CubeSat animation
if sat_movement_mode == 'Random':
    functions.random_cubesat_movement(cubesat, frame_start, frame_end, fps, angular_velocities = (0.5, 0.5, 1.5))
elif sat_movement_mode == 'File' and sat_file:
    functions.apply_movement_from_csv(cubesat, sat_file, column_map, orientation_format="Euler", fps=24)
else:
    print("No valid CubeSat movement mode selected.")

# render and video processing

# set up rendering parameters
scene.render.filepath = output_video_path
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.ffmpeg.constant_rate_factor = 'HIGH'
scene.render.ffmpeg.ffmpeg_preset = 'GOOD'

# render the animation
bpy.ops.render.render(animation=True)
print(f"Rendered animation saved to: {output_video_path}")

# apply noise effect to the rendered video (supports "Gaussian" or "Salt and Pepper")
functions.corrupt_video(output_video_path, corrupted_video_path, noise='Salt and Pepper')
print(f"Corrupted video saved to: {corrupted_video_path}")
