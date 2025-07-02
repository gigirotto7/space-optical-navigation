"""
blender_functions.py

This module contains functions to set up and animate a Blender scene,
import objects and materials from external blend files, create backgrounds,
and process video frames with noise effects.
"""

import bpy
import csv
import os
import sys
import cv2
import numpy as np
import yaml
repo_root = os.path.abspath(os.path.dirname(__file__))
config_path = os.path.join(repo_root, "input_config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
sys.path.append(config.get("site_packages_path", ""))
   

try:
    from mathutils import Vector, Matrix, Euler, Quaternion
except ImportError:
    print("ERROR: Required mathutils components not found.")
    sys.exit(1)

# scene utilities
def clear_scene_except_world():
    """
    Clears all objects from the scene except those related to the world settings.
    """
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.objects:
        bpy.data.objects.remove(block, do_unlink=True)

# object import functions
def import_object(blendfile, obj_name):
    """
    Imports an object from a .blend file.
    
    Parameters:
      blendfile (str): Path to the .blend file.
      obj_name (str): Name of the object to import.
      
    Returns:
      The imported object if successful, otherwise None.
    """
    with bpy.data.libraries.load(blendfile, link=False) as (data_from, data_to):
        if obj_name in data_from.objects:
            data_to.objects = [obj_name]
        else:
            print(f"Error: Object '{obj_name}' not found in file '{blendfile}'")
            return None
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)
            return obj
    return None

def import_object_with_materials(blendfile, obj_name):
    """
    Imports an object along with all its associated materials from a .blend file.
    
    Parameters:
      blendfile (str): Path to the .blend file.
      obj_name (str): Name of the object to import.
      
    Returns:
      The imported object if successful, otherwise None.
    """
    imported_object = None
    with bpy.data.libraries.load(blendfile, link=False) as (data_from, data_to):
        if obj_name in data_from.objects:
            data_to.objects = [obj_name]
        else:
            print(f"Error: Object '{obj_name}' not found in file '{blendfile}'")
            return None
        # Import all available materials
        data_to.materials = data_from.materials

    for obj in data_to.objects:
        if obj:
            bpy.context.collection.objects.link(obj)
            imported_object = obj

    # Link imported materials to the object
    if imported_object and imported_object.data.materials:
        for mat in data_to.materials:
            if mat and mat.name not in bpy.data.materials:
                bpy.data.materials.append(mat)
            imported_object.data.materials.append(mat)
    return imported_object

# background setup
def setup_background_plane_2D(image_path, camera, target, size, distance):
    """
    Sets up a 2D background image on a plane positioned behind the target from the camera's view.
    
    Parameters:
      image_path (str): Path to the background image file.
      camera (Object): The camera object.
      target (Object): The target object (e.g., CubeSat).
      size (float): Size of the background plane.
      distance (float): Distance from the target.
    """
    
    bpy.ops.mesh.primitive_plane_add(size=size)
    plane = bpy.context.active_object
    plane.name = "BackgroundPlane"

    # calculate camera's forward direction
    forward_vector = camera.matrix_world.to_3x3() @ Vector((0, 0, -1))
    forward_vector.normalize()

    # position and orient the plane relative to the target
    plane.location = target.location + (forward_vector * distance)
    plane.rotation_euler = camera.rotation_euler

    # create material with image texture
    mat = bpy.data.materials.new(name="BackgroundMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    tex_node = nodes.new(type="ShaderNodeTexImage")
    tex_node.location = (-400, 0)
    tex_node.image = bpy.data.images.load(image_path)

    bsdf_node = nodes.new(type="ShaderNodeBsdfDiffuse")
    bsdf_node.location = (0, 0)

    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    output_node.location = (400, 0)

    links.new(tex_node.outputs["Color"], bsdf_node.inputs["Color"])
    links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])
    plane.data.materials.append(mat)

    print(f"Background plane set up with image: {image_path}")
    

def setup_deep_space_background():
    """
    Sets up a deep space (black) background.
    """
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    nodes.clear()

    out_node = nodes.new("ShaderNodeOutputWorld")
    out_node.location = (400, 0)

    bg_node = nodes.new("ShaderNodeBackground")
    bg_node.location = (0, 0)
    bg_node.inputs["Strength"].default_value = 0.1
    bg_node.inputs["Color"].default_value = (0, 0, 0, 1)

    links.new(bg_node.outputs["Background"], out_node.inputs["Surface"])
    print("Deep space background created")

# lighting and camera setup
def setup_sunlight(angle, energy):
    """
    Adds or updates a sunlight object in the scene.
    
    Parameters:
      angle (tuple): Euler angles for the light rotation.
      energy (float): Light energy.
    """
    sunlight = bpy.data.objects.get("Sunlight")
    if not sunlight:
        bpy.ops.object.light_add(type='SUN')
        sunlight = bpy.context.active_object
        sunlight.name = "Sunlight"
    sunlight.rotation_euler = angle
    sunlight.data.energy = energy
    sunlight.data.angle = 0.53

def setup_camera(target, distance, fov):
    """
    Sets up a camera focusing on the target object.
    
    Parameters:
      target (Object): The target object (e.g., CubeSat).
      distance (Vector): Offset distance from the target.
      fov (float): Field of view in degrees.
      
    Returns:
      The camera object.
    """
    camera = bpy.data.objects.get("Camera")
    if not camera:
        bpy.ops.object.camera_add()
        camera = bpy.context.active_object
        camera.name = "Camera"

    camera.location = target.location + distance

    # calculate proper orientation for the camera to look at the target
    direction = -(target.location - camera.location).normalized()
    x_axis = Vector((0, 0, 1)).cross(direction).normalized()
    y_axis = direction.cross(x_axis).normalized()

    camera.matrix_world = Matrix([
        x_axis.to_4d(),
        y_axis.to_4d(),
        direction.to_4d(),
        camera.location.to_4d()
    ]).transposed()
    
    bpy.context.scene.camera = camera
    camera.data.lens_unit = 'FOV'
    camera.data.angle = fov * (3.14159265 / 180)
    return camera

def animate_camera_motion(camera, target, frame_start, frame_end, fps):
    """
    Animates the camera with subtle sinusoidal motion while keeping the target in view.
    
    Parameters:
      camera (Object): The camera to animate.
      target (Object): The target object (e.g., CubeSat).
      frame_start (int): Start frame of the animation.
      frame_end (int): End frame of the animation.
      fps (int): Frames per second.
    """
    total_time = (frame_end - frame_start) / fps
    initial_position = camera.location.copy()
    initial_rotation = tuple(camera.rotation_euler)

    position_amplitude = 0.1  # meters
    rotation_amplitude = 0.05  # radians

    for frame in range(frame_start, frame_end + 1):
        t = (frame - frame_start) / (frame_end - frame_start)
        camera.location = initial_position + Vector((
            position_amplitude * np.sin(2 * np.pi * t),
            position_amplitude * np.sin(2 * np.pi * t * 0.5),
            position_amplitude * np.cos(2 * np.pi * t)
        ))
        new_rotation = (
            initial_rotation[0] + rotation_amplitude * np.sin(2 * np.pi * t * 0.5),
            initial_rotation[1] + rotation_amplitude * np.cos(2 * np.pi * t),
            initial_rotation[2] + rotation_amplitude * np.sin(2 * np.pi * t)
        )
        camera.rotation_euler = Euler(new_rotation, 'XYZ')
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)

    print("Camera animation applied.")

# CubeSat movement functions
def random_cubesat_movement(cubesat, frame_start, frame_end, fps, angular_velocities):
    """
    Applies random rotational movement to the CubeSat.
    
    Parameters:
      cubesat (Object): The CubeSat object.
      frame_start (int): Start frame.
      frame_end (int): End frame.
      fps (int): Frames per second.
    """
    cubesat.keyframe_insert(data_path="location", frame=frame_start)
    cubesat.keyframe_insert(data_path="location", frame=frame_end)

    total_time = (frame_end - frame_start) / fps
    final_rotations = [v * total_time for v in angular_velocities]

    cubesat.rotation_euler = (0, 0, 0)
    cubesat.keyframe_insert(data_path="rotation_euler", frame=frame_start)
    cubesat.rotation_euler = final_rotations
    cubesat.keyframe_insert(data_path="rotation_euler", frame=frame_end)

def apply_movement_from_csv(cubesat, csv_path, column_map, orientation_format="Euler", fps=24):
    """
    Applies movement and orientation to the CubeSat based on CSV data.
    
    Parameters:
      cubesat (Object): The CubeSat object.
      csv_path (str): Path to the CSV file.
      column_map (dict): Mapping for time, position, and orientation columns.
      orientation_format (str): 'Euler' or 'Quaternion'.
      fps (int): Frames per second.
    """
    if not os.path.exists(csv_path):
        raise ValueError(f"CSV file does not exist: {csv_path}")

    with open(csv_path, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        reader.fieldnames = [key.strip() for key in reader.fieldnames]
        print("CSV header keys:", reader.fieldnames)

        for row in reader:
            if column_map["time"] not in row:
                raise KeyError(f"Column '{column_map['time']}' not found. Available: {row.keys()}")

            time_val = float(row[column_map["time"]])
            frame = int(time_val * fps)

            position = Vector((
                float(row[column_map["position"][0]]),
                float(row[column_map["position"][1]]),
                float(row[column_map["position"][2]])
            ))

            if orientation_format == "Euler":
                rotation = Euler((
                    float(row[column_map["orientation"][0]]),
                    float(row[column_map["orientation"][1]]),
                    float(row[column_map["orientation"][2]])
                ), 'XYZ')
                cubesat.rotation_euler = rotation
            elif orientation_format == "Quaternion":
                rotation = Quaternion((
                    float(row[column_map["orientation"][0]]),
                    float(row[column_map["orientation"][1]]),
                    float(row[column_map["orientation"][2]]),
                    float(row[column_map["orientation"][3]])
                ))
                cubesat.rotation_mode = 'QUATERNION'
                cubesat.rotation_quaternion = rotation

            cubesat.location = position
            cubesat.keyframe_insert(data_path="location", frame=frame)
            if orientation_format == "Euler":
                cubesat.keyframe_insert(data_path="rotation_euler", frame=frame)
            else:
                cubesat.keyframe_insert(data_path="rotation_quaternion", frame=frame)

# video processing
def add_gaussian_noise(frame, mean=0, var=0.01):
    """
    Adds Gaussian noise to a video frame.
    
    Parameters:
      frame (ndarray): Input image frame.
      mean (float): Mean of the Gaussian distribution.
      var (float): Variance of the noise.
      
    Returns:
      Noisy frame as an ndarray.
    """
    row, col, ch = frame.shape
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (row, col, ch))
    noisy_frame = np.clip(frame + gaussian * 255, 0, 255).astype(np.uint8)
    return noisy_frame

def add_salt_and_pepper_noise(frame, amount=0.0001):
    """
    Adds salt-and-pepper noise to a video frame.
    
    Parameters:
      frame (ndarray): Input image frame.
      amount (float): Fraction of pixels to be altered.
      
    Returns:
      Noisy frame as an ndarray.
    """
    noisy_frame = frame.copy()
    total_pixels = frame.shape[0] * frame.shape[1]
    num_salt = int(total_pixels * amount * 0.5)
    num_pepper = int(total_pixels * amount * 0.5)

    salt_coords = (np.random.randint(0, frame.shape[0], num_salt),
                   np.random.randint(0, frame.shape[1], num_salt))
    noisy_frame[salt_coords] = 255

    pepper_coords = (np.random.randint(0, frame.shape[0], num_pepper),
                     np.random.randint(0, frame.shape[1], num_pepper))
    noisy_frame[pepper_coords] = 0

    return noisy_frame

def corrupt_video(input_path, output_path, noise):
    """
    Processes a video by applying noise effects frame by frame.
    
    Parameters:
      input_path (str): Path to the input video.
      output_path (str): Path to save the processed video.
      noise (str): Type of noise to apply ('Gaussian' or 'Salt and Pepper').
    """
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if "gaussian" in noise.lower():
            noisy_frame = add_gaussian_noise(frame, mean=0, var=0.01)
        elif "salt" in noise.lower():
            noisy_frame = add_salt_and_pepper_noise(frame, amount=0.001)
        else:
            noisy_frame = frame
        out.write(noisy_frame)

    cap.release()
    out.release()
    print(f"Corrupted video saved to {output_path}")
