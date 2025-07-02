import subprocess
import os
import yaml

# Set project root
repo_root = os.path.abspath(os.path.dirname(__file__))

# Load config from YAML
config_path = os.path.join(repo_root, "input_config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Get Blender executable from config
blender_executable = config["blender_executable"]

# Path to the Blender script that runs inside Blender
blender_script = os.path.join(repo_root, "blender_script.py")

# Launch Blender in background
subprocess.run([
    blender_executable,
    "--background",
    "--python", blender_script
])
