import os
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLENDER_PATH_FILE = os.path.join(SCRIPT_DIR, ".blender_path.txt")
REQUIREMENTS_FILE = os.path.join(SCRIPT_DIR, "requirements_blender.txt")


def get_saved_blender_path():
    if os.path.exists(BLENDER_PATH_FILE):
        with open(BLENDER_PATH_FILE, "r") as f:
            path = f.read().strip()
            if os.path.exists(path):
                return path
    return None


def prompt_for_blender_path():
    print("Please enter the full path to Blender’s internal Python executable.")
    print("Examples:")
    print("  macOS  : /Applications/Blender.app/Contents/Resources/4.2/python/bin/python3.11")
    print("  Windows: C:\\Program Files\\Blender Foundation\\Blender 4.2\\python\\bin\\python.exe")
    print("  Linux  : /path/to/blender/4.2/python/bin/python3.11")
    path = input("Blender Python path: ").strip()
    if os.path.exists(path):
        with open(BLENDER_PATH_FILE, "w") as f:
            f.write(path)
        return path
    print("Error: Path does not exist. Aborting.")
    return None


def install_requirements(blender_python):
    print(f"Installing packages using: {blender_python}")
    subprocess.run([blender_python, "-m", "ensurepip", "--upgrade"])
    subprocess.run([blender_python, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.run([blender_python, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])


if __name__ == "__main__":
    blender_python = get_saved_blender_path()
    if not blender_python:
        blender_python = prompt_for_blender_path()
    if blender_python:
        install_requirements(blender_python)
        print("Blender environment setup complete.")
