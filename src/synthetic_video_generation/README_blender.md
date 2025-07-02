Synthetic CubeSat Video Generator with Blender
==============================================

This project automates the generation of synthetic CubeSat videos using Blender's Python API. It sets up lighting,
camera positioning, background environment, and satellite motion. Then it renders an animation and applies optional video noise.

--------------------------------------------------------------------------------

Project Overview
----------------

Folder Structure:

```
fpemtpy/
├── src/
│   └── synthetic_video_generation/
│       ├── blender_main.py           # Launches Blender in background
│       ├── blender_script.py         # Code that runs inside Blender
│       ├── blender_functions.py      # Blender helper functions
│       ├── blender_automatic_generation.py  # Batch video generator used in the thesis
│       ├── input_config_main.yaml    # Config for single video generation
│       ├── input_config_batch.yaml   # Config for batch generation from Excel
│       ├── run_blender_main.py       # Optional CLI runner
│       ├── setup_blender_env.py      # Environment setup script
│       └── requirements_blender.txt  # Python dependencies
├── templates/
│   ├── blender_models/               # .blend 3D models (e.g. CubeSat)
│   ├── backgrounds/                  # Space and sky background images
├── results/                          # Output folder for rendered and corrupted videos
```
--------------------------------------------------------------------------------

Setup Instructions (macOS and Windows)
--------------------------------------

1. Find Blender’s Python Path
   - macOS:
     Run this in terminal:
     /Applications/Blender.app/Contents/MacOS/Blender --background --python-expr "import sys; print(sys.executable)"

   - Windows:
     Open Blender > Scripting tab, then run in the Python console:
     import sys; print(sys.executable)

   Save this path when prompted by the setup script.

2. Install Python Packages into Blender. 
   Navigate to the project directory and run:

       setup_blender_env.py

   This installs all required packages into Blender’s Python environment. You will need to input the path found above when asked.

3. Find Your `site-packages` Path. 
   After installation, run:

   - macOS:
     /Applications/Blender.app/Contents/Resources/4.2/python/bin/python3.11 -m site --user-site

   - Windows:
     path\to\blender\python.exe -m site --user-site

   Copy the resulting path.

4. Update input_config_main.yaml or input_config_batch.yaml
   Open the config file and make the following changes:

   - Add the path from step 3 to:

         site_packages_path: "your/output/from/step/3"

   - Also set the path to your Blender executable:

         blender_executable: "/Applications/Blender.app/Contents/MacOS/Blender"

     (or your Blender path on Windows)

--------------------------------------------------------------------------------

Running the Generator (Single Video)
------------------------------------

1. Edit `input_config_main.yaml` to configure:
   - Input and output paths
   - Noise options
   - Cubesat model and background image
   - `site_packages_path` and `blender_executable` (as described above)

2. Run the project with:

       python blender_main.py

You do not need to run anything else.  
`blender_main.py` handles everything: loading the config, launching Blender in background mode, and running the scene generation.

--------------------------------------------------------------------------------

Batch Generation: blender_automatic_generation.py
--------------------------------------------------

The script `blender_automatic_generation.py` was developed specifically for generating all the CubeSat videos used in the thesis experiments.

- It reads parameters from an Excel file (e.g. camera position, model, lighting).
- For each row (video ID), it loads a new configuration and runs a Blender render.
- The output is stored under the `results/` folder, one video per iteration.
- Configuration for this script is stored in: `input_config_batch.yaml`

This allows high-throughput generation of many controlled variations of rendered videos using a consistent and reproducible setup.

--------------------------------------------------------------------------------

Notes
-----

- If you see a ModuleNotFoundError (e.g., for cv2), confirm your `site_packages_path` is set correctly.
- Blender does not automatically load system-level Python packages. This field helps make them available.
- Only run `blender_main.py` or `blender_automatic_generation.py`. Each script is self-contained.