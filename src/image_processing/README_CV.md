Image Processing: Tracking Evaluation
=====================================

This module evaluates various feature tracking algorithms on the synthetic CubeSat videos generated via Blender.

--------------------------------------------------------------------------------

Folder Structure:
-----------------
```
fpemtpy/
├── src/
│   └── image_processing/
│       ├── feature_algorithms.py            # Contains all feature tracking classes (SIFT, ORB, BRISK, AKAZE, KLT, etc.)
│       ├── feature_automatic_generation.py  # Applies selected algorithms to each video listed in the Excel
│       ├── main.py                          # Applies selected algorithms to a single specific video      
│       ├── input_config.yaml                  # YAML configuration file with paths and output options
├── results/
│   ├── methods_data.xlsx                    # Output Excel storing performance metrics for each method
```
--------------------------------------------------------------------------------

Overview:
---------

This module provides:
- A set of feature tracking algorithms implemented using OpenCV
- A batch-processing script that runs all methods on multiple videos
- ROI (Region of Interest) selection applied uniformly across all videos
- An Excel report summarizing tracking accuracy and performance

Algorithms supported:
- SIFT
- ORB
- BRISK
- AKAZE
- KLT
- StarFreak
- FastBrief

Metrics collected:
- Mean error magnitude
- Total matches and outliers
- Outlier ratio
- Mean processing time
- Percentage of keypoints inside ROI
- Average tracking length

--------------------------------------------------------------------------------

Configuration:
--------------

The `input_config.yaml` file should include the following fields:

```yaml
excel_path: "../../support/ID_video_generation.xlsx"
base_video_path: "../../../results"
output_excel: "methods_data.xlsx"
use_manual_roi: true
```

- `excel_path`: Path to Excel with a column `output_path` listing the videos
- `base_video_path`: Root directory where videos are stored
- `output_excel`: Where to store the metrics
- `use_manual_roi`: Whether to prompt for interactive ROI selection (set to `true`)

--------------------------------------------------------------------------------

Usage:
------

1. Make sure `input_config.yaml` is properly set up.
2. Run `main.py` to analyse a spacific video. `feature_automatic_generation.py` was greated to batch process all videos created for the thesis. 

```bash
python feature_automatic_generation.py
```

3. When prompted, select an ROI. This ROI is then reused for all videos.
4. The script will loop through each video and evaluate all methods.
5. Results are saved in the Excel file defined in `output_excel`.

--------------------------------------------------------------------------------

Notes:
------

- All videos should be accessible at paths constructed from `base_video_path + output_path`
- The ROI is selected interactively once, on the first video frame
- You can extend the evaluation to include new methods by modifying `feature_algorithms.py`

