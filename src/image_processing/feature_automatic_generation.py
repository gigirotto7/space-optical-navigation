import cv2
import numpy as np
import pandas as pd
import time
import os
import yaml

from feature_algorithms import (
    SIFT,
    ORB,
    BRISK,
    AKAZE,
    KLT,
)



def extract_video_name(video_path):
    """Extracts the video filename without extension from the path."""
    return os.path.splitext(os.path.basename(video_path))[0]

def select_roi(video_path):
    """
    Select ROI for all videos.
    """
    cap = cv2.VideoCapture(video_path)
    ret, sample_frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read a sample frame to define ROI.")
        cap.release()
        return None
    
    

    print("Select ROI, press ENTER or SPACE to confirm.")
    single_roi = cv2.selectROI("Single ROI Selection", sample_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows() 
    cap.release()  

    if single_roi[2] == 0 or single_roi[3] == 0:
        print("No valid ROI selected.")
        return None
    return single_roi 

def analyze_video(video_path, single_roi, excel_filename="analysis_results.xlsx", pop_windows = False):
    """
    1) Define ROI once.
    2) Pass that ROI to each processor's constructor so they skip interactive selection.
    3) Gather and save metrics (including outlier and tracking length analysis).
    """

    video_name = extract_video_name(video_path)

    if single_roi is None:
        print(f" Skipping {video_name} - No ROI defined.")
        return

    # 2) Define the methods 
    methods = [
        ("SIFT without RANSAC", SIFT(video_path, cubesat_roi=single_roi, use_ransac=False)),
        ("SIFT with RANSAC", SIFT(video_path, cubesat_roi=single_roi, use_ransac=True)),
        ("ORB without RANSAC", ORB(video_path, cubesat_roi=single_roi, use_ransac=False)),
        ("ORB with RANSAC", ORB(video_path, cubesat_roi=single_roi, use_ransac=True)),
        ("BRISK without RANSAC", BRISK(video_path, cubesat_roi=single_roi, use_ransac=False)),
        ("BRISK with RANSAC", BRISK(video_path, cubesat_roi=single_roi, use_ransac=True)),
        ("AKAZE without RANSAC", AKAZE(video_path, cubesat_roi=single_roi, use_ransac=False)),
        ("AKAZE with RANSAC", AKAZE(video_path, cubesat_roi=single_roi, use_ransac=True)),
        ("KLT without RANSAC", KLT(video_path, cubesat_roi=single_roi, use_ransac=False)),
        ("KLT with RANSAC", KLT(video_path, cubesat_roi=single_roi, use_ransac=True)),
    ]

    results = []
    
    for method_name, processor in methods:
        cv2.destroyAllWindows()
        print(f"\n===== Running {method_name} =====")
        cap = cv2.VideoCapture(video_path)
        ret, _ = cap.read()
        cap.release()
        if not ret:
            print(f" Error: Could not open {video_name}. Skipping...")
            continue
 
        processor.run()
        print(f"Finished {method_name}.")
        cv2.destroyAllWindows()
        
        # Collect metrics
        if processor.total_error_magnitude:
            mean_error = np.mean(processor.total_error_magnitude)
            std_error = np.std(processor.total_error_magnitude)
        else:
            mean_error = 0.0
            std_error = 0.0

        total_matches = processor.total_matches
        total_outliers = sum(processor.outliers_per_frame) if hasattr(processor, 'outliers_per_frame') else 0
        outlier_threshold = getattr(processor, 'outlier_threshold', 20)

        if total_matches > 0:
            outlier_ratio = (total_outliers / total_matches) * 100
        else:
            outlier_ratio = 0

        if processor.processing_times:
            mean_time = np.mean(processor.processing_times)
        else:
            mean_time = 0.0

        if processor.cube_sat_keypoints_percentage:
            mean_roi = np.mean(processor.cube_sat_keypoints_percentage)
        else:
            mean_roi = 0.0

        if processor.tracking_lengths:
            avg_tracking_length = np.mean(processor.tracking_lengths)
        else:
            avg_tracking_length = 0.0

        results.append({
            "Video": video_name,
            "Method": method_name,
            "Mean Error Magnitude": round(mean_error, 2),
            "Total Matches": round(total_matches, 2),
            "Total Outliers": round(total_outliers, 2),
            "Outlier Ratio (%)": round(outlier_ratio, 2),
            "Mean Processing Time (s)": round(mean_time, 5),
            "Mean % of Keypoints in ROI": round(mean_roi, 2),
            "Average Tracking Length (frames)": round(avg_tracking_length, 2)
        })
        del processor
        
    # Save results to Excel
    new_df = pd.DataFrame(results)

    try:
        existing_df = pd.read_excel(excel_filename)
        if "Video" in existing_df.columns:
            existing_df = existing_df[existing_df["Video"] != video_name]
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    except FileNotFoundError:
        final_df = new_df

    final_df.to_excel(excel_filename, index=False)
    print(f"\nAnalysis completed. Results saved to: {excel_filename}")

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "input_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    repo_root = os.path.abspath(os.path.dirname(__file__))

    common_path = config["base_video_path"]
    excel_path = os.path.abspath(os.path.join(repo_root, config["excel_path"]))
    df = pd.read_excel(excel_path)

    output_excel = config["output_excel"]

    first_video_path = os.path.abspath(os.path.join(common_path, df.iloc[0]["output_path"]))
    roi = select_roi(first_video_path)   

    for i in range(len(df)):
        row = df.iloc[i]
        VIDEO_PATH = os.path.abspath(os.path.join(common_path, row["output_path"]))
        analyze_video(VIDEO_PATH, roi, excel_filename=output_excel, pop_windows=False)
