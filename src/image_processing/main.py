import cv2
import numpy as np
import pandas as pd
import yaml
import os

from feature_algorithms import (
    SIFT,
    ORB,
    BRISK,
    AKAZE,
    StarFreak,
    FastBrief,
    KLT,
)


def analyze_video(video_path, excel_filename="analysis_results.xlsx", pop_windows=False):
    cap = cv2.VideoCapture(video_path)
    ret, sample_frame = cap.read()
    cap.release()

    if not ret:
        print("Error: could not read a sample frame to define ROI.")
        return

    window_name = "Single ROI Selection"
    print(f"\nSelect ROI once for all methods, press ENTER or SPACE to confirm.")
    single_roi = cv2.selectROI(window_name, sample_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)

    if single_roi[2] == 0 or single_roi[3] == 0:
        print("No valid ROI selected. Exiting.")
        return

    print(f"Single ROI chosen: {single_roi}")

    methods = [
        ("SIFT without RANSAC", SIFT(video_path, cubesat_roi=single_roi, use_ransac=False)),
        ("SIFT with RANSAC", SIFT(video_path, cubesat_roi=single_roi, use_ransac=True)),
        ("ORB without RANSAC", ORB(video_path, cubesat_roi=single_roi, use_ransac=False)),
        ("ORB with RANSAC", ORB(video_path, cubesat_roi=single_roi, use_ransac=True)),
        ("BRISK without RANSAC", BRISK(video_path, cubesat_roi=single_roi, use_ransac=False)),
        ("BRISK with RANSAC", BRISK(video_path, cubesat_roi=single_roi, use_ransac=True)),
        ("AKAZE without RANSAC", AKAZE(video_path, cubesat_roi=single_roi, use_ransac=False)),
        ("AKAZE with RANSAC", AKAZE(video_path, cubesat_roi=single_roi, use_ransac=True)),
        ("StarFreak without RANSAC", StarFreak(video_path, cubesat_roi=single_roi, use_ransac=False)),
        ("StarFreak with RANSAC", StarFreak(video_path, cubesat_roi=single_roi, use_ransac=True)),
        ("FastBrief without RANSAC", FastBrief(video_path, cubesat_roi=single_roi, use_ransac=False)),
        ("FastBrief with RANSAC", FastBrief(video_path, cubesat_roi=single_roi, use_ransac=True)),
        ("KLT without RANSAC", KLT(video_path, cubesat_roi=single_roi, use_ransac=False)),
        ("KLT with RANSAC", KLT(video_path, cubesat_roi=single_roi, use_ransac=True)),
    ]

    results = []

    for method_name, processor in methods:
        print(f"\n===== Running {method_name} =====")
        processor.run()

        mean_error = np.mean(processor.total_error_magnitude) if processor.total_error_magnitude else 0.0
        total_matches = processor.total_matches
        total_outliers = sum(processor.outliers_per_frame) if hasattr(processor, 'outliers_per_frame') else 0
        outlier_ratio = (total_outliers / total_matches) * 100 if total_matches > 0 else 0
        mean_time = np.mean(processor.processing_times) if processor.processing_times else 0.0
        mean_roi = np.mean(processor.cube_sat_keypoints_percentage) if processor.cube_sat_keypoints_percentage else 0.0
        avg_tracking_length = np.mean(processor.tracking_lengths) if processor.tracking_lengths else 0.0

        results.append({
            "Method": method_name,
            "Mean Error Magnitude": round(mean_error, 2),
            "Total Matches": round(total_matches, 2),
            "Outlier Ratio (%)": round(outlier_ratio, 2),
            "Mean Processing Time (s)": round(mean_time, 5),
            "Mean % of Keypoints in ROI": round(mean_roi, 2),
            "Average Tracking Length (frames)": round(avg_tracking_length, 2)
        })

    df = pd.DataFrame(results)
    df.to_excel(excel_filename, index=False)
    print(f"\nAnalysis completed. Results saved to: {excel_filename}")


if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.dirname(__file__))
    config_path = os.path.join(repo_root, "input_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    video_path = os.path.abspath(os.path.join(repo_root, config["video_path"]))
    output_excel = config["output_excel_main"]
    pop_windows = config.get("pop_windows", True)

    analyze_video(video_path, excel_filename=output_excel, pop_windows=pop_windows)
