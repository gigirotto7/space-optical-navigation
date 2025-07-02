import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.stats import norm

 
###############################################################################
# Base Class
###############################################################################
class BaseFeatureProcessor:
    def __init__(self, video_path, cubesat_roi=None, use_ransac = False, pop_windows = False):
        """
        Store the video path and optional pre-defined ROI.
        If cubesat_roi is given, we'll skip the interactive selection in run().
        """
        self.video_path = video_path
        self.cubesat_roi = cubesat_roi 
        self.use_ransac = use_ransac
        self.pop_windows = pop_windows
        

        # Common accumulators for metrics
        self.total_error_magnitude = []     # All errors (distance) across matched features
        self.processing_times = []
        self.total_matches = 0
        self.cube_sat_keypoints_percentage = []
        self.dx_errors = []  # signed differences in x
        self.dy_errors = []  # signed differences in y


        # Accumulators for outliers and tracking
        self.outlier_threshold = 20         # Threshold for outlier in pixels
        self.outliers_per_frame = []        # How many outliers in each pair of frames
        self.per_frame_mean_errors = []     # Mean error in each pair of frames 
        self.feature_id_counter = 0         # To assign unique IDs to features
        self.active_features = {}           # {feature_id: (current_pos, lifetime)}
        self.tracking_lengths = []          # Tracks feature lifetimes

        # For visualization
        self.frame_indices_for_visualization = []
        self.frames_for_visualization = []
        
        #Store per-frame errors for each feature ID
        # { feature_id: [error_frame1, error_frame2, ...] }
        self.feature_errors = {}  

    def select_roi(self, frame, window_name="Select ROI"):
        """
        Lets the user select the ROI with a drag-and-select interface.
        """
        print(f"Please select ROI for {window_name} and press ENTER or SPACE.")
        roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window_name)

        if roi[2] == 0 or roi[3] == 0:
            print("No valid ROI selected. Exiting.")
            return None

        self.cubesat_roi = roi
        return roi

    def is_within_roi(self, pt):
        """
        Checks if a keypoint coordinate (pt) is within the selected ROI.
        """
        if self.cubesat_roi is None:
            return False
        (x, y, w, h) = self.cubesat_roi
        return x <= pt[0] <= x + w and y <= pt[1] <= y + h
        


    def detect_and_compute(self, frame):
        """Subclasses must override."""
        raise NotImplementedError("Subclasses must implement detect_and_compute()")

    def match_descriptors(self, des1, des2, kp1, kp2):
        matches = self.bf.match(des1, des2)
        if not self.use_ransac:
            return matches

        # RANSAC-based filtering
        if len(matches) < 4:
            print(f" Not enough matches for Homography: {len(matches)} found. Skipping RANSAC.")
            return matches 
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        filtered_matches = [m for i, m in enumerate(matches) if mask[i]]
        return filtered_matches
    
    def track_features(self, prev_kps, curr_kps, matches):
        """
        Tracks features across multiple frames using spatial consistency.
        Assigns unique feature IDs and updates them over time.
        """
        new_active_features = {}  # Store feature IDs for this frame
        match2id = {}  # Store match-to-ID mappings

        for match in matches:
            q_idx = match.queryIdx  # Feature index in previous frame
            t_idx = match.trainIdx  # Feature index in current frame

            prev_pt = prev_kps[q_idx].pt  # Previous frame keypoint
            curr_pt = curr_kps[t_idx].pt  # Current frame keypoint

            # Check if this feature was already being tracked
            found = False
            for feature_id, (prev_pos, lifetime) in self.active_features.items():
                if np.linalg.norm(np.array(prev_pos) - np.array(prev_pt)) < 5:  # If movement is small, track feature
                    new_active_features[feature_id] = (curr_pt, lifetime + 1)
                    match2id[(q_idx, t_idx)] = feature_id
                    found = True
                    break

            # If feature is new, assign a new ID
            if not found:
                self.feature_id_counter += 1
                new_active_features[self.feature_id_counter] = (curr_pt, 1)
                match2id[(q_idx, t_idx)] = self.feature_id_counter

        # Track lost features and update active features
        for feature_id, (_, lifetime) in self.active_features.items():
            if feature_id not in new_active_features:
                self.tracking_lengths.append(lifetime)  # Feature is lost, store lifetime

        self.active_features = new_active_features  # Update feature list
        return match2id




    def run(self):
        """
        Main workflow:
          1) Open video & check frames
          2) If cubesat_roi is None, do interactive ROI selection on 1st frame
          3) Process frames, detect & match
          4) Summarize and visualize
        """
        self.active_features = {}  # Reset tracked features
        self.feature_id_counter = 0
        self.total_matches = 0
        self.total_error_magnitude = []
        self.dx_errors = []
        self.dy_errors = []
        self.outliers_per_frame = []
        self.per_frame_mean_errors = []
        self.cube_sat_keypoints_percentage = []
        self.tracking_lengths = []
        self.processing_times = []
        self.feature_errors = {}

        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 4:
            print("Video has fewer than 4 frames. Exiting.")
            cap.release()
            return

        # Pick 4 frames for visualization
        self.frame_indices_for_visualization = [
            0,
            total_frames // 3,
            (2 * total_frames) // 3,
            total_frames - 1
        ]

        # If ROI not given, do interactive selection 
        if self.cubesat_roi is None:
            frame_count = -1
            ret, sample_frame = cap.read()
            frame_count += 1
            if not ret:
                print("Error: Unable to read a sample frame for ROI.")
                cap.release()
                return

            if not self.select_roi(sample_frame, window_name=self.__class__.__name__):
                cap.release()
                return  # ROI invalid

            (x, y, w, h) = self.cubesat_roi
            cv2.rectangle(sample_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            plt.figure(figsize=(8, 6))
            plt.imshow(cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB))
            plt.title(f"Sample Frame with ROI ({self.__class__.__name__})")
            plt.axis("off")
            plt.show()

            # Reset to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = -1
        else:
            frame_count = -1

        # Read first frame
        ret, prev_frame = cap.read()
        frame_count += 1
        if not ret:
            print("Failed to read the first frame. Exiting.")
            cap.release()
            return
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        if frame_count in self.frame_indices_for_visualization:  
            self.frames_for_visualization.append(prev_gray)      
        kp1, des1 = self.detect_and_compute(prev_gray)
        if len(kp1) > 500:
            kp1 = kp1[:500]


        # Main loop
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame_count in self.frame_indices_for_visualization: 
                self.frames_for_visualization.append(gray_frame)      
            
            kp2, des2 = self.detect_and_compute(gray_frame)


            if des1 is None or des2 is None:
                prev_gray, kp1, des1 = gray_frame, kp2, des2
                continue

            # Match descriptors
            start_time = time.perf_counter()
            matches = self.match_descriptors(des1, des2, kp1, kp2)
            matches = [m for m in matches if m.queryIdx < len(kp1) and m.trainIdx < len(kp2)]
            end_time = time.perf_counter()
            self.processing_times.append(end_time - start_time)

            if matches is None or len(matches) == 0:
                prev_gray, kp1, des1 = gray_frame, kp2, des2
                continue

            self.total_matches += len(matches)

            # Track features across frames
            match2id = self.track_features(kp1, kp2, matches)


            # Calculate errors and outliers
            errors = []
            outlier_count = 0
            for match in matches:
                q_idx = match.queryIdx   
                t_idx = match.trainIdx 
                pt1 = kp1[match.queryIdx].pt
                pt2 = kp2[match.trainIdx].pt
                pt1 = kp1[q_idx].pt  # (x1, y1)
                pt2 = kp2[t_idx].pt  # (x2, y2)
                x1, y1 = pt1
                x2, y2 = pt2
                
                # Signed differences
                dx = x2 - x1
                dy = y2 - y1
                self.dx_errors.append(dx)
                self.dy_errors.append(dy)
                distance = np.sqrt(dx*dx + dy*dy)
                errors.append(distance)
                if distance > self.outlier_threshold:
                    outlier_count += 1
                    
            
                if (q_idx, t_idx) in match2id:
                    f_id = match2id[(q_idx, t_idx)]
                else:
                    continue  # Skip if not found

                if f_id not in self.feature_errors:
                    self.feature_errors[f_id] = []
                self.feature_errors[f_id].append(distance)# store the error

            self.total_error_magnitude.extend(errors)
            self.outliers_per_frame.append(outlier_count)
            self.per_frame_mean_errors.append(np.mean(errors) if errors else 0)

            # Filter keypoints based on matches retained by RANSAC
            if self.use_ransac:
                # Get indices of keypoints in kp2 that are retained by RANSAC
                filtered_kp2_indices = {m.trainIdx for m in matches}
                filtered_kp2 = [kp2[i] for i in filtered_kp2_indices]
            else:
                # Use all keypoints when RANSAC is not applied
                filtered_kp2 = kp2

            # Calculate percentage of keypoints in ROI using the filtered keypoints
            if filtered_kp2:
                cubesat_count = sum(1 for kp in filtered_kp2 if self.is_within_roi(kp.pt))
                pct_roi = (cubesat_count / len(filtered_kp2)) * 100
            else:
                pct_roi = 0

            self.cube_sat_keypoints_percentage.append(pct_roi)


            # Update for next iteration
            prev_gray, kp1, des1 = gray_frame, kp2, des2

        cap.release()

        # Finalize tracking lengths
        for _, lifetime in self.active_features.values():
            self.tracking_lengths.append(lifetime)

        # Report metrics
        self.report_metrics()
        # Show final per-feature error progression
        #if self.pop_windows: 
        #self.visualize_signed_errors()
        #self.visualize_feature_error_progression() 
        #self.visualize_matches()

    def report_metrics(self):
        """
        Summarize metrics and print them.
        """
        if len(self.total_error_magnitude) > 0:
            mean_error_magnitude = np.mean(self.total_error_magnitude)
        else:
            mean_error_magnitude = 0.0

        if len(self.processing_times) > 0:
            mean_processing_time = np.mean(self.processing_times)
        else:
            mean_processing_time = 0.0

        mean_matches = self.total_matches / len(self.per_frame_mean_errors) if len(self.per_frame_mean_errors) > 0 else 0.0

        mean_roi = (np.mean(self.cube_sat_keypoints_percentage)
                    if self.cube_sat_keypoints_percentage else 0)

        total_outliers = sum(self.outliers_per_frame)
        outlier_ratio = (total_outliers / self.total_matches * 100
                         if self.total_matches > 0 else 0)

        avg_tracking_length = np.mean(self.tracking_lengths) if self.tracking_lengths else 0

        print(f"--- {self.__class__.__name__} RESULTS ---")
        print(f"Mean Error Magnitude: {mean_error_magnitude:.2f}")
        print(f"Mean Number of Matches (per frame pair): {mean_matches:.2f}")
        print(f"Mean Processing Time: {mean_processing_time:.5f}s")
        print(f"Mean % of Keypoints in ROI: {mean_roi:.2f}%")
        print(f"Outlier Threshold: {self.outlier_threshold} px")
        print(f"Total Outliers: {total_outliers},  Outlier Ratio: {outlier_ratio:.2f}%")
        print(f"Average Tracking Length: {avg_tracking_length:.2f} frames")

    def visualize_matches(self):
        """
        Visualize feature matches only between the first two frames of the video.
        """
        # Open the video and extract the first two frames
        cap = cv2.VideoCapture(self.video_path)
        
        ret, frame1 = cap.read()  # Read the first frame
        if not ret:
            print("Error: Could not read the first frame.")
            cap.release()
            return
        
        ret, frame2 = cap.read()  # Read the second frame
        if not ret:
            print("Error: Could not read the second frame.")
            cap.release()
            return

        cap.release()  # Release the video capture

        # Convert frames to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Detect and compute keypoints and descriptors for both frames
        kp1, des1 = self.detect_and_compute(gray1)
        kp2, des2 = self.detect_and_compute(gray2)

        if des1 is None or des2 is None:
            print("No descriptors found in one of the frames.")
            return

        # Match descriptors
        matches = self.match_descriptors(des1, des2, kp1, kp2)
        if not matches:
            print("No matches found between the first two frames.")
            return

        # Sort matches by distance (best matches first)
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw the top 50 matches for visualization
        match_img = cv2.drawMatches(gray1, kp1, gray2, kp2, matches[:50], None, 
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Show the matched keypoints
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

            
    def visualize_feature_error_progression(self):
        """
        Plot each feature's error progression across the frames it was tracked.
        Each line is a separate feature; the x-axis is the 'step' for that feature.
        """
        if not self.feature_errors:
            print("No feature errors recorded.")
            return

        plt.figure(figsize=(8, 6))
        for feature_id, error_list in self.feature_errors.items():
            x_values = list(range(len(error_list)))
            plt.plot(x_values, error_list, label=f"Feat ID {feature_id}", linewidth=0.5 )

        plt.xlabel("Feature Step (each time feature was matched)")
        plt.ylabel("Error (px)")
        plt.title(f"Per-Feature Error Progression ({self.__class__.__name__})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.ylim([0, 20]) 
        plt.tight_layout()
        plt.show()

    def visualize_signed_errors(self):
        """
        Show histograms of dx_errors and dy_errors,
        with vertical lines separating the columns.
        """
        if not self.dx_errors or not self.dy_errors:
            print("No signed errors recorded.")
            return
        
        plt.figure(figsize=(12, 5))

        # Subplot 1: histogram of dx
        plt.subplot(1, 2, 1)
        plt.hist(self.dx_errors, bins=1200, color='blue', alpha=0.6, edgecolor='black', linewidth=0.5)  # Add edge color
        plt.title("Histogram of dx (signed x-error)")
        plt.xlabel("dx (px)")
        plt.ylabel("Count")
        plt.xlim([-50, 50])  

        # Subplot 2: histogram of dy
        plt.subplot(1, 2, 2)
        plt.hist(self.dy_errors, bins=1200, color='green', alpha=0.6, edgecolor='black', linewidth=0.5)  # Add edge color
        plt.title("Histogram of dy (signed y-error)")
        plt.xlabel("dy (px)")
        plt.ylabel("Count")
        plt.xlim([-50, 50])  

        plt.tight_layout()
        plt.show()


###############################################################################
# Child Classes: SIFT, ORB, BRISK, AKAZE, StarFreak, FastBrief
###############################################################################
class SIFT(BaseFeatureProcessor):
    def __init__(self, video_path, cubesat_roi=None, use_ransac=False):
        super().__init__(video_path, cubesat_roi, use_ransac)
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def detect_and_compute(self, frame):
        kp, des = self.sift.detectAndCompute(frame, None)
        if len(kp) > 500:
            kp = kp[:500]
            des = des[:500]
        return kp, des
    
    def match_descriptors(self, des1, des2, kp1, kp2):
        matches = super().match_descriptors(des1, des2, kp1, kp2)
        return matches


class ORB(BaseFeatureProcessor):
    def __init__(self, video_path, cubesat_roi=None, use_ransac=False):
        super().__init__(video_path, cubesat_roi, use_ransac)
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect_and_compute(self, frame):
        kp, des = self.orb.detectAndCompute(frame, None)
        if len(kp) > 500:
            kp = kp[:500]
            des = des[:500]
        return kp, des 

    def match_descriptors(self, des1, des2, kp1, kp2):
        matches = super().match_descriptors(des1, des2, kp1, kp2)
        return matches


class BRISK(BaseFeatureProcessor):
    def __init__(self, video_path, cubesat_roi=None, use_ransac=False):
        super().__init__(video_path, cubesat_roi, use_ransac)
        self.brisk = cv2.BRISK_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect_and_compute(self, frame):
        kp, des = self.brisk.detectAndCompute(frame, None)
        if len(kp) > 500:
            kp = kp[:500]
            des = des[:500]
        return kp, des

    def match_descriptors(self, des1, des2, kp1, kp2):
        matches = super().match_descriptors(des1, des2, kp1, kp2)
        return matches



class AKAZE(BaseFeatureProcessor):
    def __init__(self, video_path, cubesat_roi=None, use_ransac=False):
        super().__init__(video_path, cubesat_roi, use_ransac)
        self.akaze = cv2.AKAZE_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect_and_compute(self, frame):
        kp, des = self.akaze.detectAndCompute(frame, None)
        if len(kp) > 500:
            kp = kp[:500]
            des = des[:500]
        return kp, des

    def match_descriptors(self, des1, des2, kp1, kp2):
        matches = super().match_descriptors(des1, des2, kp1, kp2)
        return matches


class StarFreak(BaseFeatureProcessor):
    def __init__(self, video_path, cubesat_roi=None, use_ransac=False):
        super().__init__(video_path, cubesat_roi, use_ransac)
        self.star = cv2.xfeatures2d.StarDetector_create(
            maxSize=45,
            responseThreshold=30,
            lineThresholdProjected=10,
            lineThresholdBinarized=8,
            suppressNonmaxSize=5
        )
        self.freak = cv2.xfeatures2d.FREAK_create(
            orientationNormalized=True,
            scaleNormalized=True,
            patternScale=22.0,
            nOctaves=4
        )
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect_and_compute(self, frame):
        kp = self.star.detect(frame, None)
        if not kp:
            return None, None
        return self.freak.compute(frame, kp)

    def match_descriptors(self, des1, des2, kp1, kp2):
        matches = super().match_descriptors(des1, des2, kp1, kp2)
        return matches



class FastBrief(BaseFeatureProcessor):
    def __init__(self, video_path, cubesat_roi=None, use_ransac=False):
        super().__init__(video_path, cubesat_roi, use_ransac)
        self.fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
        self.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect_and_compute(self, frame):
        kp = self.fast.detect(frame, None)
        if not kp:
            return None, None
        if len(kp) > 500:
            kp = kp[:500]
            des = des[:500]
        return self.brief.compute(frame, kp)

    def match_descriptors(self, des1, des2, kp1, kp2):
        matches = super().match_descriptors(des1, des2, kp1, kp2)
        return matches

class KLT(BaseFeatureProcessor):
    def __init__(self, video_path, cubesat_roi=None, use_ransac=False):
        super().__init__(video_path, cubesat_roi, use_ransac)
        self.feature_params = dict(
            maxCorners=500,
            qualityLevel=0.2,
            minDistance=5,
            blockSize=7
        )
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01)
        )
        self.feature_lifetimes = {}  
        self.feature_loss_rate = []
        self.tracked_points_history = []  

    def detect_and_compute(self, frame):
        """
        Detects good features to track and returns them as keypoints.
        """
        points = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)
        if points is None:
            return [], None  # No keypoints detected

        keypoints = [cv2.KeyPoint(p[0][0], p[0][1], 1) for p in points]
        return keypoints, None

    def match_descriptors(self, prev_keypoints, curr_frame):
        """
        Track features using optical flow and apply RANSAC if enabled.
        """
        if prev_keypoints is None or len(prev_keypoints) == 0:
            return [], [], None  

        prev_pts_np = np.float32([kp.pt for kp in prev_keypoints]).reshape(-1, 1, 2)

        # Compute optical flow
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_frame, prev_pts_np, None, **self.lk_params
        )

        matches = []
        valid_new = []
        valid_old = []
        mask = None  

        if new_pts is not None and status is not None:
            for i, (new, old, s) in enumerate(zip(new_pts, prev_pts_np, status.flatten())):
                if s == 1:  
                    valid_new.append(cv2.KeyPoint(new[0][0], new[0][1], 1))
                    valid_old.append(cv2.KeyPoint(old[0][0], old[0][1], 1))
                    matches.append(cv2.DMatch(len(valid_old) - 1, len(valid_old) - 1, np.linalg.norm(new - old)))

        # Apply RANSAC if enabled
        if self.use_ransac and len(valid_old) > 4:
            src_pts = np.float32([kp.pt for kp in valid_old]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp.pt for kp in valid_new]).reshape(-1, 1, 2)

            # Apply RANSAC with Homography model
            M, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

            if mask is not None:
                mask = mask.ravel()
                filtered_matches = [matches[i] for i in range(len(matches)) if mask[i] == 1]
                valid_new = [valid_new[i] for i in range(len(valid_new)) if mask[i] == 1]
                valid_old = [valid_old[i] for i in range(len(valid_old)) if mask[i] == 1]
                matches = filtered_matches

        return matches, valid_new, mask
    
    def run(self):
        """
        Runs the KLT tracker on the video and computes all necessary metrics.
        """
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < 4:
            print("Video has fewer than 4 frames. Exiting.")
            cap.release()
            return

        ret, prev_frame = cap.read()
        if not ret:
            print("Failed to read the first frame. Exiting.")
            cap.release()
            return

        self.prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_keypoints, _ = self.detect_and_compute(self.prev_gray)

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            self.curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Track features across frames using optical flow
            start_time = time.perf_counter()
            matches, curr_keypoints, mask = self.match_descriptors(prev_keypoints, self.curr_gray)
            end_time = time.perf_counter()
            self.processing_times.append(end_time - start_time)

            if matches:
                if frame_count == 1:
                    # Store indices of features matched in the first frame
                    initial_indices = {match.queryIdx for match in matches}

                if frame_count > 1 and initial_indices is not None:
                    # Filter matches to include only initial features
                    matches = [match for match in matches if match.queryIdx in initial_indices]

                self.total_matches += len(matches)
                self.tracked_points_history.append(curr_keypoints)  

                # Compute errors
                errors = []
                outlier_count = 0

                for i, match in enumerate(matches):
                    idx = match.queryIdx
                    if idx >= len(prev_keypoints) or idx >= len(curr_keypoints):
                        continue  

                    pt1 = prev_keypoints[idx].pt
                    pt2 = curr_keypoints[idx].pt

                    dx = pt2[0] - pt1[0]
                    dy = pt2[1] - pt1[1]
                    self.dx_errors.append(dx)
                    self.dy_errors.append(dy)

                    distance = np.sqrt(dx**2 + dy**2)
                    errors.append(distance)

                    if distance > self.outlier_threshold:
                        outlier_count += 1
                    
                    if idx not in self.feature_errors:
                        self.feature_errors[idx] = []
                    self.feature_errors[idx].append(distance)

                    if idx in self.feature_lifetimes:
                        self.feature_lifetimes[idx] += 1
                    else:
                        self.feature_lifetimes[idx] = 1

                self.total_error_magnitude.extend(errors)
                self.outliers_per_frame.append(outlier_count)
                self.per_frame_mean_errors.append(np.mean(errors) if errors else 0)

                roi_before = sum(1 for kp in curr_keypoints if self.is_within_roi(kp.pt))
                total_before = len(curr_keypoints)

                if self.use_ransac:
                    roi_after = sum(1 for kp in curr_keypoints if self.is_within_roi(kp.pt))
                    total_after = len(curr_keypoints)

                if curr_keypoints:
                    cubesat_count = sum(1 for kp in curr_keypoints if self.is_within_roi(kp.pt))
                    pct_roi = (cubesat_count / len(curr_keypoints)) * 100
                else:
                    pct_roi = 0

                self.cube_sat_keypoints_percentage.append(pct_roi)

            prev_keypoints = curr_keypoints
            self.prev_gray = self.curr_gray.copy()

        cap.release()

        self.tracking_lengths.extend(self.feature_lifetimes.values())

        self.report_metrics()
        
        #self.visualize_feature_error_progression()
        #self.visualize_tracking()


    def visualize_tracking(self):
        """
        Visualizes KLT feature tracking by overlaying tracked points on 6 selected frames.
        The frames are taken every 10 frames from the video and displayed in a 2x3 grid.
        """
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Select 6 frames, skipping 10 between them
        selected_frames = []
        frame_indices = [i * 5 for i in range(6) if i * 5 < total_frames]  # Ensure we stay within bounds

        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # Jump to the specific frame
            ret, frame = cap.read()
            if not ret:
                break  # Stop if video ends early

            # Convert to grayscale for tracking
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect keypoints using KLT
            keypoints, _ = self.detect_and_compute(gray_frame)

            # Draw keypoints as green circles on the frame
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # Green filled circle

            selected_frames.append(frame)

        cap.release()  # Release the video capture

        # Plot the tracked frames in a 2x3 grid
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # 2 rows, 3 columns
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i < len(selected_frames):
                ax.imshow(cv2.cvtColor(selected_frames[i], cv2.COLOR_BGR2RGB))
                #ax.set_title(f"Frame {frame_indices[i]}")
            ax.axis("off")  # Remove axes

        plt.subplots_adjust(wspace=0, hspace=0)  # Remove spaces between images
        plt.show()

        
    def visualize_feature_error_progression(self):
        """
        Plot the error progression of features matched in the first frame,
        ensuring we only track the features detected in the first frame.
        """
        if not self.feature_errors or len(self.feature_errors) == 0:
            print("No feature errors recorded.")
            return

        error_threshold = 20  # Set the error threshold (in pixels)
        plt.figure(figsize=(10, 6))

        # Get all feature IDs that exist in the first frame
        initial_features = set(self.feature_errors.keys())

        for feature_id in initial_features:
            error_list = self.feature_errors[feature_id]
            if len(error_list) > 1:  # Only plot features with multiple matches
                # Truncate errors at the threshold, but keep the line visible
                capped_errors = [error if error <= error_threshold else error_threshold for error in error_list]
                x_values = range(len(capped_errors))  # Steps are just indices
                plt.plot(x_values, capped_errors, label=f"Feat ID {feature_id}", linewidth=0.7, alpha=0.8)

        # Add labels, legend, and title
        plt.xlabel("Feature Step (each time feature was matched)")
        plt.ylabel("Error (px)")
        plt.title(f"Per-Feature Error Progression ({self.__class__.__name__})")
        plt.ylim([0, error_threshold])  # Limit y-axis to the threshold
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.grid(alpha=0.4)
        plt.tight_layout()

        # Show the plot
        plt.show()


    def visualize_signed_errors(self):
        """
        Show histograms of dx_errors and dy_errors.
        """
        if not self.dx_errors or not self.dy_errors:
            print("No signed errors recorded.")
            return
        
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(self.dx_errors, bins=1200, color='blue', alpha=0.5, edgecolor='black')
        plt.title("Histogram of dx (signed x-error)")
        plt.xlabel("dx (px)")
        plt.ylabel("Count")
        plt.xlim([-50, 50])  

        plt.subplot(1, 2, 2)
        plt.hist(self.dy_errors, bins=1200, color='green', alpha=0.5, edgecolor='black')
        plt.title("Histogram of dy (signed y-error)")
        plt.xlabel("dy (px)")
        plt.ylabel("Count")
        plt.xlim([-50, 50])  

        plt.tight_layout()
        plt.show()

    def visualize_overall_error_histogram(self):
        """
        Plot a histogram of all feature errors over the entire video.
        """
        if not self.total_error_magnitude:
            print("No total error magnitude recorded.")
            return

        plt.figure(figsize=(8, 6))
        plt.hist(self.total_error_magnitude, bins=50, color='purple', alpha=0.7, edgecolor='black')
        plt.xlabel("Error Magnitude (px)")
        plt.ylabel("Count")
        plt.title(f"Overall Feature Tracking Error ({self.__class__.__name__})")
        plt.show()

