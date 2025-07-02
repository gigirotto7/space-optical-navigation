import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_grouped_bar_for_all_parameters(
    excel_file="7_my_feature_analysis.xlsx",
    metrics_list=["Mean Error Magnitude", "Outlier Ratio (%)", "Mean % of Keypoints in ROI", "Average Tracking Length"],
    params={
        "Lighting": {1: "Low Light", 2: "High Light"},
        "Angle": {1: "Back Angle", 2: "Side Angle", 3: "45° Angle"},
        "Noise": {1: "No Noise", 2: "Salt & Pepper", 3: "Gaussian"},
        "Distance": {1: "Low Distance", 2: "High Distance"},
        "Background": {1: "Plainfield", 2: "Partial Earth", 3: "Full Earth"}
    },
    param_digits={"Lighting": 1, "Angle": 2, "Noise": 6, "Distance": 4, "Background": 7},
    save_dir="/Users/giovannagirotto/Desktop/TUM/SemesterThesis/RESULTS/barplots/"
):
    """
    Creates grouped bar plots for each parameter (Lighting, Angle, Noise, Distance, Background)
    and saves them as separate images with clear filenames.
    """

    os.makedirs(save_dir, exist_ok=True)

    # Load Data
    df = pd.read_excel(excel_file)
    df["ID_str"] = df["Video"].astype(str)

    # Keep only videos starting with "1" and RANSAC methods
    df = df[df["ID_str"].str.startswith("3")].copy()
    df = df[df["Method"].str.contains("with RANSAC")].copy()
    df["AlgorithmFamily"] = df["Method"].apply(lambda x: x.split()[0])

    # Extract parameters from the Video ID
    for param, digit in param_digits.items():
        df[param] = pd.to_numeric(df["ID_str"].str[digit], errors="coerce")

    df = df.dropna(subset=params.keys())

    # Convert parameters to categorical labels
    for param, mapping in params.items():
        df[param] = df[param].astype(int)
        df = df[df[param].isin(mapping.keys())]
        df[param] = df[param].map(mapping)

    # Define desired order of algorithms for consistency
    desired_order = ["SIFT", "ORB", "BRISK", "AKAZE", "KLT"]

    # Y-axis labels
    y_labels = {
        "Mean Error Magnitude": "Mean Error (px)",
        "Outlier Ratio (%)": "Outlier Ratio (%)",
        "Mean % of Keypoints in ROI": "Features in ROI (%)",
        "Average Tracking Length": "Tracking Length (frames)"
    }

    # Define color palette
    selected_colors = ["#7b85d4", "#f37738", "#83c995", "#859795"]
    
    # Iterate over each parameter
    for param_name, param_labels in params.items():

        # Define save path and figure name
        fig_name = f"{param_name}.pdf"
        save_path = os.path.join(save_dir, fig_name)

        # Create a 2×2 grid for multiple metrics
        fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharey=False)
        axes = axes.flatten()

        handles, labels = None, None

        # Define a custom palette for the parameter categories
        custom_palette = dict(zip(list(param_labels.values()), selected_colors))

        for i, metric in enumerate(metrics_list):
            grouped = df.groupby(["AlgorithmFamily", param_name])[metric].mean().reset_index()
            grouped["AlgorithmFamily"] = pd.Categorical(grouped["AlgorithmFamily"], categories=desired_order, ordered=True)
            grouped = grouped.sort_values("AlgorithmFamily")

            ax = axes[i]
            sns.barplot(
                x="AlgorithmFamily",
                y=metric,
                hue=param_name,
                data=grouped,
                palette=custom_palette,
                ax=ax
            )
            ax.set_title(metric, fontweight="bold", fontsize=16)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, fontweight="bold", rotation=0)

            if i == 0:  # Capture legend handles from the first subplot
                handles, labels = ax.get_legend_handles_labels()
            ax.legend_.remove()

        # Global Legend
        legend = fig.legend(
            handles, labels, loc="lower center",
            ncol=len(param_labels), fontsize=18, frameon=True, title_fontsize=18,
            bbox_to_anchor=(0.5, -0.01), prop={'weight': 'bold'}
        )
        
        for text in legend.get_texts():
            text.set_fontsize(14)
            text.set_fontweight("bold")

        # Adjust layout
        plt.tight_layout(rect=[0, 0.1, 1, 0.93])

        # Save the plot with its name
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    print("Bar plots generated successfully for all parameters!")


if __name__ == "__main__":
    plot_grouped_bar_for_all_parameters(
        excel_file="../results/methods_data.xlsx",
        metrics_list=["Mean Error Magnitude (px)", "Outlier Ratio (%)",  "Mean % of Keypoints in ROI", "Average Tracking Length (frames)"],
        save_dir="../results/barplots/move3"
    )
