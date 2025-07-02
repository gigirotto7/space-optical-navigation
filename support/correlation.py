import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def plot_parameter_vs_parameter_heatmaps(
    excel_file="my_feature_analysis.xlsx",
    selected_algorithm="SIFT with RANSAC",
    selected_metric="Mean Error Magnitude (px)",
    params={
        "Lighting": {1: "Low Light", 2: "High Light"},
        "Angle": {1: "Back Angle", 2: "Side Angle", 3: "45° Angle"},
        "Noise": {1: "No Noise", 2: "Salt & Pepper", 3: "Gaussian"},
        "Distance": {1: "Low Distance", 2: "High Distance"},
        "Background": {1: "Plainfield", 2: "Partial Earth", 3: "Full Earth"}
    },
    param_digits={"Lighting": 1, "Angle": 2, "Noise": 6, "Distance": 4, "Background": 7},
    save_dir="../results/noise_heatmaps/Move1/"
):
    """
    Creates a heatmap for each parameter pair and saves them as separate images.
    """
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load Data
    df = pd.read_excel(excel_file)

    # Keep only videos starting with "1" and the chosen algorithm
    df = df[df["Video"].astype(str).str.startswith("1")].copy()
    df = df[df["Method"] == selected_algorithm].copy()
    df["ID_str"] = df["Video"].astype(str)

    # Ensure string length is long enough
    max_digit = max(param_digits.values())
    df = df[df["ID_str"].str.len() > max_digit].copy()

    # Extract all parameters
    for param, digit in param_digits.items():
        df[param] = pd.to_numeric(df["ID_str"].str[digit], errors="coerce")

    # Drop NaN values
    df = df.dropna(subset=params.keys())

    # Convert to int and map to descriptive labels
    for param, mapping in params.items():
        df[param] = df[param].astype(int)
        df = df[df[param].isin(mapping.keys())]
        df[param] = df[param].map(mapping)

    # Generate separate heatmaps for each parameter pair
    for i, (param_y, y_mapping) in enumerate(params.items()):
        for j, (param_x, x_mapping) in enumerate(params.items()):
            if i >= j:  # Avoid duplicate plots and diagonal
                continue

            # Create pivot table
            pivot_df = df.pivot_table(
                index=param_y,
                columns=param_x,
                values=selected_metric,
                aggfunc="mean"
            )

            # Create a new figure for each heatmap
            plt.figure(figsize=(8, 6))
            ax = sns.heatmap(
                pivot_df, annot=True, cmap="viridis", fmt=".2f",
                cbar=True, vmin=0, vmax=200, annot_kws={"size":14}
            )

            # Set labels and title
            plt.title(f"{param_y} vs {param_x} \n {selected_metric}",
                      fontweight="bold", fontsize=14)
            plt.xlabel(param_x, fontsize=12, fontweight="bold")
            plt.ylabel(param_y, fontsize=12, fontweight="bold")

            plt.xticks(rotation=0, fontsize=13)
            plt.yticks(rotation=0, fontsize=13)

            # Save the figure
            save_path = os.path.join(save_dir, f"{selected_algorithm}_{param_y}_vs_{param_x}.pdf")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()  # Close the figure to free memory


def plot_all_parameters_vs_noise(
    excel_file="my_feature_analysis.xlsx",
    selected_algorithm="SIFT with RANSAC",
    selected_metric="Mean Error Magnitude (px)",
    params={
        "Light": {1: "Low", 2: "High"},
        "Angle": {1: "Back", 2: "Side", 3: "45°"},
        "Noise": {1: "None", 2: "Salt & Pepper", 3: "Gaussian"},
        "Distance": {1: "Low", 2: "High"},
        "Background": {1: "Plainfield", 2: "Partial Earth", 3: "Full Earth"}
    },
    param_digits={"Light": 1, "Angle": 2, "Noise": 6, "Distance": 4, "Background": 7},
    save_path="/Users/giovannagirotto/Desktop/TUM/SemesterThesis/RESULTS/noise_correlations_new/Move3/"
):
    
    name = f'{selected_algorithm}.pdf'
    """
    Creates a single large heatmap with:
    - Y-axis: All parameters (Lighting, Angle, Distance, Background)
    - X-axis: Noise types (None, Salt & Pepper, Gaussian)
    - Each cell represents the mean error magnitude for that combination.
    """

    # Load Data
    df = pd.read_excel(excel_file)

    # Keep only videos starting with "1" and the chosen algorithm
    df = df[df["Video"].astype(str).str.startswith("3")].copy()
    df = df[df["Method"] == selected_algorithm].copy()
    df["ID_str"] = df["Video"].astype(str)

    # Ensure string length is long enough
    max_digit = max(param_digits.values())
    df = df[df["ID_str"].str.len() > max_digit].copy()

    # Extract all parameters
    for param, digit in param_digits.items():
        df[param] = pd.to_numeric(df["ID_str"].str[digit], errors="coerce")

    # Drop NaN values
    df = df.dropna(subset=params.keys())

    # Convert to int and map to descriptive labels
    for param, mapping in params.items():
        df[param] = df[param].astype(int)
        df = df[df[param].isin(mapping.keys())]
        df[param] = df[param].map(mapping)

    # Define X and Y groups
    x_param = "Noise"
    y_params = ["Light", "Angle", "Distance", "Background"]

    # Create labels for Y-axis
    y_labels = [f"{param}: {val}" for param in y_params for val in params[param].values()]
    x_labels = list(params["Noise"].values())

    # Prepare an empty heatmap matrix
    heatmap_matrix = np.full((len(y_labels), len(x_labels)), np.nan)

    # Fill in the matrix with mean values
    row_idx = 0
    for param_y in y_params:
        for y_value in params[param_y].values():
            col_idx = 0
            for x_value in params[x_param].values():
                mean_value = df[
                    (df[param_y] == y_value) & (df[x_param] == x_value)
                ][selected_metric].mean()

                heatmap_matrix[row_idx, col_idx] = mean_value
                col_idx += 1
            row_idx += 1

    # Plot the full heatmap with a **fixed scale of 0-200**
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        heatmap_matrix, annot=True, cmap="viridis", fmt=".2f",
        cbar=True, vmin=0, vmax=200, xticklabels=x_labels, yticklabels=y_labels,
        annot_kws={"size": 14}
    )

    # Set labels and title
    plt.title(f"Noise vs All Parameters \n {selected_metric}",
              fontweight="bold", fontsize=16)
    plt.xlabel("Noise Type", fontsize=14, fontweight="bold")
    plt.ylabel("Parameters", fontsize=14, fontweight="bold")

    plt.xticks(rotation=0, fontsize=13)
    plt.yticks(rotation=0, fontsize=13)

    plt.tight_layout()

    # Save the figure
    plt.savefig(f'{save_path}/{name}', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    #plot_parameter_vs_parameter_heatmaps(
    #    excel_file="3_my_feature_analysis.xlsx",
    #    selected_algorithm="KLT with RANSAC",
    #    selected_metric="Mean Error Magnitude (px)"
    #)
    plot_all_parameters_vs_noise(
        excel_file="../results/methods_data.xlsx",
        selected_algorithm="KLT with RANSAC",
        selected_metric="Mean Error Magnitude (px)"
    )