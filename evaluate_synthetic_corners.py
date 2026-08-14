import torch
from torch.utils.data import DataLoader

import glob
import random
import numpy as np

from dataset import CornerDataset
from model import DirectCornerRegressor, HeatmapCornerRegressor
from model import soft_argmax_2d

# Configuration
TARGET_SIZE = (256, 256)

SUCCESS_THRESHOLD = 20.0

BATCH_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train / Validation / Test split

def get_splits_with_test(paths):
    """
        80% -> training
        10% -> validation
        10% -> test
    """

    random.seed(42)

    paths_copy = paths.copy()
    random.shuffle(paths_copy)

    n = len(paths_copy)

    train_p = paths_copy[:int(0.8 * n)]
    val_p = paths_copy[int(0.8 * n):int(0.9 * n)]
    test_p = paths_copy[int(0.9 * n):]

    return train_p, val_p, test_p

# Heatmap -> coordinates
def heatmaps_to_coordinates(heatmaps):

    return soft_argmax_2d(heatmaps, temperature=50.0)

# Evaluate Direct Regression model
def evaluate_direct_model(model, test_loader):
    model.eval()

    total_error = 0.0
    total_corners = 0

    successful_images = 0
    total_images = 0

    with torch.no_grad():

        for images, true_corners in test_loader:

            images = images.to(DEVICE)
            true_corners = true_corners.to(DEVICE)

            # Prediction
            pred_corners = model(images)

            # Both predictions and targets are normalized [0, 1]
            H, W = images.shape[2], images.shape[3]

            pred_pixels = pred_corners.clone()
            true_pixels = true_corners.clone()

            pred_pixels[:, :, 0] *= W
            pred_pixels[:, :, 1] *= H

            true_pixels[:, :, 0] *= W
            true_pixels[:, :, 1] *= H

            # Euclidean distance for each corner
            distances = torch.sqrt(
                torch.sum(
                    (pred_pixels - true_pixels) ** 2,
                    dim=2
                )
            )

            total_error += distances.sum().item()
            total_corners += distances.numel()

            # Success criterion
            image_success = (
                distances < SUCCESS_THRESHOLD
            ).all(dim=1)

            successful_images += image_success.sum().item()
            total_images += images.size(0)

    mean_error = total_error / total_corners

    success_rate = (
        100.0 * successful_images / total_images
    )

    return mean_error, success_rate

# Evaluate Heatmap model

def evaluate_heatmap_model(model, test_loader):
    model.eval()

    total_error = 0.0
    total_corners = 0

    successful_images = 0
    total_images = 0

    with torch.no_grad():

        for images, true_corners in test_loader:

            images = images.to(DEVICE)
            true_corners = true_corners.to(DEVICE)

            # Model produces 4 heatmaps
            heatmaps = model(images)

            # Convert heatmaps to normalized coordinates
            pred_corners = heatmaps_to_coordinates(heatmaps)

            # Convert normalized coordinates to pixels
            H, W = images.shape[2], images.shape[3]

            pred_pixels = pred_corners.clone()
            true_pixels = true_corners.clone()

            pred_pixels[:, :, 0] *= W
            pred_pixels[:, :, 1] *= H

            true_pixels[:, :, 0] *= W
            true_pixels[:, :, 1] *= H

            # Euclidean distance
            distances = torch.sqrt(
                torch.sum(
                    (pred_pixels - true_pixels) ** 2,
                    dim=2
                )
            )

            total_error += distances.sum().item()
            total_corners += distances.numel()

            # Success criterion
            image_success = (
                distances < SUCCESS_THRESHOLD
            ).all(dim=1)

            successful_images += image_success.sum().item()
            total_images += images.size(0)

    mean_error = total_error / total_corners

    success_rate = (
        100.0 * successful_images / total_images
    )

    return mean_error, success_rate

# Main
def main():

    print("=" * 75)
    print("Synthetic Test Evaluation - Corner Detection")
    print("=" * 75)

    print(f"Device: {DEVICE}")
    print(f"Success threshold: {SUCCESS_THRESHOLD} pixels")

    # Get all scans and backgrounds
    clean_paths = glob.glob("clean_scans/*.*")
    bg_paths = glob.glob("backgrounds/*.*")

    if len(clean_paths) == 0:
        raise RuntimeError("No images found in clean_scans/")

    if len(bg_paths) == 0:
        raise RuntimeError("No images found in backgrounds/")

    # Reproduce original split
    train_paths, val_paths, test_paths = get_splits_with_test(
        clean_paths
    )

    print()
    print("Dataset split:")
    print(f"Training scans   : {len(train_paths)}")
    print(f"Validation scans : {len(val_paths)}")
    print(f"Synthetic test scans: {len(test_paths)}")

    # Synthetic test dataset
    test_dataset = CornerDataset(
        clean_scans=test_paths,
        backgrounds=bg_paths,
        target_size=TARGET_SIZE,
        is_train=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print()
    print(f"Synthetic test samples: {len(test_dataset)}")

    # 1. Direct Regression - No Dropout
    print()
    print("-" * 75)
    print("Evaluating Approach A - No Dropout")
    print("-" * 75)

    direct_no_dropout = DirectCornerRegressor(
        use_dropout=False
    ).to(DEVICE)

    direct_no_dropout.load_state_dict(
        torch.load(
            "weights/direct_corner_best(new).pth",
            map_location=DEVICE
        )
    )

    error_a_no, success_a_no = evaluate_direct_model(
        direct_no_dropout,
        test_loader
    )

    print(
        f"Mean Error   : {error_a_no:.2f} px"
    )

    print(
        f"Success Rate : {success_a_no:.2f} %"
    )

    # 2. Direct Regression - With Dropout
    print()
    print("-" * 75)
    print("Evaluating Approach A - With Dropout")
    print("-" * 75)

    direct_dropout = DirectCornerRegressor(
        use_dropout=True
    ).to(DEVICE)

    direct_dropout.load_state_dict(
        torch.load(
            "weights/direct_corner_best(dropout-new).pth",
            map_location=DEVICE
        )
    )

    error_a_dropout, success_a_dropout = evaluate_direct_model(
        direct_dropout,
        test_loader
    )

    print(
        f"Mean Error   : {error_a_dropout:.2f} px"
    )

    print(
        f"Success Rate : {success_a_dropout:.2f} %"
    )

    # 3. Heatmap - No Dropout
    print()
    print("-" * 75)
    print("Evaluating Approach B - No Dropout")
    print("-" * 75)

    heatmap_no_dropout = HeatmapCornerRegressor(
        use_dropout=False
    ).to(DEVICE)

    heatmap_no_dropout.load_state_dict(
        torch.load(
            "weights/heatmap_corner_best(new).pth",
            map_location=DEVICE
        )
    )

    error_b_no, success_b_no = evaluate_heatmap_model(
        heatmap_no_dropout,
        test_loader
    )

    print(
        f"Mean Error   : {error_b_no:.2f} px"
    )

    print(
        f"Success Rate : {success_b_no:.2f} %"
    )

    # 4. Heatmap - With Dropout
    print()
    print("-" * 75)
    print("Evaluating Approach B - With Dropout")
    print("-" * 75)

    heatmap_dropout = HeatmapCornerRegressor(
        use_dropout=True
    ).to(DEVICE)

    heatmap_dropout.load_state_dict(
        torch.load(
            "weights/heatmap_corner_best(dropout-new).pth",
            map_location=DEVICE
        )
    )

    error_b_dropout, success_b_dropout = evaluate_heatmap_model(
        heatmap_dropout,
        test_loader
    )

    print(
        f"Mean Error   : {error_b_dropout:.2f} px"
    )

    print(
        f"Success Rate : {success_b_dropout:.2f} %"
    )

    # Final table
    print()
    print()
    print("=" * 75)
    print("SYNTHETIC TEST RESULTS")
    print("=" * 75)

    print(
        f"{'Model Version':<35} | "
        f"{'Mean Error (px)':>16} | "
        f"{'Success Rate (%)':>17}"
    )

    print("-" * 75)

    print(
        f"{'Approach A (No Dropout)':<35} | "
        f"{error_a_no:>16.2f} | "
        f"{success_a_no:>17.2f}"
    )

    print(
        f"{'Approach A (With Dropout)':<35} | "
        f"{error_a_dropout:>16.2f} | "
        f"{success_a_dropout:>17.2f}"
    )

    print(
        f"{'Approach B (No Dropout)':<35} | "
        f"{error_b_no:>16.2f} | "
        f"{success_b_no:>17.2f}"
    )

    print(
        f"{'Approach B (With Dropout)':<35} | "
        f"{error_b_dropout:>16.2f} | "
        f"{success_b_dropout:>17.2f}"
    )

    print("=" * 75)

    # Best model
    results = {
        "Approach A (No Dropout)": (error_a_no, success_a_no),
        "Approach A (With Dropout)": (error_a_dropout, success_a_dropout),
        "Approach B (No Dropout)": (error_b_no, success_b_no),
        "Approach B (With Dropout)": (error_b_dropout, success_b_dropout)
    }

    best_by_error = min(
        results.items(),
        key=lambda x: x[1][0]
    )

    best_by_success = max(
        results.items(),
        key=lambda x: x[1][1]
    )

    print()
    print(
        f"Best Mean Error: "
        f"{best_by_error[0]} "
        f"({best_by_error[1][0]:.2f} px)"
    )

    print(
        f"Best Success Rate: "
        f"{best_by_success[0]} "
        f"({best_by_success[1][1]:.2f} %)"
    )


if __name__ == "__main__":
    main()