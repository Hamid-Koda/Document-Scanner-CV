import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import HeatmapCornerRegressor, soft_argmax_2d

def compare_heatmap_dropout():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Heatmap Inference on: {device}")

    img_path = "test_image(1).jpg" 
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise Exception(f"Error: Image '{img_path}' not found.")

    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_input = cv2.resize(img_rgb, (256, 256))
    tensor = (torch.from_numpy(img_input).float().permute(2, 0, 1).unsqueeze(0) / 255.).to(device)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    print("Loading Model B (No Dropout)...")
    model_no_drop = HeatmapCornerRegressor(use_dropout=False).to(device)
    model_no_drop.load_state_dict(torch.load("weights/heatmap_corner_best(new).pth", map_location=device))
    model_no_drop.eval()

    with torch.no_grad():
        logits_no_drop = model_no_drop(tensor)
        coords_no_drop = soft_argmax_2d(logits_no_drop).squeeze().cpu().numpy()

    img_no_drop = img_rgb.copy()
    for i in range(4):
        x, y = int(coords_no_drop[i, 0] * orig_w), int(coords_no_drop[i, 1] * orig_h)
        cv2.circle(img_no_drop, (x, y), 18, colors[i], -1)
        cv2.putText(img_no_drop, str(i+1), (x + 18, y - 18), cv2.FONT_HERSHEY_SIMPLEX, 2, colors[i], 4)

    print("Loading Model B (With Dropout)...")
    model_with_drop = HeatmapCornerRegressor(use_dropout=True).to(device)
    model_with_drop.load_state_dict(torch.load("weights/heatmap_corner_best(dropout-new).pth", map_location=device))
    model_with_drop.eval()

    with torch.no_grad():
        logits_with_drop = model_with_drop(tensor)
        coords_with_drop = soft_argmax_2d(logits_with_drop).squeeze().cpu().numpy()

    img_with_drop = img_rgb.copy()
    for i in range(4):
        x, y = int(coords_with_drop[i, 0] * orig_w), int(coords_with_drop[i, 1] * orig_h)
        cv2.circle(img_with_drop, (x, y), 18, colors[i], -1)
        cv2.putText(img_with_drop, str(i+1), (x + 18, y - 18), cv2.FONT_HERSHEY_SIMPLEX, 2, colors[i], 4)

    
    fig, axs = plt.subplots(1, 2, figsize=(16, 10))
    
    axs[0].imshow(img_no_drop)
    axs[0].set_title("1. Heatmap Regressor (No Dropout)", fontsize=16, fontweight='bold')
    axs[0].axis("off")

    axs[1].imshow(img_with_drop)
    axs[1].set_title("2. Heatmap Regressor (With Dropout)", fontsize=16, fontweight='bold')
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    compare_heatmap_dropout()