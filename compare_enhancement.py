import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import random
from model import EnhancementUNet

def compare_enhancement_dropout():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f" Running Enhancement Inference on: {device}")

    test_images = glob.glob("json_cropped_photos/*.jpg")
    if not test_images:
        print(" Error: No images found in 'json_cropped_photos' folder.")
        return
    
    img_path = random.choice(test_images)
    print(f" Randomly selected image: {img_path}")
    
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f" Error: Cannot read image at {img_path}")
        return

    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    target_w, target_h = 800, 1152
    img_resized = cv2.resize(img_rgb, (target_w, target_h))
    
    input_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0


    print("Loading Model (No Dropout)...")
    model_no_drop = EnhancementUNet(use_dropout=False).to(device)
    model_no_drop.load_state_dict(torch.load('weights/enhancement_unet_best(new).pth', map_location=device))
    model_no_drop.eval()

    with torch.no_grad():
        output_no_drop = model_no_drop(input_tensor)

    out_np_no_drop = output_no_drop.squeeze().cpu().numpy()
    out_np_no_drop = np.clip(out_np_no_drop, 0.0, 1.0)
    out_np_no_drop = (out_np_no_drop * 255).astype(np.uint8)
    out_rgb_no_drop = np.transpose(out_np_no_drop, (1, 2, 0))
    out_rgb_no_drop = cv2.resize(out_rgb_no_drop, (orig_w, orig_h))

    
    print("Loading Model (With Dropout)...")
    model_with_drop = EnhancementUNet(use_dropout=True).to(device)
    model_with_drop.load_state_dict(torch.load('weights/enhancement_unet_best(dropout-new).pth', map_location=device))
    model_with_drop.eval()

    with torch.no_grad():
        output_with_drop = model_with_drop(input_tensor)

    out_np_with_drop = output_with_drop.squeeze().cpu().numpy()
    out_np_with_drop = np.clip(out_np_with_drop, 0.0, 1.0)
    out_np_with_drop = (out_np_with_drop * 255).astype(np.uint8)
    out_rgb_with_drop = np.transpose(out_np_with_drop, (1, 2, 0))
    out_rgb_with_drop = cv2.resize(out_rgb_with_drop, (orig_w, orig_h))
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    axs[0].imshow(img_rgb)
    axs[0].set_title("1. Original (Cropped & Dirty)", fontsize=14, fontweight='bold')
    axs[0].axis('off')

    axs[1].imshow(out_rgb_no_drop)
    axs[1].set_title("2. Enhanced (No Dropout)", fontsize=14, fontweight='bold')
    axs[1].axis('off')

    axs[2].imshow(out_rgb_with_drop)
    axs[2].set_title("3. Enhanced (With Dropout)", fontsize=14, fontweight='bold')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    compare_enhancement_dropout()