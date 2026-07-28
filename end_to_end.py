import cv2
import numpy as np
import torch
import os
from model import EnhancementUNet, DirectCornerRegressor, HeatmapCornerRegressor, soft_argmax_2d

def end_to_end_pipeline(raw_img_path, corner_model_path, enhance_model_path, output_path, is_heatmap=False, device='cpu'):
    """7. Bonus: The End-to-End Document Scanner"""
    
    # 1. Load both trained models
    if is_heatmap:
        corner_model = HeatmapCornerRegressor().to(device)
    else:
        corner_model = DirectCornerRegressor().to(device)
    
    corner_model.load_state_dict(torch.load(corner_model_path, map_location=device))
    corner_model.eval()

    enhance_model = EnhancementUNet().to(device)
    enhance_model.load_state_dict(torch.load(enhance_model_path, map_location=device))
    enhance_model.eval()

    # 2. Read and Preprocess Raw Image
    img = cv2.imread(raw_img_path)
    if img is None:
        raise ValueError(f"Could not read image at {raw_img_path}")
        
    orig_h, orig_w = img.shape[:2]
    
    # Resize for corner model (needs 256x256 input)
    img_resized = cv2.resize(img, (256, 256))
    input_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

    # 3. Predict Corners
    with torch.no_grad():
        if is_heatmap:
            heatmaps = corner_model(input_tensor)
            corners_norm = soft_argmax_2d(heatmaps).squeeze(0).cpu().numpy()
        else:
            corners_norm = corner_model(input_tensor).squeeze(0).cpu().numpy()

    # Scale corners back to original image resolution
    corners = corners_norm.copy()
    corners[:, 0] *= orig_w
    corners[:, 1] *= orig_h

    # 4. Rectify (Warp Perspective)
    # We choose a standard document aspect ratio (e.g., A4: 1 to 1.414)
    target_w, target_h = 800, 1131
    target_corners = np.array([
        [0, 0],
        [target_w, 0],
        [target_w, target_h],
        [0, target_h]
    ], dtype=np.float32)

    # Compute Homography and warp
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), target_corners)
    rectified_img = cv2.warpPerspective(img, M, (target_w, target_h))

    # 5. Enhance the Rectified Image
    # Resize to the enhancement network's expected input size
    rectified_resized = cv2.resize(rectified_img, (256, 256))
    enhance_input = torch.from_numpy(rectified_resized).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        enhanced_tensor = enhance_model(enhance_input)

    # 6. Post-process and Save
    enhanced_np = (enhanced_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    enhanced_img = np.transpose(enhanced_np, (1, 2, 0))
    
    # Resize back to target high resolution
    final_output = cv2.resize(enhanced_img, (target_w, target_h))

    cv2.imwrite(output_path, final_output)
    print(f"✅ End-to-End scan successful! Saved to {output_path}")

if __name__ == '__main__':
    # Usage Example (Make sure weights exist before running):
    # end_to_end_pipeline(
    #     raw_img_path='real_test_photo.jpg',
    #     corner_model_path='weights/direct_corner_epoch_10.pth',
    #     enhance_model_path='weights/enhancement_unet_epoch_10.pth',
    #     output_path='final_clean_scan.jpg'
    # )
    pass