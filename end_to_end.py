import cv2
import numpy as np
import torch
import os
from model import EnhancementUNet, HeatmapCornerRegressor, soft_argmax_2d

def end_to_end_pipeline(raw_img_path, corner_model_path, enhance_model_path, output_path, device='cpu'):
    # 1. Load both trained models
    corner_model = HeatmapCornerRegressor().to(device)
    corner_model.load_state_dict(torch.load(corner_model_path, map_location=device))
    corner_model.eval()

    enhance_model = EnhancementUNet().to(device)
    enhance_model.load_state_dict(torch.load(enhance_model_path, map_location=device))
    enhance_model.eval()

    # 2. Read and Preprocess Raw Image
    img_bgr = cv2.imread(raw_img_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image at {raw_img_path}")
        
    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    img_resized = cv2.resize(img_rgb, (256, 256))
    input_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

    # 3. Predict Corners
    with torch.no_grad():
        heatmaps = corner_model(input_tensor)
        corners_norm = soft_argmax_2d(heatmaps, temperature=50.0).squeeze(0).cpu().numpy()

    corners = corners_norm.copy()
    corners[:, 0] *= orig_w
    corners[:, 1] *= orig_h

    # 4. Rectify (Warp Perspective)
    target_w, target_h = 800, 1128 
    target_corners = np.array([
        [0, 0],
        [target_w, 0],
        [target_w, target_h],
        [0, target_h]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners.astype(np.float32), target_corners)
    rectified_rgb = cv2.warpPerspective(img_rgb, M, (target_w, target_h))

    # 5. Enhance the Rectified Image (با شبکه عصبی آموزش‌دیده)
    enhance_input = torch.from_numpy(rectified_rgb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        enhanced_tensor = enhance_model(enhance_input)

    # 6. Post-process and Save
    enhanced_np = (enhanced_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    final_output_rgb = np.transpose(enhanced_np, (1, 2, 0))
    
    final_output_bgr = cv2.cvtColor(final_output_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, final_output_bgr)
    print(f"✅ End-to-End scan successful! Saved to {output_path}")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    test_image_path = 'test_image.jpg' 
    output_image_path = 'final_clean_scan.jpg'
    
    if not os.path.exists(test_image_path):
        print(f"❌ Error: Please put an image named '{test_image_path}' in the folder.")
    else:
        print("🚀 Starting End-to-End Document Scanner...")
        end_to_end_pipeline(
            raw_img_path=test_image_path,
            corner_model_path='weights/heatmap_corner_best.pth',
            enhance_model_path='weights/enhancement_unet_best.pth',
            output_path=output_image_path,
            device=device
        )