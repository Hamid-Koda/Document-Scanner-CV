import torch
import numpy as np
import cv2
import math
from skimage.metrics import structural_similarity as ssim
from model import EnhancementUNet, DirectCornerRegressor, HeatmapCornerRegressor, soft_argmax_2d

def calculate_psnr(img1, img2):
    """Calculate Peak Signal-to-Noise Ratio (PSNR)"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2):
    """Calculate Structural Similarity Index (SSIM)"""
    # channel_axis=2 is used for HWC image formats in OpenCV/Numpy
    score, _ = ssim(img1, img2, channel_axis=2, full=True)
    return score

def evaluate_model(model_path, test_loader, device='cpu'):
    """Evaluates the model on a given dataset (validation or test bucket)"""
    model = EnhancementUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_psnr = 0
    total_ssim = 0
    count = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Convert PyTorch tensors (B, C, H, W) back to Numpy HWC arrays
            outputs_np = (outputs.cpu().numpy() * 255).astype(np.uint8)
            targets_np = (targets.cpu().numpy() * 255).astype(np.uint8)
            
            for i in range(outputs_np.shape[0]):
                out_img = np.transpose(outputs_np[i], (1, 2, 0))
                tgt_img = np.transpose(targets_np[i], (1, 2, 0))
                
                total_psnr += calculate_psnr(out_img, tgt_img)
                total_ssim += calculate_ssim(out_img, tgt_img)
                count += 1

    print(f"Evaluation complete on {count} images.")
    print(f"Average PSNR: {total_psnr/count:.2f}")
    print(f"Average SSIM: {total_ssim/count:.4f}")

def inference_pipeline(model_path, image_path, output_path, device='cpu'):
    """3.4 Pipeline the process: End-to-end inference for a single rectified image"""
    model = EnhancementUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 1. Preprocess the image
    img = cv2.imread(image_path)
    orig_h, orig_w = img.shape[:2]
    img_resized = cv2.resize(img, (256, 256))
    input_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # 2. Predict the enhanced image
    with torch.no_grad():
        output_tensor = model(input_tensor.to(device))
        
    # 3. Post-process the output
    output_np = (output_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    output_img = np.transpose(output_np, (1, 2, 0))
    output_img = cv2.resize(output_img, (orig_w, orig_h))
    
    # 4. Save / Visualize
    cv2.imwrite(output_path, output_img)
    print(f"Enhanced image successfully saved to {output_path}")


def calculate_corner_error(pred_coords, target_coords, image_shape):
    """
    Calculates Euclidean distance between predicted and target corners in pixels.
    Coordinates are normalized [0, 1]. image_shape is (H, W).
    """
    h, w = image_shape
    # Scale back to pixel coordinates
    pred_scaled = pred_coords * torch.tensor([w, h], device=pred_coords.device)
    target_scaled = target_coords * torch.tensor([w, h], device=target_coords.device)
    
    # Calculate Euclidean distance for each corner
    distances = torch.norm(pred_scaled - target_scaled, dim=2) # Shape: (Batch, 4)
    return distances

def evaluate_corner_models(model, test_loader, image_shape=(256, 256), is_heatmap=False, threshold=10.0):
    """Evaluates Corner Detection models using Euclidean distance and Success Rate"""
    device = next(model.parameters()).device
    model.eval()
    
    total_distance = 0.0
    total_corners = 0
    successful_images = 0
    total_images = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if is_heatmap:
                heatmaps = model(inputs)
                pred_coords = soft_argmax_2d(heatmaps)
            else:
                pred_coords = model(inputs)
                
            distances = calculate_corner_error(pred_coords, targets, image_shape)
            
            total_distance += distances.sum().item()
            total_corners += distances.numel()
            
            # A success is if ALL 4 corners are within the threshold distance
            max_dist_per_image, _ = distances.max(dim=1)
            successful_images += (max_dist_per_image < threshold).sum().item()
            total_images += inputs.size(0)
            
    mean_error = total_distance / total_corners
    success_rate = (successful_images / total_images) * 100
    
    print(f"Mean Corner Error: {mean_error:.2f} pixels")
    print(f"Success Rate (All 4 corners < {threshold}px): {success_rate:.2f}%")
    return mean_error, success_rate

def corner_inference_pipeline(model_path, image_path, output_path, is_heatmap=False, device='cpu'):
    """5.1 Pipeline the process: End-to-end inference for a raw photo"""
    if is_heatmap:
        model = HeatmapCornerRegressor().to(device)
    else:
        model = DirectCornerRegressor().to(device)
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 1. Preprocess the image
    img = cv2.imread(image_path)
    orig_h, orig_w = img.shape[:2]
    img_resized = cv2.resize(img, (256, 256))
    input_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # 2. Predict the four corners
    with torch.no_grad():
        if is_heatmap:
            heatmaps = model(input_tensor.to(device))
            pred_coords = soft_argmax_2d(heatmaps).squeeze(0) # Shape: (4, 2)
        else:
            pred_coords = model(input_tensor.to(device)).squeeze(0) # Shape: (4, 2)
            
    # 3. Map coordinates back to original resolution
    pred_coords = pred_coords.cpu().numpy()
    pred_coords[:, 0] *= orig_w
    pred_coords[:, 1] *= orig_h
    
    # 4. Visualize the corners
    output_img = img.copy()
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)] # Different color per corner
    for i, (x, y) in enumerate(pred_coords):
        cv2.circle(output_img, (int(x), int(y)), 15, colors[i], -1)
        cv2.putText(output_img, str(i), (int(x)+15, int(y)-15), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[i], 2)
        
    cv2.imwrite(output_path, output_img)
    print(f"Corner detection visualized and saved to {output_path}")


if __name__ == '__main__':
    test_image_path = 'test_dataset/seungmin-yoon-RyFKQTYZLOg-unsplash_jpg.rf.f5741daad416f51734d2cd996f54a84b.jpg' 
    
    print("Testing Direct Corner Detection...")
    corner_inference_pipeline(
        model_path='weights/direct_corner_epoch_5.pth', 
        image_path=test_image_path, 
        output_path='corner_result_direct.jpg', 
        is_heatmap=False
    )