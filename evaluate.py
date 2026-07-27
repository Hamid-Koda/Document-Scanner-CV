import torch
import numpy as np
import cv2
import math
from skimage.metrics import structural_similarity as ssim
from model import EnhancementUNet

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

if __name__ == '__main__':
    # Usage example:
    # inference_pipeline('weights/enhancement_unet_epoch_10.pth', 'test_rectified.jpg', 'output_clean.jpg')
    pass