import torch
import numpy as np
import cv2
import glob
import random
from torch.utils.data import DataLoader
from dataset import EnhancementDataset
from model import EnhancementUNet
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

def calculate_metrics(model, dataloader, device, is_baseline=False):
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.cpu().numpy()
            
            if is_baseline:
                preds = inputs.cpu().numpy()
            else:
                with torch.autocast(device_type=device.type):
                    preds = model(inputs).cpu().numpy()
            
            for i in range(preds.shape[0]):
                img_pred = np.transpose(preds[i], (1, 2, 0))
                img_true = np.transpose(targets[i], (1, 2, 0))
                
                img_pred = np.clip(img_pred, 0, 1)
                img_true = np.clip(img_true, 0, 1)
                
                psnr_val = compute_psnr(img_true, img_pred, data_range=1.0)
                ssim_val = compute_ssim(img_true, img_pred, channel_axis=2, data_range=1.0)
                
                total_psnr += psnr_val
                total_ssim += ssim_val
                num_samples += 1
                
    return total_psnr / num_samples, total_ssim / num_samples

def evaluate_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Evaluation Pipeline on: {device}")
    
    model = EnhancementUNet().to(device)
    model.load_state_dict(torch.load('weights/enhancement_unet_best(dropout).pth', map_location=device))
    model.eval()

    all_clean_paths = glob.glob("clean_scans/*.*")
    if len(all_clean_paths) == 0:
        print("Error: No clean scans found in 'clean_scans/' folder.")
        return
        
    random.seed(42)
    random.shuffle(all_clean_paths)
    
    n = len(all_clean_paths)
    train_paths = all_clean_paths[:int(0.7 * n)]
    val_paths = all_clean_paths[int(0.7 * n):int(0.85 * n)]
    test_paths = all_clean_paths[int(0.85 * n):]
    
    print(f"Dataset Split -> Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

    train_dataset = EnhancementDataset(train_paths, epoch_size=100)
    val_dataset = EnhancementDataset(val_paths, epoch_size=100)
    test_dataset = EnhancementDataset(test_paths, epoch_size=100)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print("\n⏳ Calculating Baseline metrics on Test set (Degraded vs Clean)...")
    base_psnr, base_ssim = calculate_metrics(model, test_loader, device, is_baseline=True)

    print("⏳ Calculating Train metrics...")
    train_psnr, train_ssim = calculate_metrics(model, train_loader, device, is_baseline=False)

    print("⏳ Calculating Validation metrics...")
    val_psnr, val_ssim = calculate_metrics(model, val_loader, device, is_baseline=False)

    print("⏳ Calculating Test metrics...")
    test_psnr, test_ssim = calculate_metrics(model, test_loader, device, is_baseline=False)

    print("\n" + "="*50)
    print(f"{'Split':<15} | {'PSNR (dB)':<12} | {'SSIM':<10}")
    print("-" * 50)
    print(f"{'Baseline (Test)':<15} | {base_psnr:<12.2f} | {base_ssim:<10.4f}")
    print(f"{'Training':<15} | {train_psnr:<12.2f} | {train_ssim:<10.4f}")
    print(f"{'Validation':<15} | {val_psnr:<12.2f} | {val_ssim:<10.4f}")
    print(f"{'Test':<15} | {test_psnr:<12.2f} | {test_ssim:<10.4f}")
    print("="*50 + "\n")

if __name__ == '__main__':
    evaluate_pipeline()