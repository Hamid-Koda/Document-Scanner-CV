import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from dataset import CornerDataset

def visualize_dataset_samples(num_samples=4):
    clean_scans_paths = glob.glob("clean_scans/*.*")
    background_paths = glob.glob("backgrounds/*.*")
    
    if not clean_scans_paths or not background_paths:
        print("Error: Folders 'clean_scans' or 'backgrounds' are empty or not found!")
        return

    dataset = CornerDataset(clean_scans_paths, background_paths, target_size=(256, 256), epoch_size=10)
    
    fig, axs = plt.subplots(2, num_samples // 2, figsize=(15, 8))
    axs = axs.flatten()
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)] 
    
    for i in range(num_samples):
        img_tensor, corners_tensor = dataset[i]
        
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) 
        
        corners_np = (corners_tensor.numpy() * 256).astype(int)
        
        for j, (x, y) in enumerate(corners_np):
            cv2.circle(img_np, (x, y), radius=5, color=colors[j], thickness=-1)
            cv2.putText(img_np, str(j+1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[j], 2)
            
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        
        axs[i].imshow(img_rgb)
        axs[i].set_title(f"Sample {i+1}")
        axs[i].axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    visualize_dataset_samples(4)