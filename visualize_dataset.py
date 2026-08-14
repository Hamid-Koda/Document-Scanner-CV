import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from dataset import CornerDataset

def visualize_synthetic_data():
    print("Loading dataset generator...")
    clean_paths = glob.glob("clean_scans/*.*")
    bg_paths = glob.glob("backgrounds/*.*")
    
    dataset = CornerDataset(clean_paths, bg_paths, epoch_size=5)
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    for i in range(3):
        img_tensor, coords_tensor = dataset[i]
        
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w = img_np.shape[:2]
        
        coords = coords_tensor.numpy() * np.array([w, h])
        
        for (x, y) in coords:
            cv2.circle(img_np, (int(x), int(y)), 5, (0, 0, 255), -1)
            
        pts = coords.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_np, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        axs[i].imshow(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        axs[i].set_title(f"Synthetic Sample {i+1}")
        axs[i].axis('off')
        
    plt.tight_layout()
    plt.savefig('synthetic_samples.jpg')
    print("Saved 'synthetic_samples.jpg'.")
    plt.show()

if __name__ == '__main__':
    visualize_synthetic_data()