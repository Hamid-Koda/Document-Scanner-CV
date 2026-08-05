import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt
import glob
from dataset import EnhancementDataset
from model import EnhancementUNet

def test_overfit():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on: {device}")
    
    model = EnhancementUNet().to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    clean_paths = glob.glob("clean_scans/*.*")
    if not clean_paths:
        print("❌ Error: No images found in 'clean_scans/'")
        return
        
    dataset = EnhancementDataset(clean_paths, epoch_size=1)
    
    # گرفتن فقط یک نمونه تصادفی
    input_tensor, target_tensor = dataset[0]
    input_batch = input_tensor.unsqueeze(0).to(device)
    target_batch = target_tensor.unsqueeze(0).to(device)
    
    print("🚀 Starting Overfit Test on 1 Image (500 Epochs)...")
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        
        output = model(input_batch)
        loss = criterion(output, target_batch)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/500], Loss: {loss.item():.4f}")
            
    # نمایش خروجی نهایی
    model.eval()
    with torch.no_grad():
        final_out = model(input_batch).squeeze().cpu().permute(1, 2, 0).numpy()
        
    in_img = input_tensor.permute(1, 2, 0).numpy()
    tgt_img = target_tensor.permute(1, 2, 0).numpy()
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(in_img); axs[0].set_title("Degraded Input")
    axs[1].imshow(final_out.clip(0, 1)); axs[1].set_title("Model Output")
    axs[2].imshow(tgt_img); axs[2].set_title("Clean Target")
    plt.show()

if __name__ == '__main__':
    test_overfit()