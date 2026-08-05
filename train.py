import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import glob
from dataset import EnhancementDataset, CornerDataset
from model import EnhancementUNet, DirectCornerRegressor, HeatmapCornerRegressor

class EdgeAwareLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
    def get_gradients(self, img):
        b, c, h, w = img.shape
        img_reshaped = img.view(-1, 1, h, w)
        grad_x = F.conv2d(img_reshaped, self.sobel_x, padding=1)
        grad_y = F.conv2d(img_reshaped, self.sobel_y, padding=1)
        return torch.abs(grad_x) + torch.abs(grad_y)

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        edge_loss = self.l1(self.get_gradients(pred), self.get_gradients(target))
        return l1_loss + self.alpha * edge_loss

def generate_gaussian_heatmaps(coords, target_size, sigma=5.0):
    B, num_corners, _ = coords.shape
    H, W = target_size
    device = coords.device
    xs = coords[..., 0] * W
    ys = coords[..., 1] * H
    y_grid = torch.arange(H, dtype=torch.float32, device=device).view(1, 1, H, 1)
    x_grid = torch.arange(W, dtype=torch.float32, device=device).view(1, 1, 1, W)
    xs = xs.view(B, num_corners, 1, 1)
    ys = ys.view(B, num_corners, 1, 1)
    squared_dist = (x_grid - xs)**2 + (y_grid - ys)**2
    return torch.exp(-squared_dist / (2 * sigma**2))

def train_direct_corner_detector():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DirectCornerRegressor().to(device)
    
    # 🎯 استفاده از Huber Loss برای جلوگیری از انفجار گرادیان‌ها
    criterion = nn.SmoothL1Loss()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    clean_scans_paths = glob.glob("clean_scans/*.*")
    background_paths = glob.glob("backgrounds/*.*")
    
    train_dataset = CornerDataset(clean_scans_paths, background_paths, epoch_size=1500)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

    num_epochs = 20
    os.makedirs('weights', exist_ok=True)
    
    checkpoint_path = 'weights/direct_corner_checkpoint.pth'
    best_model_path = 'weights/direct_corner_best.pth'
    start_epoch = 0
    best_loss = float('inf')
    patience = 5
    patience_counter = 0

    if os.path.exists(checkpoint_path):
        print(f"==> Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"==> Resuming training from epoch {start_epoch+1}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Approach A - Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        epoch_loss = running_loss / len(train_loader)
        print(f"--> Approach A - Epoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss:.4f}\n")
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }
        torch.save(checkpoint_data, checkpoint_path)
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"*** New best model saved with loss: {best_loss:.4f} ***")
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {patience}")
            if patience_counter >= patience:
                print("!!! Early stopping triggered. Moving to next phase. !!!")
                break

# -----------------------------------------
#  B: Heatmap Regressor
# -----------------------------------------
def train_heatmap_corner_detector():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeatmapCornerRegressor().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    clean_scans_paths = glob.glob("clean_scans/*.*")
    background_paths = glob.glob("backgrounds/*.*")
    target_size = (256, 256)

    train_dataset = CornerDataset(clean_scans_paths, background_paths, target_size=target_size, epoch_size=1500)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

    num_epochs = 25
    os.makedirs('weights', exist_ok=True)
    
    checkpoint_path = 'weights/heatmap_corner_checkpoint.pth'
    best_model_path = 'weights/heatmap_corner_best.pth'
    start_epoch = 0
    best_loss = float('inf')
    patience = 5
    patience_counter = 0

    if os.path.exists(checkpoint_path):
        print(f"==> Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"==> Resuming training from epoch {start_epoch+1}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, target_coords) in enumerate(train_loader):
            inputs, target_coords = inputs.to(device), target_coords.to(device)
            target_heatmaps = generate_gaussian_heatmaps(target_coords, target_size, sigma=15.0)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, target_heatmaps) 
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Approach B - Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
            
        epoch_loss = running_loss / len(train_loader)
        print(f"--> Approach B - Epoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss:.4f}\n")
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }
        torch.save(checkpoint_data, checkpoint_path)
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"*** New best model saved with loss: {best_loss:.4f} ***")
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {patience}")
            if patience_counter >= patience:
                print("!!! Early stopping triggered. Training finished. !!!")
                break


def train_enhancement_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancementUNet().to(device)
    
    criterion = EdgeAwareLoss(alpha=0.15).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    clean_scans_paths = glob.glob("clean_scans/*.*")
    
    # تله شماره صفر: چک کردن پیدا شدن فایل‌ها
    print(f"📸 Total clean scans found: {len(clean_scans_paths)}")
    if len(clean_scans_paths) == 0:
        print("❌ Error: No images found!")
        return

    # باگ برطرف شد: دیگر background_paths را پاس نمی‌دهیم
    train_dataset = EnhancementDataset(clean_scans_paths, epoch_size=1500)
    
    print(f"✅ Dataset instantiated. Length: {len(train_dataset)}")

    # تله شماره یک: تست کردن خودِ دیتاست (آیا عکس خوانده می‌شود؟)
    print("⏳ Testing dataset getitem...")
    x, y = train_dataset[0]
    print(f"✅ Dataset OK! Input shape: {x.shape}, Target shape: {y.shape}")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)

    # تله شماره دو: تست کردن DataLoader
    print("⏳ Testing DataLoader...")
    x_batch, y_batch = next(iter(train_loader))
    print(f"✅ DataLoader OK! Batch shape: {x_batch.shape}")

    num_epochs = 20
    os.makedirs('weights', exist_ok=True)
    best_model_path = 'weights/enhancement_unet_best.pth'
    best_loss = float('inf')
    patience = 5
    patience_counter = 0

    print("🚀 Starting Enhancement Model Training (RGB + L1Loss + num_workers=0)...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Enhancement - Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        epoch_loss = running_loss / len(train_loader)
        print(f"--> Enhancement - Epoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss:.4f}\n")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"*** New best model saved with loss: {best_loss:.4f} ***")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("!!! Early stopping triggered. Training finished. !!!")
                break


if __name__ == '__main__':
    #train_enhancement_model() 
    #train_direct_corner_detector()
    train_heatmap_corner_detector()