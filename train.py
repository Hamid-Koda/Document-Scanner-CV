import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import glob
import matplotlib.pyplot as plt
import random
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

def get_splits(paths):
    random.seed(42)
    paths_copy = paths.copy()
    random.shuffle(paths_copy)
    n = len(paths_copy)
    train_p = paths_copy[:int(0.8 * n)]
    val_p = paths_copy[int(0.8 * n):int(0.9 * n)]
    return train_p, val_p

def train_enhancement_model(use_dropout=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancementUNet(use_dropout=use_dropout).to(device)
    criterion = EdgeAwareLoss(alpha=0.15).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    clean_paths = glob.glob("clean_scans/*.*")
    train_paths, val_paths = get_splits(clean_paths)

    train_dataset = EnhancementDataset(train_paths, epoch_size=1500, is_train=True)
    val_dataset = EnhancementDataset(val_paths, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    num_epochs = 20
    os.makedirs('weights', exist_ok=True)
    tag = "(dropout)" if use_dropout else ""
    best_model_path = f'weights/enhancement_unet_best{tag}.pth'
    best_val_loss = float('inf')

    print(f"\n🚀 Training Enhancement Model | Dropout: {use_dropout}")
    
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_train_loss += loss.item()
                
        epoch_train_loss = running_train_loss / len(train_loader)
        
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                running_val_loss += loss.item()
                
        epoch_val_loss = running_val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train: {epoch_train_loss:.4f} | Val: {epoch_val_loss:.4f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"*** Best model saved: {best_model_path} ***")

def train_direct_corner_detector(use_dropout=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DirectCornerRegressor(use_dropout=use_dropout).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    clean_paths = glob.glob("clean_scans/*.*")
    bg_paths = glob.glob("backgrounds/*.*")
    train_paths, val_paths = get_splits(clean_paths)
    
    train_dataset = CornerDataset(train_paths, bg_paths, epoch_size=1500, is_train=True)
    val_dataset = CornerDataset(val_paths, bg_paths, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    num_epochs = 20
    tag = "(dropout)" if use_dropout else ""
    best_model_path = f'weights/direct_corner_best{tag}.pth'
    best_loss = float('inf')

    print(f"\n🚀 Training Direct Corner Model | Dropout: {use_dropout}")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
                
        epoch_loss = running_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        epoch_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"*** Best model saved: {best_model_path} ***")

def train_heatmap_corner_detector(use_dropout=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeatmapCornerRegressor(use_dropout=use_dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    clean_paths = glob.glob("clean_scans/*.*")
    bg_paths = glob.glob("backgrounds/*.*")
    train_paths, val_paths = get_splits(clean_paths)
    target_size = (256, 256)

    train_dataset = CornerDataset(train_paths, bg_paths, target_size=target_size, epoch_size=1500, is_train=True)
    val_dataset = CornerDataset(val_paths, bg_paths, target_size=target_size, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    num_epochs = 25
    tag = "(dropout)" if use_dropout else ""
    best_model_path = f'weights/heatmap_corner_best{tag}.pth'
    best_loss = float('inf')

    print(f"\n🚀 Training Heatmap Model | Dropout: {use_dropout}")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, target_coords in train_loader:
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
            
        epoch_loss = running_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, target_coords in val_loader:
                inputs, target_coords = inputs.to(device), target_coords.to(device)
                target_heatmaps = generate_gaussian_heatmaps(target_coords, target_size, sigma=15.0)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, target_heatmaps)
                val_loss += loss.item()
                
        epoch_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"*** Best model saved: {best_model_path} ***")

if __name__ == '__main__':
    
    train_enhancement_model(use_dropout=False)
    train_enhancement_model(use_dropout=True)
    
    train_direct_corner_detector(use_dropout=False)
    train_direct_corner_detector(use_dropout=True)
    
    train_heatmap_corner_detector(use_dropout=False)
    train_heatmap_corner_detector(use_dropout=True)
    
    print("\n🎉 All 6 models have been successfully trained and saved!")