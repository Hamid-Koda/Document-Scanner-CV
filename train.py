import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import glob
from dataset import EnhancementDataset
from model import EnhancementUNet
from dataset import CornerDataset
from model import DirectCornerRegressor, HeatmapCornerRegressor

class EdgeAwareLoss(nn.Module):
    """Custom Loss: Combines L1 loss with Edge (Gradient) loss for sharper text"""
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        
    def get_gradients(self, img):
        # Sobel kernels for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(img.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(img.device)
        
        # We apply the filter to each channel separately
        batch_size, channels, h, w = img.shape
        img_reshaped = img.view(-1, 1, h, w)
        
        grad_x = F.conv2d(img_reshaped, sobel_x, padding=1)
        grad_y = F.conv2d(img_reshaped, sobel_y, padding=1)
        
        return torch.abs(grad_x) + torch.abs(grad_y)

    def forward(self, pred, target):
        # 1. Base color/brightness reconstruction loss
        l1_loss = self.l1(pred, target)
        
        # 2. Edge reconstruction loss (sharpness)
        pred_grad = self.get_gradients(pred)
        target_grad = self.get_gradients(target)
        edge_loss = self.l1(pred_grad, target_grad)
        
        return l1_loss + self.alpha * edge_loss

def train_enhancement_model():
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize Model, Loss, and Optimizer
    model = EnhancementUNet().to(device)
    criterion = EdgeAwareLoss(alpha=0.5)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # TODO: Replace with actual paths to your clean scans and background images
    clean_scans_paths = glob.glob("clean_scans/*.*")
    background_paths = glob.glob("backgrounds/*.*")

    # Initialize Dataset and DataLoader
    train_dataset = EnhancementDataset(clean_scans_paths, background_paths, epoch_size=500)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)

    num_epochs = 5
    
    # Create directory to save weights
    os.makedirs('weights', exist_ok=True)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        print(f"--> Epoch {epoch+1} Average Loss: {epoch_loss:.4f}\n")
        
        # Save model checkpoint
        torch.save(model.state_dict(), f'weights/enhancement_unet_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    train_enhancement_model() # Commented out until real images are provided


def generate_gaussian_heatmaps(coords, target_size, sigma=5.0):
    """
    Creates Gaussian heatmaps for given coordinates.
    coords: Tensor of shape (B, 4, 2) with normalized [x, y] in range [0, 1]
    target_size: (H, W) tuple
    """
    B, num_corners, _ = coords.shape
    H, W = target_size
    device = coords.device

    # De-normalize coordinates
    xs = coords[..., 0] * W
    ys = coords[..., 1] * H

    # Create coordinate grids
    y_grid = torch.arange(H, dtype=torch.float32, device=device).view(1, 1, H, 1)
    x_grid = torch.arange(W, dtype=torch.float32, device=device).view(1, 1, 1, W)

    # Reshape xs and ys for broadcasting
    xs = xs.view(B, num_corners, 1, 1)
    ys = ys.view(B, num_corners, 1, 1)

    # Compute Gaussian blobs
    squared_dist = (x_grid - xs)**2 + (y_grid - ys)**2
    heatmaps = torch.exp(-squared_dist / (2 * sigma**2))

    return heatmaps

def train_direct_corner_detector():
    """Approach A: Direct Regression Training Loop"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DirectCornerRegressor().to(device)
    
    # L1 or L2 loss as requested by the document for coordinate regression
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    clean_scans_paths = glob.glob("clean_scans/*.*")
    background_paths = glob.glob("backgrounds/*.*")

    train_dataset = CornerDataset(clean_scans_paths, background_paths, epoch_size=500)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    num_epochs = 5
    os.makedirs('weights', exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Approach A - Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}")
        torch.save(model.state_dict(), f'weights/direct_corner_epoch_{epoch+1}.pth')

def train_heatmap_corner_detector():
    """Approach B: Heatmap Regression Training Loop"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeatmapCornerRegressor().to(device)
    
    # Pixel-wise loss on heatmaps as requested by the document
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    clean_scans_paths = glob.glob("clean_scans/*.*")
    background_paths = glob.glob("backgrounds/*.*")
    target_size = (256, 256)

    train_dataset = CornerDataset(clean_scans_paths, background_paths, target_size=target_size, epoch_size=500)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    num_epochs = 5
    os.makedirs('weights', exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, target_coords in train_loader:
            inputs, target_coords = inputs.to(device), target_coords.to(device)
            
            # Generate target heatmaps dynamically for the batch
            target_heatmaps = generate_gaussian_heatmaps(target_coords, target_size, sigma=5.0)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target_heatmaps)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Approach B - Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}")
        torch.save(model.state_dict(), f'weights/heatmap_corner_epoch_{epoch+1}.pth')