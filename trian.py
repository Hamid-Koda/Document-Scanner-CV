import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os

# Import our custom modules
from dataset import EnhancementDataset
from model import EnhancementUNet

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
    clean_scans_paths = ["dummy_scan.jpg"] 
    background_paths = ["dummy_bg.jpg"]

    # Initialize Dataset and DataLoader
    train_dataset = EnhancementDataset(clean_scans_paths, background_paths, epoch_size=500)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)

    num_epochs = 10
    
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
    # train_enhancement_model() # Commented out until real images are provided
    pass