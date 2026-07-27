import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv2d => ReLU => Conv2d => ReLU) block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class EnhancementUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # Encoder (Downsampling)
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)
        
        # Decoder (Upsampling with Skip Connections)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256) # 256 from upconv + 256 from skip connection
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        # Final Output Layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid() # Maps output pixel values to [0, 1]

    def forward(self, x):
        # Encoder Path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool3(e3))
        
        # Decoder Path with Skip Connections
        d3 = self.upconv3(b)
        d3 = torch.cat([e3, d3], dim=1) # Applying Skip Connection
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
        
        # Final output
        out = self.out_conv(d1)
        return self.sigmoid(out)
    

class DirectCornerRegressor(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Encoder (Feature Extractor)
        # Assuming input image is resized to 256x256
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=2), # 128x128
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),          # 64x64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),         # 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),        # 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),        # 8x8
            nn.ReLU(inplace=True)
        )

        # Fully Connected layers (Regressor)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 8),   # 4 corners * 2 coordinates (x, y)
            nn.Sigmoid()         # Output normalized to [0, 1]
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        
        # Reshape output from (Batch, 8) to (Batch, 4 corners, 2 coords)
        # This matches the shape of our target tensor in CornerDataset
        return x.view(-1, 4, 2)
    

class HeatmapCornerRegressor(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super().__init__()
        # Reusing the Encoder-Decoder structure
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)
        
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        # Output layer produces 4 heatmaps (one for each corner)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder Path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool3(e3))
        
        # Decoder Path
        d3 = self.upconv3(b)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
        
        # Raw heatmaps (logits)
        heatmaps = self.out_conv(d1)
        return heatmaps

def soft_argmax_2d(heatmaps):
    """
    Extracts (x, y) coordinates from heatmaps using differentiable soft-argmax.
    heatmaps shape: (Batch, 4, H, W)
    Returns: (Batch, 4, 2) normalized coordinates [0, 1]
    """
    b, c, h, w = heatmaps.shape
    
    # 1. Apply softmax over the spatial dimensions to get probability distributions
    heatmaps_flat = heatmaps.view(b, c, -1)
    probs_flat = F.softmax(heatmaps_flat, dim=-1)
    probs = probs_flat.view(b, c, h, w)
    
    # 2. Create x and y coordinate grids normalized to [0, 1]
    y_grid = torch.linspace(0, 1, h, device=heatmaps.device).view(1, 1, h, 1)
    x_grid = torch.linspace(0, 1, w, device=heatmaps.device).view(1, 1, 1, w)
    
    # 3. Compute expected values for x and y
    y_expected = torch.sum(probs * y_grid, dim=(2, 3))
    x_expected = torch.sum(probs * x_grid, dim=(2, 3))
    
    # 4. Combine into final (x, y) coordinates
    coords = torch.stack([x_expected, y_expected], dim=-1)
    return coords