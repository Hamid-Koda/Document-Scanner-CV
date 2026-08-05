import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
from model import DirectCornerRegressor, EnhancementUNet, HeatmapCornerRegressor, soft_argmax_2d

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # -------------------------------------
    # ۱. لود کردن مدل‌ها و وزن‌های آموزش‌دیده (استفاده از Heatmap)
    # -------------------------------------
    corner_model = HeatmapCornerRegressor().to(device)
    corner_model.load_state_dict(torch.load('weights/heatmap_corner_best.pth', map_location=device))
    corner_model.eval()

    enhancement_model = EnhancementUNet().to(device)
    enhancement_model.load_state_dict(torch.load('weights/enhancement_unet_best.pth', map_location=device))
    enhancement_model.eval()

    
    image_path = 'test_image.jpg'
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Please put an image named '{image_path}' in the folder.")
        return

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    h, w, _ = original_image.shape

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(original_image).unsqueeze(0).to(device)

    
    with torch.no_grad():
        heatmaps = corner_model(input_tensor)
        predicted_coords = soft_argmax_2d(heatmaps).squeeze(0).cpu().numpy()
    
    predicted_coords = predicted_coords * np.array([w, h])
    
    warped_image = four_point_transform(original_image, predicted_coords)

    warped_resized = cv2.resize(warped_image, (256, 256))
    warped_tensor = transforms.ToTensor()(warped_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        enhanced_tensor = enhancement_model(warped_tensor)
    
    enhanced_image = enhanced_tensor.squeeze(0).cpu().numpy()
    enhanced_image = np.transpose(enhanced_image, (1, 2, 0))
    enhanced_image = np.clip(enhanced_image, 0, 1)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(original_image)
    axs[0].scatter(predicted_coords[:, 0], predicted_coords[:, 1], c='red', s=50, marker='x')
    axs[0].set_title("Original + Detected Corners")
    axs[0].axis('off')

    axs[1].imshow(warped_image)
    axs[1].set_title("Cropped & Warped")
    axs[1].axis('off')

    axs[2].imshow(enhanced_image)
    axs[2].set_title("Final Enhanced Scan")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()