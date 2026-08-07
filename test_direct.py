import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import DirectCornerRegressor

def compare_dropout_effect():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Running Inference on: {device}")

    img_path = "test_image.jpg" # ⚠️ آدرس عکس تست خودت را بگذار
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise Exception(f"❌ Error: Image '{img_path}' not found.")

    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_input = cv2.resize(img_rgb, (256, 256))
    tensor = (torch.from_numpy(img_input).float().permute(2, 0, 1).unsqueeze(0) / 255.).to(device)

    model = DirectCornerRegressor().to(device)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    # ==========================================
    # ۱. اجرای مدل بدون Dropout (وزن‌های قدیمی)
    # ==========================================
    print("Loading OLD weights (No Dropout)...")
    old_state_dict = torch.load("weights/direct_corner_best.pth", map_location=device) # ⚠️ نام فایل قدیمی‌ات را اینجا بگذار
    
    # 🎯 ترفند جادویی: آپدیت کردن ایندکس‌های قدیمی به جدید برای لود شدن در معماری جدید
    old_state_dict["regressor.4.weight"] = old_state_dict.pop("regressor.3.weight")
    old_state_dict["regressor.4.bias"] = old_state_dict.pop("regressor.3.bias")
    old_state_dict["regressor.7.weight"] = old_state_dict.pop("regressor.5.weight")
    old_state_dict["regressor.7.bias"] = old_state_dict.pop("regressor.5.bias")
    
    model.load_state_dict(old_state_dict)
    model.eval()
    with torch.no_grad():
        coords_no_drop = model(tensor).squeeze().cpu().numpy()

    img_no_drop = img_rgb.copy()
    for i in range(4):
        x, y = int(coords_no_drop[i, 0] * orig_w), int(coords_no_drop[i, 1] * orig_h)
        cv2.circle(img_no_drop, (x, y), 18, colors[i], -1)
        cv2.putText(img_no_drop, str(i+1), (x + 18, y - 18), cv2.FONT_HERSHEY_SIMPLEX, 2, colors[i], 4)

    # ==========================================
    # ۲. اجرای مدل با Dropout (وزن‌های جدید)
    # ==========================================
    print("Loading NEW weights (With Dropout)...")
    # ⚠️ نام فایلی که همین الان ترین کردی را اینجا بگذار
    model.load_state_dict(torch.load("weights/direct_corner_best(dropout).pth", map_location=device)) 
    model.eval()
    with torch.no_grad():
        coords_with_drop = model(tensor).squeeze().cpu().numpy()

    img_with_drop = img_rgb.copy()
    for i in range(4):
        x, y = int(coords_with_drop[i, 0] * orig_w), int(coords_with_drop[i, 1] * orig_h)
        cv2.circle(img_with_drop, (x, y), 18, colors[i], -1)
        cv2.putText(img_with_drop, str(i+1), (x + 18, y - 18), cv2.FONT_HERSHEY_SIMPLEX, 2, colors[i], 4)

    # ==========================================
    # ۳. رسم تصاویر کنار هم
    # ==========================================
    fig, axs = plt.subplots(1, 2, figsize=(16, 10))
    
    axs[0].imshow(img_no_drop)
    axs[0].set_title("1. Before Regularization (No Dropout)", fontsize=16, fontweight='bold')
    axs[0].axis("off")

    axs[1].imshow(img_with_drop)
    axs[1].set_title("2. After Regularization (With Dropout)", fontsize=16, fontweight='bold')
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    compare_dropout_effect()