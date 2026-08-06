import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from model import EnhancementUNet

def evaluate_cropped_triplets():
    # استفاده از CPU یا GPU سیستم شما
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting Qualitative Evaluation on: {device}")

    # 🎯 فقط مدل پاک‌سازی (Enhancement) را لود می‌کنیم
    enhance_model = EnhancementUNet().to(device)
    enhance_model.load_state_dict(torch.load('weights/enhancement_unet_best.pth', map_location=device))
    enhance_model.eval()

    # پیدا کردن تمام عکس‌های کراپ‌شده‌ی اولیه
    raw_images = glob.glob("real_test_photos/*_raw.jpg")
    if not raw_images:
        print("❌ Error: No raw images found. Make sure they end with '_raw.jpg'.")
        return

    os.makedirs('evaluation_results', exist_ok=True)

    for rect_path in raw_images:
        base_name = os.path.basename(rect_path).replace("_raw.jpg", "")
        camscanner_path = rect_path.replace("_raw.jpg", "_camscanner.jpg")

        if not os.path.exists(camscanner_path):
            print(f"⚠️ Warning: CamScanner version for {base_name} not found. Skipping...")
            continue

        print(f"📸 Processing {base_name}...")

        # ------------------------------------------------
        # ۱. خواندن عکس کراپ‌شده‌ی اولیه
        # ------------------------------------------------
        img_bgr = cv2.imread(rect_path)
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # تغییر سایز موقت به ابعادی که مضرب ۳۲ باشد تا U-Net بدون ارور کار کند
        target_w, target_h = 800, 1152 
        img_resized = cv2.resize(img_rgb, (target_w, target_h))

        # ------------------------------------------------
        # ۲. تمیز کردن با شبکه U-Net
        # ------------------------------------------------
        enhance_input = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        
        with torch.no_grad():
            enhanced_tensor = enhance_model(enhance_input)
            
        enhanced_np = (enhanced_tensor.squeeze(0).float().cpu().numpy() * 255).astype(np.uint8)
        our_output = np.transpose(enhanced_np, (1, 2, 0))
        
        # 🎯 رعایت قانون داکیومنت: برگرداندن عکس به سایز و تناسب اصلی خودش
        our_output_resized = cv2.resize(our_output, (orig_w, orig_h))

        # ------------------------------------------------
        # ۳. خواندن عکس CamScanner
        # ------------------------------------------------
        camscanner_img = cv2.cvtColor(cv2.imread(camscanner_path), cv2.COLOR_BGR2RGB)
        
        # ------------------------------------------------
        # ۴. رسم و ذخیره Triplet
        # ------------------------------------------------
        fig, axs = plt.subplots(1, 3, figsize=(18, 8))
        
        axs[0].imshow(img_rgb)
        axs[0].set_title("1. Raw Input (Raw)", fontsize=14, fontweight='bold')
        axs[0].axis('off')

        axs[1].imshow(our_output_resized)
        axs[1].set_title("2. Our Enhanced Output", fontsize=14, fontweight='bold')
        axs[1].axis('off')

        axs[2].imshow(camscanner_img)
        axs[2].set_title("3. CamScanner Reference", fontsize=14, fontweight='bold')
        axs[2].axis('off')

        plt.tight_layout()
        save_path = f"evaluation_results/{base_name}_comparison.jpg"
        plt.savefig(save_path, dpi=150)
        plt.close()
        
    print("✅ All comparisons saved successfully in 'evaluation_results/' folder!")

if __name__ == '__main__':
    evaluate_cropped_triplets()