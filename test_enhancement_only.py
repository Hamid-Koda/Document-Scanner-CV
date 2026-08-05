import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import EnhancementUNet

def test_enhancement_only():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading Enhancement Model on {device}...")

    model = EnhancementUNet().to(device)
    model.load_state_dict(torch.load('weights/enhancement_unet_best.pth', map_location=device))
    model.eval()

    img_path = 'dirty_cropped_paper.jpg' 
    img_bgr = cv2.imread(img_path)
    
    if img_bgr is None:
        print(f"❌ Error: Please put an image named '{img_path}' in the folder.")
        return

    orig_h, orig_w = img_bgr.shape[:2]

    # ✅ رفع باگِ بزرگ: تبدیل به RGB قبل از دادن به مدل!
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # پیش‌پردازش
    img_resized = cv2.resize(img_rgb, (256, 256))
    input_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

    print("Enhancing the image...")
    with torch.no_grad():
        output = model(input_tensor)

    # پس‌پردازش
    output_np = output.squeeze().cpu().numpy()
    output_np = np.clip(output_np, 0.0, 1.0)
    output_np = (output_np * 255).astype(np.uint8)
    
    # ✅ خروجی مدلِ ما مستقیماً RGB است، پس فقط ابعادش را درست می‌کنیم
    output_img_rgb = np.transpose(output_np, (1, 2, 0))
    output_img_rgb = cv2.resize(output_img_rgb, (orig_w, orig_h))

    # نمایش قبل و بعد
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    axs[0].imshow(img_rgb)
    axs[0].set_title("Original (Cropped & Dirty)")
    axs[0].axis('off')

    axs[1].imshow(output_img_rgb)
    axs[1].set_title("Enhanced (Cleaned Scan)")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    test_enhancement_only()