import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import DirectCornerRegressor

def test_direct_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading Direct Regression model...")

    # بارگذاری مدل و وزن‌ها
    model = DirectCornerRegressor().to(device)
    model.load_state_dict(torch.load("weights/direct_corner_best.pth", map_location=device))
    model.eval()

    # خواندن عکسِ تست (عکسی که روی میز یا پس‌زمینه است)
    img_path = "test_document_on_desk.jpg" 
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise Exception(f"❌ Error: Image '{img_path}' not found.")

    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # پیش‌پردازش دقیقاً مشابه زمان آموزش
    img_input = cv2.resize(img_rgb, (256, 256))
    tensor = (torch.from_numpy(img_input).float().permute(2, 0, 1).unsqueeze(0) / 255.).to(device)

    # پیش‌بینیِ مستقیم مختصات (۸ عدد)
    with torch.no_grad():
        # خروجی شامل 4 جفت مختصات (x, y) است که بین 0 و 1 نرمال شده‌اند
        coords = model(tensor).squeeze().cpu().numpy() 

    result = img_rgb.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    print("\nDetected Corners (Direct Method)")
    print("-" * 30)

    # رسم گوشه‌ها روی تصویر اصلی
    for i in range(4):
        # تبدیل مختصات نرمال‌شده (۰ تا ۱) به ابعاد عکس اصلی
        x = int(coords[i, 0] * orig_w)
        y = int(coords[i, 1] * orig_h)
        
        print(f"Corner {i+1} Location: ({x}, {y})")
        
        # رسم دایره و شماره با OpenCV
        cv2.circle(result, (x, y), 18, colors[i], -1)
        cv2.putText(result, str(i+1), (x + 18, y - 18), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[i], 3)

    # نمایش خروجی با Matplotlib
    plt.figure(figsize=(12, 10))
    plt.imshow(result)
    plt.title("Detected Corners (Approach A - Direct Regression)")
    plt.axis("off")
    plt.show()

if __name__ == '__main__':
    test_direct_model()