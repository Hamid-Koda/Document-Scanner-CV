import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def verify_ground_truth():
    # مسیر فایل‌ها (اگر متفاوت است، اصلاح کن)
    json_path = "test_dataset/_annotations.coco.json"
    img_dir = "test_dataset/"

    if not os.path.exists(json_path):
        print(f"❌ Error: JSON file not found at {json_path}")
        return

    print("⏳ Loading JSON annotations...")
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # انتخاب یک عکس تصادفی از بین عکس‌های لیبل‌خورده
    random_annotation = random.choice(coco_data['annotations'])
    image_id = random_annotation['image_id']
    
    # پیدا کردن نام فایلِ عکس
    img_info = next(item for item in coco_data['images'] if item['id'] == image_id)
    img_filename = img_info['file_name']
    img_path = os.path.join(img_dir, img_filename)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"❌ Error: Could not read image {img_filename}")
        return
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    annotated_img = img_rgb.copy()

    # ------------------------------------------------
    # ۱. استخراج گوشه‌ها از JSON و رسم آن‌ها
    # ------------------------------------------------
    # فرمت keypoints در COCO: [x1, y1, v1, x2, y2, v2, ...]
    kp = random_annotation['keypoints']
    true_corners = np.float32([
        [kp[0], kp[1]],   # گوشه ۱
        [kp[3], kp[4]],   # گوشه ۲
        [kp[6], kp[7]],   # گوشه ۳
        [kp[9], kp[10]]   # گوشه ۴
    ])

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)] # قرمز، سبز، آبی، زرد
    
    for i, pt in enumerate(true_corners):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(annotated_img, (x, y), 25, colors[i], -1)
        cv2.putText(annotated_img, str(i+1), (x + 25, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 3, colors[i], 5)

    # ------------------------------------------------
    # ۲. برش عکس (Rectification) بر اساس نقاط JSON
    # ------------------------------------------------
    # تعریف ابعاد یک کاغذ استاندارد (مثلاً A4 با نسبت 1:1.414)
    target_w = 800
    target_h = int(target_w * 1.414)
    
    # نقاط مقصد (یک مستطیل کامل)
    target_corners = np.float32([
        [0, 0], 
        [target_w, 0], 
        [target_w, target_h], 
        [0, target_h]
    ])

    # محاسبه ماتریس پرسپکتیو و اعمال آن
    M = cv2.getPerspectiveTransform(true_corners, target_corners)
    cropped_img = cv2.warpPerspective(img_rgb, M, (target_w, target_h))

    # ------------------------------------------------
    # ۳. نمایش نتیجه
    # ------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    
    axs[0].imshow(annotated_img)
    axs[0].set_title(f"1. Ground Truth from JSON\n({img_filename})", fontsize=14, fontweight='bold')
    axs[0].axis("off")

    axs[1].imshow(cropped_img)
    axs[1].set_title("2. Rectified (Cropped) Output", fontsize=14, fontweight='bold')
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # برای دیدن چند عکس مختلف، می‌توانی این اسکریپت را چند بار اجرا کنی
    verify_ground_truth()