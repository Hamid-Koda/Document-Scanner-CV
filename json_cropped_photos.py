import os
import json
import cv2
import numpy as np

def batch_crop_from_json():
    json_path = "test_dataset/_annotations.coco.json"
    img_dir = "test_dataset/"
    output_dir = "json_cropped_photos"

    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    print("⏳ Loading JSON and starting batch crop...")
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    images_dict = {img['id']: img['file_name'] for img in coco_data['images']}
    
    target_w = 800
    target_h = int(target_w * 1.414) 
    
    target_corners = np.float32([
        [0, 0], 
        [target_w, 0], 
        [target_w, target_h], 
        [0, target_h]
    ])

    success_count = 0

    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        img_filename = images_dict[image_id]
        img_path = os.path.join(img_dir, img_filename)

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Warning: Could not read {img_filename}. Skipping...")
            continue

        kp = ann['keypoints']
        true_corners = np.float32([
            [kp[0], kp[1]],
            [kp[3], kp[4]],
            [kp[6], kp[7]],
            [kp[9], kp[10]]
        ])

        M = cv2.getPerspectiveTransform(true_corners, target_corners)
        cropped_img = cv2.warpPerspective(img_bgr, M, (target_w, target_h))

        base_name = os.path.splitext(img_filename)[0]
        save_path = os.path.join(output_dir, f"{base_name}_raw.jpg")
        
        cv2.imwrite(save_path, cropped_img)
        success_count += 1
        print(f"Cropped and saved: {base_name}_raw.jpg")

    print("=" * 50)
    print(f"🎉 Successfully cropped {success_count} images!")
    print(f"📁 You can find them in the '{output_dir}' folder.")

if __name__ == '__main__':
    batch_crop_from_json()