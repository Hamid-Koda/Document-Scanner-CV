import os
import json
import cv2
import numpy as np
import torch
from model import DirectCornerRegressor, HeatmapCornerRegressor, soft_argmax_2d

def calculate_euclidean_distance(pred_pts, true_pts):
    dist = np.sqrt(np.sum((pred_pts - true_pts)**2, axis=1))
    return dist

def evaluate_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Quantitative Evaluation on: {device}")

    json_path = "test_dataset/_annotations.coco.json" 
    img_dir = "test_dataset/"
    
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return

    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    images_dict = {img['id']: img['file_name'] for img in coco_data['images']}
    
    models_to_test = {
        "Approach A (No Dropout)": {"class": DirectCornerRegressor, "weight": "weights/direct_corner_best(new).pth", "dropout": False},
        "Approach A (With Dropout)": {"class": DirectCornerRegressor, "weight": "weights/direct_corner_best(dropout-new).pth", "dropout": True},
        "Approach B (No Dropout)": {"class": HeatmapCornerRegressor, "weight": "weights/heatmap_corner_best(new).pth", "dropout": False},
        "Approach B (With Dropout)": {"class": HeatmapCornerRegressor, "weight": "weights/heatmap_corner_best(dropout-new).pth", "dropout": True}
    }

    threshold = 200.0 

    print("\n" + "="*80)
    print(f"{'Model Version':<30} | {'Mean Error (px)':<20} | {'Success Rate (%)':<20}")
    print("="*80)

    for model_name, info in models_to_test.items():
        weight_path = info["weight"]
        if not os.path.exists(weight_path):
            print(f"{model_name:<30} | {'Weight file missing':<20} | {'-':<20}")
            continue

        model = info["class"](use_dropout=info["dropout"]).to(device)
        
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        total_error = 0.0
        success_count = 0
        total_images = len(coco_data['annotations'])

        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            img_filename = images_dict[image_id]
            img_path = os.path.join(img_dir, img_filename)

            keypoints = ann['keypoints']
            true_coords = np.array([
                [keypoints[0], keypoints[1]],
                [keypoints[3], keypoints[4]],
                [keypoints[6], keypoints[7]],
                [keypoints[9], keypoints[10]]
            ], dtype=np.float32)

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            orig_h, orig_w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (256, 256))
            tensor = (torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.).to(device)

            with torch.no_grad():
                if "Approach A" in model_name:
                    preds_norm = model(tensor).squeeze().cpu().numpy()
                else:
                    heatmaps = model(tensor)
                    preds_norm = soft_argmax_2d(heatmaps).squeeze().cpu().numpy()

            pred_coords = np.zeros_like(true_coords)
            pred_coords[:, 0] = preds_norm[:, 0] * orig_w
            pred_coords[:, 1] = preds_norm[:, 1] * orig_h

            distances = calculate_euclidean_distance(pred_coords, true_coords)
            mean_dist = np.mean(distances)
            total_error += mean_dist

            if np.max(distances) <= threshold:
                success_count += 1

        final_mean_error = total_error / total_images if total_images > 0 else 0
        final_success_rate = (success_count / total_images) * 100 if total_images > 0 else 0
        
        print(f"{model_name:<30} | {final_mean_error:<17.2f} px | {final_success_rate:>13.2f} %")

    print("-" * 80)

if __name__ == '__main__':
    evaluate_models()