import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import random

class DocumentDegradation:
    def __init__(self):
        pass

    def apply_perspective_warp(self, scan_img, bg_img):
        """1. Realistic perspective warp with strict margins"""
        h, w = scan_img.shape[:2]
        bg_h, bg_w = bg_img.shape[:2]
        
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        margin_x = int(bg_w * 0.25) 
        margin_y = int(bg_h * 0.25)
        
        base_w, base_h = bg_w - 2 * margin_x, bg_h - 2 * margin_y
        max_dev_x, max_dev_y = base_w * 0.2, base_h * 0.2

        pt1 = [margin_x + random.uniform(-max_dev_x, max_dev_x), margin_y + random.uniform(-max_dev_y, max_dev_y)]
        pt2 = [bg_w - margin_x + random.uniform(-max_dev_x, max_dev_x), margin_y + random.uniform(-max_dev_y, max_dev_y)]
        pt3 = [bg_w - margin_x + random.uniform(-max_dev_x, max_dev_x), bg_h - margin_y + random.uniform(-max_dev_y, max_dev_y)]
        pt4 = [margin_x + random.uniform(-max_dev_x, max_dev_x), bg_h - margin_y + random.uniform(-max_dev_y, max_dev_y)]
        
        dst_pts = np.float32([pt1, pt2, pt3, pt4])
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_scan = cv2.warpPerspective(scan_img, M, (bg_w, bg_h))
        
        mask = np.ones((h, w), dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(mask, M, (bg_w, bg_h))
        inv_mask = cv2.bitwise_not(warped_mask)
        
        bg_cutout = cv2.bitwise_and(bg_img, bg_img, mask=inv_mask)
        warped_cutout = cv2.bitwise_and(warped_scan, warped_scan, mask=warped_mask)
        composited = cv2.add(bg_cutout, warped_cutout)
        
        return composited, dst_pts, M

    def apply_scaling(self, img):
        factor = random.uniform(2.0, 4.0)
        h, w = img.shape[:2]
        downscaled = cv2.resize(img, (int(w / factor), int(h / factor)), interpolation=cv2.INTER_AREA)
        return cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_CUBIC)

    def apply_brightness_contrast_color(self, img):
        alpha = random.uniform(0.8, 1.2) 
        beta = random.randint(-30, 30)
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        img_float = adjusted.astype(np.float32)
        b, g, r = cv2.split(img_float)
        b = np.clip(b * random.uniform(0.9, 1.1), 0, 255)
        r = np.clip(r * random.uniform(0.9, 1.1), 0, 255)
        
        return cv2.merge((b, g, r)).astype(np.uint8)
    
    def apply_illumination_and_shadows(self, img):
        h, w = img.shape[:2]
        gradient = np.ones((h, w), dtype=np.float32)
        cv2.circle(gradient, (random.randint(0, w), random.randint(0, h)), max(h, w), (random.uniform(0.6, 1.0),), -1)
        gradient = cv2.GaussianBlur(gradient, (0, 0), sigmaX=max(h, w)/3)
        
        shadow_mask = np.ones((h, w), dtype=np.float32)
        for _ in range(random.randint(1, 3)):
            pts = np.array([[random.randint(0, w), random.randint(0, h)] for _ in range(random.randint(3, 5))])
            cv2.fillPoly(shadow_mask, [pts], random.uniform(0.4, 0.8))
            
        shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), sigmaX=random.uniform(50, 150))
        img_float = img.astype(np.float32) * np.expand_dims(gradient * shadow_mask, axis=2)
        return np.clip(img_float, 0, 255).astype(np.uint8)

    def apply_occlusions(self, img):
        """NEW: Simulates fingers, thumbs, or random objects overlapping the edges"""
        if random.random() < 0.5: # 50% chance
            h, w = img.shape[:2]
            center_x = random.choice([0, w]) + random.randint(-50, 50)
            center_y = random.randint(0, h)
            axes = (random.randint(30, 80), random.randint(30, 80))
            color = (random.randint(80, 180), random.randint(100, 200), random.randint(120, 220)) 
            cv2.ellipse(img, (center_x, center_y), axes, random.randint(0, 360), 0, 360, color, -1)
            img = cv2.GaussianBlur(img, (7, 7), 0)
        return img

    def apply_blur_and_noise(self, img):
        """Fix: Motion Blur + Heavy ISO Noise"""
        if random.random() < 0.3:
            # Motion Blur
            size = random.randint(5, 11)
            kernel = np.zeros((size, size))
            if random.random() < 0.5:
                kernel[int((size-1)/2), :] = np.ones(size)
            else:
                kernel[:, int((size-1)/2)] = np.ones(size)
            kernel = kernel / size
            blurred = cv2.filter2D(img, -1, kernel)
        else:
            # Standard Blur
            blurred = cv2.GaussianBlur(img, (random.choice([3, 5]), random.choice([3, 5])), 0)
        
        # ISO Noise
        row, col, ch = blurred.shape
        gauss = np.random.normal(0, random.uniform(10, 25), (row, col, ch)).reshape(row, col, ch)
        noisy = blurred + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def apply_jpeg_compression(self, img):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(30, 80)]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(encimg, 1)

class EnhancementDataset(Dataset):
    def __init__(self, clean_scans, backgrounds, target_size=(256, 256), epoch_size=1000):
        self.clean_scans = clean_scans
        self.backgrounds = backgrounds
        self.target_size = target_size
        self.epoch_size = epoch_size
        self.degrader = DocumentDegradation()
        
    def __len__(self):
        return self.epoch_size
        
    def __getitem__(self, idx):
        scan_img = cv2.imread(random.choice(self.clean_scans))
        bg_img = cv2.imread(random.choice(self.backgrounds))
        
        composited, corners, M = self.degrader.apply_perspective_warp(scan_img, bg_img)
        img = self.degrader.apply_scaling(composited)
        img = self.degrader.apply_brightness_contrast_color(img)
        img = self.degrader.apply_illumination_and_shadows(img)
        img = self.degrader.apply_occlusions(img) 
        img = self.degrader.apply_blur_and_noise(img)
        img = self.degrader.apply_jpeg_compression(img)
        
        h, w = scan_img.shape[:2]
        inv_M = np.linalg.inv(M)
        rectified_degraded = cv2.warpPerspective(img, inv_M, (w, h))
        
        rectified_resized = cv2.resize(rectified_degraded, self.target_size)
        target_resized = cv2.resize(scan_img, self.target_size)
        
        input_tensor = torch.from_numpy(rectified_resized).float().permute(2, 0, 1) / 255.0
        target_tensor = torch.from_numpy(target_resized).float().permute(2, 0, 1) / 255.0
        
        return input_tensor, target_tensor

class CornerDataset(Dataset):
    def __init__(self, clean_scans, backgrounds, target_size=(256, 256), epoch_size=1000):
        self.clean_scans = clean_scans
        self.backgrounds = backgrounds
        self.target_size = target_size
        self.epoch_size = epoch_size
        self.degrader = DocumentDegradation()
        
    def __len__(self):
        return self.epoch_size
        
    def __getitem__(self, idx):
        scan_img = cv2.imread(random.choice(self.clean_scans))
        bg_img = cv2.imread(random.choice(self.backgrounds))
        
        composited, corners, _ = self.degrader.apply_perspective_warp(scan_img, bg_img)
        img = self.degrader.apply_scaling(composited)
        img = self.degrader.apply_brightness_contrast_color(img)
        img = self.degrader.apply_illumination_and_shadows(img)
        img = self.degrader.apply_occlusions(img) 
        img = self.degrader.apply_blur_and_noise(img)
        img = self.degrader.apply_jpeg_compression(img)
        
        orig_h, orig_w = img.shape[:2]
        img_resized = cv2.resize(img, self.target_size)
        
        corners_normalized = corners.copy()
        corners_normalized[:, 0] /= orig_w
        corners_normalized[:, 1] /= orig_h
        
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
        corners_tensor = torch.from_numpy(corners_normalized).float()
        
        return img_tensor, corners_tensor