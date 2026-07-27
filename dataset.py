import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import random

class DocumentDegradation:
    def __init__(self):
        pass

    def apply_perspective_warp(self, scan_img, bg_img):
        """1. Random perspective warp onto a random background"""
        h, w = scan_img.shape[:2]
        bg_h, bg_w = bg_img.shape[:2]
        
        # Source corners (clean scan)
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # Target corners on background (randomized)
        margin_x, margin_y = bg_w * 0.2, bg_h * 0.2
        pt1 = [random.uniform(0, margin_x), random.uniform(0, margin_y)]
        pt2 = [random.uniform(bg_w - margin_x, bg_w), random.uniform(0, margin_y)]
        pt3 = [random.uniform(bg_w - margin_x, bg_w), random.uniform(bg_h - margin_y, bg_h)]
        pt4 = [random.uniform(0, margin_x), random.uniform(bg_h - margin_y, bg_h)]
        
        dst_pts = np.float32([pt1, pt2, pt3, pt4])
        
        # Compute Homography matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Warp the clean scan
        warped_scan = cv2.warpPerspective(scan_img, M, (bg_w, bg_h))
        
        # Create a mask to blend with background
        mask = np.ones((h, w), dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(mask, M, (bg_w, bg_h))
        inv_mask = cv2.bitwise_not(warped_mask)
        
        # Composite images
        bg_cutout = cv2.bitwise_and(bg_img, bg_img, mask=inv_mask)
        warped_cutout = cv2.bitwise_and(warped_scan, warped_scan, mask=warped_mask)
        composited = cv2.add(bg_cutout, warped_cutout)
        
        return composited, dst_pts, M

    def apply_scaling(self, img):
        """2. Random downscale-upscale by a factor between 2 and 4"""
        factor = random.uniform(2.0, 4.0)
        h, w = img.shape[:2]
        
        downscaled = cv2.resize(img, (int(w / factor), int(h / factor)), interpolation=cv2.INTER_AREA)
        upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_CUBIC)
        return upscaled

    def apply_brightness_contrast_color(self, img):
        """3. Random brightness, contrast, and color-cast adjustment"""
        alpha = random.uniform(0.8, 1.2) 
        beta = random.randint(-30, 30)
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        img_float = adjusted.astype(np.float32)
        b_factor = random.uniform(0.9, 1.1)
        r_factor = random.uniform(0.9, 1.1)
        
        b, g, r = cv2.split(img_float)
        b = np.clip(b * b_factor, 0, 255)
        r = np.clip(r * r_factor, 0, 255)
        
        img_color_cast = cv2.merge((b, g, r)).astype(np.uint8)
        return img_color_cast
    
    def apply_illumination_and_shadows(self, img):
        """4. Illumination gradient and soft shadows"""
        h, w = img.shape[:2]
        
        # Base gradient
        gradient = np.ones((h, w), dtype=np.float32)
        cv2.circle(gradient, (random.randint(0, w), random.randint(0, h)), max(h, w), (random.uniform(0.6, 1.0),), -1)
        gradient = cv2.GaussianBlur(gradient, (0, 0), sigmaX=max(h, w)/3)
        
        # Add random soft shadows (polygons)
        shadow_mask = np.ones((h, w), dtype=np.float32)
        for _ in range(random.randint(1, 3)):
            pts = np.array([[random.randint(0, w), random.randint(0, h)] for _ in range(random.randint(3, 5))])
            cv2.fillPoly(shadow_mask, [pts], random.uniform(0.4, 0.8))
            
        shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), sigmaX=random.uniform(50, 150))
        
        # Apply masks to image
        combined_mask = gradient * shadow_mask
        img_float = img.astype(np.float32) * np.expand_dims(combined_mask, axis=2)
        
        return np.clip(img_float, 0, 255).astype(np.uint8)

    def apply_blur_and_noise(self, img):
        """5. Gaussian blur followed by Gaussian noise"""
        ksize = random.choice([3, 5])
        blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
        
        row, col, ch = blurred.shape
        mean = 0
        sigma = random.uniform(5, 15)
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        
        noisy = blurred + gauss
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy

    def apply_jpeg_compression(self, img):
        """6. JPEG re-encoding at a random quality between 30 and 80"""
        quality = random.randint(30, 80)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        return decimg
    

class EnhancementDataset(Dataset):
    def __init__(self, clean_scans, backgrounds, target_size=(256, 256), epoch_size=1000):
        self.clean_scans = clean_scans
        self.backgrounds = backgrounds
        self.target_size = target_size
        self.epoch_size = epoch_size
        self.degrader = DocumentDegradation()
        
    def __len__(self):
        # Generate samples on the fly, so epoch size is arbitrary
        return self.epoch_size
        
    def __getitem__(self, idx):
        scan_img = cv2.imread(random.choice(self.clean_scans))
        bg_img = cv2.imread(random.choice(self.backgrounds))
        
        # 1. Apply pipeline
        composited, corners, M = self.degrader.apply_perspective_warp(scan_img, bg_img)
        img = self.degrader.apply_scaling(composited)
        img = self.degrader.apply_brightness_contrast_color(img)
        img = self.degrader.apply_illumination_and_shadows(img)
        img = self.degrader.apply_blur_and_noise(img)
        img = self.degrader.apply_jpeg_compression(img)
        
        # 2. Rectify back to flat rectangle (for Enhancement task)
        h, w = scan_img.shape[:2]
        inv_M = np.linalg.inv(M)
        rectified_degraded = cv2.warpPerspective(img, inv_M, (w, h))
        
        # 3. Resize
        rectified_resized = cv2.resize(rectified_degraded, self.target_size)
        target_resized = cv2.resize(scan_img, self.target_size)
        
        # 4. Normalize and convert to PyTorch tensors (CHW format)
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
        
        # 1. Apply pipeline
        composited, corners, _ = self.degrader.apply_perspective_warp(scan_img, bg_img)
        img = self.degrader.apply_scaling(composited)
        img = self.degrader.apply_brightness_contrast_color(img)
        img = self.degrader.apply_illumination_and_shadows(img)
        img = self.degrader.apply_blur_and_noise(img)
        img = self.degrader.apply_jpeg_compression(img)
        
        # 2. Resize
        orig_h, orig_w = img.shape[:2]
        img_resized = cv2.resize(img, self.target_size)
        
        # 3. Normalize corners to [0, 1] range
        corners_normalized = corners.copy()
        corners_normalized[:, 0] /= orig_w
        corners_normalized[:, 1] /= orig_h
        
        # 4. Normalize image to PyTorch tensor (CHW format)
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
        corners_tensor = torch.from_numpy(corners_normalized).float()
        
        return img_tensor, corners_tensor