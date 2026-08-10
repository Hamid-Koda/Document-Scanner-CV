import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import random

class DocumentDegradation:
    def __init__(self):
        pass

    def apply_perspective_warp(self, scan_img, bg_img):
        h, w = scan_img.shape[:2]
        bg_h, bg_w = bg_img.shape[:2]
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        for _ in range(50):
            zoom_type = random.random()
            margin = 0.10 if zoom_type < 0.4 else (0.15 if zoom_type < 0.8 else 0.18)
            
            left = random.uniform(0.0, margin) * bg_w
            right = random.uniform(1 - margin, 1.0) * bg_w
            top = random.uniform(0.0, margin) * bg_h
            bottom = random.uniform(1 - margin, 1.0) * bg_h
            
            dev_x = (right - left) * 0.25
            dev_y = (bottom - top) * 0.25
            
            pt1 = [max(0, left + random.uniform(-dev_x, dev_x)), max(0, top + random.uniform(-dev_y, dev_y))]
            pt2 = [min(bg_w, right + random.uniform(-dev_x, dev_x)), max(0, top + random.uniform(-dev_y, dev_y))]
            pt3 = [min(bg_w, right + random.uniform(-dev_x, dev_x)), min(bg_h, bottom + random.uniform(-dev_y, dev_y))]
            pt4 = [max(0, left + random.uniform(-dev_x, dev_x)), min(bg_h, bottom + random.uniform(-dev_y, dev_y))]
            
            dst_pts = np.float32([pt1, pt2, pt3, pt4])
            
            if cv2.isContourConvex(dst_pts.astype(int)):
                break
        else:
            dst_pts = np.float32([[left, top], [right, top], [right, bottom], [left, bottom]])
            
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
        factor = random.uniform(1.5, 4.0)
        h, w = img.shape[:2]
        downscaled = cv2.resize(img, (int(w / factor), int(h / factor)), interpolation=cv2.INTER_AREA)
        return cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_CUBIC)

    def apply_brightness_contrast_color(self, img):
        alpha = random.uniform(0.7, 1.3) 
        beta = random.randint(-40, 40)
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        img_float = adjusted.astype(np.float32)
        b, g, r = cv2.split(img_float)
        b = np.clip(b * random.uniform(0.8, 1.2), 0, 255)
        r = np.clip(r * random.uniform(0.8, 1.2), 0, 255)
        return cv2.merge((b, g, r)).astype(np.uint8)
    
    def apply_illumination_and_shadows(self, img):
        h, w = img.shape[:2]
        gradient = np.ones((h, w), dtype=np.float32)
        cv2.circle(gradient, (random.randint(0, w), random.randint(0, h)), max(h, w), (random.uniform(0.5, 1.0),), -1)
        gradient = cv2.GaussianBlur(gradient, (0, 0), sigmaX=max(h, w)/3)
        shadow_mask = np.ones((h, w), dtype=np.float32)
        for _ in range(random.randint(1, 4)):
            pts = np.array([[random.randint(0, w), random.randint(0, h)] for _ in range(random.randint(3, 5))])
            cv2.fillPoly(shadow_mask, [pts], random.uniform(0.3, 0.8))
        shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), sigmaX=random.uniform(50, 150))
        img_float = img.astype(np.float32) * np.expand_dims(gradient * shadow_mask, axis=2)
        return np.clip(img_float, 0, 255).astype(np.uint8)

    def apply_occlusions(self, img):
        if random.random() < 0.5: 
            h, w = img.shape[:2]
            center_x = random.choice([0, w]) + random.randint(-50, 50)
            center_y = random.randint(0, h)
            axes = (random.randint(30, 100), random.randint(30, 100))
            color = (random.randint(80, 180), random.randint(100, 200), random.randint(120, 220)) 
            cv2.ellipse(img, (center_x, center_y), axes, random.randint(0, 360), 0, 360, color, -1)
            img = cv2.GaussianBlur(img, (7, 7), 0)
        return img

    def apply_blur_and_noise(self, img):
        if random.random() < 0.3:
            size = random.randint(5, 15)
            kernel = np.zeros((size, size))
            if random.random() < 0.5:
                kernel[int((size-1)/2), :] = np.ones(size)
            else:
                kernel[:, int((size-1)/2)] = np.ones(size)
            kernel = kernel / size
            blurred = cv2.filter2D(img, -1, kernel)
        else:
            blurred = cv2.GaussianBlur(img, (random.choice([3, 5]), random.choice([3, 5])), 0)
        row, col, ch = blurred.shape
        gauss = np.random.normal(0, random.uniform(15, 35), (row, col, ch)).reshape(row, col, ch)
        noisy = blurred + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def apply_jpeg_compression(self, img):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(20, 70)]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(encimg, 1)

class EnhancementDataset(Dataset):
    def __init__(self, clean_scans, target_size=(256, 256), epoch_size=1000, is_train=True):
        self.clean_scans = clean_scans
        self.target_size = target_size
        self.is_train = is_train
        self.epoch_size = epoch_size if is_train else len(clean_scans)
        self.degrader = DocumentDegradation()
        
    def __len__(self):
        return self.epoch_size
        
    def __getitem__(self, idx):
        if self.is_train:
            scan_path = random.choice(self.clean_scans)
        else:
            # Freeze state for validation
            self.r_state = random.getstate()
            self.np_state = np.random.get_state()
            random.seed(idx + 42)
            np.random.seed(idx + 42)
            scan_path = self.clean_scans[idx]
            
        scan_img = cv2.imread(scan_path)
        assert scan_img is not None, f"❌ Error: Cannot read image at {scan_path}"
        scan_img = cv2.cvtColor(scan_img, cv2.COLOR_BGR2RGB)
        
        h, w = scan_img.shape[:2]
        new_w = 800
        new_h = int(h * (new_w / w))
        scan_img = cv2.resize(scan_img, (new_w, new_h))
        
        th, tw = self.target_size
        x = random.randint(0, max(0, new_w - tw))
        y = random.randint(0, max(0, new_h - th))
        base_img = scan_img[y:y+th, x:x+tw]
        
        if base_img.shape[:2] != self.target_size:
            base_img = cv2.resize(base_img, self.target_size)
            
        target_img = base_img.copy()
        
        img = self.degrader.apply_brightness_contrast_color(base_img)
        img = self.degrader.apply_illumination_and_shadows(img)
        img = self.degrader.apply_blur_and_noise(img)
        img = self.degrader.apply_jpeg_compression(img)
        
        input_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        target_tensor = torch.from_numpy(target_img).float().permute(2, 0, 1) / 255.0
        
        if not self.is_train:
            random.setstate(self.r_state)
            np.random.set_state(self.np_state)
            
        return input_tensor, target_tensor
    

class CornerDataset(Dataset):
    def __init__(self, clean_scans, backgrounds, target_size=(256, 256), epoch_size=1000, is_train=True):
        self.clean_scans = clean_scans
        self.backgrounds = backgrounds
        self.target_size = target_size
        self.is_train = is_train
        self.epoch_size = epoch_size if is_train else len(clean_scans)
        self.degrader = DocumentDegradation()
        
    def __len__(self):
        return self.epoch_size
        
    def __getitem__(self, idx):
        if self.is_train:
            scan_path = random.choice(self.clean_scans)
            bg_path = random.choice(self.backgrounds)
        else:
            self.r_state = random.getstate()
            self.np_state = np.random.get_state()
            random.seed(idx + 100)
            np.random.seed(idx + 100)
            scan_path = self.clean_scans[idx]
            bg_path = self.backgrounds[idx % len(self.backgrounds)]
            
        scan_img = cv2.imread(scan_path)
        assert scan_img is not None, f"❌ Error: Cannot read scan at {scan_path}"
        bg_img = cv2.imread(bg_path)
        assert bg_img is not None, f"❌ Error: Cannot read background at {bg_path}"
        
        scan_img = cv2.resize(scan_img, (512, 512))
        bg_img = cv2.resize(bg_img, (512, 512))
        
        scan_img = cv2.cvtColor(scan_img, cv2.COLOR_BGR2RGB) 
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        
        composited, corners, _ = self.degrader.apply_perspective_warp(scan_img, bg_img)
        img = self.degrader.apply_scaling(composited)
        img = self.degrader.apply_brightness_contrast_color(img)
        img = self.degrader.apply_illumination_and_shadows(img)
        img = self.degrader.apply_blur_and_noise(img)
        img = self.degrader.apply_jpeg_compression(img)
        
        orig_h, orig_w = img.shape[:2]
        img_resized = cv2.resize(img, self.target_size)
        
        corners_normalized = corners.copy()
        corners_normalized[:, 0] /= orig_w
        corners_normalized[:, 1] /= orig_h
        
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
        corners_tensor = torch.from_numpy(corners_normalized).float()
        
        if not self.is_train:
            random.setstate(self.r_state)
            np.random.set_state(self.np_state)
            
        return img_tensor, corners_tensor