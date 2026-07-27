import cv2
import numpy as np
import random

class DocumentDegradation:
    def __init__(self):
        pass

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