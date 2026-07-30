import cv2
import glob

def optimize_images(folder_path, max_size=800):
    images = glob.glob(f"{folder_path}/*.*")
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"✅ Optimized: {img_path}")

if __name__ == '__main__':
    print("Starting optimization...")
    optimize_images("clean_scans")
    optimize_images("backgrounds")
    print("Done! All images are now lightweight.")