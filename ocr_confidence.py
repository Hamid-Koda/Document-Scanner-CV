import os
import easyocr
import glob

def calculate_confidence():
    print("Loading EasyOCR model...")
    reader = easyocr.Reader(['en']) 
    
    raw_images = glob.glob("real_test_photos/*_raw.jpg")
    
    if not raw_images:
        print("Error: No test images found in 'real_test_photos/'")
        return

    print("\n" + "="*70)
    print(f"{'Document':<15} | {'Version':<20} | {'Confidence Score (%)':<20}")
    print("="*70)

    for rect_path in raw_images:
        base_name = os.path.basename(rect_path).replace("_raw.jpg", "")
        camscanner_path = rect_path.replace("_raw.jpg", "_camscanner.jpg")
        
        our_model_path = f"evaluation_results/{base_name}_enhanced.jpg"

        image_paths = {
            "Raw Rectified": rect_path,
            "CamScanner": camscanner_path if os.path.exists(camscanner_path) else None,
            "Our U-Net": our_model_path if os.path.exists(our_model_path) else None
        }

        for version, path in image_paths.items():
            if path is None:
                print(f"{base_name:<15} | {version:<20} | {'File Not Found':<20}")
                continue

            results = reader.readtext(path, detail=1)
            
            if not results:
                print(f"{base_name:<15} | {version:<20} | {'0.00 (No text found)':<20}")
                continue

            confidences = [res[2] for res in results]
            
            avg_confidence = (sum(confidences) / len(confidences)) * 100
            
            print(f"{base_name:<15} | {version:<20} | {avg_confidence:>18.2f}%")
        
        print("-" * 70)

if __name__ == '__main__':
    calculate_confidence()