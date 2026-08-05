import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from model import HeatmapCornerRegressor

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")

model = HeatmapCornerRegressor().to(device)
model.load_state_dict(torch.load("weights/heatmap_corner_best.pth",
                                 map_location=device))
model.eval()

img_path = "test_image.jpg"

img_bgr = cv2.imread(img_path)

if img_bgr is None:
    raise Exception("Image not found.")

orig_h, orig_w = img_bgr.shape[:2]

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

img_input = cv2.resize(img_rgb, (256,256))

tensor = (
    torch.from_numpy(img_input)
    .float()
    .permute(2,0,1)
    .unsqueeze(0)
    /255.
).to(device)

with torch.no_grad():

    logits = model(tensor)

heatmaps = torch.sigmoid(logits).squeeze().cpu().numpy()

result = img_rgb.copy()

colors = [
    (255,0,0),
    (0,255,0),
    (0,0,255),
    (255,255,0)
]

corners=[]

print()

print("Detected Corners")

print("--------------------------")

for i in range(4):

    hm = heatmaps[i]

    confidence = hm.max()

    y,x = np.unravel_index(np.argmax(hm),hm.shape)

    x = int(x*orig_w/256)

    y = int(y*orig_h/256)

    corners.append((x,y))

    print(f"Corner {i+1}")

    print(f"Location : ({x},{y})")

    print(f"Confidence : {confidence:.4f}")

    print()

    cv2.circle(result,(x,y),18,colors[i],-1)

    cv2.putText(result,
                str(i+1),
                (x+18,y-18),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                colors[i],
                3)

#####################################################

fig = plt.figure(figsize=(18,10))

plt.subplot(2,3,1)
plt.imshow(img_rgb)
plt.title("Input Image")
plt.axis("off")

for i in range(4):

    plt.subplot(2,3,i+2)

    plt.imshow(heatmaps[i],cmap="hot")

    plt.title(f"Heatmap {i+1}")

    plt.colorbar()

    plt.axis("off")

plt.figure(figsize=(8,8))

plt.imshow(result)

plt.title("Detected Corners")

plt.axis("off")

plt.show()