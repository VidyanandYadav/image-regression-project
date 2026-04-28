"""
Run this ONCE before running main.py:
    python generate_sample_images.py
"""

import os
import cv2
import numpy as np

os.makedirs("images", exist_ok=True)

np.random.seed(42)

for i in range(1, 31):
    # Random 200x200 image with varying brightness
    brightness_base = np.random.randint(50, 220)
    img = np.full((200, 200, 3), brightness_base, dtype=np.uint8)

    # Add some random shapes to create edges
    num_shapes = np.random.randint(3, 8)
    for _ in range(num_shapes):
        x1, y1 = np.random.randint(0, 150, 2)
        x2, y2 = x1 + np.random.randint(20, 60), y1 + np.random.randint(20, 60)
        color = tuple(int(c) for c in np.random.randint(0, 255, 3))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

    # Add random circles
    for _ in range(np.random.randint(2, 5)):
        cx, cy = np.random.randint(20, 180, 2)
        r = np.random.randint(10, 40)
        color = tuple(int(c) for c in np.random.randint(0, 255, 3))
        cv2.circle(img, (int(cx), int(cy)), r, color, -1)

    filename = f"images/sample_{i:02d}.jpg"
    cv2.imwrite(filename, img)
    print(f"[CREATED] {filename}")

print("\n[DONE] 30 sample images created in 'images/' folder.")
print("       Now run: python main.py")
