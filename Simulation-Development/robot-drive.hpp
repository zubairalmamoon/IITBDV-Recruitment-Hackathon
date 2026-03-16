# To install the necessary libraries in Colab
!pip install ultralytics gdown

import os
import gdown
from ultralytics import YOLO
from google.colab.patches import cv2_imshow
import cv2

# Download the model from your link
file_id = '1uLn7PVpFewPaYy-FF355hTp6Tm2hNuQB'
model_url = f'https://drive.google.com/uc?id={file_id}'
model_path = 'YOLOv11s-Carmaker.pt'

if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

# Load the model
model = YOLO(model_path)
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from ultralytics.utils.plotting import colors

# --- Camera & Object Parameters ---
FOCAL_LENGTH = 1000
REAL_HEIGHT_CM = 30
# ----------------------------------

image_input = 'image.png'
results = model.predict(source=image_input, conf=0.25)

for r in results:
    # 1. Plot the image with NO confidence scores and NO class labels
    annotated_image = r.plot(line_width=2, conf=False, labels=False)

    # 2. Loop through every detected cone to calculate depth
    for box in r.boxes:
        # Get bounding box coordinates [x1, y1, x2, y2]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        # Get the color for this specific class
        cls_idx = int(box.cls[0])
        box_color = colors(cls_idx, True)

        # Calculate apparent height in pixels
        pixel_height = y2 - y1

        # Prevent division by zero
        if pixel_height > 0:
            # Apply the distance formula
            distance_cm = (FOCAL_LENGTH * REAL_HEIGHT_CM) / pixel_height
            distance_m = distance_cm / 100  # Convert to meters

            # 3. Format the depth text
            depth_text = f"{distance_m:.2f}m"

            # --- UPDATED POSITIONING ---
            # Position the text ABOVE the box
            # x1: aligned with the left edge of the box
            # y1 - 8: placed 8 pixels above the top edge of the box
            text_x = int(x1)
            text_y = int(y1) - 8

            # Overlay the depth text with the SAME color as the bounding box
            cv2.putText(annotated_image, depth_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2, cv2.LINE_AA)

    # 4. Show the final clean image
    cv2_imshow(annotated_image)
