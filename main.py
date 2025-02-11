import math
import time
import numpy as np
import cv2
import cvzone
from ultralytics import YOLO, solutions
from util import customReadPlate, save_to_database, save_json  # your custom functions
from sort import Sort
import torch
from datetime import datetime
from teleBot import sendPlate
import threading  # Import threading

# Video display dimensions
display_width = 1280  # Adjust as needed
display_height = 720  # Adjust as needed

# Video source (replace with your webcam if needed)
cap = cv2.VideoCapture("licensevid.mp4")

# Load the detection model
model = YOLO('best.pt')
names = model.names

# Initialize tracker and frame timers
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
prev_frame_time = 0
new_frame_time = 0

# Define the region limits for processing license plates (if needed)
limits = [200, 600, 1200, 600]

# Dictionary to store the best license plate text and score for each vehicle id
# Format: {vehicle_id: (plate_text, plate_score)}
best_plate_for_vehicle = {}

# Frame counter and OCR interval: only run OCR every "ocr_interval" frames.
frame_count = 0
ocr_interval = 3  # Adjust this value (e.g., every 3 frames)

# All plates detected (if needed)
license_plates = set()

# Timer for saving JSON
startTime = datetime.now()

while True:
    frame_count += 1
    new_frame_time = time.time()
    currentTime = datetime.now()

    success, img = cap.read()
    if not success:
        break

    # Resize frame for display
    img = cv2.resize(img, (display_width, display_height))

    # Run the model on the frame (stream=True yields results frame-by-frame)
    results = model(img, stream=True)

    # Prepare an array for detections (each row: [x1, y1, x2, y2, confidence])
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get bounding box coordinates and confidence
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Only add detections above a confidence threshold
            if conf > 0.4:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Update tracker with current detections
    resultsTracker = tracker.update(detections)

    # Process each tracked object
    for result in resultsTracker:
        x1, y1, x2, y2, track_id = result
        x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
        w, h = x2 - x1, y2 - y1

        # Crop the license plate region (adjust as needed)
        license_plate_crop = img[y1 - 3:y2 + 3, x1 - 3:x2 + 3, :]

        # Preprocess the cropped image
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        # Step 2: Apply noise reduction (Bilateral filter preserves edges well)
        blurred = cv2.bilateralFilter(gray, d=11, sigmaColor=17, sigmaSpace=17)
        # Optional: Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        # Step 3: Thresholding (Otsu's thresholding)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Step 4: (Optional) Morphological operation to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Draw a bounding box with a fancy corner effect
        cvzone.cornerRect(img, (x1, y1, w, h), colorR=(255, 255, 0), l=9, rt=2)

        # (Optional) Check if the object's center is within a region:
        cx, cy = x1 + w // 2, y1 + h // 2
        # if not (limits[0] < cx < limits[2] and limits[1] < cy < limits[3]):
        #     continue

        # Run OCR every 'ocr_interval' frames
        if frame_count % ocr_interval == 0:
            plate_text, plate_score = customReadPlate(thresh)
            if plate_text:
                # Update the best plate for this vehicle only if the new score is higher
                if (track_id not in best_plate_for_vehicle) or (plate_score > best_plate_for_vehicle[track_id][1]):
                    best_plate_for_vehicle[track_id] = (plate_text, plate_score)
                    print(f'Updated vehicle {track_id}: {plate_text} (Score: {plate_score})')


                    # Uncomment both lines once telegram bot is setup
                    # message = "🚘 Detected License Plates:\n" + plate_text
                    # threading.Thread(target=sendPlate, args=(message,)).start()

                    # Also add to the set of all unique plates detected
                    license_plates.add(plate_text)
                else:
                    print(f'Ignored vehicle {track_id} new detection: {plate_text} (Score: {plate_score})')

        # Display the stored best plate (if available)
        if track_id in best_plate_for_vehicle:
            best_text, best_score = best_plate_for_vehicle[track_id]
            cvzone.putTextRect(img, f'{best_text} {best_score}', (max(0, x1), max(35, y1 - 10)), scale=1, thickness=1)

    # Check if it's time to send the Telegram message (every 20 seconds)
    if (currentTime - startTime).seconds >= 20:
        endTime = currentTime
        # Extract only the plate numbers from the dictionary values
        jsonPlates = [plate_text for plate_text, _ in best_plate_for_vehicle.values()]

        # Format the message and send it in a separate thread
        if jsonPlates:
            save_json(jsonPlates, startTime, endTime)



        startTime = currentTime
        best_plate_for_vehicle.clear()
        license_plates.clear()

    # Calculate and update the frame rate (FPS)
    fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) != 0 else 0
    prev_frame_time = new_frame_time

    print(f'Unique plates detected: {license_plates}')

    # Show the resulting image
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
