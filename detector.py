import re 
import time 
from pathlib import Path 

import cv2 
import numpy as np 
import torch 
from ultralytics import YOLO  

# configuration 
MODELS_DIR = Path(__file__).parent / "models"

VEHICLE_MODEL_PATH = "yolov8n.pt"

VEHICLE_CLASS_IDS ={2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

PLATE_MODEL_PATH = MODELS_DIR / "license_plate_detector.pt"

HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"

MIN_PLATE_TEXT_LENGTH = 4

# clean plate text and return the uppercase
def clean_plate_text(raw_text):
  cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_text)
  
  return cleaned.upper()

# ANPR detector 
class ANPRDetector:
  def __init__(self):
    
    # pick gpu if available, otherwise fall back to cpu
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # vehicle detector
    self.vehicle_model = YOLO(VEHICLE_MODEL_PATH)
    
    # plate localizer
    self.use_yolo_plate_model = PLATE_MODEL_PATH.exists()
    
    if self.use_yolo_plate_model:
      self.plate_model = YOLO(str(PLATE_MODEL_PATH))
      print("Loaded custom YOLOv8 plate model.")
    else:
      self.plate_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
      print("No custom model found at" f"{PLATE_MODEL_PATH}.")
    
    # ocr reader 
    import easyocr 
    print("Loading easyocr reader...")
    self.ocr_reader = easyocr.Reader(["en"]) 
    gpu = (self.device=="cuda")

  # detect vehicles
  def _detect_vehicles(self, frame):
    
    # run YOLOv8 and return boxes for vehicle-class objects only
    results = self.vehicle_model(frame, verbose=False)[0]
    vehicle_boxes = []
    for box in results.boxes:
      cls_id = int(box.cls[0])
      if cls_id in VEHICLE_CLASS_IDS:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        vehicle_boxes.append(
          {
            "bbox": (x1, y1, x2, y2),
            "label": VEHICLE_CLASS_IDS[cls_id],
            "conf": float(box.conf[0]),
          }
        )
    return vehicle_boxes
  
  # detect plates yolo
  def _detect_plates_yolo(self, crop):
    # run the custom YOLOv8 plate model on a vehicle crop
    results = self.plate_model(crop, verbose=False)[0]
    plates = []
    for box in results.boxes:
      x1, y1, x2, y2 = map(int, box.xyxy[0])
      plates.append(
        {
          "bbox": (x1, y1, x2, y2),
          "conf": float(box.conf[0])
        }
      )
    return plates
  
  # OCR the cropped plate image -> (text, confidence).
  def _read_plate_text(self, plate_crop):
    if plate_crop.size == 0:
      return None, 0.0
    h, w = plate_crop.shape[:2]
    if w < 200:
      scale = 200 / w
      plate_crop = cv2.resize(plate_crop, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    ocr_results = self.ocr_reader.readtext(gray)
    if not ocr_results:
      return None, 0.0

    texts, confs = zip(*[(r[1], r[2]) for r in ocr_results])
    combined_text = clean_plate_text("".join(texts))
    avg_conf = float(np.mean(confs))

    if len(combined_text) < MIN_PLATE_TEXT_LENGTH:
      return None, 0.0

    return combined_text, avg_conf
  
  def process_frame(self, frame):
    annotated = frame.copy()
    detections = []

    vehicles = self._detect_vehicles(frame)

    for vehicle in vehicles:
      x1, y1, x2, y2 = vehicle["bbox"]
      # Draw the vehicle bounding box (subtle, thin line).
      cv2.rectangle(annotated, (x1, y1), (x2, y2), (219, 152, 52), 2)
      cv2.putText(
                annotated, vehicle["label"], (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (219, 152, 52), 2,
            )

      vehicle_crop = frame[y1:y2, x1:x2]
      if vehicle_crop.size == 0:
        continue

      # Stage 2: find plate(s) within this vehicle crop.
      if self.use_yolo_plate_model:
        plates = self._detect_plates_yolo(vehicle_crop)
      else:
        plates = self._detect_plates_haar(vehicle_crop)

      for plate in plates:
        px1, py1, px2, py2 = plate["bbox"]
        plate_crop = vehicle_crop[py1:py2, px1:px2]

        # Stage 3: OCR the plate crop.
        text, ocr_conf = self._read_plate_text(plate_crop)
        if text is None:
          continue

        # Translate plate box coords back to the FULL frame (for drawing).
        abs_x1, abs_y1 = x1 + px1, y1 + py1
        abs_x2, abs_y2 = x1 + px2, y1 + py2
        cv2.rectangle(annotated, (abs_x1, abs_y1), (abs_x2, abs_y2), (63, 209, 51), 2)
        cv2.putText(
                      annotated, text, (abs_x1, max(abs_y1 - 10, 0)),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (63, 209, 51), 2,
                  )
        overall_conf = (plate["conf"] + ocr_conf) / 2
        detections.append(
                      {
                          "plate_number": text,
                          "confidence": round(overall_conf, 3),
                          "vehicle_type": vehicle["label"],
                          "plate_crop": plate_crop,
                      }
                  )

    return annotated, detections
