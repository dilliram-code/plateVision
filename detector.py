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