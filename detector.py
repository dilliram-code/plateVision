import re 
import time 
from pathlib import Path 

import cv2 
import numpy as np 
import torch 
from ultralytics import YOLO  

# configuration 
MODELS_DIR = Path(__file__).parent / "models"

VEHICLE_CLASS_IDS ={2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

PLATE_MODEL_PATH = MODELS_DIR / "license_plate_detector.pt"

HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"

MIN_PLATE_TEXT_LENGTH = 4

# clean plate text and return the uppercase
def clean_plate_text(raw_text):
  cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_text)
  
  return cleaned.upper()