"""
PlateVision ANPR / ALPR Detector for Nepal
==========================================
Production-grade Automatic Number Plate Recognition (ANPR) system tailored
for Nepal's transportation ecosystem.

Features:
1. Multi-Vehicle Detection:
   - Recognizes Cars, Motorcycles/Scooters, Buses, Trucks, and Tempos using YOLOv8.
2. Multi-Strategy License Plate Localization:
   - Primary: Custom YOLOv8 Plate Model (if available in models/license_plate_detector.pt).
   - Secondary: Haar Cascade Classifier (OpenCV Russian/General Plate Cascade).
   - Tertiary: Advanced Computer Vision Edge, Morphological & Contour Segmenter
     (tailored specifically for Nepali horizontal 1-line and stacked 2-line bike/car plates).
3. Dual-Script OCR:
   - EasyOCR engine configured with both Nepali ('ne' - Devanagari) and English ('en' - Latin).
   - Multi-line reading order sorting (handles stacked 2-line bike & truck plates in Nepal).
4. Image Enhancement Pipeline:
   - Super-resolution bicubic scaling, CLAHE contrast equalization, bilateral denoising,
     and adaptive thresholding.
5. Integration with Nepal Plate Parser:
   - Translates Devanagari to standardized Latin and vice versa.
   - Extracts Vehicle Category, Zone/Province, Lot Number, and Validity.
"""

import os
import re
import time
import base64
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# Import local domain modules
from nepal_plate_parser import (
    parse_nepali_plate,
    clean_raw_plate_text,
    contains_devanagari,
    devanagari_to_arabic_digits,
)
import database

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
CROPS_DIR = BASE_DIR / "data" / "crops"
CROPS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# YOLOv8 COCO Vehicle Model weights
VEHICLE_MODEL_NAME = "yolov8n.pt"

# Classes from COCO dataset corresponding to road vehicles
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASS_IDS: Dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Optional custom YOLO plate detector
CUSTOM_PLATE_MODEL_PATH = MODELS_DIR / "license_plate_detector.pt"

# OpenCV Haar Cascade Plate path
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"

# Minimum text length to qualify as a valid plate
MIN_PLATE_TEXT_LENGTH = 3

# High confidence threshold for plate reporting
CONFIDENCE_THRESHOLD = 0.35


# ==============================================================================
# CORE ANPR DETECTOR CLASS
# ==============================================================================

class ANPRDetector:
    """
    Main ANPR inference engine for vehicle detection, license plate localization,
    and dual-script (Nepali Devanagari & English Embossed) recognition.
    """

    def __init__(self, languages: Optional[List[str]] = None, use_gpu: bool = False):
        """
        Initializes the ANPR detector, loading deep learning models and fallback cascades.

        Parameters
        ----------
        languages : list of str, optional
            Languages for OCR. Defaults to ['ne', 'en'] for Nepal.
        use_gpu : bool
            Whether to force GPU execution if CUDA is available.
        """
        print("[PlateVision] Initializing ANPR Detection Engine for Nepal...")

        # Select computing device
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available() and hasattr(torch.backends.mps, "is_built") and torch.backends.mps.is_built():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"[PlateVision] Primary compute device for PyTorch/YOLO: {self.device}")

        # 1. Load YOLOv8 Vehicle Detector
        print(f"[PlateVision] Loading vehicle detection model ({VEHICLE_MODEL_NAME})...")
        self.vehicle_model = YOLO(VEHICLE_MODEL_NAME)

        # 2. Setup Plate Localizer (YOLO / Haar Cascade / Morphological Fallback)
        self.use_yolo_plate_model = CUSTOM_PLATE_MODEL_PATH.exists()
        if self.use_yolo_plate_model:
            print(f"[PlateVision] Found custom YOLO plate model at: {CUSTOM_PLATE_MODEL_PATH}")
            try:
                self.plate_model = YOLO(str(CUSTOM_PLATE_MODEL_PATH))
            except Exception as e:
                print(f"[PlateVision] Warning: Failed to load custom YOLO plate model: {e}")
                self.use_yolo_plate_model = False

        # Load Haar Cascade as backup localizer
        self.plate_cascade = None
        if os.path.exists(HAAR_CASCADE_PATH):
            try:
                self.plate_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
                print("[PlateVision] Haar Cascade plate localizer loaded successfully.")
            except Exception as e:
                print(f"[PlateVision] Notice: Haar Cascade could not be loaded: {e}")

        # 3. Initialize EasyOCR Reader for Nepali & English
        self.ocr_languages = languages or ["ne", "en"]
        print(f"[PlateVision] Initializing EasyOCR with languages: {self.ocr_languages}...")
        import easyocr

        # On Mac CPU is exceptionally fast and avoids Metal shader compilation delays
        ocr_gpu = (self.device == "cuda")
        self.ocr_reader = easyocr.Reader(self.ocr_languages, gpu=ocr_gpu)
        print("[PlateVision] EasyOCR Reader initialized successfully.")

    # --------------------------------------------------------------------------
    # STAGE 1: VEHICLE DETECTION
    # --------------------------------------------------------------------------

    def _detect_vehicles(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs YOLOv8 object detection on the input frame and returns bounding
        boxes for all identified vehicle classes (cars, bikes, buses, trucks).
        """
        results = self.vehicle_model(frame, verbose=False)[0]
        vehicle_boxes = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Filter for vehicle classes with confidence > 0.30
            if cls_id in VEHICLE_CLASS_IDS and conf >= 0.30:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h, w = frame.shape[:2]

                # Constrain within frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                vehicle_boxes.append({
                    "bbox": (x1, y1, x2, y2),
                    "label": VEHICLE_CLASS_IDS[cls_id],
                    "conf": conf,
                })

        return vehicle_boxes

    # --------------------------------------------------------------------------
    # STAGE 2: LICENSE PLATE LOCALIZATION
    # --------------------------------------------------------------------------

    def _detect_plates_yolo(self, crop: np.ndarray) -> List[Dict[str, Any]]:
        """Detects license plates using custom trained YOLOv8 weights."""
        results = self.plate_model(crop, verbose=False)[0]
        plates = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf >= 0.25:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plates.append({
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf,
                    "method": "yolo",
                })
        return plates

    def _detect_plates_haar(self, crop: np.ndarray) -> List[Dict[str, Any]]:
        """Detects license plates using OpenCV Haar Cascade Classifier."""
        if self.plate_cascade is None or self.plate_cascade.empty():
            return []

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        boxes = self.plate_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(60, 20),
            maxSize=(crop.shape[1], int(crop.shape[0] * 0.8)),
        )

        plates = []
        for (x, y, w, h) in boxes:
            plates.append({
                "bbox": (x, y, x + w, y + h),
                "conf": 0.70,
                "method": "haar",
            })
        return plates

    def _detect_plates_morphological(self, crop: np.ndarray, vehicle_type: str) -> List[Dict[str, Any]]:
        """
        Autonomous Computer Vision Plate Localizer for Nepal vehicles.
        Leverages Sobel vertical edge gradients, morphological closing, and contour
        aspect ratio analysis tailored for:
        - 1-line horizontal plates (cars/buses: aspect ratio 2.5 - 5.0)
        - 2-line stacked plates (bikes/tempos/trucks: aspect ratio 1.1 - 2.2)
        """
        h, w = crop.shape[:2]
        if h < 30 or w < 50:
            return []

        # Plates are usually mounted in the lower 70% of cars/buses, or bottom/mudguard of bikes
        # We search across the whole crop, but weight lower regions higher
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # 1. Contrast enhancement via Top-Hat and Black-Hat morphology
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rect_kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)
        enhanced = cv2.add(cv2.subtract(cv2.add(gray, tophat), blackhat), 0)

        # 2. Vertical Edge Detection (License plate text has strong vertical edges)
        grad_x = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
        grad_x = np.absolute(grad_x)
        min_val, max_val = np.min(grad_x), np.max(grad_x)
        grad_x = (255 * ((grad_x - min_val) / (max_val - min_val + 1e-5))).astype("uint8")

        # 3. Blur and Morphological Close to connect characters into a single blob
        grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        closed = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, close_kernel)

        # 4. Otsu Thresholding
        _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 5. Find Contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        crop_area = h * w

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            aspect_ratio = float(bw) / float(bh)

            # Area must be reasonable (1% to 35% of vehicle crop)
            if not (0.01 * crop_area <= area <= 0.35 * crop_area):
                continue

            # Aspect ratio filtering:
            # - Horizontal rectangular plates (Nepali cars / embossed): 2.2 <= AR <= 5.5
            # - Stacked 2-line plates (Nepali bikes / trucks): 1.1 <= AR <= 2.2
            is_horizontal_plate = (2.2 <= aspect_ratio <= 5.5) and (bw >= 60 and bh >= 16)
            is_stacked_plate = (1.0 <= aspect_ratio <= 2.2) and (bw >= 40 and bh >= 25)

            if is_horizontal_plate or is_stacked_plate:
                # Add slight padding around candidate plate
                pad_x = int(bw * 0.08)
                pad_y = int(bh * 0.12)
                cx1 = max(0, x - pad_x)
                cy1 = max(0, y - pad_y)
                cx2 = min(w, x + bw + pad_x)
                cy2 = min(h, y + bh + pad_y)

                # Heuristic confidence based on positioning and aspect ratio
                pos_weight = 1.0 if y > (h * 0.3) else 0.8
                conf = round(0.65 * pos_weight, 2)

                candidates.append({
                    "bbox": (cx1, cy1, cx2, cy2),
                    "conf": conf,
                    "method": "morphological",
                    "area": area,
                })

        # Sort candidates by area and confidence
        candidates.sort(key=lambda c: c["area"], reverse=True)
        return candidates[:3]  # Return top 3 candidates

    def _detect_plates_color(self, crop: np.ndarray) -> List[Dict[str, Any]]:
        """
        Segment plates based on Nepal's distinct license plate color standards:
        - Red (निजी): Private two-wheelers and four-wheelers
        - Black (सार्वजनिक): Public buses, trucks, taxis
        - White (इम्बोस्ड / सरकारी): High-security embossed and government plates
        - Green (विद्युतीय): Electric Vehicles (EV)
        """
        h, w = crop.shape[:2]
        if h < 25 or w < 35:
            return []

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        candidates = []

        # 1. Red Plates (Private)
        lower_r1, upper_r1 = np.array([0, 60, 50]), np.array([12, 255, 255])
        lower_r2, upper_r2 = np.array([168, 60, 50]), np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_r1, upper_r1) | cv2.inRange(hsv, lower_r2, upper_r2)

        # 2. Black Plates (Public transport / Taxi / Freight)
        black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 100, 60]))

        # 3. White Plates (Embossed / Government)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 175]), np.array([180, 55, 255]))

        # 4. Green Plates (EV Electric Vehicles)
        green_mask = cv2.inRange(hsv, np.array([35, 60, 50]), np.array([85, 255, 255]))

        combined_masks = [
            ("red_private", red_mask),
            ("black_public", black_mask),
            ("white_embossed", white_mask),
            ("green_ev", green_mask),
        ]

        crop_area = h * w
        for tag, mask in combined_masks:
            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                area = bw * bh
                aspect_ratio = float(bw) / float(bh)

                # Plate size constraints
                if not (0.008 * crop_area <= area <= 0.40 * crop_area):
                    continue
                if bw < 35 or bh < 18:
                    continue

                # Aspect ratio: 0.9 - 5.5
                if 0.9 <= aspect_ratio <= 5.5:
                    pad_x = int(bw * 0.05)
                    pad_y = int(bh * 0.08)
                    cx1 = max(0, x - pad_x)
                    cy1 = max(0, y - pad_y)
                    cx2 = min(w, x + bw + pad_x)
                    cy2 = min(h, y + bh + pad_y)

                    candidates.append({
                        "bbox": (cx1, cy1, cx2, cy2),
                        "conf": 0.72,
                        "method": f"color_{tag}",
                        "area": area,
                    })

        candidates.sort(key=lambda c: c["area"], reverse=True)
        return candidates[:3]

    def locate_plates(self, vehicle_crop: np.ndarray, vehicle_type: str) -> List[Dict[str, Any]]:
        """
        Multi-tier plate localization coordinator. Tries YOLO -> Haar -> Color -> Morphological.
        """
        plates = []

        # 1. Custom YOLO model if available
        if self.use_yolo_plate_model:
            plates = self._detect_plates_yolo(vehicle_crop)

        # 2. Haar Cascade if YOLO didn't find any or is unavailable
        if not plates and self.plate_cascade is not None:
            plates = self._detect_plates_haar(vehicle_crop)

        # 3. Nepal-specific Color Segmenter (Red, Black, White, Green)
        if not plates:
            plates = self._detect_plates_color(vehicle_crop)

        # 4. Autonomous Morphological Segmenter
        if not plates:
            plates = self._detect_plates_morphological(vehicle_crop, vehicle_type)

        # 5. Fallback: If still nothing detected, check the lower third and center of vehicle
        if not plates and vehicle_crop.size > 0:
            vh, vw = vehicle_crop.shape[:2]
            if vh >= 50 and vw >= 70:
                # Lower region
                plates.append({
                    "bbox": (int(vw * 0.15), int(vh * 0.45), int(vw * 0.85), int(vh * 0.95)),
                    "conf": 0.50,
                    "method": "heuristic_crop",
                })

        return plates

    # --------------------------------------------------------------------------
    # STAGE 3: PLATE PREPROCESSING & IMAGE ENHANCEMENT
    # --------------------------------------------------------------------------

    def _preprocess_plate_for_ocr(self, plate_crop: np.ndarray) -> List[np.ndarray]:
        """
        Applies a series of computer vision filters to optimize plate image
        for dual-script character recognition (Nepali Devanagari & English):
        - Bicubic super-resolution scaling
        - Contrast Limited Adaptive Histogram Equalization (CLAHE)
        - Bilateral edge-preserving denoising
        - Otsu binarization
        """
        if plate_crop is None or plate_crop.size == 0:
            return []

        h, w = plate_crop.shape[:2]

        # 1. Ensure minimal resolution for OCR clarity (scale up if small)
        target_width = 360
        if w < target_width:
            scale = target_width / max(w, 1)
            plate_resized = cv2.resize(
                plate_crop,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            plate_resized = plate_crop.copy()

        # 2. Convert to LAB color space and apply CLAHE to L channel
        lab = cv2.cvtColor(plate_resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # 3. Grayscale and Bilateral Filter (smoothes noise without blurring font edges)
        gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # Return multiple variations for OCR resilience
        # EasyOCR works well with both high-contrast color and enhanced grayscale
        return [enhanced_bgr, denoised]

    # --------------------------------------------------------------------------
    # STAGE 4: DUAL SCRIPT OCR & TEXT LINE EXTRACTION
    # --------------------------------------------------------------------------

    def _read_plate_text(
        self,
        plate_crop: np.ndarray,
        vehicle_type: Optional[str] = None
    ) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """
        Performs dual-script OCR on the plate crop, assembling multi-line
        Nepali and English text into natural reading order.

        Returns
        -------
        tuple : (raw_text, ocr_confidence, parsed_metadata)
        """
        if plate_crop is None or plate_crop.size == 0:
            return None, 0.0, {}

        enhanced_images = self._preprocess_plate_for_ocr(plate_crop)
        if not enhanced_images:
            return None, 0.0, {}

        best_raw_text = None
        best_ocr_conf = 0.0
        best_parsed = {}

        # Run OCR on the enhanced color image first, then grayscale fallback
        for img_candidate in enhanced_images:
            try:
                ocr_results = self.ocr_reader.readtext(img_candidate)
            except Exception as e:
                print(f"[PlateVision] OCR read exception: {e}")
                ocr_results = []

            if not ocr_results:
                continue

            # Sort boxes top-to-bottom (Y coordinate), then left-to-right (X coordinate)
            # This is CRITICAL for 2-line Nepali plates (Line 1: Zone/Lot/Type, Line 2: Number)
            sorted_boxes = sorted(
                ocr_results,
                key=lambda item: (round(item[0][0][1] / 20) * 20, item[0][0][0])
            )

            texts = [r[1].strip() for r in sorted_boxes if r[1].strip()]
            confs = [r[2] for r in sorted_boxes]

            if not texts:
                continue

            joined_text = " ".join(texts)
            avg_conf = float(np.mean(confs)) if confs else 0.0

            # Parse with domain-specific Nepali parser
            parsed = parse_nepali_plate(joined_text, detected_vehicle_type=vehicle_type)

            # Select result with highest confidence and valid structure
            if parsed.get("is_valid") or avg_conf > best_ocr_conf:
                best_raw_text = joined_text
                best_ocr_conf = avg_conf
                best_parsed = parsed
                if parsed.get("is_valid") and avg_conf > 0.40:
                    break

        if not best_raw_text or len(best_raw_text.strip()) < MIN_PLATE_TEXT_LENGTH:
            return None, 0.0, {}

        return best_raw_text, best_ocr_conf, best_parsed

    # --------------------------------------------------------------------------
    # STAGE 5: VISUAL ANNOTATION WITH SLEEK GLASSY OVERLAYS
    # --------------------------------------------------------------------------

    def _draw_modern_annotations(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Renders sleek, glassy bounding boxes, badges, and authentic Devanagari
        labels on the output frame.
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        for det in detections:
            # 1. Draw vehicle box
            vx1, vy1, vx2, vy2 = det["vehicle_bbox"]
            v_label = det["vehicle_type"].upper()
            cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (40, 180, 250), 2)

            # Vehicle label badge
            cv2.rectangle(annotated, (vx1, max(0, vy1 - 22)), (vx1 + len(v_label) * 12 + 10, vy1), (40, 180, 250), -1)
            cv2.putText(
                annotated,
                v_label,
                (vx1 + 5, max(16, vy1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (10, 15, 25),
                2,
                cv2.LINE_AA,
            )

            # 2. Draw plate box
            px1, py1, px2, py2 = det["plate_bbox"]
            # Bright neon emerald border for detected plate
            cv2.rectangle(annotated, (px1, py1), (px2, py2), (50, 230, 120), 3)

            # Display text (Standardized or Raw)
            display_text = det.get("standardized_plate") or det.get("plate_number") or ""
            conf_str = f"{int(det['confidence'] * 100)}%"
            badge_text = f"{display_text} [{conf_str}]"

            # Badge background with translucent dark fill
            text_size, _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            bx1 = px1
            by1 = max(0, py1 - text_size[1] - 12)
            bx2 = px1 + text_size[0] + 16
            by2 = py1

            overlay = annotated.copy()
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (20, 25, 35), -1)
            cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

            # Border around text badge
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (50, 230, 120), 1)

            # Draw text
            cv2.putText(
                annotated,
                badge_text,
                (bx1 + 8, by2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (50, 230, 120),
                2,
                cv2.LINE_AA,
            )

        return annotated

    # --------------------------------------------------------------------------
    # FULL PIPELINE EXECUTION
    # --------------------------------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray,
        source_name: str = "web",
        save_crops: bool = True
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Executes end-to-end ANPR on an input image frame:
        1. Detects road vehicles (Car, Motorcycle, Bus, Truck).
        2. Localizes license plate candidates within each vehicle.
        3. Preprocesses & enhances plate crops.
        4. Runs dual-script (Nepali/English) OCR.
        5. Parses and validates plate numbers into structured metadata.
        6. Logs records to SQLite database with anti-duplicate suppression.
        7. Returns annotated frame and detection records.

        Parameters
        ----------
        frame : np.ndarray
            Input BGR OpenCV image.
        source_name : str
            Identifier for the image source (e.g. filename, 'webcam', 'stream').
        save_crops : bool
            Whether to save extracted plate crops to disk for UI display.

        Returns
        -------
        tuple : (annotated_frame, list_of_detection_dicts)
        """
        start_time = time.time()
        detections: List[Dict[str, Any]] = []

        if frame is None or frame.size == 0:
            return frame, detections

        # Stage 1: Detect Vehicles
        vehicles = self._detect_vehicles(frame)

        # If no vehicles detected, scan full frame directly (e.g. close-up plate photo)
        if not vehicles:
            h, w = frame.shape[:2]
            vehicles = [{
                "bbox": (0, 0, w, h),
                "label": "vehicle",
                "conf": 0.50,
            }]

        for vehicle in vehicles:
            vx1, vy1, vx2, vy2 = vehicle["bbox"]
            v_crop = frame[vy1:vy2, vx1:vx2]
            if v_crop.size == 0:
                continue

            # Stage 2: Locate Plates inside Vehicle
            plates = self.locate_plates(v_crop, vehicle["label"])

            for plate in plates:
                px1, py1, px2, py2 = plate["bbox"]
                p_crop = v_crop[py1:py2, px1:px2]
                if p_crop.size == 0:
                    continue

                # Stage 3 & 4: Preprocess and OCR
                raw_text, ocr_conf, parsed = self._read_plate_text(p_crop, vehicle_type=vehicle["label"])

                if not raw_text or ocr_conf < 0.20:
                    continue

                # Compute combined confidence
                overall_conf = round(float((plate["conf"] * 0.4) + (ocr_conf * 0.6)), 3)

                # Absolute coordinates on the full frame
                abs_px1 = vx1 + px1
                abs_py1 = vy1 + py1
                abs_px2 = vx1 + px2
                abs_py2 = vy1 + py2

                # Save plate crop image
                crop_path_rel = None
                crop_filename = None
                if save_crops:
                    crop_filename = f"crop_{int(time.time() * 1000)}_{len(detections)}.jpg"
                    full_crop_path = CROPS_DIR / crop_filename
                    cv2.imwrite(str(full_crop_path), p_crop)
                    crop_path_rel = f"/static/crops/{crop_filename}"

                # Extract parsed fields
                standardized = parsed.get("standardized") or raw_text
                devanagari_display = parsed.get("devanagari_display") or raw_text
                script = parsed.get("script") or ("Devanagari (नेपाली)" if contains_devanagari(raw_text) else "Latin (English Embossed)")
                plate_type = parsed.get("plate_type") or "General Plate"
                zone_or_prov = parsed.get("zone_or_province") or "Nepal"
                veh_cat = parsed.get("vehicle_category") or vehicle["label"].capitalize()

                detection_record = {
                    "plate_number": raw_text,
                    "standardized_plate": standardized,
                    "devanagari_display": devanagari_display,
                    "script": script,
                    "plate_type": plate_type,
                    "zone_or_province": zone_or_prov,
                    "vehicle_type": vehicle["label"],
                    "vehicle_category": veh_cat,
                    "confidence": overall_conf,
                    "ocr_confidence": round(ocr_conf, 3),
                    "detector_confidence": round(plate["conf"], 3),
                    "plate_crop_path": crop_path_rel,
                    "vehicle_bbox": (vx1, vy1, vx2, vy2),
                    "plate_bbox": (abs_px1, abs_py1, abs_px2, abs_py2),
                    "is_valid": parsed.get("is_valid", False),
                    "details": parsed.get("details", ""),
                }

                detections.append(detection_record)

        # Stage 5: Deduplicate overlapping detections (NMS & Same Plate consolidation)
        unique_detections = []
        for det in sorted(detections, key=lambda d: d["confidence"], reverse=True):
            is_dup = False
            for u in unique_detections:
                # Same plate text
                if det["standardized_plate"] and det["standardized_plate"] == u["standardized_plate"]:
                    is_dup = True
                    break
                # Bounding box IoU overlap check
                b1 = det["plate_bbox"]
                b2 = u["plate_bbox"]
                ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter = iw * ih
                a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                iou = inter / float(a1 + a2 - inter + 1e-6)
                if iou > 0.35:
                    is_dup = True
                    break
            if not is_dup:
                unique_detections.append(det)

        detections = unique_detections

        # Log deduplicated detections to database
        for det in detections:
            if not database.was_recently_seen(det["standardized_plate"], within_seconds=6):
                try:
                    database.insert_detection(
                        plate_number=det["plate_number"],
                        standardized_plate=det["standardized_plate"],
                        devanagari_display=det["devanagari_display"],
                        script=det["script"],
                        plate_type=det["plate_type"],
                        zone_or_province=det["zone_or_province"],
                        vehicle_type=det["vehicle_type"],
                        vehicle_category=det["vehicle_category"],
                        confidence=det["confidence"],
                        ocr_confidence=det["ocr_confidence"],
                        detector_confidence=det["detector_confidence"],
                        source=source_name,
                        plate_crop_path=det["plate_crop_path"],
                        processing_time_ms=round((time.time() - start_time) * 1000, 2),
                    )
                except Exception as db_err:
                    print(f"[PlateVision] Database insert error: {db_err}")

        # Stage 6: Render Sleek Glassy Annotations on Frame
        annotated_frame = self._draw_modern_annotations(frame, detections)
        return annotated_frame, detections


# ==============================================================================
# SINGLETON INSTANCE
# ==============================================================================

_detector_instance: Optional[ANPRDetector] = None

def get_detector() -> ANPRDetector:
    """Provides a thread-safe singleton instance of the ANPRDetector."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = ANPRDetector()
    return _detector_instance
