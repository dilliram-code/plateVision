"""
PlateVision Nepal - Web Application & REST API
=============================================
FastAPI backend service powering the Nepal Automatic Number Plate
Recognition (ANPR / ALPR) web platform and REST API.

Key Endpoints:
- GET  /                     : Interactive Glassmorphic Web Dashboard
- POST /api/detect/image     : Image upload for plate detection & recognition
- POST /api/detect/sample    : Quick demo with built-in Nepal vehicle samples
- POST /api/detect/frame     : Real-time video frame processing (webcam/RTSP)
- GET  /api/history          : Paginated & filtered detection registry
- GET  /api/stats            : Real-time analytics & fleet metrics
- GET  /api/export/csv       : CSV export of historical detection data
- DELETE /api/detections/{id}: Delete specific detection record
- POST /api/detections/clear : Clear database records
- GET  /api/health           : System & model operational status
"""

import os
import io
import time
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from pydantic import BaseModel

# Import detection and database layers
from detector import get_detector
import database

# ==============================================================================
# FASTAPI APP SETUP
# ==============================================================================

app = FastAPI(
    title="PlateVision Nepal ANPR",
    description="Production-grade Nepal Vehicle Number Plate Recognition API (Devanagari & English Embossed)",
    version="2.0.0",
)

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CROPS_DIR = DATA_DIR / "crops"
TEMPLATES_DIR = BASE_DIR / "templates"

# Ensure all directories exist
STATIC_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

# Mount static asset folders
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/static/crops", StaticFiles(directory=str(CROPS_DIR)), name="crops")
app.mount("/static/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ==============================================================================
# DATA MODELS
# ==============================================================================

class FramePayload(BaseModel):
    image_base64: str
    source_name: Optional[str] = "webcam"


# ==============================================================================
# HELPER UTILITIES
# ==============================================================================

def encode_image_base64(image_bgr: np.ndarray, quality: int = 88) -> str:
    """Encodes an OpenCV BGR image into a Base64 data URL string."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode(".jpg", image_bgr, encode_param)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def decode_image_base64(base64_str: str) -> Optional[np.ndarray]:
    """Decodes a Base64 data URL into an OpenCV BGR image."""
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[PlateVision] Base64 decode failed: {e}")
        return None


# ==============================================================================
# ROUTES: WEB DASHBOARD
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Renders the modern, elegant glassmorphic dashboard."""
    stats = database.get_detection_stats()
    recent = database.get_recent_detections(limit=15)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "stats": stats,
            "recent_detections": recent,
            "app_version": "2.0.0",
        },
    )


# ==============================================================================
# ROUTES: ANPR DETECTION
# ==============================================================================

@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    source_name: Optional[str] = Form(None),
):
    """
    Accepts an uploaded image file (cars, bikes, buses, trucks, or plate close-ups),
    runs the full ANPR pipeline, and returns annotated results with rich metadata.
    """
    start_time = time.time()
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file provided.")

        # Save uploaded file
        src_name = source_name or file.filename or "upload"
        timestamp_prefix = int(time.time() * 1000)
        saved_filename = f"upload_{timestamp_prefix}_{Path(src_name).name}"
        saved_path = UPLOADS_DIR / saved_filename
        with open(saved_path, "wb") as f:
            f.write(contents)

        # Execute ANPR detection
        detector = get_detector()
        annotated_frame, detections = detector.process_frame(
            image,
            source_name=src_name,
            save_crops=True,
        )

        processing_time = round((time.time() - start_time) * 1000, 2)
        annotated_b64 = encode_image_base64(annotated_frame)

        return JSONResponse({
            "success": True,
            "total_detected": len(detections),
            "processing_time_ms": processing_time,
            "annotated_image": annotated_b64,
            "original_image_url": f"/static/uploads/{saved_filename}",
            "detections": detections,
        })
    except Exception as e:
        print(f"[PlateVision] Detection error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/api/detect/sample")
async def detect_sample(sample_name: str = Form(...)):
    """
    Runs ANPR on a pre-built sample image (cars, bikes, buses, trucks, embossed plates).
    Allows instant one-click testing of all Nepali plate formats.
    """
    start_time = time.time()
    sample_file = STATIC_DIR / "samples" / sample_name

    if not sample_file.exists():
        raise HTTPException(status_code=404, detail="Sample image not found.")

    image = cv2.imread(str(sample_file))
    if image is None:
        raise HTTPException(status_code=500, detail="Could not read sample image.")

    detector = get_detector()
    annotated_frame, detections = detector.process_frame(
        image,
        source_name=f"Sample: {sample_name}",
        save_crops=True,
    )

    processing_time = round((time.time() - start_time) * 1000, 2)
    annotated_b64 = encode_image_base64(annotated_frame)

    return JSONResponse({
        "success": True,
        "sample_name": sample_name,
        "total_detected": len(detections),
        "processing_time_ms": processing_time,
        "annotated_image": annotated_b64,
        "original_image_url": f"/static/samples/{sample_name}",
        "detections": detections,
    })


@app.post("/api/detect/frame")
async def detect_frame(payload: FramePayload):
    """
    Ultra-low latency endpoint for processing real-time video frames
    streamed from a live webcam or security camera feed.
    """
    start_time = time.time()
    image = decode_image_base64(payload.image_base64)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid Base64 frame data.")

    detector = get_detector()
    annotated_frame, detections = detector.process_frame(
        image,
        source_name=payload.source_name or "webcam",
        save_crops=True,
    )

    processing_time = round((time.time() - start_time) * 1000, 2)
    annotated_b64 = encode_image_base64(annotated_frame, quality=80)

    return JSONResponse({
        "success": True,
        "total_detected": len(detections),
        "processing_time_ms": processing_time,
        "annotated_image": annotated_b64,
        "detections": detections,
    })


# ==============================================================================
# ROUTES: DATABASE, HISTORY & ANALYTICS
# ==============================================================================

@app.get("/api/history")
async def get_history(
    limit: int = Query(50, ge=1, le=500),
    search: Optional[str] = Query(None),
    vehicle_type: Optional[str] = Query("all"),
    script: Optional[str] = Query("all"),
    offset: int = Query(0, ge=0),
):
    """Fetches paginated and filtered historical detection records."""
    records = database.get_recent_detections(
        limit=limit,
        search=search,
        vehicle_type=vehicle_type,
        script=script,
        offset=offset,
    )
    return JSONResponse({"success": True, "count": len(records), "detections": records})


@app.get("/api/stats")
async def get_stats():
    """Returns aggregated analytics for glassy counters and Chart.js graphs."""
    stats = database.get_detection_stats()
    return JSONResponse({"success": True, "stats": stats})


@app.get("/api/export/csv")
async def export_csv():
    """Generates and downloads a CSV file of all recorded plate detections."""
    csv_data = database.export_detections_csv()
    filename = f"PlateVision_Nepal_Detections_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.delete("/api/detections/{detection_id}")
async def delete_detection(detection_id: int):
    """Deletes a specific detection record."""
    deleted = database.delete_detection(detection_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Detection record not found.")
    return JSONResponse({"success": True, "message": "Record deleted."})


@app.post("/api/detections/clear")
async def clear_history():
    """Clears all detection records from the database."""
    count = database.clear_all_detections()
    return JSONResponse({"success": True, "deleted_count": count})


# ==============================================================================
# ROUTES: SYSTEM HEALTH
# ==============================================================================

@app.get("/api/health")
async def get_health():
    """Provides operational health, hardware acceleration, and model status."""
    import torch
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    return JSONResponse({
        "status": "healthy",
        "service": "PlateVision Nepal ANPR",
        "version": "2.0.0",
        "device": device,
        "models": {
            "vehicle_detector": "YOLOv8 Nano (COCO)",
            "ocr_engine": "EasyOCR Dual-Script (Nepali + English)",
            "plate_localizer": "Color Segmenter + Morphology + Haar Cascade",
            "nepal_parser": "Rule-based & Regex Engine v2",
        },
    })
