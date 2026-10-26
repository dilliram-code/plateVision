# PlateVision Nepal 🇳🇵
### AI-Powered Automatic Number Plate Recognition (ANPR / ALPR) for Nepal

A production-grade, end-to-end Vehicle License Plate Detection and Recognition system specifically engineered for Nepal's transport ecosystem. It accurately detects vehicles (cars, motorcycles, scooters, buses, trucks) and recognizes license plates in **both Nepali (Devanagari script)** and **English (Latin High-Security Embossed script)**.

---

## 🌟 Key Capabilities

1. **Dual-Script Recognition**:
   - **Devanagari Script (नेपाली)**:
     - Traditional Zonal format: e.g. `बा ९२ प ५६७८` (Bagmati Two-Wheeler), `ना ५ ख १२३४` (Narayani Public Bus), `बा १५ च ४३२१` (Private Car), `बा १ ज ७८९०` (Taxi)
     - Provincial format: e.g. `बा.प्र. ०२-०३५ प ५६७८` (Bagmati Province), `प्रदेश ३ - ०१ ०२५ च ९८७६`
     - Stacked 2-line layout parsing (common on motorcycles, tempos, and trucks)
   - **English Script (High-Security Embossed)**:
     - Provincial Embossed format: e.g. `BAGMATI PROVINCE 02-035 PA 5678`, `02-035 PA 5678`, `BA 2 PA 1234`
     - Classification mapping: `PA/PB/PC` (Two-wheelers), `CA/CB` (Light 4-wheelers), `BA/BB` (Heavy vehicles), `DA` (Three-wheelers)

2. **Multi-Vehicle Classification**:
   - YOLOv8 deep learning model recognizing `car`, `motorcycle`, `bus`, and `truck` with bounding box annotations.

3. **Multi-Tier Plate Localization Pipeline**:
   - **Tier 1 (Custom YOLOv8)**: Uses `models/license_plate_detector.pt` if present.
   - **Tier 2 (OpenCV Haar Cascade)**: Multi-scale plate classifier fallback.
   - **Tier 3 (Nepal Color Segmenter)**: Segments by Nepal's official color schemes:
     - 🔴 **Red plate**: Private vehicles (निजी सवारी)
     - ⚫ **Black plate**: Public transport, buses, trucks, taxis (सार्वजनिक सवारी)
     - ⚪ **White plate**: High-security embossed and government vehicles (इम्बोस्ड / सरकारी)
     - 🟢 **Green plate**: Electric vehicles (विद्युतीय सवारी EV)
   - **Tier 4 (Morphological & Gradient)**: Vertical Sobel edge filter + rectangular aspect ratio filtering.

4. **Image Enhancement & Preprocessing**:
   - CLAHE (Contrast Limited Adaptive Histogram Equalization) on LAB color space.
   - Bilateral edge-preserving denoising.
   - Bicubic super-resolution scaling for low-resolution or distant crops.

5. **Modern Glassmorphic UI Dashboard**:
   - Frosted glass cards with `backdrop-filter: blur(24px)`.
   - Dark luxury theme with neon cyan, emerald green, and amber highlights.
   - Interactive Tabs:
     - 🔍 **Live Scanner & Upload**: Drag & drop or 1-click Quick Demo testing.
     - 📹 **Live Stream / Webcam**: Real-time video processing with live plate feed.
     - 📋 **Plate Registry**: Searchable database, crop image preview lightbox, CSV export.
     - 📊 **Analytics**: Interactive Chart.js graphs (vehicle breakdown, script ratio, hourly flow).
     - 🇳🇵 **Nepal Plate Guide**: Comprehensive educational guide on Nepal number plate formats.

---

## 📁 Project Architecture

```
plateVision/
├── main.py                   # FastAPI backend server & REST API endpoints
├── detector.py               # Production ANPR engine (YOLOv8 + EasyOCR + CV Segmenter)
├── nepal_plate_parser.py     # Domain-specific Nepal plate parsing, transliteration & rules
├── database.py               # SQLite database management with duplicate suppression & stats
├── generate_samples.py       # Realistic test sample generator (Bikes, Cars, Buses, Embossed)
├── requirements.txt          # Python dependencies
├── static/
│   ├── css/
│   │   └── style.css         # Glassmorphism styling, responsive layout & neon accents
│   ├── js/
│   │   └── app.js           # Frontend client engine (Uploads, Webcam, Charts, Modals)
│   ├── crops/                # Stored plate crops for instant lightbox inspection
│   └── samples/              # One-click demo sample images
├── templates/
│   └── index.html            # HTML5 Dashboard with modern glassmorphism design
└── data/
    └── anpr.db               # SQLite persistent database
```

---

## 🚀 Running the Application

### 1. Activate the Virtual Environment
```bash
source venv/bin/activate
```

### 2. Start the FastAPI Web Server
```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8001
```

### 3. Open in Browser
Visit:
```
http://127.0.0.1:8001
```

---

## 📡 REST API Endpoints

- `GET /` : Interactive Glassy Web Dashboard
- `POST /api/detect/image` : Upload image (`multipart/form-data`) for ANPR
- `POST /api/detect/sample` : Analyze preloaded sample (`sample_name=sample_bike_nepali.jpg`)
- `POST /api/detect/frame` : Base64 video frame processing for webcam / RTSP CCTV
- `GET /api/history` : Get paginated and filtered plate records (`?search=...&vehicle_type=...&script=...`)
- `GET /api/stats` : Real-time analytics and fleet counts
- `GET /api/export/csv` : Download all historical detections in CSV format
- `DELETE /api/detections/{id}` : Delete specific record
- `POST /api/detections/clear` : Clear all detection records
- `GET /api/health` : System diagnostics, compute device (MPS/CUDA/CPU), and model states
