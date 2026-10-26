"""
PlateVision Database Layer
==========================
Production-grade SQLite database management for Nepal ANPR / ALPR system.
Provides persistent storage, thread-safe connections, anti-duplicate checking,
rich querying/filtering, statistics aggregation, and export functionality.
"""

import io
import csv
import json
import sqlite3
import pandas as pd
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Optional, Any

# Locate database file and uploads storage
DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "anpr.db"
CROPS_DIR = DATA_DIR / "crops"
UPLOADS_DIR = DATA_DIR / "uploads"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """
    Creates and returns a thread-safe SQLite connection with Row factory enabled
    so query results can be accessed like dictionaries.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """
    Initializes the database schema with indices for fast lookups and reporting.
    Runs migration safely if columns already exist or if created anew.
    """
    conn = get_connection()
    cur = conn.cursor()

    # Create main detections table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS detections (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number        TEXT NOT NULL,
            standardized_plate  TEXT,
            devanagari_display  TEXT,
            script              TEXT,
            plate_type          TEXT,
            zone_or_province    TEXT,
            vehicle_type        TEXT,
            vehicle_category    TEXT,
            confidence          REAL NOT NULL,
            ocr_confidence      REAL DEFAULT 0.0,
            detector_confidence REAL DEFAULT 0.0,
            source              TEXT NOT NULL,
            image_path          TEXT,
            plate_crop_path     TEXT,
            processing_time_ms  REAL DEFAULT 0.0,
            timestamp           TEXT NOT NULL
        )
        """
    )

    # Indices for performance
    cur.execute("CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_detections_plate ON detections(plate_number)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_detections_std_plate ON detections(standardized_plate)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_detections_vehicle ON detections(vehicle_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_detections_script ON detections(script)")

    conn.commit()
    conn.close()


def insert_detection(
    plate_number: str,
    confidence: float,
    vehicle_type: Optional[str] = None,
    source: str = "upload",
    image_path: Optional[str] = None,
    standardized_plate: Optional[str] = None,
    devanagari_display: Optional[str] = None,
    script: Optional[str] = None,
    plate_type: Optional[str] = None,
    zone_or_province: Optional[str] = None,
    vehicle_category: Optional[str] = None,
    ocr_confidence: float = 0.0,
    detector_confidence: float = 0.0,
    plate_crop_path: Optional[str] = None,
    processing_time_ms: float = 0.0,
) -> int:
    """
    Inserts a newly recognized vehicle number plate record stamped with current timestamp.
    Returns the auto-generated primary key ID.
    """
    timestamp = datetime.now().isoformat(timespec="seconds")
    conn = get_connection()
    cur = conn.execute(
        """
        INSERT INTO detections (
            plate_number, standardized_plate, devanagari_display, script,
            plate_type, zone_or_province, vehicle_type, vehicle_category,
            confidence, ocr_confidence, detector_confidence, source,
            image_path, plate_crop_path, processing_time_ms, timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            plate_number,
            standardized_plate or plate_number,
            devanagari_display or plate_number,
            script or "Unknown",
            plate_type or "General",
            zone_or_province or "Unknown",
            vehicle_type or "vehicle",
            vehicle_category or "Vehicle",
            confidence,
            ocr_confidence,
            detector_confidence,
            source,
            image_path,
            plate_crop_path,
            processing_time_ms,
            timestamp,
        ),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def was_recently_seen(plate_number: str, within_seconds: int = 8) -> bool:
    """
    Duplicate suppression filter:
    Checks if this exact plate (or its standardized version) was recorded
    within the last N seconds (e.g. in video streams or burst frames).
    """
    if not plate_number:
        return False

    conn = get_connection()
    row = conn.execute(
        """
        SELECT timestamp FROM detections
        WHERE plate_number = ? OR standardized_plate = ?
        ORDER BY id DESC LIMIT 1
        """,
        (plate_number, plate_number),
    ).fetchone()
    conn.close()

    if row is None:
        return False

    try:
        last_seen = datetime.fromisoformat(row["timestamp"])
        elapsed = (datetime.now() - last_seen).total_seconds()
        return elapsed < within_seconds
    except Exception:
        return False


def get_recent_detections(
    limit: int = 50,
    search: Optional[str] = None,
    vehicle_type: Optional[str] = None,
    script: Optional[str] = None,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Fetches filtered and paginated recent detections for the dashboard table.
    Properly formats query parameters avoiding tuple unpacking bugs.
    """
    conn = get_connection()
    query = "SELECT * FROM detections WHERE 1=1"
    params: List[Any] = []

    if search and search.strip():
        term = f"%{search.strip()}%"
        query += " AND (plate_number LIKE ? OR standardized_plate LIKE ? OR devanagari_display LIKE ? OR zone_or_province LIKE ?)"
        params.extend([term, term, term, term])

    if vehicle_type and vehicle_type.lower() != "all":
        query += " AND LOWER(vehicle_type) = ?"
        params.append(vehicle_type.lower())

    if script and script.lower() != "all":
        if "nepali" in script.lower() or "devanagari" in script.lower():
            query += " AND script LIKE '%Devanagari%'"
        elif "english" in script.lower() or "latin" in script.lower() or "embossed" in script.lower():
            query += " AND script LIKE '%Latin%'"

    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = conn.execute(query, tuple(params)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_detection_stats() -> Dict[str, Any]:
    """
    Aggregates analytical metrics for display on the glassy dashboard counters
    and interactive Chart.js graphs.
    """
    conn = get_connection()
    cur = conn.cursor()

    # Total all-time scans
    total_detections = cur.execute("SELECT COUNT(*) FROM detections").fetchone()[0]

    # Today's scans
    today_iso = date.today().isoformat()
    today_detections = cur.execute(
        "SELECT COUNT(*) FROM detections WHERE timestamp LIKE ?",
        (f"{today_iso}%",),
    ).fetchone()[0]

    # Average confidence score
    avg_conf_row = cur.execute("SELECT AVG(confidence) FROM detections").fetchone()
    avg_confidence = round(float(avg_conf_row[0]), 3) if avg_conf_row and avg_conf_row[0] else 0.0

    # Vehicle type breakdown
    veh_rows = cur.execute(
        """
        SELECT vehicle_type, COUNT(*) as count
        FROM detections
        GROUP BY vehicle_type
        ORDER BY count DESC
        """
    ).fetchall()
    vehicle_distribution = {r["vehicle_type"] or "Unknown": r["count"] for r in veh_rows}

    # Script breakdown (Nepali Devanagari vs English Embossed)
    script_rows = cur.execute(
        """
        SELECT script, COUNT(*) as count
        FROM detections
        GROUP BY script
        ORDER BY count DESC
        """
    ).fetchall()
    script_distribution = {r["script"] or "Unknown": r["count"] for r in script_rows}

    # Hourly flow for today
    hourly_rows = cur.execute(
        """
        SELECT substr(timestamp, 12, 2) as hour, COUNT(*) as count
        FROM detections
        WHERE timestamp LIKE ?
        GROUP BY hour
        ORDER BY hour ASC
        """,
        (f"{today_iso}%",),
    ).fetchall()
    hourly_flow = {r["hour"]: r["count"] for r in hourly_rows}

    conn.close()

    return {
        "total_detections": total_detections,
        "today_detections": today_detections,
        "avg_confidence": avg_confidence,
        "vehicle_distribution": vehicle_distribution,
        "script_distribution": script_distribution,
        "hourly_flow": hourly_flow,
    }


def delete_detection(detection_id: int) -> bool:
    """Deletes a detection record by its primary key."""
    conn = get_connection()
    cur = conn.execute("DELETE FROM detections WHERE id = ?", (detection_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


def clear_all_detections() -> int:
    """Clears all records from detections table. Returns count deleted."""
    conn = get_connection()
    cur = conn.execute("DELETE FROM detections")
    conn.commit()
    deleted = cur.rowcount
    conn.close()
    return deleted


def export_detections_csv() -> str:
    """Exports all detection records to a CSV formatted string."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM detections ORDER BY id DESC", conn)
    conn.close()

    output = io.StringIO()
    df.to_csv(output, index=False)
    return output.getvalue()


# Initialize table on import
init_db()