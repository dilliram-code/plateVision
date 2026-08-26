import sqlite3 
import pandas as pd 
from datetime import datetime 
from pathlib import Path

# locate db file 
DB_PATH = Path(__file__).parent / "data" / "anpr.db"

def get_connection():
  DB_PATH.parent.mkdir(parents=True, exist_ok=True)
  conn = sqlite3.connect(DB_PATH, check_same_thread=False)
  conn.row_factory = sqlite3.Row
  return conn 

def init_db():
  conn = get_connection()
  conn.execute(
    """
    CREATE TABLE IF NOT EXISTS detections (
      id              INTEGER PRIMARY KEY AUTOINCREMENT,
      plate_number    TEXT    NOT NULL, 
      confidence      REAL    NOT NULL,
      vehicle_type    TEXT,
      source          TEXT    NOT NULL,
      image_path      TEXT,
      timestamp       TEXT    NOT NULL
    )
    """
  )
  
  conn.commit()
  conn.close()

# insert recognized-plate record, stamped with the current time 
def insert_detection(plate_number, confidence, vehicle_type, source, image_path):
  timestamp = datetime.now().isoformat(timespec="seconds")
  conn = get_connection()
  cur = conn.execute(
    """
    INSERT INTO detections(plate_number, confidence, vehicle_type, source, image_path, timestamp)
    VALUES (?, ?, ?, ?, ?, ?)
    """,
    (plate_number, confidence, vehicle_type, source, image_path, timestamp)
  )
  conn.commit()
  new_id = cur.lastrowid
  conn.close()
  
  return new_id

# implement anti-duplicate check 
def was_recently_seen(plate_number, within_seconds=10):
  conn = get_connection()
  row = conn.execute(
    """
    SELECT timestamp FROM detections
    WHERE plate_number = ?
    ORDER BY id DESC LIMIT 1
    """,
    (plate_number,),
  ).fetchone()
  conn.close()
  
  if row is None:
    return False
  
  last_seen = datetime.fromisoformat(row["timestamp"])
  elapsed = (datetime.now() - last_seen).total_seconds()
  
  return elapsed < within_seconds

def get_recent_detections(limit=50, search=None):
  conn = get_connection()
  if search:
    rows = conn.execute(
      """
      SELECT * FROM detections
      WHERE plate_number LIKE ?
      ORDER BY id DESC LIMIT ?
      """,
      (f"%{search}%", limit),
    ).fetchall()
    
  else:
    rows = conn.execute(
      "SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit)
    ).fetchall()
  conn.close()
  return [dict(r) for r in rows]