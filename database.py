import sqlite3 
import pandas as pd 
from datetime import datetime 
from pathlib import Path

# locate db file 
DB_PATH = Path(__file__).parent / "data" / "anpr.db"

def get_connections():
  DB_PATH.parent.mkdir(parents=True, exist_ok=True)
  conn = sqlite3.connect(DB_PATH, check_same_thread=False)
  conn.row_factory = sqlite3.Row
  return conn 

def init_db():
  conn = get_connections()
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