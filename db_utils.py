import argparse
import os
import sqlite3
import json
import numpy as np
import hashlib
from contextlib import contextmanager
from typing import Any, Dict, List

from tqdm import tqdm


DB_FILE = "data/distances_train.db"
MAX_FILE_SIZE = 25 * 1024 * 1024 * 1024 # 25 GB in bytes


def calculate_entry_hash(paths: list[np.ndarray], speeds: np.ndarray, rpositions: np.ndarray) -> str:
    """Calculate a hash for the entry to detect duplicates efficiently."""
    hash_obj = hashlib.sha256()
    
    for path in paths:
        hash_obj.update(path.astype(int).tobytes())
    
    hash_obj.update(speeds.astype(float).tobytes())
    hash_obj.update(rpositions.astype(float).tobytes())
    
    return hash_obj.hexdigest()


@contextmanager
def get_db_connection(filepath: str = DB_FILE):
    conn = None
    try:
        conn = sqlite3.connect(filepath)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        yield conn
    except Exception as e:
        print(f"Database error: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


def initialize_db(filepath: str = DB_FILE) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with get_db_connection(filepath) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS distance_calculations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                num_agents INTEGER NOT NULL,
                num_rewards INTEGER NOT NULL,
                max_distance REAL NOT NULL,
                paths TEXT NOT NULL,
                speeds TEXT NOT NULL,
                rpositions TEXT NOT NULL,
                entry_hash TEXT UNIQUE,
                timestamps TEXT,
                coordinates TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_num_agents
            ON distance_calculations(num_agents)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_num_rewards 
            ON distance_calculations(num_rewards)
        """)
        
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_entry_hash
            ON distance_calculations(entry_hash)
        """)
        
        conn.commit()


def save_to_db(
    paths: list[np.ndarray],
    speeds: np.ndarray,
    rpositions: np.ndarray,
    max_distance: float,
    filepath: str = DB_FILE
) -> None:
    if not os.path.exists(filepath):
        initialize_db(filepath)

    current_size = os.path.getsize(filepath)
    if current_size >= MAX_FILE_SIZE:
        print("Database size limit reached. Skipping save.")
        return
    
    try:
        if not paths or len(paths) == 0:
            print("Empty paths provided. Skipping save.")
            return
        if any(p is None or len(p) == 0 for p in paths):
            print("One or more paths are None or empty. Skipping save.")
            return

        # Calculate hash for duplicate detection
        entry_hash = calculate_entry_hash(paths, speeds, rpositions)
        
        paths_json = json.dumps([p.astype(int).tolist() for p in paths])
        speeds_json = json.dumps(speeds.astype(float).tolist())
        rpositions_json = json.dumps(rpositions.astype(float).tolist())
        timestamps, coordinates = calculate_timestamps_and_coordinates(paths, rpositions, speeds)
        timestamps_json = json.dumps(timestamps)
        coordinates_json = json.dumps(coordinates)

        with get_db_connection(filepath) as conn:
            cursor = conn.cursor()

            # Fast duplicate check using hash
            cursor.execute("""
                SELECT id FROM distance_calculations WHERE entry_hash = ?
            """, (entry_hash,))
            
            if cursor.fetchone() is not None:
                return

            cursor.execute("""
                INSERT INTO distance_calculations 
                (num_agents, num_rewards, max_distance, paths, speeds, rpositions, entry_hash, timestamps, coordinates)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                len(paths),
                len(rpositions),
                max_distance,
                paths_json,
                speeds_json,
                rpositions_json,
                entry_hash,
                timestamps_json,
                coordinates_json
            ))
            
            conn.commit()

    except Exception as e:
        print(f"Error saving calculation: {e}")


def load_from_db(
    start: int = None,
    limit: int = None,
    filepath: str = DB_FILE,
    uniform: bool = False
) -> List[Dict[str, Any]]:
    if not os.path.exists(filepath):
        print("Database file does not exist.")
        return []
    
    try:
        with get_db_connection(filepath) as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM distance_calculations"
            params = []

            if uniform:
                query += " ORDER BY RANDOM()"

            if start is not None:
                query += " OFFSET ?"
                params.append(start)

            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            results = []
            corrupted_ids = []
            for row in rows:
                try:
                    record = {
                        'id': row[0],
                        'num_agents': row[1],
                        'num_rewards': row[2],
                        'max_distance': row[3],
                        'paths': [np.array(path) for path in json.loads(row[4])],
                        'speeds': np.array(json.loads(row[5])),
                        'rpositions': np.array(json.loads(row[6])),
                        'entry_hash': row[7],
                        'timestamps': [np.array(ts) for ts in json.loads(row[8])],
                        'coordinates': [np.array(coord) for coord in json.loads(row[9])]
                    }
                    results.append(record)
                except (ValueError, TypeError, OverflowError) as e:
                    corrupted_ids.append(row[0])
                    continue            

            if corrupted_ids:
                placeholders = ','.join('?' for _ in corrupted_ids)
                cursor.execute(f"""
                    DELETE FROM distance_calculations 
                    WHERE id IN ({placeholders})
                """, corrupted_ids)
                conn.commit()
                print(f"Removed {len(corrupted_ids)} corrupted entries from the database.")

            return results
            
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return []


def count_db_entries(filepath: str = DB_FILE) -> int:
    if not os.path.exists(filepath):
        print("Database file does not exist.")
        return 0
    
    try:
        with get_db_connection(filepath) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM distance_calculations")
            count = cursor.fetchone()[0]
            return count

    except Exception as e:
        print(f"Error counting database entries: {e}")
        return 0


def add_coordinates_to_db(filepath: str = DB_FILE) -> None:
    if not os.path.exists(filepath):
        print("Database file does not exist.")
        return

    try:
        with get_db_connection(filepath) as conn:
            cursor = conn.cursor()
            
            cursor.execute("PRAGMA table_info(distance_calculations)")
            columns = [info[1] for info in cursor.fetchall()]
            
            if 'coordinates' not in columns:
                cursor.execute("""
                    ALTER TABLE distance_calculations
                    ADD COLUMN coordinates TEXT
                """)
                conn.commit()

            cursor.execute("SELECT * FROM distance_calculations")
            rows = cursor.fetchall()
            for row in tqdm(rows):
                record = {
                    'paths': [np.array(path) for path in json.loads(row[4])],
                    'rpositions': np.array(json.loads(row[6])),
                }
                coordinates = []
                for path in record['paths']:
                    path_coordinates = record['rpositions'][path].tolist()
                    coordinates.append(path_coordinates)
                coordinates_json = json.dumps(coordinates)
                cursor.execute("""
                    UPDATE distance_calculations
                    SET coordinates = ?
                    WHERE id = ?
                """, (coordinates_json, row[0]))
            conn.commit()
        
    except Exception as e:
        print(f"Error adding coordinates to database: {e}")


def add_timestamps_to_db(filepath: str = DB_FILE) -> None:
    if not os.path.exists(filepath):
        print("Database file does not exist.")
        return
    
    try:
        with get_db_connection(filepath) as conn:
            cursor = conn.cursor()
            
            cursor.execute("PRAGMA table_info(distance_calculations)")
            columns = [info[1] for info in cursor.fetchall()]
            
            if 'timestamps' not in columns:
                cursor.execute("""
                    ALTER TABLE distance_calculations
                    ADD COLUMN timestamps TEXT
                """)
                conn.commit()

            cursor.execute("SELECT * FROM distance_calculations")
            rows = cursor.fetchall()
            for row in tqdm(rows):
                record = {
                    'paths': [np.array(path) for path in json.loads(row[4])],
                    'speeds': np.array(json.loads(row[5])),
                    'rpositions': np.array(json.loads(row[6])),
                }

                timestamps, _ = calculate_timestamps_and_coordinates(record['paths'], record['rpositions'], record['speeds'])
                timestamps_json = json.dumps(timestamps)
                cursor.execute("""
                    UPDATE distance_calculations
                    SET timestamps = ?
                    WHERE id = ?
                """, (timestamps_json, row[0]))
            conn.commit()

    except Exception as e:
        print(f"Error adding timestamps to database: {e}")


def calculate_coordinates(paths, rpositions):
    coordinates = []
    for path in paths:
        path_coordinates = rpositions[path].tolist()
        coordinates.append(path_coordinates)
    return coordinates


def calculate_timestamps(coordinates, speeds):
    timestamps = []
    for path_coords, speed in zip(coordinates, speeds):
        dist = np.linalg.norm(np.diff(path_coords, axis=0), axis=1)
        time = dist / speed
        timestamps.append(np.insert(np.cumsum(time), 0, 0).tolist())
    return timestamps



def calculate_timestamps_and_coordinates(paths, rpositions, speeds):
    coordinates = calculate_coordinates(paths, rpositions)
    timestamps = calculate_timestamps(coordinates, speeds)
    return timestamps, coordinates


def print_columns(filepath: str = DB_FILE) -> None:
    with get_db_connection(filepath) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(distance_calculations)")
        columns = [info[1] for info in cursor.fetchall()]
        print(columns)


if __name__ == "__main__":
    with get_db_connection("data/distances_train.db") as conn:
        cursor = conn.cursor()
        
        cursor.execute("PRAGMA table_info(distance_calculations)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if 'coordinates' not in columns:
            cursor.execute("""
                ALTER TABLE distance_calculations
                ADD COLUMN coordinates TEXT
            """)
            conn.commit()
        if 'timestamps' not in columns:
            cursor.execute("""
                ALTER TABLE distance_calculations
                ADD COLUMN timestamps TEXT
            """)
            conn.commit()
    