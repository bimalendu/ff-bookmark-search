import os
import sqlite3
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict

def find_default_firefox_profile() -> Path:
    if os.name == "nt":
        base = Path(os.environ["APPDATA"]) / "Mozilla" / "Firefox"
    elif os.name == "posix":
        if "darwin" in os.sys.platform:
            base = Path.home() / "Library" / "Application Support" / "Firefox"
        else:
            base = Path.home() / ".mozilla" / "firefox"
    else:
        raise RuntimeError("Unsupported OS")

    profiles_dir = base / "Profiles"
    if not profiles_dir.exists():
        raise FileNotFoundError("Firefox Profiles directory not found")

    best_profile = None
    best_size = 0

    for profile_path in profiles_dir.iterdir():
        places_path = profile_path / "places.sqlite"
        if not places_path.exists():
            continue

        size = places_path.stat().st_size
        if size < 100 * 1024:
            continue

        try:
            conn = sqlite3.connect(str(places_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='moz_bookmarks';")
            has_bookmarks = cursor.fetchone() is not None
            conn.close()
        except sqlite3.Error:
            continue

        if has_bookmarks and size > best_size:
            best_profile = profile_path
            best_size = size

    if not best_profile:
        raise RuntimeError("No valid Firefox profile with bookmarks found.")

    return best_profile

def copy_places_sqlite(profile_path: Path) -> Path:
    source = profile_path / "places.sqlite"
    if not source.exists():
        raise FileNotFoundError(f"{source} not found")

    temp_dir = tempfile.mkdtemp()
    target = Path(temp_dir) / "places.sqlite"

    src_conn = sqlite3.connect(str(source))
    dst_conn = sqlite3.connect(str(target))

    with dst_conn:
        src_conn.backup(dst_conn)

    src_conn.close()
    dst_conn.close()

    return target

def get_firefox_bookmarks_from_sqlite(sqlite_path: Path) -> List[Dict]:
    conn = sqlite3.connect(str(sqlite_path))
    cursor = conn.cursor()

    try:
        query = """
        SELECT moz_bookmarks.id, moz_bookmarks.title, moz_places.url
        FROM moz_bookmarks
        JOIN moz_places ON moz_bookmarks.fk = moz_places.id
        WHERE moz_bookmarks.title IS NOT NULL
        """
        cursor.execute(query)
        bookmarks = cursor.fetchall()
    except sqlite3.OperationalError as e:
        raise RuntimeError(f"Failed to read bookmarks: {e}")
    finally:
        conn.close()

    return [
        {"id": row[0], "title": row[1], "url": row[2]}
        for row in bookmarks
    ]

def get_firefox_bookmarks() -> List[Dict]:
    profile_path = find_default_firefox_profile()
    safe_copy = copy_places_sqlite(profile_path)
    return get_firefox_bookmarks_from_sqlite(safe_copy)
