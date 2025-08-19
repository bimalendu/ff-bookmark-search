import os
import sqlite3
import shutil
import tempfile
import configparser
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

    profiles_ini = base / "profiles.ini"
    config = configparser.ConfigParser()
    config.read(profiles_ini)

    for section in config.sections():
        if config.has_option(section, "Default") and config.get(section, "Default") == "1":
            path = config.get(section, "Path")
            is_relative = config.getboolean(section, "IsRelative", fallback=True)
            return (base / path) if is_relative else Path(path)

    raise FileNotFoundError("No default Firefox profile found.")

def copy_places_sqlite(profile_path: Path) -> Path:
    source = profile_path / "places.sqlite"
    if not source.exists():
        raise FileNotFoundError(f"{source} not found")

    temp_dir = tempfile.mkdtemp()
    target = Path(temp_dir) / "places.sqlite"
    shutil.copy2(source, target)
    return target

def get_firefox_bookmarks_from_sqlite(sqlite_path: Path) -> List[Dict]:
    conn = sqlite3.connect(str(sqlite_path))
    cursor = conn.cursor()

    query = """
    SELECT moz_bookmarks.id, moz_bookmarks.title, moz_places.url
    FROM moz_bookmarks
    JOIN moz_places ON moz_bookmarks.fk = moz_places.id
    WHERE moz_bookmarks.title IS NOT NULL
    """

    cursor.execute(query)
    bookmarks = cursor.fetchall()
    conn.close()

    return [{"id": row[0], "title": row[1], "url": row[2]} for row in bookmarks]

def get_firefox_bookmarks() -> List[Dict]:
    profile_path = find_default_firefox_profile()
    safe_copy = copy_places_sqlite(profile_path)
    return get_firefox_bookmarks_from_sqlite(safe_copy)
