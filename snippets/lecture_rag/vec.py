"""
SQLite connections with the sqlite-vec extension (https://alexgarcia.xyz/sqlite-vec/) loaded,
which adds vector search to SQLite (e.g. the `vec0` virtual tables).
"""
import sqlite3
from pathlib import Path
import sqlite_vec

serialize = sqlite_vec.serialize_float32  # list[float] -> bytes, i.e. the format sqlite-vec expects for vectors


def connect(path: str | Path = ":memory:") -> sqlite3.Connection:
    """A connection to the given database file (in memory by default), with sqlite-vec loaded."""
    conn = sqlite3.connect(path)
    if not hasattr(conn, "enable_load_extension"):
        raise RuntimeError(
            "This Python's sqlite3 module cannot load extensions (it was built without them, as e.g. macOS' system Python "
            "and some pyenv builds), so sqlite-vec cannot be loaded. Use a Python that can, e.g. from Homebrew, "
            "python.org, or uv; or rebuild it with pyenv: "
            "PYTHON_CONFIGURE_OPTS=--enable-loadable-sqlite-extensions pyenv install --force <VERSION>; "
            "then re-create the virtual environment (poetry env use <PYTHON>; poetry install)."
        )
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)  # no further extensions: loading them is a security risk
    return conn
