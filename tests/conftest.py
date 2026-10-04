"""Pytest configuration ensuring jaz package is on sys.path."""

from pathlib import Path
import sys

# Add root folder to sys.path so jaz is importable
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
