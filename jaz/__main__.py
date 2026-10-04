"""Entry point for executing JAZ via `python -m jaz`."""

import importlib.util
from pathlib import Path
import subprocess
import sys


def _bootstrap_dependencies() -> None:
    """Ensure core framework dependencies (httpx, rich) are installed."""
    required = ["httpx", "rich"]
    missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]
    if missing:
        req_file = Path(__file__).resolve().parent.parent / "requirements.txt"
        print(f"[JAZ Bootstrap] Fehlende Kern-Abhängigkeiten erkannt: {', '.join(missing)}")
        print("[JAZ Bootstrap] Installiere benötigte Pakete automatisch...")
        cmd = [sys.executable, "-m", "pip", "install"]
        if req_file.is_file():
            cmd.extend(["-r", str(req_file)])
        else:
            cmd.extend(["httpx>=0.25.0", "rich>=13.0.0"])
        try:
            subprocess.run(cmd, check=True)
            print("[JAZ Bootstrap] Abhängigkeiten erfolgreich installiert.\n")
        except Exception as e:
            sys.stderr.write(f"[JAZ Bootstrap] FEHLER bei automatischer Paketinstallation: {e}\n")
            sys.exit(1)


_bootstrap_dependencies()

from .cli import main

if __name__ == "__main__":
    main()
