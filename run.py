"""Simple quick-start runner for JAZ with automatic dependency bootstrapping.

Usage:
    python run.py "Your prompt here"
    python run.py "Your prompt here" --sandbox subprocess
"""

import importlib.util
from pathlib import Path
import subprocess
import sys


def _bootstrap_dependencies() -> None:
    """Ensure core framework dependencies (httpx, rich) are installed."""
    required = ["httpx", "rich"]
    missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]
    if missing:
        req_file = Path(__file__).resolve().parent / "requirements.txt"
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


if __name__ == "__main__":
    _bootstrap_dependencies()

    from jaz.cli import main

    # If the user just runs `python run.py "My prompt..."`, automatically inject `run`
    if len(sys.argv) > 1 and sys.argv[1] not in ("run", "check", "replay", "--help", "-h"):
        sys.argv.insert(1, "run")
    main()
