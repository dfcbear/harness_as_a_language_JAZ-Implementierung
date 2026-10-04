# JAZ: Handbuch & Technische Dokumentation

Dieses Handbuch dokumentiert die Architektur, Konfiguration, Sicherheitsabschottung und praktische Anwendung der Python-Implementierung von **JAZ** (*Harness as a Language*), basierend auf dem Paper [arXiv:2609.26891 (MIT CSAIL, 2026)](https://arxiv.org/abs/2609.26891).

---

## Inhaltsverzeichnis

1. [Grundkonzept & Paradigma von JAZ](#1-grundkonzept--paradigma-von-jaz)
2. [Zentrale Konfiguration](#2-zentrale-konfiguration)
3. [Ausführung & Sicherheitsabschottung (Direct, Subprocess, Container)](#3-ausführung--sicherheitsabschottung)
4. [Terminal-Bedienung & Live-Status](#4-terminal-bedienung--live-status)
5. [Event-Stream & Vorbereitung für Web-Dashboards](#5-event-stream--vorbereitung-für-web-dashboards)
6. [Praktische Muster für eigene Agenten-Aufträge](#6-praktische-muster-für-eigene-agenten-aufträge)
7. [Testen & Spezifikationsprüfung](#7-testen--spezifikationsprüfung)

---

## 1. Grundkonzept & Paradigma von JAZ

Klassische Agenten-Frameworks (wie ReAct oder LangChain) betten ein Sprachmodell in eine starre Kontrollschleife ein und lagern Fähigkeiten wie Langzeitgedächtnis, Werkzeugaufrufe und Multi-Agenten-Orchestrierung in externe Systemkomponenten aus.

JAZ verfolgt einen grundlegend anderen, minimalistischen Ansatz:

> **Ein Agent ist nichts anderes als eine gewöhnliche Funktion `invoke(...)`, deren Körper zur Laufzeit von einem LLM in einem Python-REPL geschrieben wird.**

Dieser Ansatz basiert auf zwei definierenden Eigenschaften:

1. **Rekursion als Default**: Innerhalb der REPL-Umgebung ist die Funktion `invoke(...)` selbst als globale Variable verfügbar. Wenn ein Agent ein Problem in Teilaufgaben zerlegen möchte, ruft er einfach `invoke(...)` auf. Sub-Agenten sind somit keine Sonderkonstrukte, sondern gewöhnliche Funktionsaufrufe innerhalb des Programmcodes.
2. **Alles Sichtbare ist eine Python-Variable**:
   - Alle Eingaben an `invoke(...)` (Auftrag, Datensätze, Werkzeuge/Tools) sind gleichwertig und liegen als benannte Variablen im persistenten Namespace der REPL.
   - Der Interaktionsverlauf selbst ist als Liste von Schritten in der Variable `__history__` verfügbar. Der Agent kann seine bisherigen Ausgaben und Codeschritte direkt per Python-Code durchsuchen oder an Sub-Agenten übergeben.
   - Beendet wird ein Auftrag durch ein gewöhnliches `return <wert>` auf oberster Ebene des generierten Codes.

---

## 2. Zentrale Konfiguration

Um maximale Portabilität und Sicherheit zu gewährleisten, sind alle Einstellungen an **einer einzigen zentralen Stelle** zusammengefasst, während Geheimnisse (API-Keys) strikt über `.env` getrennt werden.

### Geheimnisse in `.env`
Lokale Zugangsdaten werden in einer `.env`-Datei abgelegt (diese wird per `.gitignore` automatisch von der Versionskontrolle ausgeschlossen):
```env
JAZ_API_KEY=dein_api_schluessel
```

### Konfigurationsdatei `jaz.toml`
Die Datei `jaz.toml` im Arbeitsverzeichnis oder Projektordner definiert die Standardparameter. Umgebungsvariablen wie `${JAZ_API_KEY}` oder `${JAZ_BASE_URL}` können direkt referenziert werden:

```toml
[llm]
# Beliebiger OpenAI-kompatibler Endpunkt (z. B. vLLM, Ollama, LM Studio oder Cloud-APIs)
base_url = "${JAZ_BASE_URL:-http://localhost:8000/v1}"
model = "${JAZ_MODEL:-default}"
api_key = "${JAZ_API_KEY:-EMPTY}"
context_window = 32768
temperature = 0.0

[repl]
# Ausführungsmodus für den generierten Code:
# "direct"     = In-Process Ausführung
# "subprocess" = Isolation im Kindprozess
# "container"  = Härtung via Podman Desktop oder Docker
sandbox = "direct"

# Container-Einstellungen (nur aktiv bei sandbox = "container")
container_runtime = "podman"      # "podman" oder "docker"
container_image = "python:3.11-slim"
```

### Automatisches Bootstrapping & Self-Healing
1. **Beim Start**: `python run.py` prüft vorab, ob `httpx` und `rich` installiert sind, und installiert sie andernfalls automatisch via `pip`.
2. **Im Agenten-REPL**: Benötigt ein Agent oder Prompt externe Bibliotheken (`numpy`, `scipy`, etc.), fängt JAZ auftretende `ModuleNotFoundError` ab und installiert die Pakete on-demand zur Laufzeit nach. Zudem steht die Funktion `ensure_packages("package1", "package2")` bereit.

### Verbindung testen
Mit folgendem Befehl kann vorab geprüft werden, ob der konfigurierte LLM-Endpunkt erreichbar ist und antwortet:

```bash
python -m jaz check
```

---

## 3. Ausführung & Sicherheitsabschottung

> [!CAUTION]
> **Wichtige Klarstellung: Ist eine `.env`-Datei oder eine `venv` eine Sandbox?**  
> **Nein.** Eine `.env`-Datei ist lediglich eine Textdatei mit Schlüssel-Wert-Paaren und bietet keinerlei Laufzeit-Isolation. Eine virtuelle Umgebung (`venv`) isoliert ausschließlich Python-Paketinstallationen. Code, der in einer `venv` ausgeführt wird, hat **vollen Zugriff** auf das Dateisystem, das Netzwerk und alle Benutzerberechtigungen. Da LLMs beliebigen Python-Code generieren, ist eine echte Schutzschicht entscheidend.

JAZ stellt drei klar differenzierte Ausführungsmodi bereit:

| Modus | Isolationsgrad | Schutzwirkung | Voraussetzungen |
|---|---|---|---|
| **`direct`** | Keine Isolation | Führt Code direkt im Hauptprozess aus. Schnellste Übergabe von Python-Objekten. Geeignet für lokale Entwicklung und vertrauenswürdige Aufgaben. | Nur Python |
| **`subprocess`** | Mittlere Isolation | Startet einen eigenen Unterprozess in einem isolierten temporären Verzeichnis. Bereinigt Umgebungsvariablen (API-Keys und Secrets werden herausgefiltert). Schutz vor unbeabsichtigten Modifikationen. | Nur Python |
| **`container`** | Strenge Härtung | Startet den REPL-Worker in einem isolierten Linux-Container mittels **Podman Desktop** oder **Docker**. Kein Netzzugriff, nur gemounteter Arbeitsordner. | Podman oder Docker |

### Modus 1: Direct (`sandbox = "direct"`)
Der LLM-Code läuft per `exec()` direkt im Python-Prozess. Alle übergebenen Python-Objekte (wie DataFrames, Datenbankverbindungen, Callbacks) bleiben im gemeinsamen Speicher und können ohne Serialisierungs-Overhead manipuliert werden.

### Modus 2: Subprocess (`sandbox = "subprocess"`)
Startet im Hintergrund einen separaten Python-Worker-Prozess (`jaz.repl.rpc`):
1. **Arbeitsbereich**: Der Prozess wird in ein separates temporäres Arbeitsverzeichnis isoliert.
2. **Key-Schutz**: Alle sensiblen Umgebungsvariablen (`API_KEY`, `SECRET`, `TOKEN`) werden aus dem Unterprozess entfernt. Generierter LLM-Code kann somit nicht über `os.environ` sensible Host-Schlüssel auslesen.
3. **RPC-Brücke**: Werkzeuge und rekursive `invoke`-Aufrufe werden über Interprozess-Kommunikation (IPC) transparent an den Hauptprozess zurückdelegiert.

### Modus 3: Container (`sandbox = "container"`)
Unterstützt gleichermaßen **Podman Desktop** und **Docker**:
- In `jaz.toml` wird hierfür `container_runtime = "podman"` oder `"docker"` gesetzt.
- *Podman Desktop* läuft nativ ohne Root-Rechte (rootless) und benötigt keine Hintergrund-Dienste mit Administratorrechten.
- Der Container läuft mit `--network none`, sodass der generierte Code keine unkontrollierten Netzwerkverbindungen aufbauen kann. Der Projektcode wird schreibgeschützt eingebunden.

---

## 4. Terminal-Bedienung & Live-Status

JAZ verfügt über eine vollwertige Kommandozeilenschnittstelle (CLI) mit einer interaktiven, farbigen **Rich-Statusanzeige**.

### Einen Auftrag ausführen
```bash
python -m jaz run "Analysiere die Datei data.csv und gib den Mittelwert der Spalte 'score' zurück."
```

### Parameter und Werkzeuge übergeben
Eigene Werkzeuge (Funktionen) können in einer separaten Python-Datei definiert und übergeben werden:

```python
# tools.py
def datenbank_abfrage(kunden_id: int) -> dict:
    """Sucht Kundendaten anhand der ID."""
    return {"id": kunden_id, "name": "Musterkunde", "umsatz": 14500}
```

Aufruf im Terminal:
```bash
python -m jaz run "Finde den Umsatz von Kunde 42" \
  -t tools.py \
  --sandbox subprocess \
  --max-iterations 20
```

### Live-Dashboard im Terminal
Während der Ausführung zeigt JAZ ein übersichtliches Terminal-Dashboard:
- **Statusleiste**: Laufzeit, Token-Verbrauch (Input/Output getrennt), aktuelle Kosten.
- **Hierarchie-Baum**: Übersicht aller laufenden Agenten und Sub-Agenten inklusive Aufruftiefe (`depth`) und Status.
- **Aktiver Schritt**: Syntax-hervorgehobener Python-Code, den das LLM gerade formuliert hat, sowie die Konsolenausgabe des REPL.

---

## 5. Event-Stream & Vorbereitung für Web-Dashboards

Jeder Durchlauf legt im Ordner `runs/<run_id>/` zwei standardisierte Dateien an:

1. `events.jsonl`: Ein zeilenweiser JSON-Stream aller Ereignisse in Echtzeit.
2. `trajectory.json`: Die vollständige, geordnete Abfolge aller Schritte, Codes und Ausgaben.

### Aufbau der `events.jsonl`
Jedes Ereignis enthält einen Zeitstempel und strukturierte Daten:
```json
{"type": "invoke_enter", "timestamp": 1727952000.1, "invoke_id": "inv_a1b2c3", "depth": 1, "inputs": {"task": "str"}}
{"type": "llm_query_enter", "timestamp": 1727952000.2, "invoke_id": "inv_a1b2c3", "model": "qwen"}
{"type": "llm_query_exit", "timestamp": 1727952002.5, "invoke_id": "inv_a1b2c3", "input_tokens": 420, "output_tokens": 85}
{"type": "repl_exec_enter", "timestamp": 1727952002.6, "invoke_id": "inv_a1b2c3", "turn": 1, "code": "print(task)"}
{"type": "repl_exec_exit", "timestamp": 1727952002.7, "invoke_id": "inv_a1b2c3", "turn": 1, "stdout": "..."}
{"type": "invoke_exit", "timestamp": 1727952003.0, "invoke_id": "inv_a1b2c3", "duration": 2.9}
```

Dieses Format ist die ideale Schnittstelle für ein späteres grafisches Web-Dashboard (z. B. via WebSocket, Server-Sent Events oder Log-Streaming).

---

## 6. Praktische Muster für eigene Agenten-Aufträge

### Muster 1: Dynamisches Scoping (`scope`)
Mit `scope` machst du Werkzeuge oder Daten für den Hauptagenten **und alle rekursiven Sub-Agenten** verfügbar:

```python
from jaz import invoke, scope
from jaz.hooks import ReturnType, IterationLimit

def web_suche(begriff: str) -> str:
    """Durchsucht das Web nach aktuellen Informationen."""
    return f"Ergebnisse für {begriff}..."

with scope(web_suche=web_suche):
    # Alle Sub-Agenten können `web_suche` aufrufen, ohne dass es explizit weitergereicht werden muss
    ergebnis = invoke(
        ReturnType(str),
        IterationLimit(15),
        task="Recherchiere die drei wichtigsten Agenten-Frameworks und erstelle einen Vergleich.",
    )
```

### Muster 2: Langzeit-Gedächtnis via Tail-Recursive Delegation (StuLife-Muster)
Wenn das Kontextfenster vollzulaufen droht, delegiert der Agent an einen frischen Sub-Agenten und übergibt die Historie verlustfrei per Referenz:

```python
from jaz import invoke
from jaz.hooks import ContextWindowWarning

# Warnt den Agenten automatisch, sobald das Kontextfenster zu 75% gefüllt ist
warning_hook = ContextWindowWarning(threshold_tokens=25000)

ergebnis = invoke(
    warning_hook,
    task="Löse eine lange Abfolge von Teilaufgaben.",
    aufgaben_liste=meine_aufgaben,
)
```
Der Agent formuliert im REPL bei drohendem Überlauf:
```python
# Der Agent übergibt die Historie verlustfrei weiter:
return invoke(
    task=task,
    prev_history=globals().get("prev_history", []) + __history__,
    restliche_schritte=remaining_steps,
)
```

### Muster 3: Fortlaufende Selbstverbesserung (AppWorld Meta/Solver-Muster)
Ein übergeordneter Meta-Agent startet Solver-Sub-Agenten, begutachtet deren `__history__` und optimiert bei Fehlern den Prompt oder die Werkzeuge für nachfolgende Aufgaben:

```python
from jaz import invoke
from jaz.hooks import RecursionLimit

meta_auftrag = """
Löse die Aufgabenreihe. Starte für jede Aufgabe einen Subagenten:
  antwort, verlauf = invoke(task=aktuelle_aufgabe, tools=hilfsfunktionen)
Wenn die Aufgabe fehlschlägt, analysiere `verlauf` und passe die Anweisungen an.
"""

invoke(RecursionLimit(3), task=meta_auftrag, datensatz=aufgaben)
```

### Muster 4: Robuste Ausführung mit `PythonLinterHook` (Auto-Repair & Diagnostik)
Insbesondere bei kleineren oder instruktionssensiblen Modellen schützt der `PythonLinterHook` vor typischen Syntax- und Namensfehlern:

```python
from jaz import invoke
from jaz.hooks import PythonLinterHook

linter = PythonLinterHook(
    auto_fix=True,              # Bereinigt Markdown-Erklärungen vor/nach dem Code automatisch
    check_undefined_tools=True, # Prüft vorab, ob aufgerufene Werkzeuge im REPL existieren
    suggest_similar_names=True, # Macht Tippfehler-Vorschläge (z. B. "Did you mean web_search?")
    disallowed_modules=["subprocess", "os.system"], # Sicherheitsfilter gegen unerwünschte Module
)

# Im Code:
result = invoke(linter, task="Analysiere die Datei", datensatz=df)
```
In der CLI kann der Linter einfach mit `--linter` aktiviert werden:
```bash
python -m jaz run "Analysiere Daten" -t tools.py --linter
```

---

## 7. Testen & Spezifikationsprüfung

Die Implementierung enthält zwei Test-Suiten:

### 1. Unit-Tests (Deterministisch, ohne API-Kosten)
Prüft die gesamte Harness-Mechanik (Variablen, Hooks, Scoping, REPL-Return, Subprozess-RPC) mittels eines skriptgesteuerten Test-LLMs:
```bash
pytest tests/unit
```

### 2. Live-Spezifikationstests (Gegen einen konfigurierten LLM-Endpunkt)
Sobald ein aktiver LLM-Endpunkt konfiguriert ist, führen die Live-Tests reale Aufgaben aus:
```bash
pytest tests/live
```
Die Live-Tests validieren:
- Ob das Modell Variablen im REPL korrekt liest (`test_live_variable_reflection`)
- Ob rekursive Sub-Agenten sauber aufgerufen werden (`test_live_subagent_recursion`)
- Ob das Weiterreichen der Historie bei langen Aufgaben funktioniert (`test_live_tail_recursive_delegation`)
