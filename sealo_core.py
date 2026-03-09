"""
sealo_core.py
The brain/tools layer for Sealo 3.0.
All tools, the agentic loop, memory, and profile management live here.
"""

import os, sys, json, subprocess, datetime, urllib.request, urllib.parse
import threading, webbrowser, sqlite3, re, logging
from pathlib import Path
from dotenv import load_dotenv
from mistralai import Mistral
from tenacity import retry, stop_after_attempt, wait_exponential

# ── Vision & Control (optional) ─────────────────────────────────────
try:
    import mss
    import pyautogui
    from PIL import Image
    import io
    pyautogui.FAILSAFE = False # Prevent exceptions if mouse hits corner during automated moves
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

# ── Voice (optional) ────────────────────────────────────────────────
try:
    import pyttsx3, speech_recognition as sr
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

# ── Logging Configuration ───────────────────────────────────────────
SEALO_DIR = Path(__file__).parent
LOG_FILE = SEALO_DIR / "sealo.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("SealoCore")

load_dotenv(SEALO_DIR / ".env")
# Mistral uses explicit API key
api_key = os.getenv("MISTRAL_API_KEY")

client = None
MODEL_ID = "mistral-large-latest" 
OFFLINE_MODE = False

if not api_key:
    logger.error("No Mistral API key found. Set MISTRAL_API_KEY in .env")
    OFFLINE_MODE = True
else:
    logger.info("Initializing Mistral client...")
    client = Mistral(api_key=api_key)

SEALO_DIR    = Path(__file__).parent
MEMORY_FILE  = SEALO_DIR / "memory.json"
PROFILE_FILE = SEALO_DIR / "user_profile.json"

# Active DB connection (shared state)
_db_conn = None
_db_path = None

# ══════════════════════════════════════════════════════════════════════
#  MEMORY
# ══════════════════════════════════════════════════════════════════════

def load_memory():
    if not MEMORY_FILE.exists():
        return []
    try:
        raw = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        return raw # Mistral uses standard dicts for messages
    except Exception:
        return []

def save_memory(history):
    try:
        # Mistral return objects; convert to dicts for JSON
        serializable_history = []
        for msg in history:
            if hasattr(msg, "model_dump"): # Mistral SDK objects
                serializable_history.append(msg.model_dump())
            elif isinstance(msg, dict):
                serializable_history.append(msg)
            else:
                serializable_history.append(str(msg))
        
        if len(serializable_history) > 80:
            serializable_history = serializable_history[-80:]
        trimmed = serializable_history
        MEMORY_FILE.write_text(json.dumps(trimmed, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"Warning: Could not save memory: {e}")

# ══════════════════════════════════════════════════════════════════════
#  USER PROFILE
# ══════════════════════════════════════════════════════════════════════

def load_profile() -> dict:
    if PROFILE_FILE.exists():
        try:
            return json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"name": None, "occupation": None, "projects": [], "preferences": [], "skills": [], "notes": [], "databases": [], "last_seen": None}

def save_profile(profile: dict):
    PROFILE_FILE.write_text(json.dumps(profile, indent=2), encoding="utf-8")

def format_profile_for_prompt(profile: dict) -> str:
    lines = ["Known user info:"]
    if profile.get("name"):       lines.append(f"  Name: {profile['name']}")
    if profile.get("occupation"): lines.append(f"  Role: {profile['occupation']}")
    if profile.get("projects"):   lines.append(f"  Projects: {', '.join(profile['projects'])}")
    if profile.get("skills"):     lines.append(f"  Skills: {', '.join(profile['skills'])}")
    if profile.get("preferences"):lines.append(f"  Preferences: {', '.join(profile['preferences'])}")
    if profile.get("databases"):  lines.append(f"  Known databases: {', '.join(profile['databases'])}")
    if profile.get("notes"):      lines.append(f"  Notes: {'; '.join(profile['notes'][-5:])}")
    return "\n".join(lines) if len(lines) > 1 else "No user profile yet."

def build_system_prompt(profile: dict) -> str:
    profile_summary = format_profile_for_prompt(profile)
    return f"""You are Sealo, an extremely capable, proactive, and intelligent personal AI assistant.
You run directly on the user's Windows computer and have real tools to interact with it.

Your personality:
- You are sharp, confident, witty, and direct. You don't waste words.
- You treat the user as a smart person and never over-explain basics unless asked.
- You proactively use your tools without being asked — if someone asks about their files, just look. If they need code run, just run it.
- You learn about the user over time. When you discover new facts about them (name, job, projects, skills), call update_user_profile immediately.
- You think through multi-step problems before acting. For complex tasks, plan, execute, then verify.
- You reference past conversations naturally when relevant.
- You are honest about your limitations and when you're not sure of something.

Your capabilities:
- Browse and read/write files anywhere on the computer
- Execute real Python code and see the actual output
- Search the web for up-to-date information
- Open applications and URLs
- Run safe system commands for diagnostics
- Remember the user across sessions (persistent memory + profile)
- Speak and listen (if voice mode is on)

{profile_summary}

Current time: {datetime.datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}

Always be honest about what you know vs. what you're inferring. When taking actions on the user's computer, briefly say what you're doing."""

# ══════════════════════════════════════════════════════════════════════
#  VOICE
# ══════════════════════════════════════════════════════════════════════
_tts_engine = None
_tts_lock = threading.Lock()

def get_tts_engine():
    global _tts_engine
    if _tts_engine is None and VOICE_AVAILABLE:
        _tts_engine = pyttsx3.init()
        _tts_engine.setProperty("rate", 170)
        voices = _tts_engine.getProperty("voices")
        for v in voices:
            if "david" in v.name.lower() or "zira" in v.name.lower():
                _tts_engine.setProperty("voice", v.id); break
    return _tts_engine

def speak(text: str):
    def _go():
        with _tts_lock:
            e = get_tts_engine()
            if e:
                clean = re.sub(r"[*`#]", "", text)
                e.say(clean); e.runAndWait()
    threading.Thread(target=_go, daemon=True).start()

def listen_from_mic() -> str:
    if not VOICE_AVAILABLE:
        return ""
    rec = sr.Recognizer()
    with sr.Microphone() as src:
        rec.adjust_for_ambient_noise(src, duration=0.5)
        try:
            audio = rec.listen(src, timeout=5, phrase_time_limit=15)
            return rec.recognize_google(audio)
        except Exception:
            return ""

# ══════════════════════════════════════════════════════════════════════
#  TOOLS
# ══════════════════════════════════════════════════════════════════════

def get_current_time() -> str:
    return datetime.datetime.now().strftime("Current time: %A, %B %d, %Y at %I:%M:%S %p")

def list_directory(path: str) -> str:
    try:
        if not os.path.isdir(path):
            return f"Error: '{path}' is not a valid directory."
        items = sorted(os.listdir(path))
        return f"Contents of '{path}':\n" + "\n".join(
            f"  {'[DIR] ' if os.path.isdir(os.path.join(path, i)) else '[FILE]'} {i}" for i in items
        )
    except Exception as e:
        return f"Error: {e}"

def read_file(path: str) -> str:
    try:
        p = Path(path)
        if not p.is_file():
            return f"Error: '{path}' not found."
        content = p.read_text(encoding="utf-8", errors="replace")
        if len(content) > 8000:
            content = content[:8000] + "\n\n[... truncated ...]"
        return f"Content of '{path}':\n```\n{content}\n```"
    except Exception as e:
        return f"Error: {e}"

def write_file(path: str, content: str) -> str:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to '{path}'."
    except Exception as e:
        return f"Error: {e}"

def execute_python(code: str, save_as: str = "") -> str:
    try:
        tmp = SEALO_DIR / "_temp_script.py"
        tmp.write_text(code, encoding="utf-8")
        r = subprocess.run([sys.executable, str(tmp)], capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace")
        out = r.stdout.strip(); err = r.stderr.strip()
        parts = []
        if out: parts.append(f"Output:\n{out}")
        if err: parts.append(f"Errors:\n{err}")
        result = "\n\n".join(parts) or "Ran successfully, no output."
        if save_as:
            Path(save_as).write_text(code, encoding="utf-8")
            result += f"\n\nSaved to '{save_as}'."
        return result
    except subprocess.TimeoutExpired:
        return "Error: Script timed out (30s)."
    except Exception as e:
        return f"Error: {e}"

# ── SQL Tools ─────────────────────────────────────────────────────────

def connect_database(path_or_connection_string: str) -> str:
    """Connect to a SQLite database file (or future: SQL Server / MySQL via connection string)."""
    global _db_conn, _db_path
    try:
        # SQLite path
        if path_or_connection_string.endswith(".db") or path_or_connection_string.endswith(".sqlite") \
                or os.path.isfile(path_or_connection_string):
            _db_conn = sqlite3.connect(path_or_connection_string, check_same_thread=False)
            _db_path = path_or_connection_string
            # Save to user profile
            profile = load_profile()
            if path_or_connection_string not in profile.get("databases", []):
                profile.setdefault("databases", []).append(path_or_connection_string)
                save_profile(profile)
            # List tables immediately
            cursor = _db_conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tables = [row[0] for row in cursor.fetchall()]
            return f"Connected to SQLite database: '{path_or_connection_string}'\nTables found: {', '.join(tables) if tables else '(none)'}"
        # Try sqlalchemy for other database types
        try:
            from sqlalchemy import create_engine, text, inspect
            engine = create_engine(path_or_connection_string)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            # Store engine reference for query use
            _db_conn = engine
            _db_path = path_or_connection_string
            profile = load_profile()
            if path_or_connection_string not in profile.get("databases", []):
                profile.setdefault("databases", []).append(path_or_connection_string[:60])
                save_profile(profile)
            return f"Connected to database via SQLAlchemy.\nTables: {', '.join(tables) if tables else '(none)'}"
        except ImportError:
            return "SQLAlchemy not installed. For non-SQLite databases, install it with: pip install sqlalchemy"
        except Exception as e:
            return f"Connection failed: {e}"
    except Exception as e:
        return f"Error connecting: {e}"

def list_tables() -> str:
    if _db_conn is None:
        return "No database connected. Use connect_database first."
    try:
        if isinstance(_db_conn, sqlite3.Connection):
            cur = _db_conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tables = [r[0] for r in cur.fetchall()]
        else:
            from sqlalchemy import inspect
            tables = inspect(_db_conn).get_table_names()
        return f"Tables in '{_db_path}':\n" + "\n".join(f"  - {t}" for t in tables) if tables else "No tables found."
    except Exception as e:
        return f"Error: {e}"

def describe_table(table_name: str) -> str:
    if _db_conn is None:
        return "No database connected."
    try:
        if isinstance(_db_conn, sqlite3.Connection):
            cur = _db_conn.cursor()
            cur.execute(f"PRAGMA table_info({table_name});")
            cols = cur.fetchall()
            if not cols:
                return f"Table '{table_name}' not found."
            lines = [f"Schema for '{table_name}':"]
            lines += [f"  - {c[1]} ({c[2]}) {'NOT NULL' if c[3] else ''} {'PK' if c[5] else ''}" for c in cols]
            cur.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cur.fetchone()[0]
            lines.append(f"\nRow count: {count:,}")
            return "\n".join(lines)
        else:
            from sqlalchemy import inspect, text
            insp = inspect(_db_conn)
            cols = insp.get_columns(table_name)
            lines = [f"Schema for '{table_name}':"] + [f"  - {c['name']} ({c['type']})" for c in cols]
            with _db_conn.connect() as conn:
                row = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).fetchone()
                lines.append(f"\nRow count: {row[0]:,}")
            return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"

def run_sql_query(query: str) -> str:
    if _db_conn is None:
        return "No database connected. Use connect_database first."
    # Safety: only allow SELECT statements (warn on others)
    stripped = query.strip().upper()
    if not stripped.startswith("SELECT") and not stripped.startswith("WITH") and not stripped.startswith("PRAGMA"):
        return "Safety: Only SELECT queries are allowed. For data-modifying queries, please run them directly in your database tool."
    try:
        if isinstance(_db_conn, sqlite3.Connection):
            cur = _db_conn.cursor()
            cur.execute(query)
            rows = cur.fetchmany(200)  # cap at 200 rows
            cols = [d[0] for d in cur.description] if cur.description else []
        else:
            from sqlalchemy import text
            with _db_conn.connect() as conn:
                result = conn.execute(text(query))
                cols = list(result.keys())
                rows = result.fetchmany(200)

        if not rows:
            return "Query returned no rows."

        # Format as a clean table string
        col_widths = [max(len(str(c)), max((len(str(r[i])) for r in rows), default=0)) for i, c in enumerate(cols)]
        header = " | ".join(str(c).ljust(w) for c, w in zip(cols, col_widths))
        sep    = "-+-".join("-" * w for w in col_widths)
        data_rows = [" | ".join(str(r[i] if r[i] is not None else "NULL").ljust(w) for i, w in enumerate(col_widths)) for r in rows]
        result = f"Query results ({len(rows)} row{'s' if len(rows) != 1 else ''}):\n{header}\n{sep}\n" + "\n".join(data_rows)
        if len(rows) == 200:
            result += "\n\n[Results capped at 200 rows]"
        return result
    except Exception as e:
        return f"SQL Error: {e}"

def get_db_status() -> dict:
    """Returns current DB connection info for the GUI to display."""
    if _db_conn is None:
        return {"connected": False, "path": None, "tables": []}
    try:
        if isinstance(_db_conn, sqlite3.Connection):
            cur = _db_conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tables = [r[0] for r in cur.fetchall()]
        else:
            from sqlalchemy import inspect
            tables = inspect(_db_conn).get_table_names()
        return {"connected": True, "path": _db_path, "tables": tables}
    except Exception:
        return {"connected": False, "path": None, "tables": []}

# ── Web & System ──────────────────────────────────────────────────────

def web_search(query: str) -> str:
    try:
        encoded = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Sealo-3.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = []
        if data.get("Answer"):       results.append(f"**Quick Answer:** {data['Answer']}")
        if data.get("AbstractText"): results.append(f"**Summary:** {data['AbstractText']}")
        if data.get("AbstractURL"):  results.append(f"**Source:** {data['AbstractURL']}")
        for t in data.get("RelatedTopics", [])[:6]:
            if isinstance(t, dict) and t.get("Text"):
                results.append(f"- {t['Text']}")
                if t.get("FirstURL"): results.append(f"  → {t['FirstURL']}")
        return (f"Search: **{query}**\n\n" + "\n".join(results)) if results else f"No results for '{query}'."
    except Exception as e:
        return f"Search error: {e}"

def open_application(name_or_path: str) -> str:
    apps = {"notepad": "notepad.exe", "calculator": "calc.exe", "explorer": "explorer.exe",
            "chrome": "chrome", "edge": "msedge", "word": "WINWORD.EXE", "excel": "EXCEL.EXE",
            "powerpoint": "POWERPNT.EXE", "vscode": "code", "terminal": "wt.exe",
            "cmd": "cmd.exe", "task manager": "taskmgr.exe", "paint": "mspaint.exe", "spotify": "spotify"}
    target = apps.get(name_or_path.lower().strip(), name_or_path)
    try:
        subprocess.Popen(target, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"Launched '{name_or_path}'."
    except Exception as e:
        return f"Error: {e}"

def open_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    webbrowser.open(url)
    return f"Opened {url} in browser."

def update_user_profile(field: str, value: str) -> str:
    profile = load_profile()
    list_fields = ["projects", "skills", "preferences", "notes", "databases"]
    if field in list_fields:
        if value not in profile.get(field, []):
            profile.setdefault(field, []).append(value)
            save_profile(profile)
            return f"Added '{value}' to {field}."
        return f"'{value}' already in {field}."
    elif field in profile:
        profile[field] = value
        save_profile(profile)
        return f"Updated {field} to '{value}'."
    return f"Unknown field '{field}'."

def run_command(command: str) -> str:
    blocked = ["format ", "shutdown"] # Removed 'del', 'rm' to give it agentic power if needed, but keeping safety blocks for destructive system commands
    for b in blocked:
        if b.lower() in command.lower():
            return f"Blocked: '{b.strip()}' is not allowed."
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=15, encoding="utf-8", errors="replace")
        out = r.stdout.strip() or r.stderr.strip()
        return out[:4000] or "(No output)"
    except subprocess.TimeoutExpired:
        return "Timed out."
    except Exception as e:
        return f"Error: {e}"

def run_terminal_command(command: str, cwd: str = ".") -> str:
    """Executes a terminal/PowerShell command on the host machine and returns the output.
    USE WITH CAUTION. This gives Sealo powerful access to the user's system."""
    try:
        # Default to powershell on Windows to have access to full terminal features
        process = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30, # longer timeout for complex commands
            encoding="utf-8",
            errors="replace"
        )
        output = process.stdout.strip()
        if process.stderr:
            output += f"\nSTDERR:\n{process.stderr.strip()}"
        if not output.strip():
            output = f"Command executed successfully with exit code {process.returncode} (No output)"
        return output[:8000] # Cap output
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        return f"Error executing terminal command: {str(e)}"

def modify_file_content(path: str, target_text: str, replacement_text: str) -> str:
    """Finds exact target_text in the file at path and replaces it with replacement_text."""
    try:
        p = Path(path).resolve()
        if not p.exists():
            return f"Error: File '{path}' does not exist."
        content = p.read_text(encoding="utf-8")
        if target_text not in content:
            return f"Error: The target text to replace was not found in '{path}'. Please provide EXACT text."
        
        new_content = content.replace(target_text, replacement_text)
        p.write_text(new_content, encoding="utf-8")
        return f"Successfully modified '{path}'. Replaced text."
    except Exception as e:
        return f"Error modifying file: {str(e)}"

# ── Vision & System Control ───────────────────────────────────────────

def analyze_screen(prompt: str = "Explain what is on the screen right now.") -> str:
    """Takes a screenshot and passes it to Mistral Vision (base64)."""
    print(f"DEBUG: execute analyze_screen with prompt: {prompt}")
    if not VISION_AVAILABLE or not client:
        return "Error: Vision tools or Mistral client unavailable."
    try:
        import base64
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            sct_img = sct.grab(monitor)
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            
            # Save to buffer
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

            # Mistral Vision request
            resp = client.chat.complete(
                model=MODEL_ID,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64_img}"}
                        ]
                    }
                ]
            )
            return f"Screenshot analyzed. Result: {resp.choices[0].message.content}"
    except Exception as e:
        return f"Error analyzing screen: {e}"

def analyze_image(path: str, prompt: str = "Analyze this image.") -> str:
    """Passes a local image file to Mistral Vision."""
    print(f"DEBUG: execute analyze_image with path: {path}")
    if not client:
        return "Error: Mistral client unavailable."
    
    img_path = Path(path)
    if not img_path.exists():
        return f"Error: File not found at {path}"

    try:
        import base64
        with open(img_path, "rb") as image_file:
            b64_img = base64.b64encode(image_file.read()).decode("utf-8")
        
        # Determine mime type roughly
        ext = img_path.suffix.lower()
        mime = "image/jpeg"
        if ext == ".png": mime = "image/png"
        elif ext == ".webp": mime = "image/webp"

        resp = client.chat.complete(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": f"data:{mime};base64,{b64_img}"}
                    ]
                }
            ]
        )
        return f"Image analyzed: {path}\nResult: {resp.choices[0].message.content}"
    except Exception as e:
        logger.error(f"Error in analyze_image: {e}")
        return f"Error analyzing image: {e}"

def move_mouse(x: int, y: int) -> str:
    if not VISION_AVAILABLE: return "Error: pyautogui not installed."
    try:
        pyautogui.moveTo(x, y, duration=0.2)
        return f"Moved mouse to ({x}, {y})."
    except Exception as e:
        return f"Error moving mouse: {e}"

def click(x: int = None, y: int = None, button: str = "left", clicks: int = 1) -> str:
    if not VISION_AVAILABLE: return "Error: pyautogui not installed."
    try:
        if x is not None and y is not None:
            pyautogui.click(x=x, y=y, button=button, clicks=clicks)
            loc_str = f" at ({x}, {y})"
        else:
            pyautogui.click(button=button, clicks=clicks)
            loc_str = " at current position"
        return f"Clicked {button} {clicks} time(s){loc_str}."
    except Exception as e:
        return f"Error clicking: {e}"

def type_text(text: str, interval: float = 0.02) -> str:
    if not VISION_AVAILABLE: return "Error: pyautogui not installed."
    try:
        pyautogui.write(text, interval=interval)
        return f"Typed: '{text}'."
    except Exception as e:
        return f"Error typing: {e}"

def press_key(key: str, times: int = 1) -> str:
    if not VISION_AVAILABLE: return "Error: pyautogui not installed."
    try:
        pyautogui.press(key, presses=times)
        return f"Pressed '{key}' {times} time(s)."
    except Exception as e:
        return f"Error pressing key: {e}"

# ── TOOLS ─────────────────────────────────────────────────────────────

MISTRAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local date and time.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files/folders in a directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Directory path."}},
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file's content.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path."}},
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write/create a text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path."},
                    "content": {"type": "string", "description": "Content to write."}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "modify_file_content",
            "description": "Modify an existing file by replacing exact target text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path."},
                    "target_text": {"type": "string", "description": "Exact text to replace."},
                    "replacement_text": {"type": "string", "description": "New text."}
                },
                "required": ["path", "target_text", "replacement_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Write and EXECUTE Python code. Returns real output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code."},
                    "save_as": {"type": "string", "description": "Optional path to save."}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet via DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_screen",
            "description": "Take a screenshot and explain what is visible (Vision).",
            "parameters": {
                "type": "object",
                "properties": {"prompt": {"type": "string", "description": "What to look for."}}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_application",
            "description": "Launch a Windows application.",
            "parameters": {
                "type": "object",
                "properties": {"name_or_path": {"type": "string"}},
                "required": ["name_or_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_terminal_command",
            "description": "Run a PowerShell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_user_profile",
            "description": "Update user's name, hobbies, or info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "value": {"type": "string"}
                },
                "required": ["field", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": "Analyze a local image file (Vision).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Local path to image file."},
                    "prompt": {"type": "string", "description": "What to look for in the image."}
                },
                "required": ["path"]
            }
        }
    }
]
# ... more tools could be added following this pattern

TOOL_MAP = {
    "get_current_time": get_current_time,
    "list_directory": list_directory,
    "read_file": read_file,
    "write_file": write_file,
    "modify_file_content": modify_file_content,
    "execute_python": execute_python,
    "connect_database": connect_database,
    "list_tables": list_tables,
    "describe_table": describe_table,
    "run_sql_query": run_sql_query,
    "web_search": web_search,
    "open_application": open_application,
    "open_url": open_url,
    "update_user_profile": update_user_profile,
    "run_command": run_command,
    "run_terminal_command": run_terminal_command,
    "analyze_screen": analyze_screen,
    "analyze_image": analyze_image,
    "move_mouse": move_mouse,
    "click": click,
    "type_text": type_text,
    "press_key": press_key,
}

# ══════════════════════════════════════════════════════════════════════
#  SEALO AGENT (Mistral Best Practices)
# ══════════════════════════════════════════════════════════════════════

class SealoAgent:
    def __init__(self, api_key: str, model_id: str = "mistral-large-latest"):
        self.api_key = api_key
        self.model_id = model_id
        if api_key:
            self.client = Mistral(api_key=api_key)
        else:
            self.client = None
        
        self.history = []
        self.system_prompt = ""
        self.tools = MISTRAL_TOOLS
        self.tool_map = TOOL_MAP
        self._stop_event = threading.Event()

    def stop(self):
        """Signal the agent to stop."""
        self._stop_event.set()

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt

    def load_history(self, history: list):
        # Sanitize for Mistral format
        self.history = []
        for msg in history:
            if isinstance(msg, dict) and msg.get("role") in ["user", "assistant"]:
                self.history.append({"role": msg["role"], "content": str(msg["content"])})

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_mistral(self, messages, tools=None):
        if not self.client:
            raise Exception("Mistral client not initialized. Check your API key.")

        try:
            response = self.client.chat.complete(
                model=self.model_id,
                messages=messages,
                tools=tools,
                tool_choice="auto" if tools else None
            )
            return response
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Mistral API error: {error_msg}")
            if "400" in error_msg and "API key" in error_msg:
                raise Exception("Invalid Mistral API key. Update your .env file.")
            elif "403" in error_msg:
                raise Exception("API key rejected. Check permissions or quota.")
            else:
                raise Exception(f"Mistral API error: {error_msg}")

    def chat(self, user_input: str, on_tool_call=None) -> str:
        """Main entry point for chatting with the agent."""
        if user_input:
            self.history.append({"role": "user", "content": user_input})

        # Prepend system prompt
        messages = [{"role": "system", "content": self.system_prompt}] + self.history
        self._stop_event.clear()

        while True:
            if self._stop_event.is_set():
                logger.info("Agent stop event triggered.")
                return "Activity cancelled by user."

            try:
                response = self._call_mistral(messages, tools=self.tools)
            except Exception as e:
                return f"Error calling Mistral: {e}"

            message = response.choices[0].message
            messages.append(message)
            self.history.append(message) # Keep track for next turns

            if not message.tool_calls:
                break

            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                import json
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except Exception:
                    fn_args = {}
                
                # Type sanitization (best practice)
                if fn_name == "click":
                    if "clicks" in fn_args: fn_args["clicks"] = int(fn_args["clicks"])
                    if "x" in fn_args and fn_args.get("x") is not None: fn_args["x"] = int(fn_args["x"])
                    if "y" in fn_args and fn_args.get("y") is not None: fn_args["y"] = int(fn_args["y"])
                elif fn_name == "type_text":
                    if "interval" in fn_args: fn_args["interval"] = float(fn_args["interval"])
                elif fn_name == "press_key":
                    if "times" in fn_args: fn_args["times"] = int(fn_args["times"])

                fn = self.tool_map.get(fn_name)
                result = fn(**fn_args) if fn else "Tool not found"
                
                if on_tool_call:
                    on_tool_call(fn_name, fn_args, result)
                
                tool_msg = {
                    "role": "tool",
                    "name": fn_name,
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                }
                logger.debug(f"Tool {fn_name} result: {result}")
                messages.append(tool_msg)
                
                # Convert to dict if it's a message object before appending to history
                if hasattr(tool_msg, "model_dump"):
                    self.history.append(tool_msg.model_dump())
                else:
                    self.history.append(tool_msg)

        # Return final content and update history to keep it clean (dicts only)
        final_content = messages[-1].content
        
        # Trim history to keep it performant
        if len(self.history) > 100:
            self.history = self.history[-100:]
        
        return final_content

def run_agent_loop(history: list, system_prompt: str, on_tool_call=None):
    """Legacy wrapper for backward compatibility."""
    agent = SealoAgent(api_key=api_key, model_id=MODEL_ID)
    agent.set_system_prompt(system_prompt)
    agent.load_history(history)
    
    # We add the last user message manually because load_history sanitizes
    # But wait, sealo.py/sealo_gui.py already append the user message to history.
    # So if we load it, it's already there. 
    # EXCEPT we need to return the updated history in the old format.
    
    # Mistral chat() handles the loop
    # In legacy mode, we don't pass user_input to chat() because it's already in history
    final_text = agent.chat("", on_tool_call=on_tool_call)
    
    # Convert history back to dicts for legacy callers
    serializable = []
    for m in agent.history:
        if hasattr(m, "model_dump"):
            serializable.append(m.model_dump())
        elif isinstance(m, dict):
            serializable.append(m)
            
    return final_text, serializable
