"""
sealo_core.py — The "Brain" of Sealo AI Assistant (v3.0)
========================================================
This module contains the core logic for the Sealo AI assistant. It handles:
1. LLM API Communication (via the OpenAI SDK pointed at Groq Cloud)
2. Tool execution and definition (the things Sealo can actually *do*)
3. Persistent memory (saving chat history to JSON)
4. User profile management (saving facts about the user to JSON)
5. Optional Voice/Vision features (TTS, microphone, screenshots)

Architecture Details for Reviewers:
- We use the `openai` Python package because Groq's API is fully OpenAI-compatible.
  This allows us to seamlessly swap between OpenAI, Groq, or local Ollama just by changing the `base_url`.
- Every tool (action) is decorated with `@sealo_tool`, which handles logging and catches
  crashes so that a failed tool doesn't crash the entire AI loop.
- The `TOOLS` list at the bottom defines the strict JSON Schema for what arguments
  each tool requires using OpenAI's function calling standard.
- The `run_agent_loop` function is the core engine: it sends a prompt -> gets a tool call ->
  executes the Python function -> sends the result back -> gets the final human-readable answer.

Pi Portability Notes:
- To run this on a Raspberry Pi, ensure SQlite3 is installed (usually default in Python).
- If running headless without GUI, the vision tools (pyautogui, mss, PIL) will fail.
  Our `try/except` imports gracefully handle this by setting `VISION_AVAILABLE = False`.
- The Groq API requires internet access. If the Pi will be offline, you can swap the
  `base_url` back to `localhost:11434` and run a lightweight model on Ollama.
"""

import os, sys, json, subprocess, datetime, urllib.request, urllib.parse
import threading, webbrowser, sqlite3, re
from pathlib import Path
from dotenv import load_dotenv
import logging
import functools
import traceback
from openai import OpenAI

# ── Logging Setup ────────────────────────────────────────────────────
# We write debug logs to sealo_debug.log to trace tool execution and API errors
SEALO_DIR = Path(__file__).parent
log_file = SEALO_DIR / "sealo_debug.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)
logger = logging.getLogger("sealo_core")

# ── Vision & Control (optional) ─────────────────────────────────────
# We wrap these imports in try/except so the script still works on headless
# servers (like a Raspberry Pi without a desktop environment).
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
# Same as above, if pyttsx3/SpeechRecognition aren't installed, voice is disabled.
try:
    import pyttsx3, speech_recognition as sr
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

# ── Initialization ───────────────────────────────────────────────────
# Load environment variables (.env file in the same folder)
load_dotenv(Path(__file__).parent / ".env")

# Try Gemini first (generous free tier: 1500 req/day), fallback to Groq
gemini_key = os.getenv("GEMINI_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

if groq_key:
    # Groq: fast, works right now. Rate-limited at ~30 req/min on free tier.
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key, max_retries=0)
    MODEL_ID = "llama-3.1-8b-instant"
    LLM_PROVIDER = "groq"
elif gemini_key:
    # Gemini: generous 1500 req/day free tier, use when Groq key is removed
    client = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=gemini_key)
    MODEL_ID = "gemini-2.0-flash"
    LLM_PROVIDER = "gemini"
else:
    raise RuntimeError("No API key found. Set GROQ_API_KEY or GEMINI_API_KEY in .env")

SEALO_DIR    = Path(__file__).parent
MEMORY_FILE  = SEALO_DIR / "memory.json"
PROFILE_FILE = SEALO_DIR / "user_profile.json"

class DatabaseManager:
    """
    Encapsulates active database connection state.
    This avoids global variables and makes it easier to track which DB is open.
    """
    def __init__(self):
        self.conn = None
        self.path = None

    def set_connection(self, conn, path):
        self.conn = conn
        self.path = path

db_manager = DatabaseManager()

# ══════════════════════════════════════════════════════════════════════
#  DECORATORS 
# ══════════════════════════════════════════════════════════════════════

def sealo_tool(func):
    """
    Decorator wrapped around every tool function.
    Safely executes the tool, logs its arguments, and catches exceptions
    so the AI receives the error text instead of crashing the Python process.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tool_name = func.__name__
        try:
            logger.debug(f"Executing tool {tool_name} with args={args} kwargs={kwargs}")
            result = func(*args, **kwargs)
            logger.info(f"Tool {tool_name} executed successfully.")
            return result
        except Exception as e:
            logger.error(f"Error in {tool_name}: {e}", exc_info=True)
            return f"Error executing {tool_name}: {e}"
    return wrapper

# ══════════════════════════════════════════════════════════════════════
#  MEMORY MANAGEMENT
# ══════════════════════════════════════════════════════════════════════

def load_memory():
    """Reads the JSON array of past conversation messages."""
    if not MEMORY_FILE.exists():
        return []
    try:
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []

def save_memory(history):
    """Saves the last 80 messages to disk to preserve token limits."""
    try:
        trimmed = history[-80:]
        MEMORY_FILE.write_text(json.dumps(trimmed, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"Warning: Could not save memory: {e}")

# ══════════════════════════════════════════════════════════════════════
#  USER PROFILE
# ══════════════════════════════════════════════════════════════════════

def load_profile() -> dict:
    """Loads factual assertions about the user that the AI has learned over time."""
    if PROFILE_FILE.exists():
        try:
            return json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"name": None, "occupation": None, "projects": [], "preferences": [], "skills": [], "notes": [], "databases": [], "last_seen": None}

def save_profile(profile: dict):
    PROFILE_FILE.write_text(json.dumps(profile, indent=2), encoding="utf-8")

def format_profile_for_prompt(profile: dict) -> str:
    """Converts the JSON profile into a plain text block injected into the System Prompt."""
    lines = ["Known user info:"]
    if profile.get("name"):       lines.append(f"  Name: {profile['name']}")
    if profile.get("occupation"): lines.append(f"  Role: {profile['occupation']}")
    if profile.get("projects"):   lines.append(f"  Projects: {', '.join(profile['projects'])}")
    if profile.get("skills"):     lines.append(f"  Skills: {', '.join(profile['skills'])}")
    if profile.get("preferences"):lines.append(f"  Preferences: {', '.join(profile['preferences'])}")
    if profile.get("databases"):  lines.append(f"  Known databases: {', '.join(profile['databases'])}")
    if profile.get("notes"):      lines.append(f"  Notes: {'; '.join(profile['notes'][-5:])}")
    return "\n".join(lines) if len(lines) > 1 else "No user profile yet."

# ══════════════════════════════════════════════════════════════════════
#  VOICE ENGINE
# ══════════════════════════════════════════════════════════════════════
class VoiceManager:
    """Encapsulates TTS engine state to avoid globals and thread issues."""
    def __init__(self):
        self.engine = None
        self.lock = threading.Lock()
        self.available = VOICE_AVAILABLE
        
    def get_engine(self):
        # Lazy initialization of pyttsx3
        if self.engine is None and self.available:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", 170)
                voices = self.engine.getProperty("voices")
                # Attempt to find common Windows voices
                for v in voices:
                    if "david" in v.name.lower() or "zira" in v.name.lower():
                        self.engine.setProperty("voice", v.id)
                        break
            except Exception:
                self.available = False
        return self.engine

    def speak(self, text: str):
        if not self.available:
            return
            
        def _go():
            with self.lock:
                e = self.get_engine()
                if e:
                    # Strip markdown symbols before speaking
                    clean = re.sub(r"[*`#]", "", text)
                    try:
                        e.say(clean)
                        e.runAndWait()
                    except Exception:
                        pass # Ignore TTS errors during runtime
        
        # Run in background daemon so the CLI doesn't block while speaking
        threading.Thread(target=_go, daemon=True).start()

voice_manager = VoiceManager()

def speak(text: str):
    """Facade for backward compatibility with older gui components"""
    voice_manager.speak(text)

def listen_from_mic() -> str:
    """Uses SpeechRecognition and PyAudio to transcribe 15 seconds of mic input."""
    if not VOICE_AVAILABLE:
        return ""
    rec = sr.Recognizer()
    with sr.Microphone() as src:
        rec.adjust_for_ambient_noise(src, duration=0.5)
        try:
            audio = rec.listen(src, timeout=5, phrase_time_limit=15)
            return rec.recognize_google(audio) # Requires internet access
        except Exception:
            return ""

# ══════════════════════════════════════════════════════════════════════
#  TOOLS (ACTIONS THE AI CAN TAKE)
# ══════════════════════════════════════════════════════════════════════

@sealo_tool
def get_current_time() -> str:
    """Returns the current date and time."""
    return datetime.datetime.now().strftime("Current time: %A, %B %d, %Y at %I:%M:%S %p")

@sealo_tool
def list_directory(path: str) -> str:
    """Lists files and folders at the given path."""
    if not os.path.isdir(path):
        return f"Error: '{path}' is not a valid directory."
    items = sorted(os.listdir(path))
    return f"Contents of '{path}':\n" + "\n".join(
        f"  {'[DIR] ' if os.path.isdir(os.path.join(path, i)) else '[FILE]'} {i}" for i in items
    )

@sealo_tool
def read_file(path: str) -> str:
    """Reads text file contents. Truncates big files at 8000 chars to save tokens."""
    p = Path(path)
    if not p.is_file():
        return f"Error: '{path}' not found."
    content = p.read_text(encoding="utf-8", errors="replace")
    if len(content) > 8000:
        content = content[:8000] + "\n\n[... truncated ...]"
    return f"Content of '{path}':\n```\n{content}\n```"

@sealo_tool
def write_file(path: str, content: str) -> str:
    """Creates a new file or overwrites an existing one."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to '{path}'."

@sealo_tool
def execute_python(code: str, save_as: str = "") -> str:
    """
    DANGEROUS BUT POWERFUL: Writes arbitrary Python code generated by the LLM
    to a temp file, runs it, and captures the stdout/stderr.
    This allows Sealo to do math, make charts, web scrape, etc autonomously.
    """
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

# ── SQL Database Tools ─────────────────────────────────────────────────────────

@sealo_tool
def connect_database(path_or_connection_string: str) -> str:
    """Connects to a SQLite database file or a SQL Server/Postgres string via SQLAlchemy."""
    
    # Simple SQLite path handling
    if path_or_connection_string.endswith(".db") or path_or_connection_string.endswith(".sqlite") \
            or os.path.isfile(path_or_connection_string):
        conn = sqlite3.connect(path_or_connection_string, check_same_thread=False)
        db_manager.set_connection(conn, path_or_connection_string)
        
        # Save historical DB paths to user profile so Sealo remembers them tomorrow
        profile = load_profile()
        if path_or_connection_string not in profile.get("databases", []):
            profile.setdefault("databases", []).append(path_or_connection_string)
            save_profile(profile)
            
        # Return a preemptive list of tables to save the LLM a step
        if hasattr(db_manager.conn, "cursor"):
            cursor = db_manager.conn.cursor() # type: ignore
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tables = [row[0] for row in cursor.fetchall()]
        else:
            tables = []
        return f"Connected to SQLite database: '{path_or_connection_string}'\nTables found: {', '.join(tables) if tables else '(none)'}"
    
    # Advanced SQLAlchemy handling for other database types (if installed)
    try:
        from sqlalchemy import create_engine, text, inspect
        engine = create_engine(path_or_connection_string)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        db_manager.set_connection(engine, path_or_connection_string)
        
        profile = load_profile()
        if path_or_connection_string not in profile.get("databases", []):
            profile.setdefault("databases", []).append(path_or_connection_string[:60])
            save_profile(profile)
            
        return f"Connected to database via SQLAlchemy.\nTables: {', '.join(tables) if tables else '(none)'}"
    except ImportError:
        return "SQLAlchemy not installed. For non-SQLite databases, install it with: pip install sqlalchemy"

@sealo_tool
def list_tables() -> str:
    """Lists tables in the currently connected database."""
    if db_manager.conn is None:
        return "No database connected. Use connect_database first."
    
    if hasattr(db_manager.conn, "cursor"): # SQLite check
        cur = db_manager.conn.cursor() # type: ignore
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = [r[0] for r in cur.fetchall()]
    else:
        from sqlalchemy import inspect
        tables = inspect(db_manager.conn).get_table_names()
    return f"Tables in '{db_manager.path}':\n" + "\n".join(f"  - {t}" for t in tables) if tables else "No tables found."

@sealo_tool
def describe_table(table_name: str) -> str:
    """Returns the schema/columns of a given database table."""
    if db_manager.conn is None:
        return "No database connected."
    
    if hasattr(db_manager.conn, "cursor"): # SQLite check
        cur = db_manager.conn.cursor() # type: ignore
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
        insp = inspect(db_manager.conn) # type: ignore
        cols = insp.get_columns(table_name)
        lines = [f"Schema for '{table_name}':"] + [f"  - {c['name']} ({c['type']})" for c in cols]
        with db_manager.conn.connect() as conn: # type: ignore
            row = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).fetchone()
            lines.append(f"\nRow count: {row[0]:,}")
        return "\n".join(lines)

@sealo_tool
def run_sql_query(query: str) -> str:
    """Executes a SQL SELECT query and parses the output into a markdown table string."""
    if db_manager.conn is None:
        return "No database connected. Use connect_database first."
    
    # Safety: only allow standard read queries to prevent the AI from accidentally dropping tables
    stripped = query.strip().upper()
    if not stripped.startswith("SELECT") and not stripped.startswith("WITH") and not stripped.startswith("PRAGMA"):
        return "Safety: Only SELECT queries are allowed. For data-modifying queries, please run them directly in your database tool."
    
    if hasattr(db_manager.conn, "cursor"): # SQLite implementation
        cur = db_manager.conn.cursor() # type: ignore
        cur.execute(query)
        rows = cur.fetchmany(200)  # cap at 200 rows to prevent flooding context window
        cols = [d[0] for d in cur.description] if cur.description else []
    else:
        from sqlalchemy import text # SQLAlchemy implementation
        with db_manager.conn.connect() as conn: # type: ignore
            result = conn.execute(text(query))
            cols = list(result.keys())
            rows = result.fetchmany(200)

    if not rows:
        return "Query returned no rows."

    # Format the SQL results as a clean ASCII/Markdown table
    col_widths = [max(len(str(c)), max((len(str(r[i])) for r in rows), default=0)) for i, c in enumerate(cols)]
    header = " | ".join(str(c).ljust(w) for c, w in zip(cols, col_widths))
    sep    = "-+-".join("-" * w for w in col_widths)
    data_rows = [" | ".join(str(r[i] if r[i] is not None else "NULL").ljust(w) for i, w in enumerate(col_widths)) for r in rows]
    result = f"Query results ({len(rows)} row{'s' if len(rows) != 1 else ''}):\n{header}\n{sep}\n" + "\n".join(data_rows)
    if len(rows) == 200:
        result += "\n\n[Results capped at 200 rows]"
    return result

def get_db_status() -> dict:
    """Utility function (non LLM tool) used by GUIs to display DB connection state."""
    if db_manager.conn is None:
        return {"connected": False, "path": None, "tables": []}
    try:
        if hasattr(db_manager.conn, "cursor"): # SQLite check
            cur = db_manager.conn.cursor() # type: ignore
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tables = [r[0] for r in cur.fetchall()]
        else:
            from sqlalchemy import inspect
            tables = inspect(db_manager.conn).get_table_names() # type: ignore
        return {"connected": True, "path": db_manager.path, "tables": tables}
    except Exception:
        return {"connected": False, "path": None, "tables": []}

# ── File & Code Tools ───────────────────────────────────────────────────────

@sealo_tool
def write_file(path: str, content: str) -> str:
    """Creates or overwrites a file with the given content."""
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Successfully wrote to {path}"

@sealo_tool
def execute_python(code: str, save_as: str = None) -> str:
    """Executes arbitrary Python code in a controlled dict state and captures stdout."""
    import sys
    import io
    
    # Save code to disk if requested
    if save_as:
        p = Path(save_as).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(code, encoding="utf-8")
        save_msg = f"(Saved code to {save_as})\n"
    else:
        save_msg = ""
        
    # Capture print() statements from the executed code
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    
    try:
        # Execute the code in a clean namespace dictionary
        namespace = {}
        exec(code, namespace)
        output = redirected_output.getvalue()
        if not output.strip():
            output = "Code executed successfully. (No console output returned)"
        return save_msg + output[:4000]
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        return save_msg + f"Python Execution Error:\n{err}"
    finally:
        sys.stdout = old_stdout

# ── Web & System Tools ──────────────────────────────────────────────────────

@sealo_tool
def web_search(query: str) -> str:
    """Uses the DuckDuckGo Instant Answer API for lightweight, free web searches."""
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

@sealo_tool
def open_application(name_or_path: str) -> str:
    """Launches executables async so the Python process isn't blocked waiting for them to close."""
    apps = {"notepad": "notepad.exe", "calculator": "calc.exe", "explorer": "explorer.exe",
            "chrome": "chrome", "edge": "msedge", "word": "WINWORD.EXE", "excel": "EXCEL.EXE",
            "powerpoint": "POWERPNT.EXE", "vscode": "code", "terminal": "wt.exe",
            "cmd": "cmd.exe", "task manager": "taskmgr.exe", "paint": "mspaint.exe", "spotify": "spotify"}
    target = apps.get(name_or_path.lower().strip(), name_or_path)
    # Using Popen without .wait() makes it fire-and-forget
    subprocess.Popen(target, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return f"Launched '{name_or_path}'."

@sealo_tool
def open_url(url: str) -> str:
    """Opens links in the user's default system browser."""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    webbrowser.open(url)
    return f"Opened {url} in browser."

@sealo_tool
def update_user_profile(field: str, value: str) -> str:
    """Modifies the user_profile.json file to give Sealo persistent memory about the human."""
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

@sealo_tool
def run_command(command: str) -> str:
    """(Legacy) Simple shell executor with basic blocklists."""
    blocked = ["format ", "shutdown"]
    for b in blocked:
        if b.lower() in command.lower():
            return f"Blocked: '{b.strip()}' is not allowed."
    r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=15, encoding="utf-8", errors="replace")
    out = r.stdout.strip() or r.stderr.strip()
    return out[:4000] or "(No output)"

@sealo_tool
def run_terminal_command(command: str, cwd: str = ".") -> str:
    """Executes a terminal/PowerShell command on the host machine and returns the output."""
    process = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
        cwd=cwd, capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace"
    )
    output = process.stdout.strip()
    if process.stderr:
        output += f"\nSTDERR:\n{process.stderr.strip()}"
    if not output.strip():
        output = f"Command executed successfully with exit code {process.returncode} (No output)"
    return output[:8000]

@sealo_tool
def modify_file_content(path: str, target_text: str, replacement_text: str) -> str:
    """Finds exact target_text in the file at path and replaces it with replacement_text."""
    p = Path(path).resolve()
    if not p.exists():
        return f"Error: File '{path}' does not exist."
    content = p.read_text(encoding="utf-8")
    if target_text not in content:
        return f"Error: The target text to replace was not found in '{path}'. Please provide EXACT text."
    
    new_content = content.replace(target_text, replacement_text)
    p.write_text(new_content, encoding="utf-8")
    return f"Successfully modified '{path}'."

# ── Vision & System Control ───────────────────────────────────────────

@sealo_tool
def analyze_screen(prompt: str = "Explain what is on the screen right now.") -> str:
    """
    Takes a screenshot using MSS, converts via PIL, and sends it to Gemini 2.5 Flash
    (Note: Groq currently doesn't support visual inputs for Llama 3 yet, so this specific
    function would require swapping the LLM backend or using a separate client if needed.)
    """
    if not VISION_AVAILABLE:
        return "Error: Vision tools (mss, pillow, pyautogui) not installed."
    with mss.mss() as sct:
        monitor = sct.monitors[1] # Primary monitor
        sct_img = sct.grab(monitor)
        
        # Convert raw BGRA pixels to standard RGB Image for the API
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        
        # NOTE: This call relies on the `google.genai` client which was removed 
        # when we switched to Groq. This logic needs updating if the user wants 
        # vision features restored using a compatible Multimodal model.
        # resp = client.models.generate_content(model=MODEL_ID, contents=[img, prompt])
        # return f"Screenshot taken. Analysis: {resp.text}"
        return "Screenshot taken. (Note: Vision API client currently disabled while on Groq)"

@sealo_tool
def move_mouse(x: int, y: int) -> str:
    """Automates mouse movement."""
    if not VISION_AVAILABLE: return "Error: pyautogui not installed."
    pyautogui.moveTo(x, y, duration=0.2)
    return f"Moved mouse to ({x}, {y})."

@sealo_tool
def click(x: int = None, y: int = None, button: str = "left", clicks: int = 1) -> str:
    """Simulates physical mouse clicks."""
    if not VISION_AVAILABLE: return "Error: pyautogui not installed."
    if x is not None and y is not None:
        pyautogui.click(x=x, y=y, button=button, clicks=clicks)
        loc_str = f" at ({x}, {y})"
    else:
        pyautogui.click(button=button, clicks=clicks)
        loc_str = " at current position"
    return f"Clicked {button} {clicks} time(s){loc_str}."

@sealo_tool
def type_text(text: str, interval: float = 0.02) -> str:
    """Automates keyboard typing string combinations."""
    if not VISION_AVAILABLE: return "Error: pyautogui not installed."
    pyautogui.write(text, interval=interval)
    return f"Typed: '{text}'."

@sealo_tool
def press_key(key: str, times: int = 1) -> str:
    """Simulates pressing special keys like Enter, Win, Space."""
    if not VISION_AVAILABLE: return "Error: pyautogui not installed."
    pyautogui.press(key, presses=times)
    return f"Pressed '{key}' {times} time(s)."

# ══════════════════════════════════════════════════════════════════════
#  TOOL SCHEMAS (JSON Schema Format)
# ══════════════════════════════════════════════════════════════════════
# The LLM needs these strict JSON descriptions to understand WHAT tools
# it has access to, WHAT arguments they take, and HOW to call them. 
# This matches the standard OpenAI "function calling" specification.

def _str(desc): return {"type": "string", "description": desc}
def _obj(props, req=None):
    return {"type": "object", "properties": props, "required": req or list(props.keys())}
def _decl(name, desc, params=None):
    kwargs = {"name": name, "description": desc}
    if params: kwargs["parameters"] = params
    else: kwargs["parameters"] = {"type": "object", "properties": {}}
    return {"type": "function", "function": kwargs}

TOOLS = [
    _decl("get_current_time",   "Get the current local date and time."),
    _decl("list_directory",     "List files/folders in a directory.",  _obj({"path": _str("Directory path.")})),
    _decl("read_file",          "Read a text file's content.",         _obj({"path": _str("File path.")})),
    _decl("write_file",         "Write/create a text file.",           _obj({"path": _str("File path."), "content": _str("Content to write.")})),
    _decl("modify_file_content","Modify an existing file by replacing exact target text with replacement text. Very useful for editing code.",
          _obj({"path": _str("File path."), "target_text": _str("Exact text to replace."), "replacement_text": _str("New text.")})),
    _decl("execute_python",     "Write and EXECUTE Python code. Returns real output. Use for math, data analysis, automation, visualization.",
          _obj({"code": _str("Python code."), "save_as": _str("Optional path to also save the script.")}, req=["code"])),
    _decl("connect_database",   "Connect to a database. Pass a SQLite .db file path or a SQLAlchemy connection string.",
          _obj({"path_or_connection_string": _str("SQLite path or connection string.")})),
    _decl("list_tables",        "List all tables in the currently connected database. Call connect_database first."),
    _decl("describe_table",     "Show column schema and row count for a table.",  _obj({"table_name": _str("Table name.")})),
    _decl("run_sql_query",      "Execute a SELECT SQL query and return formatted results. Only SELECT is allowed for safety.",
          _obj({"query": _str("SQL SELECT query.")})),
    _decl("web_search",         "Search the web for real-time info, news, docs, etc.", _obj({"query": _str("Search query.")})),
    _decl("open_application",   "Open a Windows app by name (notepad, calculator, chrome, spotify, vscode, etc.).",
          _obj({"name_or_path": _str("App name or path.")})),
    _decl("open_url",           "Open a URL in the browser.",          _obj({"url": _str("URL to open.")})),
    _decl("update_user_profile","Permanently save info about the user. Fields: name, occupation, projects, skills, preferences, notes.",
          _obj({"field": _str("Field name."), "value": _str("Value to save.")})),
    _decl("run_command",        "Run a safe Windows shell command.", _obj({"command": _str("Command to run.")})),
    _decl("run_terminal_command","Execute ANY terminal or PowerShell command on the host Windows machine and get output. Full laptop access.",
          _obj({"command": _str("Terminal/PowerShell command to execute."), "cwd": _str("Working directory.")}, req=["command"])),
    _decl("analyze_screen",     "Take a screenshot of the user's primary monitor and analyze it with visual AI.",
          _obj({"prompt": _str("Specific question or instruction about the screen. Defaults to 'Explain what is on the screen'.")})),
    _decl("move_mouse",         "Move mouse cursor to specific X, Y coordinates on screen.",
          _obj({"x": {"type": "integer", "description": "X coordinate"}, "y": {"type": "integer", "description": "Y coordinate"}})),
    _decl("click",              "Click the mouse at specific coordinates or current location.",
          _obj({"x": {"type": "integer", "description": "Optional X"}, "y": {"type": "integer", "description": "Optional Y"}, "button": _str("'left', 'right', or 'middle'"), "clicks": {"type": "integer", "description": "Number of clicks"}}, req=["button", "clicks"])),
    _decl("type_text",          "Simulate keyboard typing.",
          _obj({"text": _str("Text to type.")})),
    _decl("press_key",          "Press a specific keyboard key (e.g. 'enter', 'tab', 'esc', 'win').",
          _obj({"key": _str("Key name."), "times": {"type": "integer", "description": "Times to press"}})),
]

# Map string tool names returned by the LLM directly to actual Python functions
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
    "move_mouse": move_mouse,
    "click": click,
    "type_text": type_text,
    "press_key": press_key,
}

# ══════════════════════════════════════════════════════════════════════
#  AGENT LOOP ENGINE
# ══════════════════════════════════════════════════════════════════════

def build_system_prompt(profile: dict) -> list:
    """Constructs the master instructions fed to the AI at the start of every message context."""
    content = f"""You are Sealo, a very capable personal AI assistant running on the user's Windows computer.
You have real tools to interact with their computer, databases, and the web.

Rules:
- Sharp, direct, witty. Don't over-explain.
- Use tools ONLY when the user's request clearly requires one. Do NOT call tools for simple greetings or chitchat.
- ONLY call update_user_profile when the user explicitly shares genuinely new personal info (e.g. their name, job, a new project). Do NOT call it for casual conversation.
- For complex requests, plan briefly, then execute tool calls efficiently.
- Never call the same tool twice in a row with identical arguments.

Capabilities: filesystem, live Python execution, SQL databases (SQLite, SQL Server, MySQL, Postgres), web search, opening apps/URLs, voice, system commands.

{format_profile_for_prompt(profile)}

Current time: {datetime.datetime.now().strftime('%A %B %d %Y %I:%M %p')}"""
    return [{"role": "system", "content": content}]

def run_agent_loop(history: list, system_prompt_msg: list, on_tool_call=None):
    """
    Runs the LLM agentic loop for Groq / OpenAI formatting.
    This handles recursive tool chaining (e.g. Model decides to write a file, we run it, 
    send success back to model, model decides to read file, we run it... etc).
    
    Args:
        history: List of dictionary messages (user/assistant/tool interaction history)
        system_prompt_msg: List containing the single System Prompt dictionary
        on_tool_call: Optional UI callback triggered when a python tool function fires
        
    Returns:
        (final_text, history): The Assistant string response and updated memory array.
    """
    MAX_TOOL_ROUNDS = 3 # Safety cap: prevent infinite loops if model calls tools endlessly.
    
    # OPIMIZATION: Filter out old tool calls and tool results from the history to save massive amounts of tokens.
    # LLMs only need to see the final text answer they gave, not the raw JSON logs of how they got there.
    slim_history = []
    for msg in history:
        if msg.get("role") == "tool":
            continue # Drop raw tool execution results
        
        # Deep copy to avoid mutating the actual saved history dict
        clean_msg = dict(msg)
        if clean_msg.get("role") == "assistant" and "tool_calls" in clean_msg:
            del clean_msg["tool_calls"] # Drop the JSON tool call
            if not clean_msg.get("content"):
                continue # If it was a purely tool-calling message with no text, drop it entirely
        
        slim_history.append(clean_msg)
        
    messages = system_prompt_msg + slim_history
    for _round in range(MAX_TOOL_ROUNDS + 1):
        # On the final safety round, we force 'tool_choice="none"' which forces 
        # the model to stop writing JSON tools and just output regular text.
        extra = {} if _round < MAX_TOOL_ROUNDS else {"tool_choice": "none"}
        
        # 1. Ping the remote LLM via OpenAI compatible SDK
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                tools=TOOLS,
                tool_choice=extra.get("tool_choice", "auto"),
            )
            msg = response.choices[0].message
        except Exception as e:
            err_str = str(e)
            logger.warning(f"API error: {err_str}")
            
            # Groq throws 400 Bad Request if the Llama model hallucinates weird JSON tool syntax internally.
            if "tool_use_failed" in err_str or "Failed to call a function" in err_str:
                class MockMsg:
                    role = "assistant"
                    content = "I apologize, my internal systems encountered an error while typing out that tool command (likely a syntax issue with special characters). Could you rephrase or try a simpler path?"
                    tool_calls = None
                    def model_dump(self, exclude_none=True):
                        return {"role": self.role, "content": self.content}
                
                msg = MockMsg()
            elif "429" in err_str or "too many requests" in err_str.lower() or "RESOURCE_EXHAUSTED" in err_str:
                return "**Rate limit hit.** Please wait a moment and try again.", history
            else:
                return f"**API Error:** {err_str}", history
        
        # 2. Add raw LLM response to context memory.
        #    We must dump Pydantic objects to dicts, excluding nulls, because Ollama/Groq crash on `<nil>` JSON fields.
        msg_dict = msg.model_dump(exclude_none=True)
        if "content" not in msg_dict or msg_dict["content"] is None:
            msg_dict["content"] = ""
            
        history.append(msg_dict)
        messages.append(msg_dict)

        # 3. If there are no tool calls in this turn, the loop finishes and returns the text response!
        if not getattr(msg, "tool_calls", None):
            break

        # 4. If there ARE tool calls, execute them in Python locally.
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            
            # Safely parse JSON args (catch empty or malformed model output)
            try:
                fn_args = json.loads(tool_call.function.arguments) or {}
            except Exception:
                fn_args = {}
                
            # Get the exact python function mapping and execute it!
            fn = TOOL_MAP.get(fn_name)
            result = fn(**fn_args) if fn else f"Unknown tool '{fn_name}'"
            
            # Fire UI callback so the terminal/window can draw "Using tool: X..."
            if on_tool_call:
                on_tool_call(fn_name, fn_args, result)
                
            # 5. Pack the tool execution result into the strict 'tool' message schema
            tool_msg = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": fn_name,
                "content": str(result)
            }
            # Append to history so on the NEXT loop iteration, the LLM reads its own action results!
            history.append(tool_msg)
            messages.append(tool_msg)

    return getattr(msg, "content", "") or "", history
