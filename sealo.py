import os
import sys
import json
import subprocess
import datetime
import urllib.request
import urllib.parse
import threading
import webbrowser
from pathlib import Path
from dotenv import load_dotenv
from mistralai import Mistral
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
import sealo_core as core
import logging

logger = logging.getLogger("SealoCLI")

# --- Optional voice imports ---
try:
    import pyttsx3
    import speech_recognition as sr
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

# Force UTF-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Load environment variables
load_dotenv()
MODEL_ID = "mistral-large-latest"
console = Console()

# --- File Paths ---
SEALO_DIR = Path(__file__).parent
MEMORY_FILE = SEALO_DIR / "memory.json"
PROFILE_FILE = SEALO_DIR / "user_profile.json"

# ═══════════════════════════════════════════════════════════════════
#  VOICE ENGINE
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
#  VOICE ENGINE
# ═══════════════════════════════════════════════════════════════════

_tts_engine = None
_tts_lock = threading.Lock()

def get_tts_engine():
    global _tts_engine
    if _tts_engine is None and VOICE_AVAILABLE:
        _tts_engine = pyttsx3.init()
        _tts_engine.setProperty('rate', 175)
        _tts_engine.setProperty('volume', 1.0)
        voices = _tts_engine.getProperty('voices')
        for v in voices:
            if 'david' in v.name.lower() or 'zira' in v.name.lower():
                _tts_engine.setProperty('voice', v.id)
                break
    return _tts_engine

def speak(text: str):
    def _speak():
        with _tts_lock:
            engine = get_tts_engine()
            if engine:
                clean = text.replace('**', '').replace('*', '').replace('`', '').replace('#', '')
                engine.say(clean)
                engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

def listen_from_mic() -> str:
    if not VOICE_AVAILABLE:
        return ""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        console.print("[bold yellow]Listening...[/bold yellow] (speak now)")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)
            console.print("[dim]Processing speech...[/dim]")
            return recognizer.recognize_google(audio)
        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            console.print("[dim yellow]Could not understand audio.[/dim yellow]")
            return ""
        except Exception as e:
            console.print(f"[dim red]Speech error: {e}[/dim red]")
            return ""

# ═══════════════════════════════════════════════════════════════════
#  TOOLS
# ═══════════════════════════════════════════════════════════════════

def get_current_time() -> str:
    now = datetime.datetime.now()
    return now.strftime("Current date and time: %A, %B %d, %Y at %I:%M:%S %p")

def list_directory(path: str) -> str:
    try:
        if not os.path.isdir(path):
            return f"Error: '{path}' is not a valid directory."
        items = os.listdir(path)
        if not items:
            return f"The directory '{path}' is empty."
        formatted = "\n".join(
            f"  {'[DIR] ' if os.path.isdir(os.path.join(path, i)) else '[FILE]'} {i}"
            for i in sorted(items)
        )
        return f"Contents of '{path}':\n{formatted}"
    except Exception as e:
        return f"Error: {e}"

def read_file(path: str) -> str:
    try:
        if not os.path.isfile(path):
            return f"Error: '{path}' is not a valid file."
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        if len(content) > 6000:
            content = content[:6000] + "\n\n[... truncated at 6000 characters ...]"
        return f"Content of '{path}':\n```\n{content}\n```"
    except Exception as e:
        return f"Error: {e}"

def write_file(path: str, content: str) -> str:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding='utf-8')
        return f"Successfully wrote {len(content)} characters to '{path}'."
    except Exception as e:
        return f"Error writing file: {e}"

def execute_python(code: str, save_as: str = "") -> str:
    """Write Python code to a temp file, execute it, and return the output."""
    try:
        script_path = SEALO_DIR / "_temp_script.py"
        script_path.write_text(code, encoding='utf-8')
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=30, encoding='utf-8', errors='replace'
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        output_parts = []
        if stdout:
            output_parts.append(f"Output:\n{stdout}")
        if stderr:
            output_parts.append(f"Errors/Warnings:\n{stderr}")
        if not output_parts:
            output_parts.append("Script ran successfully with no output.")
        output = "\n\n".join(output_parts)
        # If user wants to save the file under a real name
        if save_as:
            save_path = Path(save_as)
            save_path.write_text(code, encoding='utf-8')
            output += f"\n\nScript also saved to '{save_as}'."
        return output
    except subprocess.TimeoutExpired:
        return "Error: Script timed out after 30 seconds."
    except Exception as e:
        return f"Error executing Python: {e}"

def web_search(query: str) -> str:
    try:
        encoded = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={'User-Agent': 'Sealo-AI/2.0'})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        results = []
        if data.get('Answer'):
            results.append(f"**Quick Answer:** {data['Answer']}")
        if data.get('AbstractText'):
            results.append(f"**Summary:** {data['AbstractText']}")
            if data.get('AbstractURL'):
                results.append(f"**Source:** {data['AbstractURL']}")
        for topic in data.get('RelatedTopics', [])[:5]:
            if isinstance(topic, dict) and topic.get('Text'):
                results.append(f"- {topic['Text']}")
        if not results:
            return f"No direct results found for '{query}'. The information might be too recent or obscure."
        return f"Web search: '{query}'\n\n" + "\n".join(results)
    except Exception as e:
        return f"Error searching: {e}"

def open_application(name_or_path: str) -> str:
    """Open an application by name or full path."""
    try:
        # Common app shortcuts
        app_map = {
            'notepad': 'notepad.exe',
            'calculator': 'calc.exe',
            'explorer': 'explorer.exe',
            'chrome': 'chrome',
            'edge': 'msedge',
            'word': 'WINWORD.EXE',
            'excel': 'EXCEL.EXE',
            'powerpoint': 'POWERPNT.EXE',
            'vscode': 'code',
            'terminal': 'wt.exe',
            'cmd': 'cmd.exe',
            'task manager': 'taskmgr.exe',
            'paint': 'mspaint.exe',
            'spotify': 'spotify',
        }
        target = app_map.get(name_or_path.lower().strip(), name_or_path)
        os.startfile(target) if os.path.isabs(target) else subprocess.Popen(
            target, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return f"Launched '{name_or_path}' successfully."
    except Exception as e:
        return f"Error launching '{name_or_path}': {e}"

def open_url(url: str) -> str:
    """Open a URL in the default web browser."""
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        webbrowser.open(url)
        return f"Opened '{url}' in your browser."
    except Exception as e:
        return f"Error opening URL: {e}"

def update_user_profile(field: str, value: str) -> str:
    """Update the user's persistent profile. Fields: name, occupation, projects, skills, preferences, notes."""
    profile = core.load_profile()
    list_fields = ['projects', 'skills', 'preferences', 'notes']
    if field in list_fields:
        if value not in profile[field]:
            profile[field].append(value)
            core.save_profile(profile)
            return f"Added '{value}' to your {field}."
        return f"'{value}' was already in your {field}."
    elif field in profile:
        profile[field] = value
        core.save_profile(profile)
        return f"Updated your {field} to '{value}'."
    else:
        return f"Unknown profile field '{field}'. Valid fields: name, occupation, projects, skills, preferences, notes."

def run_command(command: str) -> str:
    blocked = ['del ', 'rm ', 'rmdir /s', 'format ', 'shutdown', ': delete', 'rd /s']
    for b in blocked:
        if b.lower() in command.lower():
            return f"Error: The command contains a blocked keyword '{b.strip()}' for safety reasons."
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=15,
            encoding='utf-8', errors='replace'
        )
        output = result.stdout.strip() or result.stderr.strip()
        return output[:4000] if output else "(No output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out."
    except Exception as e:
        return f"Error: {e}"

# ═══════════════════════════════════════════════════════════════════
#  TOOL SCHEMAS
# ═══════════════════════════════════════════════════════════════════

# ── Mistral Tool Mapping ─────────────────────────────────────────────

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
            "description": "List files and folders inside a directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path to the directory."}},
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the text content of a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path to the file."}},
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or create a text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write to."},
                    "content": {"type": "string", "description": "Full text content."}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Write and EXECUTE Python code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The Python code to execute."},
                    "save_as": {"type": "string"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet.",
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
            "name": "open_application",
            "description": "Open an application.",
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
            "name": "open_url",
            "description": "Open a URL.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_user_profile",
            "description": "Update user profile.",
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
            "name": "run_command",
            "description": "Run shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            }
        }
    }
]

TOOL_MAP = {
    "get_current_time": get_current_time,
    "list_directory": list_directory,
    "read_file": read_file,
    "write_file": write_file,
    "execute_python": execute_python,
    "web_search": web_search,
    "open_application": open_application,
    "open_url": open_url,
    "update_user_profile": update_user_profile,
    "run_command": run_command,
}

# ═══════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════

def main():
    profile = core.load_profile()
    profile["last_seen"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    core.save_profile(profile)

    # Initialize Agent
    agent = core.SealoAgent(api_key=core.api_key, model_id=MODEL_ID)
    agent.set_system_prompt(core.build_system_prompt(profile))
    
    # Load past memory
    initial_history = core.load_memory()
    agent.load_history(initial_history)

    # Build greeting
    greeting = "Sealo AI - A.I Assistant"
    if profile.get("name"):
        subtitle = f"Welcome back, {profile['name']}. Ready to work."
    else:
        subtitle = "All systems online. Ready to assist."

    console.print(Panel.fit(f"[bold cyan]{greeting}[/bold cyan]\n[dim]{subtitle}[/dim]"))
    commands_help = "[bold]Commands:[/bold] 'exit' | '!voice' | '!mem clear' | '!profile'"
    console.print(commands_help)
    if VOICE_AVAILABLE:
        console.print("[dim]Voice mode available. Type !voice to toggle.[/dim]")

    if initial_history:
        console.print(f"[dim]Resumed with {len(initial_history)} messages in memory.[/dim]")
    console.print()

    voice_mode = False

    while True:
        try:
            # Input
            if voice_mode and VOICE_AVAILABLE:
                user_input = listen_from_mic()
                if user_input:
                    console.print(f"[bold green]You (voice) >[/bold green] {user_input}")
                else:
                    continue
            else:
                user_input = console.input("[bold green]You > [/bold green]")

            if not user_input.strip():
                continue

            # --- Built-in commands ---
            cmd = user_input.lower().strip()
            if cmd in ['exit', 'quit']:
                console.print("\n[bold cyan]Sealo >[/bold cyan] Signing off. See you next time!")
                break

            if cmd == '!voice':
                if not VOICE_AVAILABLE:
                    console.print("[red]Voice unavailable.[/red] Install pyttsx3 and SpeechRecognition.")
                else:
                    voice_mode = not voice_mode
                    status = "[bold green]ON[/bold green]" if voice_mode else "[bold red]OFF[/bold red]"
                    console.print(f"[bold cyan]Sealo >[/bold cyan] Voice mode {status}")
                continue

            if cmd == '!mem clear':
                if MEMORY_FILE.exists():
                    MEMORY_FILE.unlink()
                    agent.history = []
                    console.print("[bold cyan]Sealo >[/bold cyan] Memory cleared.")
                else:
                    console.print("[bold cyan]Sealo >[/bold cyan] Memory was already empty.")
                continue

            if cmd == '!profile':
                p = core.load_profile()
                console.print("\n[bold cyan]User Profile:[/bold cyan]")
                console.print(Markdown(f"```json\n{json.dumps(p, indent=2)}\n```"))
                continue

            # Update system prompt with potential profile changes
            agent.set_system_prompt(core.build_system_prompt(core.load_profile()))

            with console.status("[dim]Sealo is thinking...[/dim]", spinner="dots"):
                # chat() will append user_input to agent.history
                final_text = agent.chat(user_input)

            console.print(f"\n[bold cyan]Sealo >[/bold cyan]")
            console.print(Markdown(final_text))
            console.print()

            if voice_mode:
                speak(final_text)

            # Save historical dicts
            core.save_memory(agent.history)

        except KeyboardInterrupt:
            if agent.is_thinking if hasattr(agent, "is_thinking") else False:
                console.print("\n[bold yellow]Sealo >[/bold yellow] Stopping agent...")
                agent.stop()
            else:
                console.print("\n[bold cyan]Sealo >[/bold cyan] Interrupted. Shutting down!")
                break
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
            logger.exception("Unexpected error in CLI main loop")

if __name__ == "__main__":
    main()
