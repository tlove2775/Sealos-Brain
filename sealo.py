"""
sealo.py — CLI Frontend for Sealo AI Assistant (v3.0)
=====================================================
This is the command-line interface (terminal UI) for the Sealo personal AI assistant.
It handles user input (text + optional voice), sends prompts to the Groq-hosted LLM,
displays formatted responses using the Rich library, and manages conversation memory.

Architecture Overview:
  sealo.py (this file) ─── the "face" of Sealo (UI, input/output, display)
  sealo_core.py ────────── the "brain" of Sealo (LLM client, tools, agent loop)

The entire AI logic (model calls, tool schemas, tool execution) lives in sealo_core.py.
This file only handles the terminal UI layer.

Key Dependencies:
  - rich:    Beautiful terminal formatting (panels, markdown, spinners)
  - dotenv:  Loads API keys from .env file
  - pyttsx3: Optional text-to-speech for voice mode
  - speech_recognition: Optional mic input for voice mode

Pi Compatibility Notes:
  - The 'rich' library works on any terminal with ANSI color support
  - pyttsx3 voice may need 'espeak' installed on Linux/Pi: sudo apt install espeak
  - speech_recognition requires pyaudio: sudo apt install portaudio19-dev
"""

# ═══════════════════════════════════════════════════════════════════════
#  IMPORTS
# ═══════════════════════════════════════════════════════════════════════

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
from dotenv import load_dotenv       # Reads .env file for API keys
from rich.console import Console     # Rich terminal output engine
from rich.markdown import Markdown   # Renders markdown in terminal
from rich.panel import Panel         # Draws bordered panels
from rich.rule import Rule           # Draws horizontal separator lines

# --- Optional voice imports ---
# These are wrapped in try/except so Sealo still works without them.
# On a Pi, you'd need: pip install pyttsx3 SpeechRecognition pyaudio
try:
    import pyttsx3                   # Text-to-speech engine
    import speech_recognition as sr  # Microphone speech-to-text
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False          # Gracefully disable voice features

# Force UTF-8 output to prevent encoding crashes on Windows terminals
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Load environment variables from .env file (contains GROQ_API_KEY)
load_dotenv()

# ═══════════════════════════════════════════════════════════════════════
#  IMPORT CORE ENGINE
# ═══════════════════════════════════════════════════════════════════════
# All AI logic, tool definitions, and the agent loop live in sealo_core.py.
# We import only the public API we need for the CLI.
from sealo_core import (
    client, MODEL_ID, TOOLS, TOOL_MAP,               # LLM client + tool registry
    load_memory, save_memory, load_profile, save_profile,  # Persistence
    format_profile_for_prompt, build_system_prompt, run_agent_loop,  # Agent engine
    listen_from_mic, speak,                            # Voice functions
    MEMORY_FILE, PROFILE_FILE                          # File paths for !mem and !profile
)

# Initialize the Rich console (handles all terminal output formatting)
console = Console()

# ═══════════════════════════════════════════════════════════════════════
#  TOOL CALL CALLBACK
# ═══════════════════════════════════════════════════════════════════════

def _on_tool_call(fn_name, fn_args, result):
    """
    Callback triggered by sealo_core.run_agent_loop() whenever a tool executes.
    This prints a subtle cyan line in the terminal so the user can see which
    tools Sealo is using in real-time while it "thinks".
    
    Example output:  > Using tool: web_search(query='weather in Tokyo')
    """
    console.print(f"  [dim cyan]> Using tool: {fn_name}({', '.join(f'{k}={repr(v)[:40]}' for k,v in fn_args.items())})[/dim cyan]")

# ═══════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION LOOP
# ═══════════════════════════════════════════════════════════════════════

def main():
    """
    Main entry point for the Sealo CLI application.
    
    Flow:
    1. Load user profile and update last_seen timestamp
    2. Display welcome banner
    3. Load conversation memory (previous messages)
    4. Enter input loop:
       a. Get user input (keyboard or microphone)
       b. Check for built-in commands (!voice, !mem clear, !profile, exit)
       c. Send message to LLM via run_agent_loop()
       d. Display formatted response
       e. Save updated conversation history
    """
    # --- Load and update user profile ---
    profile = load_profile()
    profile["last_seen"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    save_profile(profile)

    # --- Build and display the welcome banner ---
    greeting = "Sealo AI - Your A.I Assistant"
    if profile.get("name"):
        subtitle = f"Welcome back, {profile['name']}. Ready to work."
    else:
        subtitle = "All systems online. Ready to assist."

    console.print(Panel.fit(f"[bold cyan]{greeting}[/bold cyan]\n[dim]{subtitle}[/dim]"))
    commands_help = "[bold]Commands:[/bold] 'exit' | '!voice' | '!mem clear' | '!profile'"
    console.print(commands_help)
    if VOICE_AVAILABLE:
        console.print("[dim]Voice mode available. Type !voice to toggle.[/dim]")

    # --- Load previous conversation history from memory.json ---
    history = load_memory()
    if history:
        console.print(f"[dim]Resumed with {len(history)} messages in memory.[/dim]")
    console.print()

    voice_mode = False  # Toggle for speech input/output mode

    # --- Main conversation loop ---
    while True:
        try:
            # --- Get user input (text or voice) ---
            if voice_mode and VOICE_AVAILABLE:
                user_input = listen_from_mic()    # Listen via microphone
                if user_input:
                    console.print(f"[bold green]You (voice) >[/bold green] {user_input}")
                else:
                    continue  # No speech detected, loop again
            else:
                user_input = console.input("[bold green]You > [/bold green]")  # Keyboard input

            # ── Built-in Commands ────────────────────────────────────
            # These are handled locally without sending to the LLM.

            # EXIT: Quit the application
            if user_input.lower() in ['exit', 'quit']:
                console.print("\n[bold cyan]Sealo >[/bold cyan] Signing off. See you next time!")
                break

            # !VOICE: Toggle voice input/output mode
            if user_input.strip().lower() == '!voice':
                if not VOICE_AVAILABLE:
                    console.print("[red]Voice unavailable.[/red] Install pyttsx3 and SpeechRecognition.")
                else:
                    voice_mode = not voice_mode
                    status = "[bold green]ON[/bold green]" if voice_mode else "[bold red]OFF[/bold red]"
                    console.print(f"[bold cyan]Sealo >[/bold cyan] Voice mode {status}")
                continue

            # !MEM CLEAR: Wipe conversation history
            if user_input.strip().lower() == '!mem clear':
                if MEMORY_FILE.exists():
                    MEMORY_FILE.unlink()    # Delete memory.json
                    history = []            # Clear in-memory history
                    console.print("[bold cyan]Sealo >[/bold cyan] Memory cleared.")
                else:
                    console.print("[bold cyan]Sealo >[/bold cyan] Memory was already empty.")
                continue

            # !PROFILE: Display current user profile as JSON
            if user_input.strip().lower() == '!profile':
                profile = load_profile()
                console.print("\n[bold cyan]User Profile:[/bold cyan]")
                console.print(Markdown(f"```json\n{json.dumps(profile, indent=2)}\n```"))
                continue

            # Skip empty input
            if not user_input.strip():
                continue

            # ── Send to LLM ──────────────────────────────────────────
            # Append the user's message to conversation history
            history.append({"role": "user", "content": user_input})
            
            # Build the system prompt (includes user profile + current time)
            system_prompt = build_system_prompt(load_profile())

            # Call the agent loop — this sends the message to Groq's API,
            # handles any tool calls the LLM makes, and returns the final text.
            # The spinner animation shows while we wait for the response.
            with console.status("[dim]Sealo is thinking...[/dim]", spinner="dots"):
                final_text, history = run_agent_loop(history, system_prompt, on_tool_call=_on_tool_call)

            # --- Display the response ---
            console.print(f"\n[bold cyan]Sealo >[/bold cyan]")
            console.print(Markdown(final_text))   # Render as markdown for formatting
            console.print()

            # If voice mode is on, speak the response aloud
            if voice_mode:
                speak(final_text)

            # Save updated conversation history to disk (persists between sessions)
            save_memory(history)

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            console.print("\n[bold cyan]Sealo >[/bold cyan] Interrupted. Shutting down!")
            break
        except Exception as e:
            # Catch and display any errors without crashing
            console.print(f"\n[bold red]Error:[/bold red] {e}")
            import traceback
            traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
