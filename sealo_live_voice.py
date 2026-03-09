"""
sealo_live_voice.py  —  Sealo 3.0 Voice Mode (Mistral Version)
Re-implemented to use Mistral AI for chat and pyttsx3 for local TTS.
"""

import threading
import os
from pathlib import Path
from dotenv import load_dotenv
import sealo_core as core

# ── PyAudio ────────────────────────────────────────────────────────────
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

# ── SpeechRecognition ─────────────────────────────────────────────────
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

class LiveVoiceSession:
    """
    Voice conversation loop:
    1. Listen to mic → Google STT (via SpeechRecognition) → text
    2. Send text to Mistral chat model → response text
    3. Use local core.speak (pyttsx3) → Audio output
    4. Repeat
    """

    def __init__(self, on_status=None, on_transcript=None,
                 on_user_text=None, user_name="there"):
        self.on_status    = on_status    or (lambda m: print(f"[Live] {m}"))
        self.on_transcript = on_transcript or (lambda t: print(f"[Sealo] {t}"))
        self.on_user_text = on_user_text or (lambda t: print(f"[You] {t}"))
        self.user_name    = user_name
        self._stop_event  = threading.Event()
        self._thread      = None
        self._parent_gui_stop = None

        # Initialize Agent
        self.agent = core.SealoAgent(api_key=core.api_key, model_id=core.MODEL_ID)
        self.agent.set_system_prompt("") # Will be set in loop

    def start(self) -> bool:
        # Check requirements from core
        if not core.VOICE_AVAILABLE or not SR_AVAILABLE:
            self.on_status("Voice/Speech tools missing. Install pyttsx3 & speech_recognition.")
            return False
        if core.OFFLINE_MODE or not core.client:
            self.on_status("Mistral API key missing or offline.")
            return False
            
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="SealoVoice")
        self._thread.start()
        return True

    def stop(self):
        self._stop_event.set()

    def _loop(self):
        """Main voice conversation loop."""
        profile = core.load_profile()
        system = (
            f"You are Sealo, a sharp and witty personal AI assistant talking "
            f"to {self.user_name}. You are in voice conversation mode so keep "
            f"responses SHORT (1-2 sentences). No markdown."
        )

        self.on_status("Voice mode ready. Speak now!")

        while not self._stop_event.is_set():
            # 1. Listen
            self.on_status("Listening...")
            user_text = core.listen_from_mic() # Uses SpeechRecognition

            if self._stop_event.is_set():
                break

            if not user_text:
                continue

            self.on_user_text(user_text)
            self.on_status("Thinking...")

            try:
                # Use Agent with voice-specific system prompt
                self.agent.set_system_prompt(system)
                reply_text = self.agent.chat(user_text)

                self.on_transcript(reply_text)
                self.on_status("Speaking...")

                # 3. Speak
                core.speak(reply_text)
                
            except Exception as e:
                err_str = str(e)
                self.on_status(f"Error: {err_str}")
                if "401" in err_str or "403" in err_str:
                    self.on_transcript("Mistral API key error.")
                    break
                if self._stop_event.is_set():
                    break

        self.on_status("Voice session ended.")
