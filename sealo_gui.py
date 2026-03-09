"""
sealo_gui.py  —  Sealo 3.0 GUI
A modern, dark-mode desktop AI assistant with SQL integration.
Run with: python sealo_gui.py
"""

import sys, json, threading, datetime, re
from pathlib import Path
import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
import sealo_core as core
import logging

logger = logging.getLogger("SealoGUI")

try:
    from sealo_live_voice import LiveVoiceSession, PYAUDIO_AVAILABLE
    LIVE_VOICE_AVAILABLE = PYAUDIO_AVAILABLE  # Live voice needs pyaudio
except Exception as _live_err:
    print(f"Warning: Live voice unavailable: {_live_err}")
    LIVE_VOICE_AVAILABLE = False
    PYAUDIO_AVAILABLE = False
    LiveVoiceSession = None

# ──────────────────────────── Theme ──────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

COLORS = {
    "bg":          "#0f1117",
    "sidebar":     "#161b27",
    "panel":       "#1a2035",
    "card":        "#1e2640",
    "input_bg":    "#232b3e",
    "accent":      "#4f8cff",
    "accent2":     "#7c5cfc",
    "user_bubble": "#1e3a5f",
    "ai_bubble":   "#1a2540",
    "tool_bg":     "#12191f",
    "success":     "#2ecc71",
    "warning":     "#f39c12",
    "error":       "#e74c3c",
    "text":        "#e8eaf6",
    "dim":         "#7b8ab8",
    "border":      "#2a3555",
}

FONT_FAMILY = "Segoe UI"

# ──────────────────────────── App ────────────────────────────────────

class SealoChatBubble(ctk.CTkFrame):
    """A single chat message bubble with basic markdown-ish rendering."""
    def __init__(self, parent, text: str, role: str, tool_info: str = "", **kwargs):
        color = COLORS["user_bubble"] if role == "user" else COLORS["ai_bubble"]
        super().__init__(parent, fg_color=color, corner_radius=12, **kwargs)

        # Role label
        label = "You" if role == "user" else "Sealo"
        label_color = COLORS["accent"] if role == "user" else COLORS["accent2"]
        ctk.CTkLabel(self, text=label, font=(FONT_FAMILY, 11, "bold"),
                     text_color=label_color, anchor="w").pack(anchor="w", padx=12, pady=(8, 2))

        # Message text — use CTkTextbox for selective styling
        self.text_area = ctk.CTkTextbox(
            self, font=(FONT_FAMILY, 13), text_color=COLORS["text"],
            fg_color="transparent", border_width=0, wrap="word",
            height=20, # Initial small height, will expand
            activate_scrollbars=False
        )
        self.text_area.pack(fill="x", padx=6, pady=(0, 2))
        
        # Tags for "markdown"
        self.text_area.tag_config("bold", foreground=COLORS["accent"])
        self.text_area.tag_config("code", foreground=COLORS["accent2"])
        self.text_area.tag_config("code_block", foreground=COLORS["dim"])

        self._render_markdown(text)
        
        # Auto-size height (rough estimate)
        num_lines = text.count('\n') + 2
        extra_height = (len(text) // 80) * 15
        self.text_area.configure(height=num_lines * 20 + extra_height)
        self.text_area.configure(state="disabled")

        # Timestamp
        ts = datetime.datetime.now().strftime("%I:%M %p")
        ctk.CTkLabel(self, text=ts, font=(FONT_FAMILY, 10),
                     text_color=COLORS["dim"], anchor="e").pack(anchor="e", padx=12, pady=(0, 6))

        # Tool info (if any)
        if tool_info:
            tool_frame = ctk.CTkFrame(self, fg_color=COLORS["tool_bg"], corner_radius=6)
            tool_frame.pack(fill="x", padx=12, pady=(0, 8))
            ctk.CTkLabel(tool_frame, text=tool_info, font=("Consolas", 10),
                         text_color=COLORS["dim"], anchor="w", wraplength=600, justify="left"
                         ).pack(anchor="w", padx=8, pady=4)

    def _render_markdown(self, text: str):
        """Simple regex-based bold and code highlighting."""
        # This is a very basic parser
        parts = re.split(r'(\*\*.*?\*\*|`.*?`|```.*?```)', text, flags=re.DOTALL)
        for part in parts:
            if part.startswith('```') and part.endswith('```'):
                content = part[3:-3].strip()
                self.text_area.insert("end", content + "\n", "code_block")
            elif part.startswith('`') and part.endswith('`'):
                self.text_area.insert("end", part[1:-1], "code")
            elif part.startswith('**') and part.endswith('**'):
                self.text_area.insert("end", part[2:-2], "bold")
            else:
                self.text_area.insert("end", part)




class SealoGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Sealo — AI Assistant  v3.0")
        self.geometry("1350x820")
        self.minsize(900, 600)
        self.configure(fg_color=COLORS["bg"])

        self.history = core.load_memory()
        self.profile = core.load_profile()
        self.profile["last_seen"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        core.save_profile(self.profile)
        
        # Initialize Agent
        self.agent = core.SealoAgent(api_key=core.api_key, model_id=core.MODEL_ID)
        self.agent.set_system_prompt(core.build_system_prompt(self.profile))
        self.agent.load_history(self.history)

        self.voice_mode = False
        self.is_thinking = False
        self._pending_tool_lines = []
        self._live_session = None
        self._live_active = False

        self._build_layout()
        self._update_db_panel()
        self._show_welcome()

    # ── Layout ──────────────────────────────────────────────────────

    def _build_layout(self):
        # Top bar
        topbar = ctk.CTkFrame(self, fg_color=COLORS["sidebar"], height=52, corner_radius=0)
        topbar.pack(fill="x", side="top")
        topbar.pack_propagate(False)
        ctk.CTkLabel(topbar, text="  SEALO", font=(FONT_FAMILY, 18, "bold"),
                     text_color=COLORS["accent"]).pack(side="left", padx=16)
        ctk.CTkLabel(topbar, text="A.I Assistant  ·  v3.0", font=(FONT_FAMILY, 12),
                     text_color=COLORS["dim"]).pack(side="left")

        # Top-right controls
        right = ctk.CTkFrame(topbar, fg_color="transparent")
        right.pack(side="right", padx=12)

        # Live Voice button (primary feature)
        self.live_voice_btn = ctk.CTkButton(
            right, text="Live Voice", width=110, height=30,
            fg_color="#7c5cfc", hover_color="#9b7dff",
            font=(FONT_FAMILY, 12, "bold"), command=self._toggle_live_voice
        )
        self.live_voice_btn.pack(side="left", padx=4)

        self.voice_btn = ctk.CTkButton(right, text="TTS: OFF", width=100, height=30,
                                       fg_color=COLORS["card"], hover_color=COLORS["border"],
                                       font=(FONT_FAMILY, 12), command=self._toggle_voice)
        self.voice_btn.pack(side="left", padx=4)
        ctk.CTkButton(right, text="Clear Memory", width=110, height=30,
                      fg_color=COLORS["card"], hover_color="#c0392b",
                      font=(FONT_FAMILY, 12), command=self._clear_memory).pack(side="left", padx=4)

        # Main 3-column layout
        main = ctk.CTkFrame(self, fg_color=COLORS["bg"])
        main.pack(fill="both", expand=True)

        # Left: DB Sidebar
        self.sidebar = ctk.CTkFrame(main, fg_color=COLORS["sidebar"], width=220, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=0)
        self.sidebar.pack_propagate(False)
        self._build_sidebar()

        # Center: Chat
        center = ctk.CTkFrame(main, fg_color=COLORS["bg"])
        center.pack(side="left", fill="both", expand=True)
        self._build_chat_area(center)
        self._build_input_area(center)

        # Right: Tool activity panel
        right_panel = ctk.CTkFrame(main, fg_color=COLORS["panel"], width=290, corner_radius=0)
        right_panel.pack(side="right", fill="y")
        right_panel.pack_propagate(False)
        self._build_tool_panel(right_panel)

    def _build_sidebar(self):
        ctk.CTkLabel(self.sidebar, text="DATABASE", font=(FONT_FAMILY, 11, "bold"),
                     text_color=COLORS["dim"]).pack(anchor="w", padx=14, pady=(16, 4))

        self.db_status_label = ctk.CTkLabel(
            self.sidebar, text="No database connected", font=(FONT_FAMILY, 11),
            text_color=COLORS["dim"], wraplength=190, justify="left"
        )
        self.db_status_label.pack(anchor="w", padx=14, pady=(0, 8))

        ctk.CTkButton(self.sidebar, text="Connect SQLite DB", height=32,
                      fg_color=COLORS["accent"], hover_color=COLORS["accent2"],
                      font=(FONT_FAMILY, 12), command=self._connect_db_dialog
                      ).pack(fill="x", padx=12, pady=2)

        ctk.CTkButton(self.sidebar, text="Connect SQL Server", height=32,
                      fg_color=COLORS["card"], hover_color=COLORS["accent2"],
                      font=(FONT_FAMILY, 12), command=self._connect_sqlserver_dialog
                      ).pack(fill="x", padx=12, pady=2)

        sep = ctk.CTkFrame(self.sidebar, fg_color=COLORS["border"], height=1)
        sep.pack(fill="x", padx=12, pady=10)

        ctk.CTkLabel(self.sidebar, text="TABLES", font=(FONT_FAMILY, 11, "bold"),
                     text_color=COLORS["dim"]).pack(anchor="w", padx=14, pady=(0, 4))

        self.table_frame = ctk.CTkScrollableFrame(self.sidebar, fg_color="transparent")
        self.table_frame.pack(fill="both", expand=True, padx=8)

        # Bottom: profile
        sep2 = ctk.CTkFrame(self.sidebar, fg_color=COLORS["border"], height=1)
        sep2.pack(fill="x", padx=12, pady=6)
        self.profile_label = ctk.CTkLabel(
            self.sidebar, text=self._short_profile(), font=(FONT_FAMILY, 11),
            text_color=COLORS["dim"], wraplength=196, justify="left"
        )
        self.profile_label.pack(anchor="w", padx=14, pady=(0, 14))

    def _build_chat_area(self, parent):
        self.chat_scroll = ctk.CTkScrollableFrame(parent, fg_color=COLORS["bg"], label_text="")
        self.chat_scroll.pack(fill="both", expand=True, padx=8, pady=8)

    def _build_input_area(self, parent):
        bar = ctk.CTkFrame(parent, fg_color=COLORS["panel"], corner_radius=14, height=68)
        bar.pack(fill="x", padx=8, pady=(0, 10))
        bar.pack_propagate(False)

        self.input_box = ctk.CTkEntry(
            bar, placeholder_text="Message Sealo...",
            font=(FONT_FAMILY, 14), fg_color=COLORS["input_bg"],
            border_color=COLORS["border"], text_color=COLORS["text"],
            height=44, corner_radius=10
        )
        self.input_box.pack(side="left", fill="x", expand=True, padx=(10, 6), pady=12)
        self.input_box.bind("<Return>", lambda e: self._send())
        self.input_box.bind("<Shift-Return>", lambda e: None)

        self.attach_btn = ctk.CTkButton(
            bar, text="+", width=40, height=44,
            fg_color=COLORS["card"], hover_color=COLORS["accent"],
            font=(FONT_FAMILY, 18, "bold"), corner_radius=10, command=self._attach_file
        )
        self.attach_btn.pack(side="left", padx=(0, 6), pady=12)

        self.send_btn = ctk.CTkButton(bar, text="Send", width=70, height=44,
                                      fg_color=COLORS["accent"], hover_color=COLORS["accent2"],
                                      font=(FONT_FAMILY, 13), corner_radius=10, command=self._send)
        self.send_btn.pack(side="left", padx=(0, 6), pady=12)

        self.stop_btn = ctk.CTkButton(
            bar, text="Stop", width=70, height=44,
            fg_color=COLORS["card"], hover_color=COLORS["error"],
            font=(FONT_FAMILY, 13), corner_radius=10, command=self._stop_thinking,
            state="disabled"
        )
        self.stop_btn.pack(side="left", padx=(0, 6), pady=12)

        self.mic_btn = ctk.CTkButton(
            bar, text="Mic", width=60, height=44,
            fg_color=COLORS["card"], hover_color=COLORS["accent"],
            font=(FONT_FAMILY, 12), corner_radius=10, command=self._voice_input
        )
        self.mic_btn.pack(side="left", padx=(0, 10), pady=12)

    def _build_tool_panel(self, parent):
        ctk.CTkLabel(parent, text="TOOL ACTIVITY", font=(FONT_FAMILY, 11, "bold"),
                     text_color=COLORS["dim"]).pack(anchor="w", padx=14, pady=(14, 4))
        self.tool_log = ctk.CTkTextbox(
            parent, font=("Consolas", 11), fg_color=COLORS["tool_bg"],
            text_color=COLORS["dim"], wrap="word", corner_radius=8
        )
        self.tool_log.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.tool_log.configure(state="disabled")

    # ── Helpers ──────────────────────────────────────────────────────

    def _short_profile(self) -> str:
        p = self.profile
        lines = []
        if p.get("name"):       lines.append(f"Name: {p['name']}")
        if p.get("occupation"): lines.append(f"Role: {p['occupation']}")
        if p.get("skills"):     lines.append(f"Skills: {', '.join(p['skills'][:3])}")
        return "\n".join(lines) if lines else "No profile yet"

    def _add_bubble(self, text: str, role: str, tool_info: str = ""):
        bubble = SealoChatBubble(self.chat_scroll, text=text,
                                 role=role, tool_info=tool_info)
        bubble.pack(fill="x", pady=6, padx=8)
        self.update_idletasks()
        self._smooth_scroll()

    def _smooth_scroll(self, steps=10, current_step=0):
        """Animated scroll to bottom."""
        target = 1.0
        start = self.chat_scroll._parent_canvas.yview()[1]
        if start >= 0.99 or current_step >= steps:
            self.chat_scroll._parent_canvas.yview_moveto(1.0)
            return
        
        next_val = start + (target - start) * 0.3
        self.chat_scroll._parent_canvas.yview_moveto(next_val)
        self.after(20, lambda: self._smooth_scroll(steps, current_step + 1))

    def _log_tool(self, name: str, args: dict, result: str):
        self.tool_log.configure(state="normal")
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        arg_str = ", ".join(f"{k}={repr(v)[:30]}" for k, v in args.items())
        self.tool_log.insert("end", f"\n[{ts}] {name}({arg_str})\n")
        # Truncate result for display
        short_result = result[:200] + "..." if len(result) > 200 else result
        self.tool_log.insert("end", f"→ {short_result}\n{'─'*34}\n")
        self.tool_log.see("end")
        self.tool_log.configure(state="disabled")
        self._pending_tool_lines.append(f"[{name}] {arg_str[:50]}")

    def _set_thinking(self, state: bool):
        self.is_thinking = state
        if state:
            self.send_btn.configure(text="...", state="disabled", fg_color=COLORS["dim"])
            self.stop_btn.configure(state="normal", fg_color=COLORS["card"])
            self.attach_btn.configure(state="disabled")
            self.input_box.configure(state="disabled")
        else:
            self.send_btn.configure(text="Send", state="normal", fg_color=COLORS["accent"])
            self.stop_btn.configure(state="disabled", fg_color=COLORS["card"])
            self.attach_btn.configure(state="normal")
            self.input_box.configure(state="normal")
            self.input_box.focus()

    def _show_welcome(self):
        name = self.profile.get("name") or "there"
        mem_count = len(self.history)
        msg = f"Hey {name}! I'm Sealo v3.0 — Beast Mode activated.\n"
        if mem_count:
            msg += f"I have {mem_count} messages in memory from our last session.\n"
        if core._db_conn:
            msg += f"Database connected: {core._db_path}"
        else:
            msg += "Connect a database using the sidebar to start querying your data."
        self._add_bubble(msg, "sealo")

    def _update_db_panel(self):
        status = core.get_db_status()
        # Clear table list
        for w in self.table_frame.winfo_children():
            w.destroy()
        if status["connected"]:
            db_name = Path(status["path"]).name if status["path"] else "connected"
            self.db_status_label.configure(text=f"Connected:\n{db_name}", text_color=COLORS["success"])
            for tbl in status["tables"]:
                btn = ctk.CTkButton(
                    self.table_frame, text=tbl, height=28, anchor="w",
                    fg_color="transparent", hover_color=COLORS["card"],
                    font=("Consolas", 12), text_color=COLORS["text"],
                    command=lambda t=tbl: self._quick_describe(t)
                )
                btn.pack(fill="x", pady=1)
        else:
            self.db_status_label.configure(text="No database connected", text_color=COLORS["dim"])

    def _quick_describe(self, table_name: str):
        """When user clicks a table in sidebar, describe it via chat."""
        self.input_box.delete(0, "end")
        self.input_box.insert(0, f"Describe the {table_name} table for me")
        self._send()

    def _stop_thinking(self):
        """Action for the Stop button."""
        if self.is_thinking:
            self.agent.stop()
            self._log_tool("USER_INTERRUPT", {}, "Stopping agent loop...")

    def _attach_file(self):
        """Open a file dialog and suggest analysis to the agent."""
        path = filedialog.askopenfilename(
            title="Select File to Analyze",
            filetypes=[
                ("All Supported", "*.txt *.py *.md *.json *.sql *.png *.jpg *.jpeg *.webp"),
                ("Text Files", "*.txt *.py *.md *.json *.sql"),
                ("Images", "*.png *.jpg *.jpeg *.webp"),
                ("All Files", "*.*")
            ]
        )
        if path:
            file_path = Path(path)
            ext = file_path.suffix.lower()
            
            # Auto-prepare a message for the user to confirm or edit
            if ext in [".png", ".jpg", ".jpeg", ".webp"]:
                suggestion = f"Analyze this image: {path}"
            else:
                suggestion = f"Read and explain this file: {path}"
            
            self.input_box.delete(0, "end")
            self.input_box.insert(0, suggestion)
            # We don't auto-send to give the user a chance to add their own prompt
            self._log_tool("FILE_ATTACH", {"path": path}, "File selected. Press Send to analyze.")

    # ── Actions ──────────────────────────────────────────────────────

    def _connect_db_dialog(self):
        path = filedialog.askopenfilename(
            title="Select SQLite Database",
            filetypes=[("SQLite", "*.db *.sqlite *.sqlite3"), ("All", "*.*")]
        )
        if path:
            result = core.connect_database(path)
            self._add_bubble(result, "sealo")
            self._update_db_panel()

    def _connect_sqlserver_dialog(self):
        dialog = ctk.CTkInputDialog(
            text="Enter SQLAlchemy connection string:\n\nExamples:\nmssql+pyodbc://user:pw@server/db?driver=ODBC+Driver+17+for+SQL+Server\nmysql+pymysql://user:pw@host/db\npostgresql://user:pw@host/db",
            title="Connect Database"
        )
        conn_str = dialog.get_input()
        if conn_str and conn_str.strip():
            result = core.connect_database(conn_str.strip())
            self._add_bubble(result, "sealo")
            self._update_db_panel()

    def _toggle_live_voice(self):
        if not LIVE_VOICE_AVAILABLE:
            messagebox.showwarning(
                "Live Voice Unavailable",
                "Requires pyaudio. Run: pip install pyaudio\nAlso ensure GEMINI_API_KEY is set."
            )
            return

        if self._live_active:
            # Stop the session
            if self._live_session:
                self._live_session.stop()
                self._live_session = None
            self._live_active = False
            self.live_voice_btn.configure(text="Live Voice", fg_color="#7c5cfc")
            self._add_bubble("Live voice session ended.", "sealo")
        else:
            # Start the session
            self._live_active = True
            self.live_voice_btn.configure(text="Listening...", fg_color=COLORS["success"])
            self._add_bubble("Voice mode on! Speak and I'll respond with Gemini's AI voice.", "sealo")

            def _on_status(msg):
                self.after(0, lambda m=msg: self._log_tool("Voice", {}, m))
                # Update button text with current status
                if "listening" in msg.lower():
                    self.after(0, lambda: self.live_voice_btn.configure(text="Listening..."))
                elif "speaking" in msg.lower():
                    self.after(0, lambda: self.live_voice_btn.configure(text="Speaking..."))
                elif "thinking" in msg.lower():
                    self.after(0, lambda: self.live_voice_btn.configure(text="Thinking..."))
                if "ended" in msg.lower() or "closed" in msg.lower():
                    self.after(0, self._on_live_ended)

            def _on_transcript(text):
                if text.strip():
                    self.after(0, lambda t=text: self._add_bubble(t, "sealo"))

            def _on_user_text(text):
                if text.strip():
                    self.after(0, lambda t=text: self._add_bubble(t, "user"))

            profile = core.load_profile()
            self._live_session = LiveVoiceSession(
                on_status=_on_status,
                on_transcript=_on_transcript,
                on_user_text=_on_user_text,
                user_name=profile.get("name") or "there"
            )
            self._live_session.start()

    def _on_live_ended(self):
        self._live_active = False
        self._live_session = None
        self.live_voice_btn.configure(text="Live Voice", fg_color="#7c5cfc")

    def _toggle_voice(self):
        if not core.VOICE_AVAILABLE:
            messagebox.showwarning("TTS Unavailable", "Install pyttsx3 and SpeechRecognition for TTS support.")
            return
        self.voice_mode = not self.voice_mode
        if self.voice_mode:
            self.voice_btn.configure(text="TTS: ON", fg_color=COLORS["success"])
        else:
            self.voice_btn.configure(text="TTS: OFF", fg_color=COLORS["card"])


    def _clear_memory(self):
        if messagebox.askyesno("Clear Memory", "Delete all conversation history? This cannot be undone."):
            if core.MEMORY_FILE.exists():
                core.MEMORY_FILE.unlink()
            self.history = []
            for w in self.chat_scroll.winfo_children():
                w.destroy()
            self._add_bubble("Memory cleared. Fresh start!", "sealo")

    def _voice_input(self):
        if not core.VOICE_AVAILABLE:
            messagebox.showwarning("Voice Unavailable", "Install pyttsx3 and SpeechRecognition.")
            return
        self.mic_btn.configure(text="...", state="disabled")
        def _listen():
            text = core.listen_from_mic()
            self.after(0, lambda: self._after_voice(text))
        threading.Thread(target=_listen, daemon=True).start()

    def _after_voice(self, text: str):
        self.mic_btn.configure(text="Mic", state="normal")
        if text:
            self.input_box.delete(0, "end")
            self.input_box.insert(0, text)
            self._send()

    def _send(self):
        if self.is_thinking:
            return
        text = self.input_box.get().strip()
        if not text:
            return
        self.input_box.delete(0, "end")
        self._add_bubble(text, "user")
        self._set_thinking(True)
        self._pending_tool_lines = []

        self.history.append({"role": "user", "content": text})
        profile = core.load_profile()
        system_prompt = core.build_system_prompt(profile)

        def _worker():
            try:
                # Refresh profile info each turn
                self.agent.set_system_prompt(core.build_system_prompt(core.load_profile()))
                
                final_text = self.agent.chat(
                    user_input=text, 
                    on_tool_call=lambda name, args, result: self.after(0, lambda: self._log_tool(name, args, result))
                )
                
                # Update local state from agent state
                self.history = self.agent.history
                core.save_memory(self.history)
                
                self.after(0, lambda: self._on_response(final_text))
            except Exception as e:
                logger.exception("Unexpected error in GUI worker thread")
                self.after(0, lambda: self._on_error(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_response(self, text: str):
        tool_summary = "\n".join(self._pending_tool_lines) if self._pending_tool_lines else ""
        self._add_bubble(text, "sealo", tool_info=tool_summary)
        self._set_thinking(False)
        self._update_db_panel()
        # Refresh profile label
        self.profile = core.load_profile()
        self.profile_label.configure(text=self._short_profile())
        if self.voice_mode:
            core.speak(text)

    def _on_error(self, error: str):
        self._add_bubble(f"Error: {error}", "sealo")
        self._set_thinking(False)


# ─────────────────────────────────────────────────────────────────────

def main():
    app = SealoGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
