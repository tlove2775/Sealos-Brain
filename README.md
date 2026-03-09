# 🌌 SeaLo 3.0 — Beast Mode

SeaLo is a powerful, modern, and versatile AI Desktop Assistant built to handle everything from coding and system control to deep data analysis. version 3.0 introduces high-performance **Mistral AI** integration, a robust vision system, and advanced file analysis capabilities.

![SeaLo GUI](https://via.placeholder.com/800x450.png?text=SeaLo+3.0+GUI+Interface) *<!-- Replace with actual screenshot after pushing! -->*

---

## 🚀 Key Features

### 🧠 Advanced Intelligence (v3.0)
- **Mistral AI Integration**: Powered by `mistral-large-latest` for elite reasoning and coding.
- **Graceful Interrupt**: New **"Stop"** button in GUI and `Ctrl+C` support in CLI to cancel long-running tasks instantly.
- **Centralized Logging**: Comprehensive `sealo.log` captures all system events, API calls, and tool results for easy debugging.

### 📂 File & Data Analysis
- **GUI Attachment Support**: Click the `+` button to analyze any local file (text, code, or images).
- **Vision Protocol**: Automatically uses Mistral Vision to "see" and describe local images or your current screen.
- **SQL & Database Mastery**: Connect to SQLite or SQL Server databases via the sidebar to query and describe tables naturally.

### 🎙️ Multi-Modal Interaction
- **Live Voice (Optional)**: Real-time voice interaction (requires `pyaudio`).
- **Vision Tools**: `analyze_screen` and `analyze_image` allow SeaLo to understand your visual workspace.
- **System Control**: SeaLo can move mice, click, type, and launch applications on Windows.

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/SeaLo.git
   cd SeaLo
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure you have `customtkinter`, `mistralai`, `python-dotenv`, `tenacity`, and `rich` installed.)*

3. **Optional (Live Voice)**:
   ```bash
   pip install pyaudio speech_recognition pyttsx3
   ```

---

## ⚙️ Setup

Create a `.env` file in the root directory and add your API key:

```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

---

## 🖥️ Usage

### Modern GUI
Launch the sleek, dark-mode desktop interface:
```bash
python sealo_gui.py
```

### Powerful CLI
Run the terminal-based assistant:
```bash
python sealo.py
```

---

## 📜 Logging & Debugging

SeaLo 3.0 includes a built-in diagnostic system. If you encounter issues, check:
- **`sealo.log`**: Standard application logs and tool results.
- **`sealo_debug.log`**: Detailed traceback and communication logs.

---

## 🛡️ License

Creative Commons / Private Use — *Modify as needed.*
