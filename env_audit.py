import os, sys, sealo_core, sealo_live_voice
print("--- MODULE AUDIT ---")
for name in sorted(sys.modules.keys()):
    if "google" in name or "genai" in name or "gemini" in name:
        print(f"Loaded: {name}")

print("\n--- CODE CHECK ---")
with open(sealo_core.__file__, 'r', encoding='utf-8') as f:
    content = f.read()
    if "google" in content:
        print("WARNING: 'google' keyword found in sealo_core.py!")
    if "models.generate_content" in content:
        print("DANGER: 'models.generate_content' found in sealo_core.py!")

print("\n--- ENV CHECK ---")
for k, v in os.environ.items():
    if "API_KEY" in k:
        print(f"{k} is set (length={len(v)})")
