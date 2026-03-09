
import os
import sys
import sealo_core
import sealo_live_voice
import sealo_gui

print("--- PATH DIAGNOSTIC ---")
print(f"sealo_core file: {sealo_core.__file__}")
print(f"sealo_live_voice file: {sealo_live_voice.__file__}")
print(f"sealo_gui file: {sealo_gui.__file__}")
print("\n--- SYS PATH ---")
for p in sys.path:
    print(p)
print("\n--- SEARCHING FOR OTHER sealo_core.py ---")
import subprocess
try:
    # Try to find all sealo_core.py on the drive (capped)
    result = subprocess.run(["where", "/r", "C:\\Users\\demol", "sealo_core.py"], capture_output=True, text=True)
    print(result.stdout)
except Exception as e:
    print(f"Search failed: {e}")
