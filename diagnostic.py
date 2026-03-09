
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

print("--- DIAGNOSTIC START ---")
print(f"Python Version: {sys.version}")
print(f"Executable: {sys.executable}")
print(f"CWD: {os.getcwd()}")

# Load .env
env_path = Path(".env")
if env_path.exists():
    print(f"Found .env at {env_path.absolute()}")
    load_dotenv(env_path)
else:
    print(".env NOT FOUND!")

# Check keys
mistral_key = os.getenv("MISTRAL_API_KEY")
print(f"MISTRAL_API_KEY: {'[SET - length ' + str(len(mistral_key)) + ']' if mistral_key else '[NOT SET]'}")
print(f"GEMINI_API_KEY: {'[SET]' if os.getenv('GEMINI_API_KEY') else '[NOT SET]'}")
print(f"GOOGLE_API_KEY: {'[SET]' if os.getenv('GOOGLE_API_KEY') else '[NOT SET]'}")

# Try Mistral call
try:
    from mistralai import Mistral
    print("Mistral SDK found.")
    client = Mistral(api_key=mistral_key)
    print("Testing Mistral (mistral-tiny)...")
    resp = client.chat.complete(
        model="mistral-tiny",
        messages=[{"role": "user", "content": "Test diagnostic"}],
        max_tokens=5
    )
    print("Mistral Success!")
    print(f"Response: {resp.choices[0].message.content}")
except Exception as e:
    print(f"Mistral Failure Error: {e}")
    # Print the full error if possible
    import traceback
    traceback.print_exc()

print("\n--- LOADED MODULES ---")
for m in sorted(sys.modules.keys()):
    if "google" in m or "genai" in m or "mistral" in m:
        print(f"  {m}")

print("\n--- DIAGNOSTIC END ---")
