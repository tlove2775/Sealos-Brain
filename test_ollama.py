"""Quick smoke test to verify the Llama 3.1 local Ollama integration."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from sealo_core import run_agent_loop, build_system_prompt, load_profile

profile = load_profile()
system_prompt = build_system_prompt(profile)

print("--- Testing: simple greeting ---")
response, _ = run_agent_loop([{"role": "user", "content": "Hi! What can you do?"}], system_prompt)
print(f"Sealo: {response}\n")

print("--- Testing: tool call (time) ---")
response2, _ = run_agent_loop([{"role": "user", "content": "What time is it right now?"}], system_prompt)
print(f"Sealo: {response2}\n")

print("All tests complete!")
