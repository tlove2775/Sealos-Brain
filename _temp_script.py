import re

file_path = r"C:\\Users\\demol\\Desktop\\Sealos Code\\sealo.py"
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find any "Beast Mode" in the CLI
matches = re.findall(r'Beast Mode', content)
print(f"Found {len(matches)} matches in sealo.py")

# Print the surrounding lines for context
for match in re.finditer(r'Beast Mode', content):
    start = max(0, match.start() - 30)
    end = min(len(content), match.end() + 30)
    print(f"Context: ...{content[start:end]}...")