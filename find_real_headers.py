import json
import re

# Search for actual chapter/book title pages (not TOC)
pattern = re.compile(
    r"(BOOK\s+\d{1,2})[\s\n]+(.*?PARVA)",
    re.IGNORECASE | re.DOTALL
)

with open("data/parsed_text/parsed_pages.jsonl", encoding="utf-8") as f:
    combined_text = ""
    for line in f:
        page = json.loads(line)
        combined_text += "\n\n" + page["text"]

# Remove index lines first
combined_text = re.sub(r"The Mahabharata.*?Index[^\n]*", "", combined_text, flags=re.IGNORECASE | re.DOTALL)

# Find all book matches
matches = list(re.finditer(pattern, combined_text))

print(f"Found {len(matches)} potential book matches\n")

for match in matches:
    # Get context: 200 chars before and 300 after
    start = max(0, match.start() - 100)
    end = min(len(combined_text), match.end() + 200)
    
    before = combined_text[start:match.start()].replace("\n", "\\n")[-50:]
    matched = match.group(0).replace("\n", "\\n")
    after = combined_text[match.end():end].replace("\n", "\\n")[:100]
    
    print(f"Match: {match.group(1)} / {match.group(2)[:30]}")
    print(f"  Context: ...{before}[{matched}]{after}...")
    print()
