import json

with open("data/parsed_text/parsed_pages.jsonl", encoding="utf-8") as f:
    for line in list(f)[50:80]:
        page = json.loads(line)
        text = page["text"]
        if "BOOK" in text and "PARVA" in text:
            print(f"PAGE {page['page_number']}: {text[:200]}")
