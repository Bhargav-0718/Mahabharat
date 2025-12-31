import json
import re
import itertools

fname = 'data/parsed_text/parsed_pages.jsonl'
start = 0

with open(fname, encoding='utf-8') as f:
    pages = [json.loads(l) for l in f]

# find book 8 boundaries by scan
book8_pages = []
for p in pages:
    text_upper = p['text'].upper()
    if 'BOOK 8' in text_upper or 'BOOK VIII' in text_upper:
        book8_pages.append(p['page_number'])

print('book8 markers pages:', book8_pages[:20])

# check section headers regex
sec_re = re.compile(
    r'^[ \t]*SECTION[ \t]+([IVXLCDM]+|\d+)\b.*$',
    re.IGNORECASE | re.MULTILINE
)

# find first page after first BOOK 8 marker with SECTION matches
if book8_pages:
    start_page = min(book8_pages)
else:
    start_page = 0

matches_found = False

for p in pages:
    if p['page_number'] < start_page:
        continue

    matches = list(sec_re.finditer(p['text']))
    if matches:
        m = matches[0]
        print(
            'First section in/after book8 at page',
            p['page_number'],
            'example:',
            p['text'][m.start():m.end()]
        )
        matches_found = True
        break

print('sections found?', matches_found)
