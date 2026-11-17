import json

INPUT_FILE = 'data/article5/pantun-anotated.jsonl' # sesuaikan nama file Anda

SUSPECT_TYP_NOT_ASG = {
    'respon', 'telpon', 'nelfon', 'merespon', 'vaksin', 'kualitas', 
    'aktivitas', 'kreatif', 'kompleks', 'modern', 'spesial', 'auto'
}

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    text = f.read()

decoder = json.JSONDecoder()
pos = 0
idx = 0

while pos < len(text):
    # Lewati spasi atau baris kosong
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos >= len(text):
        break
        
    idx += 1
    try:
        obj, end_pos = decoder.raw_decode(text, pos)
        pos = end_pos
    except json.JSONDecodeError as e:
        print(f"[Entri #{idx}] ERROR SYNTAX JSON di karakter {pos}: {e}")
        break

    tokens = obj.get('tokens', [])
    tags = obj.get('ner_tags')

    if tags is None:
        print(f"[Entri #{idx}] HILANG 'ner_tags': {' '.join(tokens[:6])}...")
        continue

    if len(tokens) != len(tags):
        print(f"[Entri #{idx}] PANJANG BEDA (tokens={len(tokens)}, tags={len(tags)}): {' '.join(tokens[:6])}...")
        continue

    for tok, tag in zip(tokens, tags):
        if tok.lower() in SUSPECT_TYP_NOT_ASG and tag == 'B-ASG':
            print(f"[Entri #{idx}] '{tok}' berlabel {tag} -> ubah ke B-TYP/O")