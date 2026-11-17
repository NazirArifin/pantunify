import csv, re, sys, joblib, importlib.util
from collections import defaultdict, Counter
from pathlib import Path

DATA = Path("data/article5/manual_bi_random_sample_augmented.csv")
MODEL = Path("data/article5/crf_layer3_bi_baru.joblib")
SCRIPT = Path("scripts/article5/train_crf_layer3_bi.py")

# load feature function from training script
spec = importlib.util.spec_from_file_location("train_bi", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

crf = joblib.load(MODEL)

def count_b(bi): return sum(1 for c in bi if c == "B")

# 1) targeted hard cases
targets = {"pantai", "cintai", "dicintai", "mencintaimu", "menghargai", "dihargai", "hargai", "bagai", "sebagai", "berbagai", "menyukai", "melukai"}
by_token = defaultdict(list)

with DATA.open(encoding="utf-8", newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        t = (row.get("token") or "").strip().lower()
        if t in targets:
            units = (row.get("units") or "").strip().split()
            gold = (row.get("manual_bi_final") or "").strip()
            if units and gold and len(units) == len(gold):
                by_token[t].append((tuple(units), gold))

print("=== TARGETED CASES ===")
for t in sorted(targets):
    if not by_token[t]:
        print(f"{t:12} | MISSING in dataset")
        continue
    pair = Counter(by_token[t]).most_common(1)[0][0]  # most frequent units/gold pair
    units, gold = pair
    units = list(units)
    X = [[mod.phone_features(units, i, token=t) for i in range(len(units))]]
    pred = "".join(crf.predict(X)[0])
    status = "OK" if pred == gold else "FAIL"
    print(f"{t:12} | gold={gold:12} pred={pred:12} | {status}")

# 2) global stress subset ai/ui/oi
total = exact = syl_ok = 0
with DATA.open(encoding="utf-8", newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        token = (row.get("token") or "").strip().lower()
        if not re.search(r"(ai|ui|oi)", token):
            continue
        units = (row.get("units") or "").strip().split()
        gold = (row.get("manual_bi_final") or "").strip()
        if not units or not gold or len(units) != len(gold):
            continue
        X = [[mod.phone_features(units, i, token=token) for i in range(len(units))]]
        pred = "".join(crf.predict(X)[0])
        total += 1
        if pred == gold:
            exact += 1
        if count_b(pred) == count_b(gold):
            syl_ok += 1

print("\n=== SUBSET ai/ui/oi ===")
print(f"samples={total}")
print(f"word_exact_match={exact/total:.4f}" if total else "word_exact_match=NA")
print(f"syllable_count_accuracy={syl_ok/total:.4f}" if total else "syllable_count_accuracy=NA")
