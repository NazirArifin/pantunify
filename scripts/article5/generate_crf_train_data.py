import pandas as pd

VOWEL_UNITS = {
    "a",
    "i",
    "u",
    "e",
    "o",
    "ə",
    "ɛ",
    "ɔ",
    "ɪ",
    "ʊ",
    "ɑ",
    "æ",
    "ʌ",
    "ɝ",
    "ɚ",
    "eɪ",
    "aɪ",
    "aʊ",
    "ɔɪ",
    "oʊ",
}


def _coerce_g2p_output_to_str(raw) -> str:
    if isinstance(raw, tuple):
        return _coerce_g2p_output_to_str(raw[0])
    if isinstance(raw, list):
        if raw and isinstance(raw[0], list):
            raw = [ph for sub in raw for ph in sub if ph]
        return " ".join(str(part) for part in raw if part)
    return str(raw).strip()


def run_g2p_pipeline(token: str) -> str:
    try:
        from g2p_id import G2P as G2P_ID
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            "g2p-id belum terpasang. Install dulu: pip install g2p-id"
        ) from err

    g2p_model = G2P_ID()
    try:
        result = g2p_model.to_phoneme(token)
    except AttributeError:
        try:
            result = g2p_model(token)
        except Exception:
            result = g2p_model.convert(token)

    return _coerce_g2p_output_to_str(result).replace(" ", "")


def tokenize_phonemes(phoneme_str: str) -> list[str]:
    cleaned = "".join(ch for ch in str(phoneme_str) if not ch.isspace() and ch != "-")
    units: list[str] = []
    i = 0
    while i < len(cleaned):
        if i + 1 < len(cleaned) and cleaned[i : i + 2] in VOWEL_UNITS:
            units.append(cleaned[i : i + 2])
            i += 2
            continue
        units.append(cleaned[i])
        i += 1
    return units

# Daftar kata target dan manual_bi_final yang sudah dikoreksi
data_baru = [
    ("menemui", "BIBIBIB", "baru mengembalikan buku disuruh menemui ibumu"),
    ("melalui", "BIBIBIB", "jalan berliku harus dilalui dengan sabar"),
    ("mengetahui", "BIBIBIBIB", "tidak ada yang mengetahui rahasia hati"),
    ("mengakui", "BIBIBIB", "berani berbuat harus berani mengakui kesalahan"),
    ("memenuhi", "BIBIBIIB", "janji manis harus segera dipenuhi"),
    ("mendahului", "BIIBIBIBIB", "jangan suka mendahului kehendak tuhan"),
    ("menyetujui", "BIBIBIBIIB", "ayah dan ibu sudah menyetujui lamaranmu"),
    ("merestui", "BIBIIBIB", "semoga orang tua merestui hubungan kita"),
    ("dihargai", "BIBIIBIB", "tutuplah aurat agar engkau dihargai"),
    ("menghargai", "BIIBIIBIB", "sesama teman harus saling menghargai"),
    ("menyukai", "BIBIBIB", "banyak orang menyukai sifatmu yang ramah"),
    ("melukai", "BIBIBIB", "berteman jangan sampai saling melukai"),
    ("menandai", "BIBIIBIB", "senyum manis menandai awal bahagia"),
    ("memaknai", "BIBIIBIB", "kita harus bijak memaknai setiap cobaan"),
    ("menguasai", "BIBIBBIB", "ilmu agama harus kita kuasai"),
]

rows = []
for idx, (token, gold_bi, ctx) in enumerate(data_baru, start=9001):
    # GANTI fungsi di bawah ini dengan fungsi G2P & tokenizer pipeline inferensi Anda
    phoneme_str = run_g2p_pipeline(token)          # contoh output: 'mənəmui'
    unit_list = tokenize_phonemes(phoneme_str)     # contoh output: ['m', 'ə', 'n', 'ə', 'm', 'u', 'i']
    
    # Validasi panjang unit vs panjang label BI
    assert len(unit_list) == len(gold_bi), f"Mismatch panjang pada '{token}': {len(unit_list)} vs {len(gold_bi)}"
    
    rows.append({
        "task_id": f"r{idx}_t001",
        "record_id": idx,
        "token_id": 1,
        "token": token,
        "ner_tag": "O",
        "layer2_phoneme": phoneme_str,
        "units": " ".join(unit_list),
        "unit_count": len(unit_list),
        "silver_bi_auto": "",
        "manual_bi_annotator1": "",
        "manual_bi_annotator2": "",
        "manual_bi_final": gold_bi,
        "qa_status": "verified",
        "annotator_notes": "synthetic_hiatus_fix",
        "context_text": ctx
    })

df_synthetic = pd.DataFrame(rows)
df_synthetic.to_csv("synthetic_hiatus_train.csv", index=False)
print("Berhasil membuat", len(df_synthetic), "baris data training tervalidasi.")
