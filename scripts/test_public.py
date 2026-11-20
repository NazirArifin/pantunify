import argparse
import os
import pandas as pd

# ==============================================================================
# KONFIGURASI UTAMA (SWITCHER MODE & PATH FILE)
# ==============================================================================
# Ubah MODE menjadi "unpivot" untuk memecah kolom konflik ke tabel ringkas,
# atau ubah menjadi "merge" untuk menggabungkan hasil isian kembali ke CSV utama.
MODE = "merge"  # Pilihan: "unpivot" | "merge"

MASTER_CSV = "data/article5/pantun_golden_review.csv"
LONG_CSV = "data/article5/pantun_review_ringkas.csv"
OUTPUT_MERGED_CSV = "data/article5/pantun_golden_final.csv"

FIELD_SPECS = [
    ("Skema Rima", "Semua", "utils_skema_rima", "ib_skema_rima", "golden_skema_rima"),
    ("Silabel", "B1", "utils_silabel_b1", "ib_silabel_b1", "golden_silabel_b1"),
    ("Silabel", "B2", "utils_silabel_b2", "ib_silabel_b2", "golden_silabel_b2"),
    ("Silabel", "B3", "utils_silabel_b3", "ib_silabel_b3", "golden_silabel_b3"),
    ("Silabel", "B4", "utils_silabel_b4", "ib_silabel_b4", "golden_silabel_b4"),
    ("Rima", "B1", "utils_rima_b1", "ib_rima_b1", "golden_rima_b1"),
    ("Rima", "B2", "utils_rima_b2", "ib_rima_b2", "golden_rima_b2"),
    ("Rima", "B3", "utils_rima_b3", "ib_rima_b3", "golden_rima_b3"),
    ("Rima", "B4", "utils_rima_b4", "ib_rima_b4", "golden_rima_b4"),
]


def _extract_line_context(text_pantun: str, ib_seg: str, baris: str) -> tuple[str, str]:
    """Ambil potongan teks & pemenggalan IndoBERT khusus baris terkait agar mudah dicek."""
    if baris == "Semua":
        return text_pantun.strip(), ib_seg.strip()

    idx = int(baris.replace("B", "")) - 1
    lines = [ln.strip() for ln in text_pantun.splitlines() if ln.strip()]
    line_text = lines[idx] if 0 <= idx < len(lines) else text_pantun.strip()

    segs = [sg.strip() for sg in ib_seg.split("|") if sg.strip()]
    line_seg = segs[idx] if 0 <= idx < len(segs) else ib_seg.strip()
    return line_text, line_seg


def run_unpivot(master_path: str, long_path: str) -> None:
    if not os.path.exists(master_path):
        raise FileNotFoundError(f"File master tidak ditemukan: {master_path}")

    df = pd.read_csv(master_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    long_rows = []

    for _, row in df.iterrows():
        if row.get("status_review", "").strip() != "PERLU_REVIEW":
            continue

        no_pantun = row.get("no", "").strip()
        text_pantun = row.get("text_pantun", "")
        ib_seg = row.get("indobert_pemenggalan", "")

        for aspek, baris, u_col, ib_col, g_col in FIELD_SPECS:
            u_val = row.get(u_col, "").strip()
            ib_val = row.get(ib_col, "").strip()
            g_val = row.get(g_col, "").strip()

            # Hanya ekspor kolom golden_* yang masih kosong (belum resolved)
            if g_val == "":
                konteks_teks, pemenggalan_ib = _extract_line_context(text_pantun, ib_seg, baris)
                long_rows.append({
                    "no": no_pantun,
                    "aspek": aspek,
                    "baris": baris,
                    "target_col": g_col,
                    "konteks_teks": konteks_teks,
                    "pemenggalan_ib": pemenggalan_ib,
                    "nilai_utils": u_val,
                    "nilai_ib": ib_val,
                    "golden_final": "",
                })

    df_long = pd.DataFrame(long_rows)
    df_long.to_csv(long_path, index=False, encoding="utf-8-sig")

    print("=== [MODE: UNPIVOT] SELESAI ===")
    print(f"Total item konflik diekspor : {len(df_long)} baris (dari {df['status_review'].eq('PERLU_REVIEW').sum()} pantun)")
    print(f"File ringkas tersimpan di   : {long_path}")
    print("Petunjuk: Buka file tersebut, isi kolom 'golden_final', lalu jalankan MODE = 'merge'.")


def run_merge(master_path: str, long_path: str, output_path: str) -> None:
    if not os.path.exists(master_path):
        raise FileNotFoundError(f"File master tidak ditemukan: {master_path}")
    if not os.path.exists(long_path):
        raise FileNotFoundError(f"File hasil anotasi ringkas tidak ditemukan: {long_path}")

    df_master = pd.read_csv(master_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df_long = pd.read_csv(long_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)

    updates = {}
    empty_in_long = 0
    for _, r in df_long.iterrows():
        no_key = r.get("no", "").strip()
        col_key = r.get("target_col", "").strip()
        val = r.get("golden_final", "").strip()
        if val:
            updates[(no_key, col_key)] = val
        else:
            empty_in_long += 1

    golden_cols = [spec[4] for spec in FIELD_SPECS]
    updated_cells = 0
    resolved_pantun = 0
    still_incomplete = 0

    for idx, row in df_master.iterrows():
        no_key = row.get("no", "").strip()
        for g_col in golden_cols:
            if (no_key, g_col) in updates:
                df_master.at[idx, g_col] = updates[(no_key, g_col)]
                updated_cells += 1

        all_filled = all(df_master.at[idx, c].strip() != "" for c in golden_cols)
        if row.get("status_review", "").strip() == "PERLU_REVIEW":
            if all_filled:
                df_master.at[idx, "status_review"] = "SELESAI_REVIEW"
                resolved_pantun += 1
            else:
                still_incomplete += 1

    df_master.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("=== [MODE: MERGE] SELESAI ===")
    print(f"Sel golden diperbarui       : {updated_cells} sel")
    print(f"Pantun tuntas direview      : {resolved_pantun} pantun (status -> SELESAI_REVIEW)")
    if empty_in_long > 0 or still_incomplete > 0:
        print(f"PERHATIAN                   : Masih ada {empty_in_long} item kosong di tabel ringkas ({still_incomplete} pantun belum lengkap).")
    print(f"File final tersimpan di     : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unpivot/Merge Helper untuk Anotasi Golden Standard Pantun")
    parser.add_argument("--mode", choices=["unpivot", "merge"], default=MODE, help="Mode operasi: unpivot atau merge")
    parser.add_argument("--master", default=MASTER_CSV, help="Path CSV master 32 kolom")
    parser.add_argument("--long", default=LONG_CSV, help="Path CSV ringkas 9 kolom")
    parser.add_argument("--output", default=OUTPUT_MERGED_CSV, help="Path CSV hasil merge final")
    args = parser.parse_args()

    if args.mode == "unpivot":
        run_unpivot(args.master, args.long)
    elif args.mode == "merge":
        run_merge(args.master, args.long, args.output)