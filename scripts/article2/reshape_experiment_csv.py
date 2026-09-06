#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def _normalize_setting(raw: str) -> str:
    s = (raw or "").strip().lower()
    if "zero" in s:
        return "zero_shot"
    if "few" in s:
        return "few_shot"
    return "unknown"


def _clean(text: str) -> str:
    return (text or "").strip()


def _detect_delimiter(input_path: Path) -> str:
    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=["\t", ",", ";", "|"])
        return dialect.delimiter
    except csv.Error:
        # Fallback aman untuk file eksperimen yang umumnya bertab.
        return "\t"


def reshape_wide_to_long(input_path: Path, output_path: Path, keep_empty: bool = False) -> int:
    delimiter = _detect_delimiter(input_path)

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header_1 = next(reader, None)
        header_2 = next(reader, None)

        if not header_1 or not header_2:
            raise ValueError("File CSV tidak memiliki dua baris header yang dibutuhkan.")

        max_len = max(len(header_1), len(header_2))
        header_1 = header_1 + [""] * (max_len - len(header_1))
        header_2 = header_2 + [""] * (max_len - len(header_2))

        # Kolom metadata diasumsikan ada sebelum kolom hasil model pertama.
        first_result_idx = None
        for i, h2 in enumerate(header_2):
            if "hasil" in (h2 or "").lower():
                first_result_idx = i
                break

        if first_result_idx is None:
            raise ValueError("Tidak ditemukan kolom hasil model (baris header kedua tidak memuat 'Hasil').")

        meta_names = [h.strip() for h in header_1[:first_result_idx]]
        model_names = [h.strip() for h in header_1[first_result_idx:]]
        setting_labels = [h.strip() for h in header_2[first_result_idx:]]

        output_rows = []
        for row in reader:
            if not any(cell.strip() for cell in row):
                continue

            row = row + [""] * (max_len - len(row))
            meta_values = row[:first_result_idx]

            meta = {}
            for k, v in zip(meta_names, meta_values):
                key = k.strip().lower().replace(" ", "_")
                meta[key] = _clean(v)

            model_values = row[first_result_idx:]
            for model_name, setting_label, generated in zip(model_names, setting_labels, model_values):
                generated_text = _clean(generated)
                if not keep_empty and not generated_text:
                    continue

                output_rows.append(
                    {
                        "no": meta.get("no.", meta.get("no", "")),
                        "id_data_asli": meta.get("id_data_asli", ""),
                        "jenis_pantun": meta.get("jenis_pantun", ""),
                        "pantun_asli_lengkap": meta.get("pantun_asli_lengkap", ""),
                        "sampiran_asli": meta.get("sampiran_asli", ""),
                        "isi_input_model": meta.get("isi_untuk_input_model", ""),
                        "model": model_name,
                        "setting": _normalize_setting(setting_label),
                        "setting_label": setting_label,
                        "sampiran_generated": generated_text,
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "no",
        "id_data_asli",
        "jenis_pantun",
        "pantun_asli_lengkap",
        "sampiran_asli",
        "isi_input_model",
        "model",
        "setting",
        "setting_label",
        "sampiran_generated",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    return len(output_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ubah dataset eksperimen pantun dari format wide (multi-model) ke long format."
    )
    parser.add_argument(
        "--input",
        default="data/article2/100_Pantun_Eksperimen.csv",
        help="Path CSV input format wide.",
    )
    parser.add_argument(
        "--output",
        default="data/article2/100_Pantun_Eksperimen_long.csv",
        help="Path CSV output format long.",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Tetap simpan baris dengan output model kosong.",
    )

    args = parser.parse_args()
    total = reshape_wide_to_long(Path(args.input), Path(args.output), keep_empty=args.keep_empty)
    print(f"Selesai. Total baris long-format: {total}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
