# import pandas as pd
# import numpy as np
# import re
# from collections import Counter

# # 1. Load dataset utama
# df = pd.read_csv("data/pantun_dataset.csv")
# print(f"Total baris: {len(df)}, Total kolom: {len(df.columns)}")
# assert len(df) == 6640 and len(df.columns) == 17, "Jumlah baris/kolom tidak sesuai!"

# # Helper pemecah 4 baris eksplisit dari literal '\n'
# lines_split = df["text_pantun"].apply(lambda x: x.split("\\n"))
# for i in range(4):
#     df[f"line_{i+1}"] = lines_split.apply(lambda lst: lst[i] if len(lst) > i else "")

# print("\n" + "="*70)
# print("KOMENTAR #8: STATISTIK DESKRIPTIF 17 VARIABEL (OVERALL & BY SKEMA_RIMA)")
# print("="*70)

# num_cols = [f"suku_kata_baris_{i}" for i in range(1, 5)] + [f"jumlah_kata_baris_{i}" for i in range(1, 5)]

# def summarize_numeric(sub_df, label):
#     print(f"\n--- {label} (n = {len(sub_df)}) ---")
#     rows = []
#     for col in num_cols:
#         s = sub_df[col]
#         q25, q50, q75 = np.percentile(s, [25, 50, 75])
#         rows.append({
#             "Variable": col,
#             "Mean": round(s.mean(), 2),
#             "SD": round(s.std(), 2),
#             "Median": round(q50, 2),
#             "IQR": round(q75 - q25, 2),
#             "Min": int(s.min()),
#             "Max": int(s.max())
#         })
#     print(pd.DataFrame(rows).to_string(index=False))

# summarize_numeric(df, "ALL STANZAS")
# for schema, group in df.groupby("skema_rima"):
#     summarize_numeric(group, f"SKEMA: {schema}")

# # Ringkasan variabel string/kategorikal (melengkapi 17 variabel)
# print("\n--- Ringkasan Variabel Non-Numerik (17 Variabel Lengkap) ---")
# str_cols = ["id", "text_pantun", "baris_sampiran", "baris_isi", "skema_rima"] + [f"rima_akhir_baris_{i}" for i in range(1, 5)]
# str_summary = []
# for col in str_cols:
#     char_len = df[col].astype(str).str.len()
#     str_summary.append({
#         "Variable": col,
#         "Unique_Values": df[col].nunique(),
#         "Top_Value": repr(df[col].value_counts().index[0])[:35],
#         "Top_Freq": int(df[col].value_counts().iloc[0]),
#         "Mean_Char_Len": round(char_len.mean(), 2),
#         "SD_Char_Len": round(char_len.std(), 2)
#     })
# print(pd.DataFrame(str_summary).to_string(index=False))

# print("\n" + "="*70)
# print("KOMENTAR #9: ANALISIS EKSPANSI METRIK PADA BARIS KE-4")
# print("="*70)

# # Distribusi frekuensi suku kata (8..12) per posisi baris
# syl_dist = {}
# for i in range(1, 5):
#     vc = df[f"suku_kata_baris_{i}"].value_counts().reindex([8, 9, 10, 11, 12], fill_value=0)
#     syl_dist[f"Baris_{i}_n"] = vc
#     syl_dist[f"Baris_{i}_%"] = (vc / len(df) * 100).round(2)
# print("\nDistribusi Jumlah Suku Kata (8-12) per Posisi Baris:")
# print(pd.DataFrame(syl_dist))

# # Analisis penanda linguistik penyebab ekspansi suku kata pada baris 1..4
# clitic_pattern = re.compile(r"\b\w+(lah|kah|pun|nya)\b", re.IGNORECASE)
# prep_conj_pattern = re.compile(r"\b(di|ke|dari|pada|untuk|dengan|yang|dan|jika|kalau|agar|supaya|jangan|marilah|hendaklah)\b", re.IGNORECASE)

# ling_rows = []
# for i in range(1, 5):
#     lines = df[f"line_{i}"]
#     has_clitic = lines.apply(lambda s: bool(clitic_pattern.search(s))).mean() * 100
#     clitic_count = lines.apply(lambda s: len(clitic_pattern.findall(s))).mean()
#     func_word_count = lines.apply(lambda s: len(prep_conj_pattern.findall(s))).mean()
#     avg_word_len = lines.apply(lambda s: np.mean([len(w) for w in s.split()]) if s.split() else 0).mean()
    
#     # Khusus baris dengan suku kata >= 11 (ekspansi metrik)
#     long_mask = df[f"suku_kata_baris_{i}"] >= 11
#     clitic_in_long = lines[long_mask].apply(lambda s: bool(clitic_pattern.search(s))).mean() * 100
    
#     ling_rows.append({
#         "Baris": f"Line {i}",
#         "Proporsi_>=11_SukuKata(%)": round(long_mask.mean() * 100, 2),
#         "Baris_Berklitik(%)": round(has_clitic, 2),
#         "Rerata_Klitik_per_Baris": round(clitic_count, 3),
#         "Berklitik_pada_Baris_>=11(%)": round(clitic_in_long, 2),
#         "Rerata_Kata_Tugas/Preposisi": round(func_word_count, 3),
#         "Rerata_Panjang_Kata(huruf)": round(avg_word_len, 2)
#     })
# print("\nIndikator Morfosintaksis per Posisi Baris:")
# print(pd.DataFrame(ling_rows).to_string(index=False))

# print("\n" + "="*70)
# print("KOMENTAR #10: DAMPAK DUPLIKASI SAMPIRAN & ISI TERHADAP VARIANS LEKSIKAL")
# print("="*70)

# dup_sampiran_rows = df["baris_sampiran"].duplicated(keep="first").sum()
# dup_isi_rows = df["baris_isi"].duplicated(keep="first").sum()
# unique_sampiran = df["baris_sampiran"].nunique()
# unique_isi = df["baris_isi"].nunique()

# def lexical_metrics(text_series):
#     tokens = [w.lower() for line in text_series for w in line.replace("\\n", " ").split()]
#     total_tokens = len(tokens)
#     unique_tokens = len(set(tokens))
#     ttr = unique_tokens / total_tokens if total_tokens else 0
#     bigrams = list(zip(tokens[:-1], tokens[1:]))
#     ubr = len(set(bigrams)) / len(bigrams) if bigrams else 0
#     return total_tokens, unique_tokens, round(ttr, 5), round(ubr, 5)

# samp_all = lexical_metrics(df["baris_sampiran"])
# samp_dedup = lexical_metrics(df["baris_sampiran"].drop_duplicates())
# isi_all = lexical_metrics(df["baris_isi"])
# isi_dedup = lexical_metrics(df["baris_isi"].drop_duplicates())

# print(f"Sampiran: {unique_sampiran} unik, {dup_sampiran_rows} baris duplikat ({dup_sampiran_rows/len(df)*100:.2f}%)")
# print(f"Isi     : {unique_isi} unik, {dup_isi_rows} baris duplikat ({dup_isi_rows/len(df)*100:.2f}%)")
# var_df = pd.DataFrame([
#     {"Segment": "Sampiran (Full 6,640)", "Total_Tokens": samp_all[0], "Vocab_Size": samp_all[1], "TTR": samp_all[2], "Unique_Bigram_Ratio": samp_all[3]},
#     {"Segment": f"Sampiran (Unique {unique_sampiran})", "Total_Tokens": samp_dedup[0], "Vocab_Size": samp_dedup[1], "TTR": samp_dedup[2], "Unique_Bigram_Ratio": samp_dedup[3]},
#     {"Segment": "Isi (Full 6,640)", "Total_Tokens": isi_all[0], "Vocab_Size": isi_all[1], "TTR": isi_all[2], "Unique_Bigram_Ratio": isi_all[3]},
#     {"Segment": f"Isi (Unique {unique_isi})", "Total_Tokens": isi_dedup[0], "Vocab_Size": isi_dedup[1], "TTR": isi_dedup[2], "Unique_Bigram_Ratio": isi_dedup[3]},
# ])
# print(var_df.to_string(index=False))

# print("\n" + "="*70)
# print("KOMENTAR #11: DISTRIBUSI KLUSTER FONETIK RIMA AKHIR")
# print("="*70)

# all_rhymes = pd.concat([df[f"rima_akhir_baris_{i}"] for i in range(1, 5)], ignore_index=True)
# total_lines = len(all_rhymes)
# rhyme_counts = all_rhymes.value_counts()

# print(f"Total unit rima unik di seluruh {total_lines} larik: {len(rhyme_counts)}")
# top_rhyme_df = pd.DataFrame({
#     "Rhyme_Cluster": rhyme_counts.head(15).index,
#     "Count": rhyme_counts.head(15).values,
#     "Percentage(%)": (rhyme_counts.head(15).values / total_lines * 100).round(2),
#     "Cumulative(%)": (rhyme_counts.head(15).values.cumsum() / total_lines * 100).round(2)
# })
# print("\nTop 15 Kluster Rima Akhir:")
# print(top_rhyme_df.to_string(index=False))

# # Klasifikasi struktur fonotaktik kluster rima (Vokal Tunggal, Diftong/VV, Vokal+Konsonan)
# vowels = set("aiueo")
# def classify_rhyme_structure(r):
#     r = str(r).lower().strip()
#     if len(r) == 1 and r in vowels:
#         return "Open Single Vowel (V)"
#     elif len(r) >= 2 and all(ch in vowels for ch in r):
#         return "Diphthong / Vowel Cluster (VV)"
#     else:
#         return "Closed Vowel-Consonant (VC / VCC)"

# struct_counts = all_rhymes.apply(classify_rhyme_structure).value_counts()
# struct_df = pd.DataFrame({
#     "Phonotactic_Class": struct_counts.index,
#     "Count": struct_counts.values,
#     "Percentage(%)": (struct_counts.values / total_lines * 100).round(2)
# })
# print("\nDistribusi Kelas Struktur Fonotaktik Rima:")
# print(struct_df.to_string(index=False))

# print("\n" + "="*70)
# print("KOMENTAR #6 & #3: AUDIT ORTOGRAFI & DATA TERELIMINASI")
# print("="*70)
# has_upper = df["text_pantun"].apply(lambda s: any(c.isupper() for c in s)).sum()
# has_hyphen = df["text_pantun"].apply(lambda s: "-" in s).sum()
# print(f"Bait mempertahankan huruf kapital: {has_upper} ({has_upper/len(df)*100:.2f}%)")
# print(f"Bait mengandung tanda hubung (-): {has_hyphen} ({has_hyphen/len(df)*100:.2f}%)")

# try:
#     ex_df = pd.read_csv("data/excluded_pantun_dataset.csv")
#     print(f"\nFile excluded_pantun_dataset.csv ditemukan: {len(ex_df)} baris.")
#     print("Kolom excluded:", list(ex_df.columns))
# except FileNotFoundError:
#     print("\nFile excluded_pantun_dataset.csv tidak ditemukan di direktori ini.")

import pandas as pd
ex = pd.read_csv("data/excluded_pantun_dataset.csv")
kategori = ex["alasan"].str.split(":").str[0].str.strip()
print(kategori.value_counts())
print((kategori.value_counts(normalize=True) * 100).round(2))