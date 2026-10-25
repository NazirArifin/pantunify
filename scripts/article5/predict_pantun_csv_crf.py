import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from HybridBiLSTMCRFPOSTagger import HybridBiLSTMCRFPOSTagger


TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def infer_model_hparams_from_checkpoint(checkpoint: Dict) -> Dict[str, int]:
    state = checkpoint["model_state_dict"]
    num_chars, char_emb_dim = state["char_embeddings.weight"].shape
    vocab_size, word_emb_dim = state["word_embeddings.weight"].shape
    num_classes, two_word_hidden = state["emission.weight"].shape
    word_hidden_dim = two_word_hidden // 2
    char_hidden_dim = state["char_lstm.weight_ih_l0"].shape[0] // 4
    return {
        "num_chars": int(num_chars),
        "char_emb_dim": int(char_emb_dim),
        "char_hidden_dim": int(char_hidden_dim),
        "vocab_size": int(vocab_size),
        "word_emb_dim": int(word_emb_dim),
        "word_hidden_dim": int(word_hidden_dim),
        "num_classes": int(num_classes),
    }


def encode_tokens(
    tokens: List[str],
    word_vocab: Dict[str, int],
    char_vocab: Dict[str, int],
    lowercase: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    unk_word = word_vocab["<UNK>"]
    unk_char = char_vocab["<UNK>"]
    pad_char = char_vocab["<PAD>"]

    norm_tokens = [t.lower() if lowercase else t for t in tokens]
    word_ids = [word_vocab.get(t, unk_word) for t in norm_tokens]
    char_lists = []
    for token in norm_tokens:
        chars = [char_vocab.get(ch, unk_char) for ch in token]
        char_lists.append(chars if chars else [unk_char])

    max_char_len = max((len(x) for x in char_lists), default=1)
    sentence_word_ids = torch.tensor([word_ids], dtype=torch.long)
    sentence_char_ids = torch.full((1, len(tokens), max_char_len), pad_char, dtype=torch.long)
    word_lengths = torch.ones((1, len(tokens)), dtype=torch.long)
    sentence_lengths = torch.tensor([len(tokens)], dtype=torch.long)

    for j, chars in enumerate(char_lists):
        clen = max(1, min(len(chars), max_char_len))
        word_lengths[0, j] = clen
        sentence_char_ids[0, j, :clen] = torch.tensor(chars[:clen], dtype=torch.long)

    return sentence_word_ids, sentence_char_ids, word_lengths, sentence_lengths


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict POS tags for pantun CSV rows (BiLSTM-CRF)")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--text-column", default="text_pantun")
    parser.add_argument("--lowercase", action="store_true")
    args = parser.parse_args()

    with open(args.vocab_json, "r", encoding="utf-8") as f:
        vocab_payload = json.load(f)
    word_vocab: Dict[str, int] = vocab_payload["word_vocab"]
    char_vocab: Dict[str, int] = vocab_payload["char_vocab"]
    tag_vocab: Dict[str, int] = vocab_payload["tag_vocab"]
    id_to_tag = {idx: tag for tag, idx in tag_vocab.items()}

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    hparams = infer_model_hparams_from_checkpoint(checkpoint)
    model = HybridBiLSTMCRFPOSTagger(
        num_chars=hparams["num_chars"],
        char_emb_dim=hparams["char_emb_dim"],
        char_hidden_dim=hparams["char_hidden_dim"],
        vocab_size=hparams["vocab_size"],
        word_emb_dim=hparams["word_emb_dim"],
        word_hidden_dim=hparams["word_hidden_dim"],
        num_classes=hparams["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processed_pantun = 0
    written = 0
    with open(args.input_csv, "r", encoding="utf-8-sig", newline="") as fin, out_path.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        fieldnames = ["id", "pantun_index", "line_no", "line_text", "token_count", "tokens", "pred_tags", "token_tag_pairs"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for pantun_index, row in enumerate(reader, start=1):
            if pantun_index > args.limit:
                break
            processed_pantun = pantun_index
            text_raw = row.get(args.text_column, "") or ""
            text = text_raw.replace("\\n", "\n")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

            for line_no, line_text in enumerate(lines, start=1):
                tokens = tokenize(line_text)
                if not tokens:
                    continue

                sentence_word_ids, sentence_char_ids, word_lengths, sentence_lengths = encode_tokens(
                    tokens=tokens,
                    word_vocab=word_vocab,
                    char_vocab=char_vocab,
                    lowercase=args.lowercase,
                )
                sentence_word_ids = sentence_word_ids.to(device)
                sentence_char_ids = sentence_char_ids.to(device)
                word_lengths = word_lengths.to(device)
                sentence_lengths = sentence_lengths.to(device)

                paths, _ = model.predict(
                    sentence_word_ids=sentence_word_ids,
                    sentence_char_ids=sentence_char_ids,
                    word_lengths=word_lengths,
                    sentence_lengths=sentence_lengths,
                )
                pred_ids = paths[0] if paths else []
                pred_tags = [id_to_tag[int(i)] for i in pred_ids]
                pairs = [f"{tok}/{tag}" for tok, tag in zip(tokens, pred_tags)]

                writer.writerow(
                    {
                        "id": row.get("id", ""),
                        "pantun_index": pantun_index,
                        "line_no": line_no,
                        "line_text": line_text,
                        "token_count": len(tokens),
                        "tokens": " ".join(tokens),
                        "pred_tags": " ".join(pred_tags),
                        "token_tag_pairs": " | ".join(pairs),
                    }
                )
                written += 1

    print(f"[OK] output: {out_path}")
    print(f"[INFO] pantun processed: {processed_pantun}")
    print(f"[INFO] line predictions: {written}")


if __name__ == "__main__":
    main()
