import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


TokenTagSentence = List[Tuple[str, str]]


def parse_conllu(path: Path) -> List[TokenTagSentence]:
    sentences: List[TokenTagSentence] = []
    current: TokenTagSentence = []

    with path.open("r", encoding="utf-8") as f:
        for line_idx, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            if line.startswith("#"):
                continue

            cols = line.split("\t")
            if len(cols) < 4:
                raise ValueError(f"Invalid CoNLL-U columns at {path}:{line_idx}")

            token_id = cols[0]
            if "-" in token_id or "." in token_id:
                continue

            form = cols[1]
            upos = cols[3]
            if upos == "_":
                continue
            current.append((form, upos))

    if current:
        sentences.append(current)
    return sentences


def normalize_token(token: str, lowercase: bool) -> str:
    return token.lower() if lowercase else token


def build_word_vocab(sentences: Sequence[TokenTagSentence], lowercase: bool) -> Dict[str, int]:
    vocab: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for sentence in sentences:
        for token, _ in sentence:
            norm = normalize_token(token, lowercase)
            if norm not in vocab:
                vocab[norm] = len(vocab)
    return vocab


def build_char_vocab(sentences: Sequence[TokenTagSentence], lowercase: bool) -> Dict[str, int]:
    vocab: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for sentence in sentences:
        for token, _ in sentence:
            norm = normalize_token(token, lowercase)
            for ch in norm:
                if ch not in vocab:
                    vocab[ch] = len(vocab)
    return vocab


def build_tag_vocab(sentences: Iterable[TokenTagSentence]) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    for sentence in sentences:
        for _, upos in sentence:
            if upos not in vocab:
                vocab[upos] = len(vocab)
    return vocab


def encode_sentence(
    sentence: TokenTagSentence,
    word_vocab: Dict[str, int],
    char_vocab: Dict[str, int],
    tag_vocab: Dict[str, int],
    lowercase: bool,
) -> Dict[str, List[int]]:
    word_ids: List[int] = []
    char_ids: List[List[int]] = []
    tag_ids: List[int] = []
    oov_mask: List[int] = []

    unk_word_id = word_vocab[UNK_TOKEN]
    unk_char_id = char_vocab[UNK_TOKEN]

    for token, upos in sentence:
        norm = normalize_token(token, lowercase)
        wid = word_vocab.get(norm, unk_word_id)
        word_ids.append(wid)
        oov_mask.append(1 if wid == unk_word_id else 0)

        chars = [char_vocab.get(ch, unk_char_id) for ch in norm]
        char_ids.append(chars if chars else [unk_char_id])

        if upos not in tag_vocab:
            raise ValueError(f"UPOS '{upos}' tidak ada dalam tag_vocab")
        tag_ids.append(tag_vocab[upos])

    return {
        "word_ids": word_ids,
        "char_ids": char_ids,
        "tag_ids": tag_ids,
        "oov_mask": oov_mask,
    }


def write_jsonl(
    out_path: Path,
    sentences: Sequence[TokenTagSentence],
    word_vocab: Dict[str, int],
    char_vocab: Dict[str, int],
    tag_vocab: Dict[str, int],
    lowercase: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for sentence in sentences:
            row = encode_sentence(sentence, word_vocab, char_vocab, tag_vocab, lowercase)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CoNLL-U to JSONL for train_hybrid_bilstm_pos.py")
    parser.add_argument("--train-conllu", nargs="+", required=True, help="Satu atau banyak file train .conllu")
    parser.add_argument("--dev-conllu", nargs="+", default=None, help="Satu atau banyak file dev .conllu")
    parser.add_argument("--out-dir", required=True, help="Folder output")
    parser.add_argument("--train-out-name", default="train.jsonl", help="Nama output train jsonl")
    parser.add_argument("--dev-out-name", default="dev.jsonl", help="Nama output dev jsonl")
    parser.add_argument("--vocab-name", default="vocab.json", help="Nama output vocabulary")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase token sebelum di-encode")
    args = parser.parse_args()

    train_sentences: List[TokenTagSentence] = []
    for p in args.train_conllu:
        parsed = parse_conllu(Path(p))
        train_sentences.extend(parsed)

    dev_sentences: List[TokenTagSentence] = []
    if args.dev_conllu:
        for p in args.dev_conllu:
            parsed = parse_conllu(Path(p))
            dev_sentences.extend(parsed)

    if not train_sentences:
        raise ValueError("Tidak ada kalimat train yang berhasil diparse.")

    word_vocab = build_word_vocab(train_sentences, lowercase=args.lowercase)
    char_vocab = build_char_vocab(train_sentences, lowercase=args.lowercase)
    tag_vocab = build_tag_vocab(train_sentences + dev_sentences)

    out_dir = Path(args.out_dir)
    train_path = out_dir / args.train_out_name
    write_jsonl(train_path, train_sentences, word_vocab, char_vocab, tag_vocab, lowercase=args.lowercase)

    if dev_sentences:
        dev_path = out_dir / args.dev_out_name
        write_jsonl(dev_path, dev_sentences, word_vocab, char_vocab, tag_vocab, lowercase=args.lowercase)

    vocab_payload = {
        "special_tokens": {"pad": PAD_TOKEN, "unk": UNK_TOKEN},
        "word_vocab": word_vocab,
        "char_vocab": char_vocab,
        "tag_vocab": tag_vocab,
        "stats": {
            "train_sentences": len(train_sentences),
            "dev_sentences": len(dev_sentences),
            "word_vocab_size": len(word_vocab),
            "char_vocab_size": len(char_vocab),
            "tag_size": len(tag_vocab),
        },
    }
    vocab_path = out_dir / args.vocab_name
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(vocab_payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] train jsonl: {train_path}")
    if dev_sentences:
        print(f"[OK] dev jsonl: {out_dir / args.dev_out_name}")
    print(f"[OK] vocab: {vocab_path}")
    print(f"[INFO] train_sentences={len(train_sentences)}, dev_sentences={len(dev_sentences)}")
    print(f"[INFO] word_vocab={len(word_vocab)}, char_vocab={len(char_vocab)}, tags={len(tag_vocab)}")


if __name__ == "__main__":
    main()
