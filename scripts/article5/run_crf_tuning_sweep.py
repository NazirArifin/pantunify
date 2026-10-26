import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run 6-run CRF tuning sweep")
    parser.add_argument("--train-script", default="scripts/article5/train_hybrid_bilstm_crf_pos.py")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--valid-jsonl", required=True)
    parser.add_argument("--vocab-json", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--word-emb-dim", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--aux-ce-loss-weight", type=float, default=0.0)
    parser.add_argument("--class-weighting", choices=("none", "critical"), default="none")
    parser.add_argument("--critical-class-weight", type=float, default=2.0)
    parser.add_argument("--non-critical-class-weight", type=float, default=1.0)
    parser.add_argument("--sampling-strategy", choices=("none", "oov_critical"), default="none")
    parser.add_argument("--sampling-oov-alpha", type=float, default=1.5)
    parser.add_argument("--sampling-critical-alpha", type=float, default=1.0)
    parser.add_argument("--sampling-num-samples", type=int, default=0)
    parser.add_argument("--sampling-no-replacement", action="store_true")
    parser.add_argument("--enable-disemvowel-augmentation", action="store_true")
    parser.add_argument("--disemvowel-tags", default="ADV,PART,PRON,CCONJ,SCONJ")
    parser.add_argument("--disemvowel-prob", type=float, default=0.5)
    parser.add_argument("--disemvowel-vowels", default="aiueoAIUEO")
    parser.add_argument("--disemvowel-allow-unk-word-rewrite", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def build_sweep_configs() -> List[Dict]:
    return [
        {
            "name": "run_01_lr1e3_crf3e3_do03_wd1e5_gc1",
            "lr": 1e-3,
            "lr_crf": 3e-3,
            "dropout": 0.3,
            "weight_decay": 1e-5,
            "grad_clip": 1.0,
        },
        {
            "name": "run_02_lr1e3_crf5e3_do03_wd1e5_gc1",
            "lr": 1e-3,
            "lr_crf": 5e-3,
            "dropout": 0.3,
            "weight_decay": 1e-5,
            "grad_clip": 1.0,
        },
        {
            "name": "run_03_lr2e3_crf3e3_do03_wd1e5_gc1",
            "lr": 2e-3,
            "lr_crf": 3e-3,
            "dropout": 0.3,
            "weight_decay": 1e-5,
            "grad_clip": 1.0,
        },
        {
            "name": "run_04_lr2e3_crf5e3_do03_wd1e5_gc1",
            "lr": 2e-3,
            "lr_crf": 5e-3,
            "dropout": 0.3,
            "weight_decay": 1e-5,
            "grad_clip": 1.0,
        },
        {
            "name": "run_05_lr1e3_crf3e3_do05_wd1e4_gc5",
            "lr": 1e-3,
            "lr_crf": 3e-3,
            "dropout": 0.5,
            "weight_decay": 1e-4,
            "grad_clip": 5.0,
        },
        {
            "name": "run_06_lr2e3_crf5e3_do05_wd1e4_gc5",
            "lr": 2e-3,
            "lr_crf": 5e-3,
            "dropout": 0.5,
            "weight_decay": 1e-4,
            "grad_clip": 5.0,
        },
    ]


def load_vocab_sizes(vocab_json_path: Path) -> Dict[str, int]:
    with open(vocab_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return {
        "num_chars": len(obj["char_vocab"]),
        "vocab_size": len(obj["word_vocab"]),
        "num_classes": len(obj["tag_vocab"]),
    }


def build_command(
    python_exe: str,
    train_script: Path,
    config: Dict,
    vocab_sizes: Dict[str, int],
    args: argparse.Namespace,
) -> List[str]:
    run_dir = Path(args.out_root) / config["name"]
    cmd = [
        python_exe,
        str(train_script),
        "--train-jsonl",
        args.train_jsonl,
        "--valid-jsonl",
        args.valid_jsonl,
        "--save-dir",
        str(run_dir),
        "--num-chars",
        str(vocab_sizes["num_chars"]),
        "--vocab-size",
        str(vocab_sizes["vocab_size"]),
        "--num-classes",
        str(vocab_sizes["num_classes"]),
        "--word-emb-dim",
        str(args.word_emb_dim),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--lr",
        str(config["lr"]),
        "--lr-crf",
        str(config["lr_crf"]),
        "--dropout",
        str(config["dropout"]),
        "--weight-decay",
        str(config["weight_decay"]),
        "--grad-clip",
        str(config["grad_clip"]),
        "--best-metric",
        "composite",
        "--score-weight-acc",
        "0.5",
        "--score-weight-oov-acc",
        "0.3",
        "--score-weight-critical-f1",
        "0.2",
        "--critical-tags",
        "ADV,PART,NOUN,VERB",
        "--tag-vocab-json",
        args.vocab_json,
        "--vocab-json",
        args.vocab_json,
        "--early-stopping-patience",
        "4",
        "--metrics-jsonl",
        str(run_dir / "metrics.jsonl"),
        "--aux-ce-loss-weight",
        str(args.aux_ce_loss_weight),
        "--class-weighting",
        args.class_weighting,
        "--critical-class-weight",
        str(args.critical_class_weight),
        "--non-critical-class-weight",
        str(args.non_critical_class_weight),
        "--sampling-strategy",
        args.sampling_strategy,
        "--sampling-oov-alpha",
        str(args.sampling_oov_alpha),
        "--sampling-critical-alpha",
        str(args.sampling_critical_alpha),
        "--sampling-num-samples",
        str(args.sampling_num_samples),
    ]
    if args.sampling_no_replacement:
        cmd.append("--sampling-no-replacement")
    if args.enable_disemvowel_augmentation:
        cmd.extend(
            [
                "--enable-disemvowel-augmentation",
                "--disemvowel-tags",
                args.disemvowel_tags,
                "--disemvowel-prob",
                str(args.disemvowel_prob),
                "--disemvowel-vowels",
                args.disemvowel_vowels,
            ]
        )
        if args.disemvowel_allow_unk_word_rewrite:
            cmd.append("--disemvowel-allow-unk-word-rewrite")
    return cmd


def main() -> None:
    args = build_arg_parser().parse_args()
    train_script = Path(args.train_script)
    if not train_script.exists():
        raise FileNotFoundError(f"train script not found: {train_script}")
    vocab_json_path = Path(args.vocab_json)
    if not vocab_json_path.exists():
        raise FileNotFoundError(f"vocab json not found: {vocab_json_path}")

    vocab_sizes = load_vocab_sizes(vocab_json_path)
    print("[INFO] Vocab sizes:", vocab_sizes)

    configs = build_sweep_configs()
    python_exe = sys.executable
    print(f"[INFO] Python: {python_exe}")
    print(f"[INFO] Execute mode: {args.execute}")

    for i, config in enumerate(configs, start=1):
        cmd = build_command(
            python_exe=python_exe,
            train_script=train_script,
            config=config,
            vocab_sizes=vocab_sizes,
            args=args,
        )
        pretty = " ".join(shlex.quote(x) for x in cmd)
        print(f"\n[RUN {i:02d}] {config['name']}")
        print(pretty)
        if args.execute:
            subprocess.run(cmd, check=True)

    if not args.execute:
        print("\n[INFO] Dry run selesai. Tambahkan --execute untuk menjalankan semua eksperimen.")


if __name__ == "__main__":
    main()
