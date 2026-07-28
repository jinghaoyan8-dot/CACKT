#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run four Laptop14 ablation experiments for absa.run_joint.

Ablations:
  1) w/o SCB
  2) w/o sentiment prior
  3) w/o ACF
  4) w/o CA-SCD

Usage:
  python run_laptop14_4ablation_seed58.py

Optional:
  python run_laptop14_4ablation_seed58.py --root out/Res/laptop14_4ablation_seed58
  python run_laptop14_4ablation_seed58.py --dry_run
"""

import argparse
import datetime as _dt
import re
import subprocess
import sys
from pathlib import Path


def build_common_args(args):
    return [
        "--data_dir", args.data_dir,
        "--train_file", args.train_file,
        "--predict_file", args.predict_file,
        "--bert_config_file", args.bert_config_file,
        "--vocab_file", args.vocab_file,
        "--init_checkpoint", args.init_checkpoint,

        "--num_train_epochs", str(args.num_train_epochs),
        "--train_batch_size", str(args.train_batch_size),
        "--predict_batch_size", str(args.predict_batch_size),
        "--learning_rate", str(args.learning_rate),
        "--max_seq_length", str(args.max_seq_length),
        "--save_proportion", str(args.save_proportion),

        "--candidate_threshold_mode", "dynamic",
        "--dynamic_threshold_scale", "0.22",
        "--candidate_min_keep", "1",
        "--confidence_temperature", "1.0",
        "--min_candidate_confidence", "0.54",

        "--use_sentiment_conditioned_boundary", "True",
        "--sentiment_prior_weight", "0.30",
        "--direct_sentiment_boundary_output", "False",
        "--drop_other_predictions", "True",

        "--use_ac_confidence_filter", "True",
        "--ac_min_confidence", "0.44",
        "--ac_other_margin", "0.03",

        "--random_train", "0.9",
        "--expectation_start_step", "200",
        "--pseudo_confidence_scale", "0.30",

        "--weight_span", "1e-7",
        "--weight_ac", "1",
        "--weight_pair_verifier", "0.30",

        "--n_best_size", "8",
        "--use_consistency_decoding", "True",
        "--consistency_pool_multiplier", "2",
        "--consistency_final_top_k", "0",
        "--pair_nms_overlap", "0.5",

        "--use_pair_verifier", "True",
        "--pair_verifier_alpha", "1.4",

        "--seed", str(args.seed),
    ]


ABLATIONS = [
    (
        "01_wo_SCB",
        [
            "--use_sentiment_conditioned_boundary", "False",
            "--sentiment_prior_weight", "0.00",
        ],
    ),
    (
        "02_wo_sentiment_prior",
        [
            "--use_sentiment_conditioned_boundary", "True",
            "--sentiment_prior_weight", "0.00",
        ],
    ),
    (
        "03_wo_ACF",
        [
            "--use_ac_confidence_filter", "False",
            "--ac_min_confidence", "0.00",
            "--ac_other_margin", "0.00",
        ],
    ),
    (
        "04_wo_CA_SCD",
        [
            "--use_consistency_decoding", "False",
            "--consistency_pool_multiplier", "1",
            "--consistency_final_top_k", "0",
        ],
    ),
]


def run_one(name, extra_args, common_args, root, log_dir, dry_run=False):
    out_dir = root / name
    log_path = log_dir / f"{name}.log"

    cmd = [
        sys.executable,
        "-m",
        "absa.run_joint",
        *common_args,
        "--output_dir",
        str(out_dir),
        *extra_args,
    ]

    print("\n" + "=" * 80)
    print(f"Running : {name}")
    print(f"Output  : {out_dir}")
    print(f"Log     : {log_path}")
    print("=" * 80)
    print(" ".join(cmd))

    if dry_run:
        return 0

    if out_dir.exists():
        # Avoid loading old checkpoint. Ensure each ablation trains from scratch.
        import shutil
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()

    with (root / "status.txt").open("a", encoding="utf-8") as status_file:
        status_file.write(f"{name}\texit_code={return_code}\n")

    if return_code != 0:
        print(f"[WARN] {name} failed with exit code {return_code}; continuing.")

    return return_code


def parse_last_metrics(text):
    all_match = re.findall(
        r"P_all:\s*([0-9.]+),\s*R_all:\s*([0-9.]+),\s*F1_all:\s*([0-9.]+)",
        text,
    )
    ae_match = re.findall(
        r"P_ae:\s*([0-9.]+),\s*R_ae:\s*([0-9.]+),\s*F1_ae:\s*([0-9.]+)",
        text,
    )
    ac_match = re.findall(r"Acc_ac:\s*([0-9.]+)", text)

    if all_match:
        p_all, r_all, f1_all = all_match[-1]
    else:
        p_all = r_all = f1_all = "NA"

    if ae_match:
        p_ae, r_ae, f1_ae = ae_match[-1]
    else:
        p_ae = r_ae = f1_ae = "NA"

    acc_ac = ac_match[-1] if ac_match else "NA"
    return p_all, r_all, f1_all, p_ae, r_ae, f1_ae, acc_ac


def collect_summary(root):
    log_dir = root / "logs"
    rows = []

    for log_path in sorted(log_dir.glob("*.log")):
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        rows.append((log_path.stem, *parse_last_metrics(text)))

    summary_path = root / "summary.tsv"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("exp\tP_all\tR_all\tF1_all\tP_ae\tR_ae\tF1_ae\tAcc_ac\n")
        for row in rows:
            f.write("\t".join(row) + "\n")

    print("\n" + "=" * 80)
    print(f"Saved summary: {summary_path}")
    print("=" * 80)
    print("exp\tP_all\tR_all\tF1_all\tP_ae\tR_ae\tF1_ae\tAcc_ac")
    for row in rows:
        print("\t".join(row))

    return summary_path


def parse_args():
    timestamp = _dt.datetime.now().strftime("%m%d_%H%M")
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", default=f"out/Res/laptop14_4ablation_seed58_{timestamp}",
                        help="Root directory for all ablation outputs.")
    parser.add_argument("--data_dir", default="data/absa")
    parser.add_argument("--train_file", default="split_laptop14_train.txt")
    parser.add_argument("--predict_file", default="laptop14_test.txt")
    parser.add_argument("--bert_config_file", default="bert-large-uncased/bert_config.json")
    parser.add_argument("--vocab_file", default="bert-large-uncased/vocab.txt")
    parser.add_argument("--init_checkpoint", default="bert-large-uncased/pytorch_model.bin")

    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--predict_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", default="2e-5")
    parser.add_argument("--max_seq_length", type=int, default=85)
    parser.add_argument("--save_proportion", default="0.5")
    parser.add_argument("--seed", type=int, default=58)

    parser.add_argument("--dry_run", action="store_true",
                        help="Only print commands; do not run experiments.")

    return parser.parse_args()


def main():
    args = parse_args()

    root = Path(args.root)
    log_dir = root / "logs"
    root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    common_args = build_common_args(args)

    print(f"Results root: {root}")
    print(f"Logs dir    : {log_dir}")

    failed = []
    for name, extra_args in ABLATIONS:
        code = run_one(
            name=name,
            extra_args=extra_args,
            common_args=common_args,
            root=root,
            log_dir=log_dir,
            dry_run=args.dry_run,
        )
        if code != 0:
            failed.append(name)

    if not args.dry_run:
        collect_summary(root)

    print("\nAll requested ablations finished.")
    if failed:
        print("Failed runs:", ", ".join(failed))
    print(f"Root: {root}")


if __name__ == "__main__":
    main()
