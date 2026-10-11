#!/usr/bin/env python
"""Download the evaluation sets the existing data layouts do not have yet.

    python scripts/download_extra_data.py [--data-root data] [--oc-data-root <COMPASS_DATA_CACHE>]
        [--parts truthfulqa winogrande d1]

Into <data-root>/eval/<name>/ (Future_dLLM layout, read by scripts/run_eval.sh), each with a
SOURCE.json:

    truthfulqa/generation.parquet   817   truthfulqa/truthful_qa generation (lm-eval truthfulqa_gen)
    countdown/test.parquet          256   d1 dataset/countdown_cd3_test.jsonl
    sudoku/test.parquet             500   d1 dataset/4x4_test_sudoku.csv
    sudoku/test256.parquet          256   the 256 puzzles d1's eval.py scores (see below)

Into <oc-data-root>/data/ (OpenCompass's local layout, beside ARC/, piqa/, gpqa/):

    winogrande/dev.jsonl           1267   OpenCompass's own winogrande.zip (winogrande_gen reads dev.jsonl)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

D1_REPO = "dllm-reasoning/d1"
# Commit the Countdown / Sudoku files are read from (main at the time of writing).
D1_COMMIT = "faf85d6dcea40477c722c467023c73b08a46cb82"
D1_RAW = f"https://raw.githubusercontent.com/{D1_REPO}/{D1_COMMIT}/"


def _write_parquet(frame: pd.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return len(frame)


def _write_source(path: Path, **payload) -> None:
    payload.setdefault("downloaded", date.today().isoformat())
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as src:
        return src.read()


def download_truthfulqa(root: Path, cache: str) -> None:
    """The generation config lm-eval's truthfulqa_gen reads (question, best/correct/incorrect
    answers)."""
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "generation", cache_dir=cache)
    rows = {"generation.parquet": _write_parquet(
        ds["validation"].to_pandas(), root / "eval/truthfulqa/generation.parquet")}
    _write_source(root / "eval/truthfulqa/SOURCE.json", source_dataset="truthfulqa/truthful_qa",
                  source_config="generation", split="validation", rows=rows)


OC_WINOGRANDE_URL = "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/winogrande.zip"


def download_oc_winogrande(oc_root: Path) -> None:
    """OpenCompass's winogrande.zip (the URL in its utils/datasets_info.py), unpacked into
    <oc_root>/data/winogrande/ where WinograndeDatasetV2 looks for dev.jsonl."""
    data = oc_root / "data"
    data.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(_fetch(OC_WINOGRANDE_URL))) as zf:
        zf.extractall(data)
    dev = data / "winogrande/dev.jsonl"
    print(f"{dev}: {sum(1 for _ in dev.open())} rows")


def download_d1(root: Path) -> None:
    """Countdown (3 numbers) and 4x4 Sudoku test sets from d1 (arXiv:2504.12216).

    d1's eval.py scores all 256 Countdown items and a 256-of-500 Sudoku subsample:
    init_seed(42), then np.random.choice(500, 256, replace=False) with nothing else
    drawing from numpy in between. test256.parquet is that subsample, in that order;
    its puzzle set matches d1's published eval_baselines generations."""
    lines = _fetch(D1_RAW + "dataset/countdown_cd3_test.jsonl").decode("utf-8").splitlines()
    items = [json.loads(line) for line in lines if line.strip()]
    countdown = pd.DataFrame({"numbers": [it["input"] for it in items],
                              "target": [it["output"] for it in items]})
    rows = {"test.parquet": _write_parquet(countdown, root / "eval/countdown/test.parquet")}
    _write_source(root / "eval/countdown/SOURCE.json", source_repo=D1_REPO, source_commit=D1_COMMIT,
                  source_file="dataset/countdown_cd3_test.jsonl",
                  columns={"numbers": "jsonl 'input'", "target": "jsonl 'output'"}, rows=rows)

    sudoku = pd.read_csv(io.BytesIO(_fetch(D1_RAW + "dataset/4x4_test_sudoku.csv")),
                         dtype={"Puzzle": str, "Solution": str})
    np.random.seed(42)
    subsample = np.random.choice(len(sudoku), 256, replace=False)
    sub = sudoku.iloc[subsample].reset_index(drop=True)
    sub.insert(0, "source_index", subsample)
    rows = {"test.parquet": _write_parquet(sudoku, root / "eval/sudoku/test.parquet"),
            "test256.parquet": _write_parquet(sub, root / "eval/sudoku/test256.parquet")}
    _write_source(root / "eval/sudoku/SOURCE.json", source_repo=D1_REPO, source_commit=D1_COMMIT,
                  source_file="dataset/4x4_test_sudoku.csv",
                  subsample="np.random.seed(42); np.random.choice(500, 256, replace=False)",
                  rows=rows)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=str(repo / "data"))
    parser.add_argument("--oc-data-root",
                        default=os.environ.get("COMPASS_DATA_CACHE", str(repo.parents[0] / "dllm/oc")),
                        help="OpenCompass COMPASS_DATA_CACHE (holds data/ARC, data/piqa, ...)")
    parser.add_argument("--hf-cache", default=str(repo / ".cache/hf_datasets"),
                        help="datasets cache used while downloading; not needed afterwards")
    parser.add_argument("--parts", nargs="+", choices=("truthfulqa", "winogrande", "d1"),
                        default=["truthfulqa", "winogrande", "d1"])
    args = parser.parse_args()
    root = Path(args.data_root).expanduser().resolve()
    if "truthfulqa" in args.parts:
        download_truthfulqa(root, args.hf_cache)
    if "winogrande" in args.parts:
        download_oc_winogrande(Path(args.oc_data_root).expanduser().resolve())
    if "d1" in args.parts:
        download_d1(root)
    print(f"datasets written under {root}/eval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
