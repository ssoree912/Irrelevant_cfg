"""Countdown and 4x4 Sudoku with d1's own evaluation (dllm-reasoning/d1, arXiv:2504.12216).

Neither set has an lm-eval task; d1's eval/ is the reference protocol, and this follows it:

  data      Countdown: all 256 of dataset/countdown_cd3_test.jsonl
            Sudoku:    d1's 256-of-500 subsample of dataset/4x4_test_sudoku.csv
            (scripts/download_extra_data.py writes both under $DATA_DIR/eval/)
  prompt    d1's system prompt + "\\n\\n" + question as one user turn, the tokenizer's chat
            template with the generation prompt, then "<reasoning>" prefilled
            (eval/countdown.py, eval/sudoku.py, GSM8KDataset.create_prompt)
  decode    the generated tokens with special tokens kept (eval.py)
  score     parse_countdown_answers / parse_sudoku_answers from eval/parse_and_get_acc.py:
            Countdown accuracy = correct expressions / items,
            Sudoku accuracy = correctly filled empty cells / empty cells (summed over puzzles)

Generation is this repo's: the LLaDA_cfg lm-eval model's generate_until (single block,
cache-free, any vanilla / neg_* / id_* config, family defaults, LLADA_CFG_STORE resume and
LLADA_CFG_SHARD sharding). d1 itself decodes gen_length 128 as 4 blocks of 32 in 64 steps.

    python eval/d1_eval.py --task countdown|sudoku --model_args "pretrained=<dir>,config=vanilla" \\
        --output_path <dir> [--limit N]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm_eval_model import LLaDACFG  # noqa: E402

# d1 eval/countdown.py, verbatim.
CTD_SYSTEM_PROMPT = (
    "Using only the provided numbers, create an arithmetic expression that evaluates to exactly the provided target number. You may use the operations +, -, *, and / as needed, but each number must be used exactly once. Think step-by-step. After reasoning, provide only your final expression inside \\boxed"
    + "{}"
    + " tags without including an equals sign or the target number. For example: \\boxed{a + b * c}"
    + """Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""
)

# d1 eval/sudoku.py, verbatim.
SUDOKU_SYSTEM_PROMPT = """
Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

Rules:
- Fill empty cells with digits 1-4
- Each row must contain digits 1-4 exactly once
- Each column must contain digits 1-4 exactly once
- Each 2x2 box must contain digits 1-4 exactly once

Important: Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid.

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
[16-character solution string with no spaces or separators]
</answer>
"""


def load_items(task, data_dir):
    """(question, ground_truth) pairs as d1's datasets build them."""
    if task == "countdown":
        frame = pd.read_parquet(Path(data_dir) / "eval/countdown/test.parquet")
        items = []
        for numbers_str, target in zip(frame["numbers"], frame["target"]):
            numbers = [int(num) for num in numbers_str.split(",")]
            items.append({"question": f"Numbers: {numbers}\nTarget: {int(target)}",
                          "ground_truth": [numbers, int(target)]})
        return items, CTD_SYSTEM_PROMPT
    frame = pd.read_parquet(Path(data_dir) / "eval/sudoku/test256.parquet")
    return ([{"question": f"Solve the following Sudoku puzzle: {puzzle}\n", "ground_truth": solution}
             for puzzle, solution in zip(frame["Puzzle"], frame["Solution"])],
            SUDOKU_SYSTEM_PROMPT)


def build_prompt(tokenizer, system_prompt, question):
    messages = [{"role": "user", "content": system_prompt + "\n\n" + question}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                         tokenize=False) + "<reasoning>"


# d1 eval/parser_helper.py, verbatim.
def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return string

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"

        return s[len(left) : -1]
    except:  # noqa: E722
        return s


# The per-item bodies of d1 eval/parse_and_get_acc.py's parse_countdown_answers and
# parse_sudoku_answers, without the tiktoken token count.
def score_countdown(item):
    def validate_equation(equation_str, available_numbers):
        try:
            numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
            return sorted(numbers_in_eq) == sorted(available_numbers)
        except:  # noqa: E722
            return False

    def evaluate_equation(equation_str):
        try:
            if not re.match(r"^[\d+\-*/().\s]+$", equation_str):
                raise ValueError("Invalid characters in equation.")
            return eval(equation_str.strip(), {"__builtins__": None}, {})
        except Exception:
            return float("Inf")

    numbers, target = item["ground_truth"]
    generated_text = item["generations"]
    equation = ""
    try:
        equation = remove_boxed(last_boxed_only_string(generated_text))
    except:  # noqa: E722
        answer_match = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)
        equation = answer_match.group(1).strip() if answer_match else generated_text
    equation = equation.replace(r"\div", "/").replace(r"\times", "*").replace(r"\cdot", "*")
    equation_match = re.search(r"([0-9+\-*/() ]+)=[0-9. ]+", equation)
    if equation_match:
        equation = equation_match.group(1).strip()
    is_correct, result = False, None
    if validate_equation(equation, numbers):
        result = evaluate_equation(equation)
        if target is not None and abs(result - target) < 1e-5:
            is_correct = True
    return {"extracted_answer": equation, "evaluation_result": result, "is_correct": is_correct,
            "correct": int(is_correct), "total": 1}


def score_sudoku(item):
    question, ground_truth, raw_generation = (item["question"], item["ground_truth"],
                                              item["generations"])
    puzzle_str = re.search(r"Sudoku puzzle: ([0-9]{16})", question).group(1)
    empty_indices = [i for i in range(16) if puzzle_str[i] == "0"]
    solution_str = ""
    patterns = [
        r"<answer>.*?```\s*([\d\s]+)```",
        r"<answer>(.*?)(?:<\|eot_id\|>|<\|endoftext\|>|</answer>)",
        r"</answer>\s*(.*?)(?:<\|eot_id\|>|<\|endoftext\|>|$)",
        r".*?(\d{16})\s*</answer>",
        r"\b(\d{16})\b",
    ]
    for pattern in patterns:
        if solution_str:
            break
        match = re.search(pattern, raw_generation, re.DOTALL)
        if match and match.group(1).strip():
            solution_str = match.group(1).strip()
    solution_str = re.sub(r"\s", "", solution_str)
    if not solution_str:
        correct_cells = 0
    else:
        solution_str = (solution_str + "0" * 16)[:16]
        correct_cells = sum(1 for i in empty_indices if solution_str[i] == ground_truth[i])
    return {"extracted_answer": solution_str, "correct": correct_cells,
            "total": len(empty_indices), "is_correct": correct_cells == len(empty_indices)}


def stored_tokens(store_path):
    """doc_id -> generated token ids from the LLaDA_cfg store and its shard files."""
    tokens = {}
    paths = [store_path] + sorted(glob.glob(glob.escape(store_path) + ".shard*"))
    for path in (p for p in paths if os.path.exists(p)):
        with open(path) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tokens[rec["doc_id"]] = rec["tokens"]
    return tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=("countdown", "sudoku"))
    parser.add_argument("--model_args", required=True,
                        help="LLaDA_cfg model args, e.g. pretrained=<dir>,config=vanilla")
    parser.add_argument("--data_dir", default=str(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out = Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)
    store_path = os.environ.get("LLADA_CFG_STORE") or str(out / "generations.jsonl")
    os.environ["LLADA_CFG_STORE"] = store_path
    n_shards = int(os.environ.get("LLADA_CFG_SHARD", "0/1").split("/")[1])

    model_args = dict(kv.split("=", 1) for kv in args.model_args.split(",") if kv)
    lm = LLaDACFG(**model_args)
    items, system_prompt = load_items(args.task, args.data_dir)
    if args.limit:
        items = items[:args.limit]
    for item in items:
        item["prompt_input"] = build_prompt(lm.tokenizer, system_prompt, item["question"])
    lm.generate_until([SimpleNamespace(args=(item["prompt_input"], {"until": []}), doc_id=i)
                       for i, item in enumerate(items)])
    if n_shards > 1:
        return 0  # shard pass: generations only, the unsharded pass scores

    tokens = stored_tokens(store_path)
    score = score_countdown if args.task == "countdown" else score_sudoku
    scored = []
    for i, item in enumerate(items):
        item["generations"] = lm.tokenizer.decode(tokens[i], skip_special_tokens=False)
        scored.append({**item, **score(item)})
    correct = sum(s["correct"] for s in scored)
    total = sum(s["total"] for s in scored)
    results = {"task": args.task, "model_args": args.model_args, "n": len(scored),
               "accuracy": correct / total, "correct": correct, "total": total}
    if args.task == "sudoku":
        results["solved"] = sum(s["is_correct"] for s in scored) / len(scored)
    # d1's generations file layout, so eval/parse_and_get_acc.py can re-score it.
    with open(out / f"{args.task}_generations.json", "w") as fh:
        json.dump({"generations": [{k: item[k] for k in ("question", "prompt_input", "generations",
                                                          "ground_truth")} for item in items]},
                  fh, indent=2)
    with open(out / "samples.jsonl", "w") as fh:
        for s in scored:
            fh.write(json.dumps(s) + "\n")
    with open(out / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
