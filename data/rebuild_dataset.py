#!/usr/bin/env python3
"""Rebuild the two dataset files from the problem lists committed in this repo.

The problem statements belong to LeetCode, so the repo doesn't ship them. It ships only the list of problems
(slug + difficulty, in the original order): data/problem_list.json and data/new_problem_list.json.
This script downloads the statements for exactly those problems, so every split, cache and result in the repo
lines up again:

    python data/rebuild_dataset.py            # both files (~3,200 requests, about an hour with the default pause)
    python data/rebuild_dataset.py --only old # just data/leetcode_problems_dataset.json
    python data/rebuild_dataset.py --only new # just data/leetcode_new_problems.json

No login needed (all problems are free). A stopped run resumes where it ended. The difficulty labels are taken
from the lists (the labels used in the experiments), not from today's LeetCode.

Caveat: LeetCode occasionally edits a statement, and a problem can become premium-only (then it's skipped with a
warning), so a rebuilt dataset can differ slightly from the one the results were measured on.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `config` imports when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))      # ... and fetch_new_problems
from config import CFG  # noqa: E402
from fetch_new_problems import QUESTION_QUERY, graphql, make_session  # noqa: E402

DATA = Path(__file__).resolve().parent
TARGETS = {  # list file -> dataset file (the paths the notebooks read)
    "old": (DATA / "problem_list.json", CFG.dataset_path),
    "new": (DATA / "new_problem_list.json", CFG.dataset_path.with_name("leetcode_new_problems.json")),
}


def rebuild(list_path: Path, out_path: Path, session, pause: float):
    problems = json.load(open(list_path))["problems"]
    have = json.load(open(out_path)) if out_path.exists() else {}
    todo = [p for p in problems if p["slug"] not in have]
    print(f"{out_path.name}: {len(problems)} problems, {len(problems) - len(todo)} already there, {len(todo)} to download")

    missing = []
    for n, p in enumerate(todo, 1):
        q = graphql(session, QUESTION_QUERY, {"titleSlug": p["slug"]})["question"]
        if q and q.get("content"):
            have[p["slug"]] = {**{k: v for k, v in p.items() if k != "slug"}, "content": q["content"]}
        else:
            missing.append(p["slug"])
        if n % 20 == 0 or n == len(todo):
            _write(have, problems, out_path)
            print(f"  {n}/{len(todo)}")
        time.sleep(pause)
    _write(have, problems, out_path)
    if missing:
        print(f"  ⚠ {len(missing)} problems have no free statement any more and were skipped: {missing[:5]}...")


def _write(have: dict, problems: list, out_path: Path):
    # always in the order of the list: the train/test split depends on it
    ordered = {p["slug"]: have[p["slug"]] for p in problems if p["slug"] in have}
    json.dump(ordered, open(out_path, "w"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=sorted(TARGETS), help="rebuild just one of the two files")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between requests (default 1)")
    args = ap.parse_args()
    session = make_session()
    for name in ([args.only] if args.only else list(TARGETS)):
        rebuild(*TARGETS[name], session, args.sleep)


if __name__ == "__main__":
    main()
