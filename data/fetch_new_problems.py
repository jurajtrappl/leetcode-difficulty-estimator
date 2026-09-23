#!/usr/bin/env python3
"""Download LeetCode problems published AFTER the original dataset, as an unseen test set.

Why: Llama 3.1 was trained on web data up to December 2023, so it may simply remember the difficulty of the
problems in leetcode_problems_dataset.json (downloaded in December 2023). Problems LeetCode published later
cannot be in its training data, so they show how well a model judges difficulty from the text alone.

    python data/fetch_new_problems.py                  # problems with ID >= 3000 (~January 2024 onwards)
    python data/fetch_new_problems.py --min-id 3100    # stricter cut-off

No login needed (only free problems are downloaded). The result is written to data/leetcode_new_problems.json
in the same format as the original dataset (plus "frontend_id" and "title"). The download resumes if you stop
it, and it waits between requests to be polite to LeetCode.

About the cut-off: LeetCode doesn't publish a creation date, but problem IDs grow with time (~25 new problems a
month). The original dataset ends at ID 2950 (early December 2023); --min-id 3000 skips the rest of December
2023 as a safety margin around the model's training cut-off.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `config` imports when run as a script
from config import CFG  # also loads .env

GRAPHQL = "https://leetcode.com/graphql/"
LIST_QUERY = """
query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {
  problemsetQuestionList: questionList(categorySlug: $categorySlug, limit: $limit, skip: $skip, filters: $filters) {
    total: totalNum
    questions: data { questionFrontendId title titleSlug difficulty isPaidOnly }
  }
}"""
QUESTION_QUERY = """
query questionData($titleSlug: String!) {
  question(titleSlug: $titleSlug) { questionFrontendId title titleSlug content difficulty isPaidOnly }
}"""


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Content-Type": "application/json",
        "Referer": "https://leetcode.com/problemset/",
        "Origin": "https://leetcode.com",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/126.0 Safari/537.36",
    })
    s.get("https://leetcode.com/problemset/", timeout=30)  # picks up the csrftoken cookie
    if "csrftoken" in s.cookies:
        s.headers["x-csrftoken"] = s.cookies["csrftoken"]
    return s


def graphql(session, query, variables, retries=5):
    for attempt in range(retries):
        r = session.post(GRAPHQL, json={"query": query, "variables": variables}, timeout=30)
        if r.status_code == 429 or r.status_code >= 500:           # rate limited / server hiccup: back off
            time.sleep(5 * 2 ** attempt)
            continue
        r.raise_for_status()
        body = r.json()
        if body.get("errors"):
            raise RuntimeError(f"LeetCode GraphQL error: {body['errors']}")
        return body["data"]
    raise RuntimeError(f"LeetCode kept refusing requests (last status {r.status_code})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-id", type=int, default=3000, help="first problem ID to download (default 3000)")
    ap.add_argument("--out", type=Path, default=CFG.dataset_path.with_name("leetcode_new_problems.json"))
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between requests (default 1)")
    args = ap.parse_args()

    old = set(json.load(open(CFG.dataset_path)))
    result = json.load(open(args.out)) if args.out.exists() else {}
    session = make_session()

    # 1. list of all problems (100 per page), keep the free ones with ID >= min_id
    candidates, skip = [], 0
    while True:
        page = graphql(session, LIST_QUERY, {"categorySlug": "", "limit": 100, "skip": skip, "filters": {}})
        page = page["problemsetQuestionList"]
        for q in page["questions"]:
            fid = q["questionFrontendId"]
            if fid.isdigit() and int(fid) >= args.min_id and not q["isPaidOnly"] and q["titleSlug"] not in old:
                candidates.append(q)
        skip += 100
        if skip >= page["total"]:
            break
        time.sleep(args.sleep)
    todo = [q for q in candidates if q["titleSlug"] not in result]
    print(f"{len(candidates)} free problems with ID >= {args.min_id}; {len(result)} already downloaded, {len(todo)} to go")

    # 2. statement + difficulty of each, saved every 20 problems so a stopped run resumes
    for n, q in enumerate(todo, 1):
        data = graphql(session, QUESTION_QUERY, {"titleSlug": q["titleSlug"]})["question"]
        if data and data.get("content"):
            result[q["titleSlug"]] = {"difficulty": data["difficulty"], "content": data["content"],
                                      "frontend_id": int(data["questionFrontendId"]), "title": data["title"]}
        if n % 20 == 0 or n == len(todo):
            json.dump(result, open(args.out, "w"))
            print(f"  {n}/{len(todo)} downloaded")
        time.sleep(args.sleep)

    counts = {d: sum(v["difficulty"] == d for v in result.values()) for d in ("Easy", "Medium", "Hard")}
    ids = [v["frontend_id"] for v in result.values()]
    print(f"saved {len(result)} problems to {args.out} (IDs {min(ids, default='-')}-{max(ids, default='-')}): {counts}")


if __name__ == "__main__":
    main()
