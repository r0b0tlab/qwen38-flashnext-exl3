#!/usr/bin/env python3
"""
Prepare the manual-review pair for a Q200v2 run's hard_reasoning rows
(mirrors the 27B campaign's procedure): extracts the rows into
manual-review-input.json and writes a manual-evidence.json skeleton
(ids + content hashes pre-bound) for the reviewer to fill.

Usage:
  python3 scripts/make_manual_review.py <run_dir> [--reviewer hermes-agent]
Outputs (in run_dir):
  manual-review-input.json   — the hard_reasoning rows, full schema
  manual-evidence.json       — skeleton: {schema, dataset_sha256, run_identity_sha256,
                               reviewer, method, rows:[{id, content_sha256, passed, rationale}]}
"""
import argparse
import glob
import json
import os

MANUAL_EVIDENCE_SCHEMA = "r0b0tlab.qwen38.manual_evidence.v1"
MANUAL_REVIEW_METHOD = "independent_manual_review"  # must match the kit constant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--reviewer", default = "hermes-agent")
    args = ap.parse_args()

    rows = []
    for path in glob.glob(os.path.join(args.run_dir, "*.rows.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    hard = [r for r in rows if r.get("family") == "hard_reasoning"]
    if not hard:
        raise SystemExit("no hard_reasoning rows found")

    dataset_sha = {r.get("dataset_sha256") for r in hard}
    dataset_sha256 = dataset_sha.pop() if len(dataset_sha) == 1 else (hard[0].get("dataset_sha256") or "")
    # The kit validates the evidence against the RUNNER-computed run identity
    # (_sha256_json(identity), recorded in the runner's summary.json) -- NOT the
    # row-level identity_sha256 field.
    run_identity_sha256 = ""
    for spath in glob.glob(os.path.join(args.run_dir, "*.summary.json")):
        try:
            with open(spath) as f:
                run_identity_sha256 = json.load(f).get("run_identity_sha256") or ""
        except Exception:
            pass
    if not run_identity_sha256:
        raise SystemExit("run_identity_sha256 not found (need the runner's *.summary.json)")

    with open(os.path.join(args.run_dir, "manual-review-input.json"), "w") as f:
        json.dump(hard, f, indent = 2)

    evidence = {
        "schema": MANUAL_EVIDENCE_SCHEMA,
        "dataset_sha256": dataset_sha256,
        "run_identity_sha256": run_identity_sha256,
        "reviewer": args.reviewer,
        "method": MANUAL_REVIEW_METHOD,
        "rows": [
            {"id": r["id"], "content_sha256": r.get("content_sha256"),
             "passed": None, "rationale": ""}
            for r in hard
        ],
    }
    with open(os.path.join(args.run_dir, "manual-evidence.json"), "w") as f:
        json.dump(evidence, f, indent = 2)

    print(json.dumps({
        "run_dir": os.path.abspath(args.run_dir),
        "hard_reasoning_rows": len(hard),
        "dataset_sha256": dataset_sha256,
        "run_identity_sha256": run_identity_sha256,
        "wrote": ["manual-review-input.json", "manual-evidence.json (skeleton: fill passed/rationale)"],
    }, indent = 2))


if __name__ == "__main__":
    main()
