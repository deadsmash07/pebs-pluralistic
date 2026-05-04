"""Submission-integrity gate: verify every claim in PAPER_CLAIMS.json resolves to an artifact on disk.

Usage:
  python3 IMPLEMENTATION/scripts/check_paper_claims_artifacts.py
  python3 IMPLEMENTATION/scripts/check_paper_claims_artifacts.py --strict  # exit non-zero if any missing

Artifact path resolution:
  - If path starts with "IMPLEMENTATION/", resolve relative to <DATA_ROOT>/
  - Else, resolve relative to the track's repo root (via claim.track field)
  - "{a,b}" brace-expansion expanded to multiple candidates
  - " + "-separated multi-path artifacts each checked individually
  - "*" glob patterns expanded via glob.glob

Purpose: catch regressions where a paper claim's supporting artifact has been
deleted or moved. Designed to be run from CI or as a pre-submission gate.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = "<HOME>/<DATA_ROOT>/nw"
IMPL = f"{ROOT}/IMPLEMENTATION"
TRACK_ROOTS = {
    1: f"{IMPL}/1_Causal_RLHF",
    2: f"{IMPL}/2_Delay_Aware_RLHF",
    3: f"{IMPL}/3_PILSD_Standalone",
}


def resolve(track: int, p: str) -> tuple[bool, str]:
    p = p.strip()
    if not p:
        return False, ""
    if p.startswith("IMPLEMENTATION/"):
        full = f"{ROOT}/{p}"
    elif p.startswith("/"):
        full = p
    else:
        full = f"{TRACK_ROOTS.get(track, IMPL)}/{p}"
    if "*" in full:
        return bool(glob.glob(full)), full
    return os.path.exists(full), full


def expand_paths(artifact: str) -> list[str]:
    """Split a claim's artifact string into individual paths."""
    # strip parenthetical size annotations
    a = artifact.split(" (")[0]
    parts: list[str] = []
    if "{" in a and "}" in a:
        pre, rest = a.split("{", 1)
        opts, post = rest.split("}", 1)
        for opt in opts.split(","):
            parts.append(f"{pre}{opt.strip()}{post}")
    else:
        for seg in a.split(" + "):
            parts.append(seg.strip())
    return [p for p in parts if p]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero if any artifact missing")
    ap.add_argument("--claims", default=f"{IMPL}/PAPER_CLAIMS.json")
    args = ap.parse_args()

    claims = json.loads(open(args.claims).read())
    missing: list[tuple[str, str, str]] = []
    found: list[tuple[str, str]] = []
    total_artifact_claims = 0

    for c in claims["claims"]:
        a = c.get("artifact", "")
        if not a:
            continue
        total_artifact_claims += 1
        for p in expand_paths(a):
            exists, full = resolve(c.get("track", 0), p)
            if exists:
                found.append((c["id"], p))
            else:
                missing.append((c["id"], p, full))

    print(f"Paper version: {claims.get('paper_version', '?')}")
    print(f"Total claims: {len(claims['claims'])}")
    print(f"  with artifact: {total_artifact_claims}")
    print(f"  paths resolved: {len(found)}/{len(found)+len(missing)}")

    if missing:
        print(f"\n  MISSING ({len(missing)}):")
        for i, p, full in missing:
            print(f"    {i:25s}  {p}  (looked at: {full})")
        if args.strict:
            return 1
        return 0

    print("\n  ✓ All artifact paths resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
