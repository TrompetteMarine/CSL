#!/usr/bin/env bash
# Initialise a clean git repository for CSL and make the first commit on 'main'.
# Safe to run once in a freshly copied folder. Does not push (see PUSH_INSTRUCTIONS.md).
set -euo pipefail
cd "$(dirname "$0")"

if [ -d .git ]; then
  echo "A .git directory already exists here; aborting to avoid clobbering it."
  exit 1
fi

git init -q
git add -A
git commit -q -m "CSL v1.0.0 — Endogenous Credibility in Social Learning"
git branch -M main

echo "Initialised git repo on branch 'main' with $(git ls-files | wc -l | tr -d ' ') tracked files."
echo "Next: create the GitHub repo and push (see PUSH_INSTRUCTIONS.md)."
