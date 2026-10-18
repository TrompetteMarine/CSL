#!/usr/bin/env python3
"""Parse repository Python sources without importing or executing simulation code."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".lake", "__pycache__", ".venv", "venv", "node_modules", "downloads", "runtime", "rebuilt"}
files = [p for p in ROOT.rglob("*.py") if not SKIP.intersection(p.relative_to(ROOT).parts)]
for file in files:
    ast.parse(file.read_text(encoding="utf-8"), filename=str(file.relative_to(ROOT)))
print(f"Python syntax passed: {len(files)} files (no simulations executed).")
