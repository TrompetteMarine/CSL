#!/usr/bin/env python3
"""Check the code-only release boundary, pinned sources and saved Lean records.

Saved records are provenance, not a new Lean kernel run. Rebuild with
``make verify-lean`` using the pinned Lean environment for fresh certification.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

from prepare_data import ARRAY_ASSETS, METADATA_ROOT, METADATA_TOP, ROOT, check_file, read_manifests, sha256_file

ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
SKIP_PARTS = {".git", ".lake", "__pycache__", ".venv", "venv", "node_modules", "downloads", "runtime", "rebuilt"}
FORBIDDEN_DIRS = {"article", "theory", "literature", "submission", "checkpoints"}
FORBIDDEN_NAMES = re.compile(r"(?:NHB_Manuscript|NHB_Submission|Supplementary_Material|Original_arxiv|Original_Concept|Manuscript_Redline)", re.I)
SECRET_PATTERNS = [
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def release_files(root: Path) -> list[Path]:
    """Inspect tracked files once Git exists; otherwise the staged release tree."""
    probe = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=root, capture_output=True, text=True)
    if probe.returncode == 0 and probe.stdout.strip() == "true":
        result = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "."], cwd=root, capture_output=True, check=True)
        names = [x.decode() for x in result.stdout.split(b"\0") if x]
        return [root / name for name in names if (root / name).is_file()]
    return [p for p in root.rglob("*") if p.is_file() and not any(x in SKIP_PARTS for x in p.relative_to(root).parts)
            and not p.relative_to(root).as_posix().startswith("data/results/")
            and not p.relative_to(root).as_posix().startswith("baseline/nhb_smallworld_20261012/results/")
            and not (p.parent.relative_to(root).as_posix() == METADATA_ROOT and p.name in METADATA_TOP - {"requirements.txt"})
            and not p.relative_to(root).as_posix().startswith("NHB_Source_Data/")]


def verify_release_boundary(root: Path) -> dict:
    files = release_files(root)
    for file in files:
        name = file.relative_to(root).as_posix()
        parts = file.relative_to(root).parts
        require(not file.is_symlink(), f"Symlink in release source: {name}")
        require(not FORBIDDEN_DIRS.intersection(parts), f"Manuscript directory in code release: {name}")
        require(not FORBIDDEN_NAMES.search(file.name), f"Manuscript document in code release: {name}")
        require(file.suffix.lower() != ".zip", f"ZIP should be a release asset, not a Git file: {name}")
        require(file.stat().st_size <= 100 * 1024**2, f"File exceeds GitHub 100 MiB limit: {name}")
        if file.suffix.lower() in {".tex", ".pdf"}:
            require(parts[0] == "figures", f"Document outside standalone figures: {name}")
        if file.suffix.lower() in {".py", ".cpp", ".c", ".h", ".js", ".mjs", ".json", ".md", ".yaml", ".yml", ".toml", ".txt", ".sh"} or file.name in {"Makefile", "CITATION.cff", ".gitignore"}:
            text = file.read_text(encoding="utf-8", errors="replace")
            for pattern in SECRET_PATTERNS:
                require(not pattern.search(text), f"Possible hard-coded credential in {name}")
    return {"source_files_checked": len(files), "manuscript_and_supplement_excluded": True,
            "no_large_git_files_or_embedded_release_zips": True, "credential_pattern_scan": "passed"}


def verify_source_manifest(root: Path, required: bool, full_source: bool = False) -> dict:
    git_path = root / "replication/git_source_manifest.json"
    path = root / "replication/source_manifest.json" if full_source or not git_path.exists() else git_path
    if not path.exists():
        require(not required, "Missing replication/source_manifest.json")
        return {"status": "not supplied"}
    manifest_path = path
    content = json.loads(manifest_path.read_text())
    records = content.get("files", content)
    if isinstance(records, dict):
        records = [{"path": name, **(value if isinstance(value, dict) else {"sha256": value})} for name, value in records.items()]
    require(isinstance(records, list) and bool(records), "Empty or invalid source manifest")
    names = set()
    for record in records:
        name = record["path"]
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts, f"Invalid source manifest path: {name}")
        require(name not in names, f"Duplicate source manifest path: {name}")
        names.add(name)
        source = root / path
        require(source.is_file(), f"Source manifest file missing: {name}")
        require(sha256_file(source) == record["sha256"], f"Published source hash differs: {name}")
        if "bytes" in record:
            require(source.stat().st_size == record["bytes"], f"Published source byte count differs: {name}")
    if manifest_path.name == "git_source_manifest.json":
        availability = json.loads((root / "replication/DATA_AVAILABILITY.json").read_text())
        deferred = {item["path"] for item in availability["deferred_source_files"]}
        manifests = {"replication/source_manifest.json", "replication/git_source_manifest.json"}
        actual = {p.relative_to(root).as_posix() for p in release_files(root)} - deferred - manifests
        require(names == actual, "Git source manifest does not cover the exact published file set")
    return {"status": "passed", "manifest": manifest_path.name, "files": len(names)}


def verify_lean_records(root: Path) -> dict:
    formal = root / "formal"
    hashes = json.loads((formal / "verified_source_hashes.json").read_text())
    require(len(hashes) == 18, "Expected 18 frozen Lean source hashes")
    for name, expected in hashes.items():
        require(sha256_file(formal / name) == expected, f"Frozen Lean source changed: {name}")
    inventory = json.loads((formal / "declaration_audit.json").read_text())
    summary = json.loads((formal / "verification_summary.json").read_text())
    require(len(inventory) == 117 and summary["total_declarations"] == 117, "Expected 117 saved Lean declarations")
    require(len({x["name"] for x in inventory}) == 117, "Duplicate Lean declaration record")
    counts = Counter("proof" if x["kind"] in {"theorem", "lemma"} else "definition" for x in inventory)
    require(counts == {"proof": 93, "definition": 24}, f"Unexpected Lean declaration inventory: {counts}")
    for record in inventory:
        require(set(record["axioms"]) <= ALLOWED_AXIOMS, f"Unexpected saved axiom: {record['name']}")
        require(record["status"] == "kernel_checked_locally", f"Unexpected saved status: {record['name']}")
    base = formal / "reviewed_baseline/Aristotle_Reviewed_Source_and_Exploration_2026-10-11"
    checker_path = base / "original_submission/check_proofs.py"
    spec = importlib.util.spec_from_file_location("frozen_contract_checker", checker_path)
    require(spec is not None and spec.loader is not None, "Cannot load retained contract checker")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    contract = json.loads((base / "original_submission/Proof_Contract.json").read_text())
    lock = json.loads((formal / "project/lake-manifest.json").read_text())
    pins = {item["name"]: item["rev"] for item in lock["packages"] if item["type"] == "git"}
    require(len(pins) == 9 and pins == contract["git_dependency_revisions"], "Lean dependency pins differ from frozen contract")
    require((formal / "project/lean-toolchain").read_text().strip() == "leanprover/lean4:v4.28.0", "Lean toolchain pin changed")
    require(summary["mathlib"] == pins["mathlib"] and summary["toolchain"] == "Lean4.28.0", "Saved verification toolchain disagrees with source pins")
    protected = 0
    for name, expected in contract["protected_sources"].items():
        require(checker.source_interface(formal / "project" / name) == expected, f"Protected original interface changed: {name}")
        protected += len(expected["declarations"])
    require(protected == 28 and len(contract["theorems"]) == 20,
            "Expected 28 protected original declaration interfaces including 20 proof contracts")
    for source in (base / "reviewed_source").rglob("*.lean"):
        relative = source.relative_to(base / "reviewed_source")
        if relative.parts[0] in {"RepairCore", "RepairTasks", "RepairProb"} or relative.as_posix() in {"RepairCore.lean", "RepairTasks.lean", "RepairProb.lean"}:
            require(source.read_bytes() == (formal / "project" / relative).read_bytes(), f"Reviewed original Lean bytes differ: {relative}")
    actual = []
    for directory in ["RepairCore", "RepairTasks", "RepairProb", "RepairSupplement"]:
        for source in sorted((formal / "project" / directory).glob("*.lean")):
            clean = checker.without_comments(source.read_text())
            forbidden = re.findall(r"\b(?:sorry|admit|sorryAx|axiom|unsafe|native_decide|implemented_by|extern|trustCompiler|ofReduceBool)\b", clean)
            require(not forbidden, f"Forbidden Lean source token in {source.name}: {forbidden}")
            actual += ["CSLRepair." + name for _, name in re.findall(r"(?m)^(theorem|lemma|(?:noncomputable )?def)\s+(\w+)\b", clean)]
    require(set(actual) == {x["name"] for x in inventory} and len(actual) == 117, "Lean source inventory differs from saved records")
    require(summary["build"] == "passed" and summary["type_audit"] == "passed" and summary["axiom_audit"] == "passed", "Retained Lean verification summary did not pass")
    return {"frozen_source_files": 18, "saved_declarations": 117, "proofs": 93, "definitions": 24,
            "protected_original_interfaces": 28, "protected_original_proof_contracts": 20, "pinned_git_dependencies": 9,
            "status": "saved records and source identity validated; no new kernel build"}


def verify_payloads(root: Path, with_data: bool) -> dict:
    assets, payloads = read_manifests(root / "replication")
    arrays = [record for asset in ARRAY_ASSETS for record in payloads[asset]]
    require(len(arrays) == 664 and len({x["path"] for x in arrays}) == 664, "Expected 664 unique raw population arrays")
    if with_data:
        targets = [*ARRAY_ASSETS, "replication_metadata.zip"]
        for asset in targets:
            for record in payloads[asset]:
                check_file(root / record["path"], record)
        actual = {p.relative_to(root).as_posix() for p in (root / "data/results").glob("*.npz")}
        require(actual == {x["path"] for x in arrays}, "Unexpected or missing raw array names")
    return {"release_assets": len(assets), "raw_arrays_in_manifest": 664,
            "restored_payload_hashes_checked": with_data}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--with-data", action="store_true", help="Also hash all 664 restored arrays and all configuration metadata")
    parser.add_argument("--require-source-manifest", action="store_true")
    parser.add_argument("--full-source", action="store_true", help="Verify the complete prepared snapshot, including derived files whose GitHub upload is pending")
    args = parser.parse_args()
    root = args.root.resolve()
    report = {"release_boundary": verify_release_boundary(root),
              "published_sources": verify_source_manifest(root, args.require_source_manifest, args.full_source),
              "lean_saved_records": verify_lean_records(root),
              "release_payloads": verify_payloads(root, args.with_data)}
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as error:
        print(f"verify_package: {error}", file=sys.stderr)
        raise SystemExit(1)
