#!/usr/bin/env python3
"""Restore checksum-pinned replication payloads; never restore manuscript archives.

Examples:
  python tools/prepare_data.py --download
  python tools/prepare_data.py --archives-dir /path/to/release --metadata-only

Only the committed payload manifest grants extraction permission. Archives and
members are checked before replacing destination files. No GitHub credentials
are needed for the public release. This command does not execute payload code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARRAY_ASSETS = ("agent_study_arrays.zip", "many_communities_arrays.zip")
METADATA_ASSET = "replication_metadata.zip"
WORKBOOK_ASSET = "source_data_workbooks.zip"
METADATA_ROOT = "baseline/nhb_smallworld_20261012"
METADATA_TOP = {
    "Choice_Only_Freeze.json", "Design_Freeze.json", "Environment.json",
    "Execution_Summary.json", "Followup_Freeze.json", "Many_Communities_Freeze.json",
    "SHA256SUMS.json", "requirements.txt",
}
CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Hash potentially large assets without loading them into memory."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            result.update(chunk)
    return result.hexdigest()


def canonical_path(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError(f"Invalid payload path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(x in ("", ".", "..") or x.startswith(".") for x in name.split("/")):
        raise ValueError(f"Unsafe payload path: {name!r}")
    if str(path) != name or ":" in name:
        raise ValueError(f"Non-canonical payload path: {name!r}")
    return path


def member_allowed(asset: str, name: str) -> bool:
    """Additional structural allowlist, independent of hashes in the manifest."""
    try:
        path = canonical_path(name)
    except ValueError:
        return False
    if asset in ARRAY_ASSETS:
        return bool(re.fullmatch(r"data/results/[cfmxw][0-9]{4}\.npz", name))
    if asset == METADATA_ASSET:
        if path.parent.as_posix() == METADATA_ROOT and path.name in METADATA_TOP:
            return True
        return (path.parent.as_posix() == METADATA_ROOT + "/results"
                and path.suffix in (".json", ".csv")
                and bool(re.fullmatch(r"[A-Za-z0-9_-]+\.(?:json|csv)", path.name)))
    if asset == WORKBOOK_ASSET:
        return path.parts[0] == "NHB_Source_Data" and path.suffix in {
            ".csv", ".xlsx", ".json", ".py", ".npz", ".md", ".mjs",
        }
    return False


def read_manifests(directory: Path) -> tuple[dict, dict]:
    release = json.loads((directory / "release_manifest.json").read_text())
    payload = json.loads((directory / "payload_manifest.json").read_text())
    if release.get("schema_version") != 1 or payload.get("schema_version") != 1:
        raise ValueError("Unsupported replication manifest schema")
    assets = {x["name"]: x for x in release["assets"]}
    if len(assets) != len(release["assets"]):
        raise ValueError("Duplicate release asset")
    if set(assets) != set(payload["archives"]):
        raise ValueError("Release and payload asset lists disagree")
    all_paths = set()
    for name, asset in assets.items():
        if name not in (*ARRAY_ASSETS, METADATA_ASSET, WORKBOOK_ASSET):
            raise ValueError(f"Unrecognised release asset: {name}")
        records = payload["archives"][name]
        paths = [x["path"] for x in records]
        if len(paths) != len(set(paths)) or all_paths.intersection(paths):
            raise ValueError("Duplicate payload destination")
        all_paths.update(paths)
        if len(records) != asset["entry_count"] or sum(x["bytes"] for x in records) != asset["uncompressed_bytes"]:
            raise ValueError(f"Payload count or byte total disagrees: {name}")
        for record in [asset, *records]:
            if not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) or type(record["bytes"]) is not int or record["bytes"] < 0:
                raise ValueError(f"Invalid size or SHA-256 in {name}")
        for path in paths:
            if not member_allowed(name, path):
                raise ValueError(f"Payload not allowed for {name}: {path}")
    return assets, payload["archives"]


def check_file(path: Path, record: dict) -> None:
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        raise ValueError(f"Missing file or byte-size mismatch: {path}")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")


def download_asset(path: Path, record: dict, repository: str, tag: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Repository must be OWNER/REPOSITORY")
    url = f"https://github.com/{repository}/releases/download/{quote(tag, safe='')}/{record['name']}"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    digest = hashlib.sha256()
    total = 0
    try:
        request = Request(url, headers={"User-Agent": "decision-generated-credibility-replication/1"})
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            for chunk in iter(lambda: response.read(CHUNK), b""):
                total += len(chunk)
                if total > record["bytes"]:
                    raise ValueError(f"Download exceeds pinned size: {record['name']}")
                digest.update(chunk)
                output.write(chunk)
        if total != record["bytes"] or digest.hexdigest() != record["sha256"]:
            raise ValueError(f"Downloaded file fails manifest: {record['name']}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def destination(root: Path, name: str) -> Path:
    """Reject symlinked parents and destinations even inside the output root."""
    relative = canonical_path(name)
    target = root.joinpath(*relative.parts)
    for candidate in [target, *target.parents]:
        if candidate == root.parent:
            break
        if candidate.is_symlink():
            raise ValueError(f"Symlink in extraction destination: {candidate}")
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Extraction destination escapes root: {name}")
    return target


def restore_archive(archive: Path, asset: dict, records: list[dict], root: Path) -> int:
    """Verify the whole archive, then individually verify streamed member bytes."""
    check_file(archive, asset)
    expected = {r["path"]: r for r in records}
    restored = 0
    with zipfile.ZipFile(archive) as packed:
        entries = packed.infolist()
        if len(entries) != len(expected) or len({x.filename for x in entries}) != len(entries):
            raise ValueError(f"Duplicate or missing archive entries: {archive.name}")
        if set(x.filename for x in entries) != set(expected):
            raise ValueError(f"Archive members disagree with pinned manifest: {archive.name}")
        for info in entries:
            record = expected[info.filename]
            filetype = stat.S_IFMT(info.external_attr >> 16)
            if (info.is_dir() or filetype not in (0, stat.S_IFREG)
                    or info.flag_bits & 1 or info.file_size != record["bytes"]
                    or not member_allowed(archive.name, info.filename)):
                raise ValueError(f"Disallowed archive member: {info.filename}")
        for info in entries:
            record = expected[info.filename]
            target = destination(root, info.filename)
            if target.exists() and target.is_file() and target.stat().st_size == record["bytes"] and sha256_file(target) == record["sha256"]:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".", suffix=".part", delete=False) as output:
                    temporary = Path(output.name)
                    digest = hashlib.sha256()
                    total = 0
                    with packed.open(info) as source:
                        for chunk in iter(lambda: source.read(CHUNK), b""):
                            total += len(chunk)
                            if total > record["bytes"]:
                                raise ValueError(f"Member exceeds pinned size: {info.filename}")
                            digest.update(chunk)
                            output.write(chunk)
                if total != record["bytes"] or digest.hexdigest() != record["sha256"]:
                    raise ValueError(f"Member SHA-256 mismatch: {info.filename}")
                os.replace(temporary, target)
                restored += 1
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    return restored


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archives-dir", type=Path, default=ROOT / "downloads", help="Directory containing release ZIPs (default: downloads/)")
    parser.add_argument("--download", action="store_true", help="Fetch missing release ZIPs over HTTPS")
    parser.add_argument("--repository", default="TrompetteMarine/CSL")
    parser.add_argument("--tag", default="v2026.10.16")
    parser.add_argument("--skip-arrays", "--metadata-only", dest="skip_arrays", action="store_true", help="Restore configuration metadata without the 664 large arrays")
    parser.add_argument("--extract-workbooks", action="store_true", help="Also fetch and unpack the optional source-data workbook bundle")
    parser.add_argument("--output-root", type=Path, default=ROOT, help="Destination checkout root")
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "replication")
    args = parser.parse_args(argv)
    assets, payloads = read_manifests(args.manifest_dir)
    names = [METADATA_ASSET]
    if not args.skip_arrays:
        names += list(ARRAY_ASSETS)
    if args.extract_workbooks:
        names.append(WORKBOOK_ASSET)
    root = args.output_root.absolute()
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        archive = args.archives_dir / name
        if not archive.exists():
            if not args.download:
                raise ValueError(f"Missing {archive}; provide --archives-dir or add --download")
            print(f"Downloading {name} ...", flush=True)
            download_asset(archive, assets[name], args.repository, args.tag)
        print(f"Verifying and restoring {name} ...", flush=True)
        count = restore_archive(archive, assets[name], payloads[name], root)
        print(f"  {len(payloads[name])} members verified; {count} restored.", flush=True)
    print("Replication payload restoration passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        print(f"prepare_data: {error}", file=sys.stderr)
        raise SystemExit(1)
