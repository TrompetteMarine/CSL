#!/usr/bin/env python3
"""Publish allowlisted replication assets to the existing CSL repository.

The default is a local plan. --execute is required for GitHub writes. No token
is read or stored by this script; authentication stays with GitHub CLI.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(args, *, capture=False):
    return subprocess.run(args, cwd=ROOT, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-dir', type=Path, required=True)
    parser.add_argument('--repository', default='TrompetteMarine/CSL')
    parser.add_argument('--tag', default='v2026.10.16')
    parser.add_argument('--target', required=True, help='Exact existing CSL commit SHA for the release tag')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    owner, name = args.repository.split('/')
    manifest = json.loads((ROOT / 'replication/release_manifest.json').read_text())
    files = []
    for asset in manifest['assets']:
        path = args.assets_dir.resolve() / asset['name']
        if not path.is_file() or path.stat().st_size != asset['bytes'] or sha(path) != asset['sha256']:
            raise SystemExit('Missing or changed release asset: ' + asset['name'])
        files.append(path)
    for name in ['release_manifest.json', 'payload_manifest.json', 'SHA256SUMS']:
        path = args.assets_dir.resolve() / name
        if path.read_bytes() != (ROOT / 'replication' / name).read_bytes():
            raise SystemExit('Release manifest differs from reviewed repository: ' + name)
        files.append(path)
    run([sys.executable, 'tools/verify_package.py', '--require-source-manifest'])
    print(json.dumps({'repository': args.repository, 'target': args.target,
                      'tag': args.tag, 'assets': [p.name for p in files],
                      'execute': args.execute}, indent=2), flush=True)
    if not args.execute:
        return
    if not shutil.which('gh'):
        raise SystemExit('Install and authenticate GitHub CLI before using --execute.')
    login = run(['gh', 'api', 'user', '--jq', '.login'], capture=True).stdout.strip()
    if login != owner:
        raise SystemExit('Authenticated GitHub account differs from the repository owner.')
    if len(args.target) != 40 or any(x not in '0123456789abcdef' for x in args.target):
        raise SystemExit('Supply an exact 40-character lowercase commit SHA.')
    run(['gh', 'api', 'repos/' + args.repository + '/commits/' + args.target], capture=True)
    probe = subprocess.run(['gh', 'api', 'repos/' + args.repository + '/releases/tags/' + args.tag],
                           cwd=ROOT, text=True, capture_output=True)
    if probe.returncode == 0:
        raise SystemExit('Release already exists; inspect it before changing published assets.')
    if '404' not in probe.stderr:
        raise SystemExit('Could not establish that the release is absent; no writes attempted.')
    run(['gh', 'release', 'create', args.tag, *map(str, files), '--repo', args.repository,
         '--target', args.target, '--title', 'Replication snapshot 2026-10-16',
         '--notes-file', str(ROOT / 'docs/RELEASE_NOTES.md')])
    release = json.loads(run(['gh', 'api', 'repos/' + args.repository + '/releases/tags/' + args.tag],
                             capture=True).stdout)
    remote = {a['name']: a for a in release['assets']}
    for path in files:
        asset = remote.get(path.name)
        if not asset or asset['size'] != path.stat().st_size:
            raise SystemExit('Published release asset is missing or differs in size: ' + path.name)
        if asset.get('digest') != 'sha256:' + sha(path):
            raise SystemExit('Published release digest is absent or differs: ' + path.name)
    print('Published branch and release assets verified: ' + release['html_url'])


if __name__ == '__main__':
    main()
