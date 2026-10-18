#!/usr/bin/env python3
"""Check the original source contract, then rebuild and audit its twenty proofs.

Use the contract shipped with the original package, not one supplied by the
returned project. New modules require separate statement and axiom review.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile


def without_comments(text):
    """Remove nested Lean block comments and line comments in these modules."""
    out, i, depth = [], 0, 0
    while i < len(text):
        if text.startswith('/-', i):
            depth += 1
            i += 2
        elif depth and text.startswith('-/', i):
            depth -= 1
            i += 2
            if not depth:
                out.append(' ')
        elif depth:
            i += 1
        elif text.startswith('--', i):
            end = text.find('\n', i)
            i = len(text) if end < 0 else end
        else:
            out.append(text[i])
            i += 1
    if depth:
        raise ValueError('Unclosed source comment')
    return ''.join(out)


def digest(text):
    tokens = re.findall(r"[A-Za-z_][A-Za-z_0-9']*|[0-9]+|[^\s]", text)
    return hashlib.sha256('\0'.join(tokens).encode()).hexdigest()


def source_interface(path):
    text = without_comments(path.read_text())
    starts = list(re.finditer(r'(?m)^(theorem|(?:noncomputable )?def)\s+(\w+)\b', text))
    if not starts:
        raise ValueError(f'No declarations in {path}')
    result = {'preamble_sha256': digest(text[:starts[0].start()]), 'declarations': {}}
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        section = text[match.start():end]
        if match.group(1) == 'theorem':
            signature, separator, proof = section.partition(':= by')
            if not separator:
                raise ValueError(f'Expected tactic proof for {match.group(2)}')
            result['declarations'][match.group(2)] = {'kind': 'theorem', 'signature_sha256': digest(signature)}
        else:
            result['declarations'][match.group(2)] = {'kind': 'definition', 'definition_sha256': digest(section)}
    return result


def run(command, project):
    result = subprocess.run(command, cwd=project, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end='')
    if result.returncode:
        raise RuntimeError(f'Command failed with exit code {result.returncode}: {command}')
    return result.stdout


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path, default=here / 'lean')
    parser.add_argument('--contract', type=Path, default=here / 'Proof_Contract.json')
    args = parser.parse_args()
    project = args.project_dir.resolve()
    contract = json.loads(args.contract.read_text())
    for relative, expected in contract['protected_sources'].items():
        actual = source_interface(project / relative)
        if actual != expected:
            raise RuntimeError(f'Source contract changed: {relative}. Review the changed types, definitions or preamble.')
    if (project / 'lean-toolchain').read_text().strip() != contract['lean_toolchain']:
        raise RuntimeError('Lean toolchain changed')
    manifest = json.loads((project / 'lake-manifest.json').read_text())
    pins = {item['name']: item['rev'] for item in manifest['packages'] if item['type'] == 'git'}
    if pins != contract['git_dependency_revisions']:
        raise RuntimeError('Pinned dependency revisions changed')
    run(['lake', 'build', 'RepairCore', 'RepairTasks'], project)
    with tempfile.TemporaryDirectory(prefix='csl-proof-audit-') as temporary:
        audit = Path(temporary) / 'Audit.lean'
        audit.write_text('import RepairCore\nimport RepairTasks\n' + ''.join(f'#print axioms {name}\n' for name in contract['theorems']))
        output = run(['lake', 'env', 'lean', str(audit)], project)
    reports = {name: {x.strip() for x in axioms.split(',') if x.strip()}
               for name, axioms in re.findall(r"'([^']+)' depends on axioms: \[(.*?)\]", output)}
    if set(reports) != set(contract['theorems']):
        raise RuntimeError('Axiom report omitted a required declaration')
    allowed = set(contract['allowed_axioms'])
    for name, axioms in reports.items():
        if not axioms <= allowed:
            raise RuntimeError(f'Unaccepted axioms in {name}: {sorted(axioms - allowed)}')
    print(json.dumps({'accepted_original_declarations': len(reports), 'source_contract': 'unchanged', 'axiom_audit': 'passed', 'additional_modules': 'require separate review'}, indent=2))


if __name__ == '__main__':
    main()
