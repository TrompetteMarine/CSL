#!/usr/bin/env python3
"""Rebuild exact source declarations, preserving the reviewed baseline contract.

No theorem is accepted from an old log. `#check @name` records every exact type;
`#print axioms name` audits its transitive logical dependencies after rebuilding.
"""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / 'project'
BASE = ROOT / 'reviewed_baseline/Aristotle_Reviewed_Source_and_Exploration_2026-10-11'
LOGS = ROOT / 'logs'
LOGS.mkdir(exist_ok=True)
env = os.environ.copy()
local_bin = Path(env.get('CSL_LEAN_BIN', str(ROOT/'runtime/lean-4.28.0-linux/bin')))
if local_bin.is_dir():
    env['PATH'] = str(local_bin) + os.pathsep + env['PATH']
local_shim = ROOT/'runtime/proc_self_compat.so'
if local_shim.exists():
    env['LD_PRELOAD'] = str(local_shim)
env['TAR_OPTIONS'] = '--no-same-owner'
steps = []

def run(command, logname, cwd=PROJECT):
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (LOGS/logname).open('w') as f:
        r = subprocess.run(command, cwd=cwd, env=env, text=True, stdout=f, stderr=subprocess.STDOUT)
    steps.append(dict(command=command, start_utc=started,
        end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        exit_code=r.returncode, log=logname))
    (ROOT/'execution_steps.json').write_text(json.dumps(steps, indent=2)+'\n')
    print(logname, r.returncode, flush=True)
    if r.returncode:
        raise RuntimeError((LOGS/logname).read_text())
    return (LOGS/logname).read_text()

spec = importlib.util.spec_from_file_location('contract_checker', BASE/'original_submission/check_proofs.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
contract = json.loads((BASE/'original_submission/Proof_Contract.json').read_text())
for name, expected in contract['protected_sources'].items():
    if checker.source_interface(PROJECT/name) != expected:
        raise RuntimeError('Original source interface changed: '+name)

baseline_hashes = {}
for p in (BASE/'reviewed_source').rglob('*.lean'):
    relative = p.relative_to(BASE/'reviewed_source')
    if relative.parts[0] not in ('RepairCore','RepairTasks','RepairProb') and str(relative) not in ('RepairCore.lean','RepairTasks.lean','RepairProb.lean'):
        continue
    if (PROJECT/relative).read_bytes() != p.read_bytes():
        raise RuntimeError('Reviewed source changed: '+str(relative))
    baseline_hashes[str(relative)] = hashlib.sha256(p.read_bytes()).hexdigest()

manifest = json.loads((PROJECT/'lake-manifest.json').read_text())
pins = {p['name']:p['rev'] for p in manifest['packages'] if p['type']=='git'}
if pins != contract['git_dependency_revisions']:
    raise RuntimeError('Dependency pins changed')
dependency_audit = {}
for name, rev in pins.items():
    p=PROJECT/'.lake/packages'/name
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip()
    tracked=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=p,text=True)
    if head!=rev or tracked:
        raise RuntimeError('Pinned dependency is not clean: '+name)
    dependency_audit[name]=dict(expected=rev,actual=head,tracked_sources_clean=True)
(ROOT/'dependency_audit.json').write_text(json.dumps(dependency_audit,indent=2)+'\n')
(ROOT/'baseline_contract_audit.json').write_text(json.dumps(dict(
    original_interfaces='unchanged',reviewed_source_bytes='unchanged',
    sha256=baseline_hashes),indent=2)+'\n')

run(['lean','--version'],'lean_version.log')
run(['lake','build','RepairCore','RepairTasks','RepairProb','RepairSupplement'],'complete_build.log')

inventory=[]
for folder in ['RepairCore','RepairTasks','RepairProb','RepairSupplement']:
    for p in sorted((PROJECT/folder).glob('*.lean')):
        source=checker.without_comments(p.read_text())
        forbidden=re.findall(r'\b(?:sorry|admit|sorryAx|axiom|unsafe|native_decide|implemented_by|extern|trustCompiler|ofReduceBool)\b',source)
        if forbidden:
            raise RuntimeError(f'Forbidden source tokens in {p}: {forbidden}')
        for m in re.finditer(r'(?m)^(theorem|lemma|(?:noncomputable )?def)\s+(\w+)\b',source):
            inventory.append(dict(name='CSLRepair.'+m[2],kind=m[1],
                source=str(p.relative_to(PROJECT)),new_revision=folder=='RepairSupplement'))
prefix='import RepairCore\nimport RepairTasks\nimport RepairProb\nimport RepairSupplement\n'
(PROJECT/'ReviewAllTypes.lean').write_text(prefix+'set_option pp.universes true\n'+''.join('#check @'+i['name']+'\n' for i in inventory))
(PROJECT/'ReviewAllAxioms.lean').write_text(prefix+''.join('#print axioms '+i['name']+'\n' for i in inventory))
run(['lake','env','lean','ReviewAllTypes.lean'],'all_declaration_types.log')
axiom_text=run(['lake','env','lean','ReviewAllAxioms.lean'],'all_declaration_axioms.log')
axioms={n:[x.strip() for x in a.split(',') if x.strip()]
    for n,a in re.findall(r"'([^']+)' depends on axioms: \[(.*?)\]",axiom_text,re.S)}
axioms.update({n:[] for n in re.findall(r"'([^']+)' does not depend on any axioms",axiom_text)})
if set(axioms)!={i['name'] for i in inventory}:
    raise RuntimeError('Axiom audit omitted declarations')
for i in inventory:
    i['axioms']=axioms[i['name']]
    if not set(i['axioms']) <= {'propext','Classical.choice','Quot.sound'}:
        raise RuntimeError('Unaccepted transitive axiom in '+i['name'])
    i['status']='kernel_checked_locally'
(ROOT/'declaration_audit.json').write_text(json.dumps(inventory,indent=2)+'\n')
summary=dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    toolchain='Lean4.28.0', mathlib=pins['mathlib'], source_contract='unchanged',
    baseline_proofs=sum(i['kind'] in ('theorem','lemma') and not i['new_revision'] for i in inventory),
    new_proofs=sum(i['kind'] in ('theorem','lemma') and i['new_revision'] for i in inventory),
    total_declarations=len(inventory), allowed_axioms=['propext','Classical.choice','Quot.sound'],
    build='passed',type_audit='passed',axiom_audit='passed',
    runtime_compatibility='readlink redirects only /proc/<getpid>/exe to /proc/self/exe; no Lean proof or kernel code changed',
    exclusions=['Full large-community stochastic coupling theorem','Feller invariant-law existence',
        'Sparse local weak convergence and empirical LLN','Watts–Strogatz graph-limit probability estimates',
        'Joint sparse/large-community limit','Simulation outputs'])
(ROOT/'verification_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
