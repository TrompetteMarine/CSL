"""Independent numerical audit of the frozen October 2026 NHB study.

Read-only with respect to the original records; writes solely to this directory.
No simulated agent or trial is treated as an independent population replicate.
"""
from pathlib import Path
from collections import Counter
import hashlib, json, re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
META=ROOT/'baseline/nhb_smallworld_20261012/results'
DATA=ROOT/'data/results'
OUT=Path(__file__).resolve().parent
configs={p.stem:json.loads(p.read_text()) for p in META.glob('*.json') if re.fullmatch('[cfmxw][0-9]{4}',p.stem)}
fresh={k:c for k,c in configs.items() if k[0]!='w'}
assert len(fresh)==652 and len(configs)==664
assert {p.stem for p in DATA.glob('*.npz')}==set(configs)
errors=Counter();dtypes=Counter();rows=[];matrix_checks=[];withdrawal=[]
max_gain=0.;clips=0;equivalence=[];entry_replay=[]

def update_error(key,a,b):
    errors[key]=max(errors[key],float(abs(a-b)))

for label,c in sorted(configs.items()):
    with np.load(DATA/(label+'.npz')) as z:
        path,q,psi,tail=z['path'],z['final_Q'],z['final_psi'],z['tail']
        T,R,N,M=c['T'],c['reps'],c['N'],c['M'];window=c['window']
        assert path.shape==(T,R,6) and q.shape==(R,N,2) and psi.shape==(R,M) and tail.shape==(R,M)
        assert np.all(z['sizes']==c['n'])
        assert all(np.isfinite(a).all() for a in [path,q,psi,tail])
        assert np.min(q)>=0 and np.max(q)<=1 and np.max(abs(psi))<=1+1e-12
        assert np.min(path[:,:,:2])>=0 and np.max(path[:,:,:2])<=1
        assert np.min(tail)>=0 and np.max(tail)<=1+1e-12
        dtypes[str(q.dtype)]+=1
        update_error('regret_path_identity',np.max(abs(path[:,:,5]-(c['parameters']['mu'][0]-c['parameters']['mu'][1])*(1-path[:,:,0]))),0.)
        update_error('community_vs_population_window',np.max(abs(tail.mean(1)-path[-window:,:,0].mean(0))),0.)
        assert c['audit']['clips']==0
        clips+=c['audit']['clips'];max_gain=max(max_gain,c['audit']['gain_max'])
        for name,matrix in [('B',c['B']),('choice_B',c.get('choice_B',c['B']))]:
            assert len(matrix)==M
            adj=[]
            for i,row in enumerate(matrix):
                assert len(set(j for j,v in row))==len(row)
                assert all(0<=j<M and v>=0 for j,v in row)
                assert abs(sum(v for j,v in row)-1)<1e-12
                # Keep zero-weight neighbour entries when Gamma=0: graph itself still exists.
                adj.append({j for j,v in row if j!=i})
            assert all(i not in nb for i,nb in enumerate(adj))
            assert all(i in adj[j] for i,nb in enumerate(adj) for j in nb)
            k=1 if M==2 else c.get('k',4)
            assert sum(map(len,adj))==k*M
            if c['geometry'] in ['ring','broadcast','regular'] or (name=='B' and c['group']=='choiceonly'):
                assert all(len(nb)==k for nb in adj)
            matrix_checks.append(dict(label=label,matrix=name,vertices=M,edges=sum(map(len,adj))//2))
        rr=json.loads((META/(label+'_rows.json')).read_text()) if (META/(label+'_rows.json')).exists() else []
        if label[0]!='w':
            assert len(rr)==R
            for r in rr:
                j=r['rep'];a=path[-window:,j,0].mean();private=(q[j,:,0]>q[j,:,1]).mean();gap=(q[j,:,0]-q[j,:,1]).mean()
                expected=dict(optimal_fraction=a,private_correct=private,private_gap=gap,latent_public_gap=private-a,
                              mean_removed_probability=path[-100:,j,3].mean(),regret=path[:,j,5].sum(),
                              wrong_communities=(tail[j]<.5).mean(),community_variance=tail[j].var())
                for key,v in expected.items(): update_error(key,v,r[key])
                assert r['wrong']==int(a<.5)
                row={**r,**expected,'gamma':c['gamma'],'k':c.get('k',4),'T':T,'source_dtype':str(q.dtype)}
                rows.append(row)
        elif int(label[1:])>=4:
            for j in range(R):
                withdrawal.append(dict(label=label,graph_id=c['graph_id'],rep=j,lam=c['lam'],entry_label=c['entry_label'],
                    optimal_fraction=float(path[-window:,j,0].mean()),wrong=int(path[-window:,j,0].mean()<.5),
                    private_correct=float((q[j,:,0]>q[j,:,1]).mean()),first_step_probability=float(path[0,j,4])))

assert errors['optimal_fraction']<1e-12 and errors['private_correct']==0
assert errors['private_gap']<1e-7 and errors['latent_public_gap']<1e-12
assert errors['mean_removed_probability']<1e-12 and errors['regret']<1e-8
assert errors['wrong_communities']==0 and errors['community_variance']<1e-12
assert errors['regret_path_identity']==0 and errors['community_vs_population_window']<1e-12
df=pd.DataFrame(rows);df.to_csv(OUT/'recomputed_replications.csv',index=False)
keys=['group','N','n','M','geometry','initial','mode','lam','gamma','k','T']
summary=df.groupby(keys).agg(populations=('rep','size'),batches=('graph_id','nunique'),
    optimal_fraction=('optimal_fraction','mean'),private_correct=('private_correct','mean'),
    wrong_count=('wrong','sum'),wrong_communities=('wrong_communities','mean'),
    private_gap=('private_gap','mean'),regret=('regret','mean')).reset_index()
summary.to_csv(OUT/'recomputed_cells.csv',index=False)
batch=df.groupby(keys+['graph_id']).agg(populations=('rep','size'),
    optimal_fraction=('optimal_fraction','mean'),private_correct=('private_correct','mean'),
    wrong_communities=('wrong_communities','mean')).reset_index()
batch.to_csv(OUT/'recomputed_batches.csv',index=False)
wd=pd.DataFrame(withdrawal);wd.to_csv(OUT/'recomputed_withdrawal.csv',index=False)
recorded_wd=pd.DataFrame(json.loads((META/'withdrawal.json').read_text()))
for _,r in wd.iterrows():
    old=recorded_wd.query('graph_id==@r.graph_id and rep==@r.rep and lam==@r.lam').iloc[0]
    for key in ['optimal_fraction','private_correct','first_step_probability']:
        update_error('withdrawal_'+key,r[key],old[key])

for geo in ['ring','ws0.1','regular','broadcast']:
    for gid in range(4):
        cs=[c for c in fresh.values() if c['group']=='controls' and c['geometry']==geo and c['graph_id']==gid and c['initial']=='misleading_random']
        strong=next(c for c in cs if c['mode']==2)
        doubled=next(c for c in cs if c['mode']==1 and c['lam']==1.2)
        with np.load(DATA/(strong['label']+'.npz')) as a,np.load(DATA/(doubled['label']+'.npz')) as b:
            # Emitted signals differ by factor 2 although the state and choice paths coincide.
            e={key:float(np.max(abs(a[key]-b[key]))) for key in ['path','tail','final_Q']}
            e['psi_rescaling']=float(np.max(abs(a['final_psi']-2*b['final_psi'])))
            assert max(e.values())==0
            equivalence.append(dict(geometry=geo,graph_id=gid,strong=strong['label'],doubled=doubled['label'],max_error=e))

for gid in range(4):
    entry=configs[f'w{gid:04d}']
    orig=next(c for c in fresh.values() if c['group']=='primary' and c['n']==100 and c['geometry']=='broadcast' and c['initial']=='misleading_random' and c['graph_id']==gid)
    with np.load(DATA/(entry['label']+'.npz')) as a,np.load(DATA/(orig['label']+'.npz')) as b:
        e=float(np.max(abs(a['path']-b['path'])))
        assert e==0 and np.array_equal(a['tail'],b['tail'])
        entry_replay.append(dict(entry=entry['label'],original=orig['label'],path_error=e,
            terminal_storage_error=float(np.max(abs(a['final_Q']-b['final_Q'])))))
    retained=configs[f'w{gid+4:04d}'];removed=configs[f'w{gid+8:04d}']
    assert retained['entry_label']==removed['entry_label']==entry['label']
    assert retained['seed']==removed['seed'] and retained['B']==removed['B'] and retained['theta']==removed['theta']==1
    # Direct first-decision intervention: identical scores and lagged signals; no further learning involved.
    with np.load(DATA/(entry['label']+'.npz')) as z:
        q=z['final_Q'];psi=z['final_psi'];delta=q[:,:,0]-q[:,:,1]
        for lam in [.6,0.]:
            direct=(1/(1+np.exp(-12*(delta+lam*psi.mean(1)[:,None])))).mean(1)
            observed=wd.query('graph_id==@gid and lam==@lam').sort_values('rep').first_step_probability.to_numpy()
            update_error('withdrawal_direct_first_decision',np.max(abs(direct-observed)),0.)
assert errors['withdrawal_direct_first_decision']<1e-12

cohorts=[]
for group,cs in __import__('itertools').groupby(sorted(fresh.values(),key=lambda c:c['group']),key=lambda c:c['group']):
    cs=list(cs);cohorts.append(dict(group=group,configurations=len(cs),populations=sum(c['reps'] for c in cs),
        agent_trials=sum(c['N']*c['T']*c['reps'] for c in cs),horizons=sorted(set(c['T'] for c in cs))))
ledger=dict(configurations=652,population_trajectories=2512,
    fresh_start_agent_trials=sum(c['N']*c['T']*c['reps'] for c in fresh.values()),
    entry_replay_populations=16,paired_continuations=32,
    replay_and_continuation_agent_trials=sum(c['N']*c['T']*c['reps'] for k,c in configs.items() if k[0]=='w'),
    cohorts=cohorts,raw_archives_verified=len(configs),raw_array_dtypes=dict(dtypes),
    max_summary_errors=dict(errors),zero_substantive_clips=clips==0,max_reported_gain=max_gain,
    graph_weight_matrices_checked=len(matrix_checks),constant_equivalence=equivalence,entry_replays=entry_replay,
    engine_sha256=hashlib.sha256((META.parent/'src/sim.cpp').read_bytes()).hexdigest(),
    interpretation='Trajectories share paired innovation indices across conditions. Counts do not imply independence across configurations.')
(OUT/'numeric_claims.json').write_text(json.dumps(ledger,indent=2))
print(json.dumps({k:v for k,v in ledger.items() if k not in ['constant_equivalence','entry_replays']},indent=2))
