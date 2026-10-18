"""Exact one-trial action noise and routing covariance at saved entry states.

The ring comparison changes only the linear routing operator, holding the
actual broadcast state's p, d, and conditional signal variances fixed.
It is not a simulated alternative next-choice law.
"""
from pathlib import Path
import json,math
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
BASE=ROOT/'baseline/nhb_smallworld_20261012/results';DATA=ROOT/'data/results'
records=[];community_rows=[];matrices={k:[] for k in ['signal_variances','signal_expectations','broadcast_covariance','ring_covariance','public_signal_covariance']}
state_ids=[]
for gid in range(4):
    c=json.loads((BASE/(f'w{gid:04d}.json')).read_text());N=c['N'];M=c['M'];sizes=np.full(M,c['n']);omega=sizes/N
    B=np.zeros((M,M))
    for i,row in enumerate(c['B']):
        for j,v in row:B[i,j]=v
    C=np.broadcast_to(omega,(M,M))
    with np.load(DATA/(f'w{gid:04d}.npz')) as z:Q=z['final_Q'];psi=z['final_psi']
    for r in range(4):
        state_id=f'g{gid}_r{r}';delta=Q[r,:,0]-Q[r,:,1];field=float(omega@psi[r]);p=1/(1+np.exp(-12*(delta+.6*field)));d=np.maximum(p,1-p)
        pv=p*(1-p);v=4*(d*d*pv).reshape(M,c['n']).sum(1)/(sizes*sizes)
        expected_signal=(d*(2*p-1)).reshape(M,c['n']).mean(1)
        cov_public_signal=2*(d*pv).reshape(M,c['n']).sum(1)/(N*sizes)
        var_public=float(pv.sum()/(N*N))
        bc=(C*v[None,:])@C.T;rc=(B*v[None,:])@B.T
        bsd=np.sqrt(np.diag(bc));rsd=np.sqrt(np.diag(rc));sigsd=np.sqrt(v)
        bcor=bc/np.outer(bsd,bsd);rcor=rc/np.outer(rsd,rsd);off=~np.eye(M,dtype=bool)
        global_b=float(omega@bc@omega);global_r=float(omega@rc@omega)
        assert np.max(abs(bcor-1))<1e-12
        assert abs(global_b-global_r)<1e-16 # regular ring is doubly stochastic
        rec=dict(state_id=state_id,graph_id=gid,rep=r,mean_better_propensity=float(p.mean()),
            conditional_public_sd=math.sqrt(var_public),community_signal_rms_sd=float(np.sqrt(v.mean())),
            community_signal_sd_mean=float(sigsd.mean()),community_signal_sd_min=float(sigsd.min()),community_signal_sd_max=float(sigsd.max()),
            broadcast_field_sd=float(bsd[0]),broadcast_offdiagonal_covariance=float(bc[0,1]),
            broadcast_offdiagonal_correlation=float(bcor[0,1]),broadcast_covariance_rank=int(np.linalg.matrix_rank(bc)),
            ring_field_rms_sd=float(np.sqrt(np.diag(rc).mean())),ring_field_sd_mean=float(rsd.mean()),
            ring_field_sd_min=float(rsd.min()),ring_field_sd_max=float(rsd.max()),
            ring_offdiagonal_covariance_mean=float(rc[off].mean()),ring_offdiagonal_correlation_mean=float(rcor[off].mean()),
            ring_covariance_rank=int(np.linalg.matrix_rank(rc)),same_population_mean_field_sd=math.sqrt(global_b),
            ring_to_broadcast_field_rms_sd_ratio=float(np.sqrt(np.diag(rc).mean())/bsd[0]),
            public_broadcast_field_correlation=float((cov_public_signal@omega)/(math.sqrt(var_public)*bsd[0])))
        records.append(rec);state_ids.append(state_id)
        for j in range(M):
            community_rows.append(dict(state_id=state_id,graph_id=gid,rep=r,community=j,
                signal_variance=float(v[j]),signal_sd=float(sigsd[j]),expected_signal=float(expected_signal[j]),
                broadcast_field_sd=float(bsd[j]),ring_field_sd=float(rsd[j]),public_signal_covariance=float(cov_public_signal[j])))
        matrices['signal_variances'].append(v);matrices['signal_expectations'].append(expected_signal)
        matrices['broadcast_covariance'].append(bc);matrices['ring_covariance'].append(rc);matrices['public_signal_covariance'].append(cov_public_signal)
pop=pd.DataFrame(records);pop.to_csv(OUT/'conditional_noise_population.csv',index=False)
pd.DataFrame(community_rows).to_csv(OUT/'conditional_noise_community.csv',index=False)
metrics=[x for x in pop.columns if x not in ['state_id','graph_id','rep']]
batch=pop.groupby('graph_id')[metrics].mean().reset_index();batch.to_csv(OUT/'conditional_noise_batch.csv',index=False)
summary=[]
for metric in metrics:
    values=batch[metric].to_numpy();mean=values.mean();se=values.std(ddof=1)/2;half=student_t.ppf(.975,3)*se
    summary.append(dict(metric=metric,mean=float(mean),mcse=float(se),ci95_low=float(mean-half),ci95_high=float(mean+half),df=3,batches=4,entry_states=16,
        interpretation='Monte Carlo uncertainty of the average conditional-noise summary over entry states; not the conditional action-noise SD itself.'))
pd.DataFrame(summary).to_csv(OUT/'conditional_noise_ensemble_summary.csv',index=False)
np.savez_compressed(OUT/'conditional_noise_covariances.npz',state_ids=np.array(state_ids),**{k:np.array(v) for k,v in matrices.items()})
out=dict(conditioning='Complete pre-choice Q and lagged psi at all16 saved double-precision broadcast entry states; fixed matrices and parameters.',
    routing_comparison='Only C changes from actual full broadcast to retained ring B. Actual p,d,v held fixed. No alternative choices are recomputed.',
    community_size=100,communities=100,agents=10000,entry_states=16,
    broadcast_covariance_ranks=sorted(set(pop.broadcast_covariance_rank)),ring_covariance_ranks=sorted(set(pop.ring_covariance_rank)),
    broadcast_offdiagonal_correlations=[float(pop.broadcast_offdiagonal_correlation.min()),float(pop.broadcast_offdiagonal_correlation.max())],
    average_summaries={r['metric']:r['mean'] for r in summary})
(OUT/'Conditional_Noise_Summary.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
