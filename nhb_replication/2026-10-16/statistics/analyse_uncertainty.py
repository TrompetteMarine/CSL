"""Seed-aware Monte Carlo uncertainty and covariance propagation.

No agents or time points are treated as independent observations. Independent
seed batches are the outer sampling unit. Interacting-population innovations
within batches are retained for conditional uncertainty and paired contrasts.
"""
from pathlib import Path
import hashlib, json, math
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
BASE=ROOT/'baseline/nhb_smallworld_20261012/results';DATA=ROOT/'data/results'
df=pd.read_csv(ROOT/'audit/recomputed_replications.csv')
wd=pd.read_csv(ROOT/'audit/recomputed_withdrawal.csv')
KEYS=['group','N','n','M','geometry','initial','mode','lam','theta','gamma','k','T']
METRICS=['optimal_fraction','private_correct','private_gap','regret','wrong_communities','mean_removed_probability','latent_public_gap','community_variance']
BOUNDS={m:(0.,1.) for m in METRICS}
BOUNDS.update(private_gap=(-1.,1.),latent_public_gap=(-1.,1.),community_variance=(0.,.25),first_step_probability=(0.,1.))

def clean_json(x):
    if isinstance(x,dict):return {str(k):clean_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean_json(v) for v in x]
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    return x

def estimate(values,batch_ids,rep_ids=None,bounds=None):
    """Marginal batch-t interval and separate graph-conditional MCSE.

    Var_cond(mean) = sum_b s_b^2 / (B^2 R_b). This is not added to
    s_batchmean^2/B, which already includes the innovation contribution.
    """
    x=np.asarray(values,dtype=float);g=np.asarray(batch_ids)
    ids=np.unique(g);B=len(ids);assert B>=2
    xg=[x[g==b] for b in ids];R=np.array([len(a) for a in xg])
    assert np.all(R==R[0]),'Current estimator uses equal batch weighting.'
    means=np.array([a.mean() for a in xg]);mean=means.mean()
    between=float(np.var(means,ddof=1)/B);se=math.sqrt(between)
    critical=float(student_t.ppf(.975,B-1))
    within=np.array([a.var(ddof=1) if len(a)>1 else np.nan for a in xg])
    components=within/(B*B*R)
    conditional=float(components.sum());conditional_se=math.sqrt(conditional)
    denominator=float(np.sum(components**2/(R-1)))
    conditional_df=conditional**2/denominator if denominator else float('inf')
    conditional_critical=float(student_t.ppf(.975,conditional_df))
    result=dict(mean=float(mean),mcse=se,ci95_low=float(mean-critical*se),ci95_high=float(mean+critical*se),
        df=B-1,batches=B,populations=len(x),trajectories_per_batch=int(R[0]),variance=between,
        conditional_mcse=conditional_se,conditional_df=conditional_df,
        conditional_ci95_low=float(mean-conditional_critical*conditional_se),
        conditional_ci95_high=float(mean+conditional_critical*conditional_se),
        raw_structural_variance_estimate=between-conditional,
        interpretation='Approximate seed-batch t Monte Carlo interval; conditional uncertainty is a separate target.')
    if bounds is not None:
        width=bounds[1]-bounds[0];half=width*math.sqrt(math.log(40.)/(2*B))
        result.update(natural_min=bounds[0],natural_max=bounds[1],
            hoeffding95_low=float(mean-half),hoeffding95_high=float(mean+half),
            hoeffding95_support_low=float(max(bounds[0],mean-half)),
            hoeffding95_support_high=float(min(bounds[1],mean+half)))
    return result

# Check that actual within-batch stream seeds do not collide in this design.
mask=(1<<64)-1;action_constant=0x9e3779b97f4a7c15;reward_constant=0xd1b54a32d192ed03
stream_records=[]
for seed_family,batch_ids,reps in [('fresh',range(16),range(4)),('withdrawal',range(4),range(4))]:
    for b in batch_ids:
        base=(702000 if seed_family=='fresh' else 820000)+b*100
        for r in reps:
            stream_records.append(dict(family=seed_family,batch=b,rep=r,
                action_seed=(base+action_constant*(r+1))&mask,reward_seed=(base+reward_constant*(r+1))&mask))
for family in ['fresh','withdrawal']:
    s=[a[k] for a in stream_records if a['family']==family for k in ['action_seed','reward_seed']]
    assert len(s)==len(set(s))
pd.DataFrame(stream_records).to_csv(OUT/'seed_streams.csv',index=False)

df.to_csv(OUT/'population_outcomes.csv',index=False)
endpoint=[];batch_rows=[]
groups=[(key,x.copy()) for key,x in df.groupby(KEYS,sort=True)]
expanded=df[(df.group.isin(['primary','followup']))&(df.initial=='misleading_contiguous')&(df.n.isin([50,100,200]))&(df.geometry.isin(['ring','ws0.01','ws0.1']))].copy()
expanded['group']='expanded'
groups.extend((key,x.copy()) for key,x in expanded.groupby(KEYS,sort=True))
for key,x in groups:
    meta=dict(zip(KEYS,key))
    for metric in METRICS:
        bounds=(0.,.1*float(meta['T'])) if metric=='regret' else BOUNDS[metric]
        endpoint.append(dict(**meta,metric=metric,**estimate(x[metric],x.graph_id,bounds=bounds)))
    for b,y in x.groupby('graph_id'):
        batch_rows.append(dict(**meta,graph_id=int(b),populations=len(y),**{m:float(y[m].mean()) for m in METRICS}))
end=pd.DataFrame(endpoint);end.to_csv(OUT/'endpoint_intervals.csv',index=False)
pd.DataFrame(batch_rows).to_csv(OUT/'batch_outcomes.csv',index=False)

withdraw_metrics=['first_step_probability','optimal_fraction','private_correct']
wd.to_csv(OUT/'withdrawal_population_outcomes.csv',index=False)
withdraw_intervals=[]
for lam,x in wd.groupby('lam'):
    for metric in withdraw_metrics:
        withdraw_intervals.append(dict(lam=lam,metric=metric,**estimate(x[metric],x.graph_id,bounds=(0.,1.))))
pd.DataFrame(withdraw_intervals).to_csv(OUT/'withdrawal_intervals.csv',index=False)
wd.groupby(['lam','graph_id'])[withdraw_metrics].mean().reset_index().to_csv(OUT/'withdrawal_batch_outcomes.csv',index=False)

contrasts=[];contrast_populations=[];contrast_batches=[];covariances={}
def paired(contrast_id,ref,inter,metric,reference,intervention,bounds,extra=None):
    pair=ref[['graph_id','rep',metric]].merge(inter[['graph_id','rep',metric]],on=['graph_id','rep'],suffixes=('_reference','_intervention'),validate='one_to_one')
    assert len(pair)==len(ref)==len(inter)
    pair['difference']=pair[metric+'_intervention']-pair[metric+'_reference']
    pair['contrast_id']=contrast_id;pair['metric']=metric
    stat=estimate(pair.difference,pair.graph_id,bounds=bounds)
    sr=estimate(pair[metric+'_reference'],pair.graph_id)
    si=estimate(pair[metric+'_intervention'],pair.graph_id)
    batches=pair.groupby('graph_id')[[metric+'_reference',metric+'_intervention','difference']].mean()
    joint=np.cov(batches[[metric+'_reference',metric+'_intervention']].to_numpy().T,ddof=1)/len(batches)
    propagated=float(joint[0,0]+joint[1,1]-2*joint[0,1])
    assert abs(propagated-stat['variance'])<max(1e-12,stat['variance']*1e-10)
    record=dict(contrast_id=contrast_id,metric=metric,reference=reference,intervention=intervention,
        effect=stat.pop('mean'),**stat,reference_mean=sr['mean'],reference_mcse=sr['mcse'],
        reference_ci95_low=sr['ci95_low'],reference_ci95_high=sr['ci95_high'],
        intervention_mean=si['mean'],intervention_mcse=si['mcse'],
        intervention_ci95_low=si['ci95_low'],intervention_ci95_high=si['ci95_high'],
        covariance_of_condition_means=float(joint[0,1]),
        independent_conditions_mcse=math.sqrt(joint[0,0]+joint[1,1]),
        batch_correlation=float(np.corrcoef(batches.iloc[:,0],batches.iloc[:,1])[0,1]) if joint[0,0]>0 and joint[1,1]>0 else float('nan'))
    if extra:record.update(extra)
    contrasts.append(record)
    contrast_populations.extend(dict(contrast_id=contrast_id,metric=metric,graph_id=int(r.graph_id),rep=int(r.rep),
        reference_value=float(r[metric+'_reference']),intervention_value=float(r[metric+'_intervention']),effect=float(r.difference)) for _,r in pair.iterrows())
    contrast_batches.extend(dict(contrast_id=contrast_id,metric=metric,graph_id=int(b),reference_mean=float(r.iloc[0]),intervention_mean=float(r.iloc[1]),effect=float(r.iloc[2])) for b,r in batches.iterrows())
    covariances[contrast_id+'__'+metric]=dict(order=[reference,intervention],covariance_of_means=joint.tolist(),propagated_variance=propagated)
    return record

main_ref=df[(df.group=='primary')&(df.n==100)&(df.initial=='misleading_random')&(df.geometry=='ring')]
main_inter=df[(df.group=='primary')&(df.n==100)&(df.initial=='misleading_random')&(df.geometry=='broadcast')]
for metric in ['optimal_fraction','private_correct','regret']:
    paired('broadcast_minus_ring',main_ref,main_inter,metric,'Ring','Broadcast',(-100.,100.) if metric=='regret' else (-1.,1.),dict(N=10000,n=100,T=1000))
for metric in withdraw_metrics:
    paired('withdrawn_minus_retained',wd[wd.lam==.6],wd[wd.lam==0.],metric,'Retained','Withdrawn',(-1.,1.),dict(N=10000,n=100,T=1000))
for n in [50,100,200]:
    for g0,g1 in [('ring','ws0.01'),('ring','ws0.1'),('ws0.01','ws0.1')]:
        for metric in ['optimal_fraction','private_correct','wrong_communities','regret']:
            a=expanded[(expanded.n==n)&(expanded.geometry==g0)];b=expanded[(expanded.n==n)&(expanded.geometry==g1)]
            paired(f'expanded_n{n}_{g1}_minus_{g0}',a,b,metric,g0,g1,(-100.,100.) if metric=='regret' else (-1.,1.),dict(N=10000,n=n,T=1000,exploratory=True))
pd.DataFrame(contrasts).to_csv(OUT/'paired_contrasts.csv',index=False)
pd.DataFrame(contrast_populations).to_csv(OUT/'paired_contrast_populations.csv',index=False)
pd.DataFrame(contrast_batches).to_csv(OUT/'paired_contrast_batches.csv',index=False)
(OUT/'paired_covariances.json').write_text(json.dumps(clean_json(covariances),indent=2,allow_nan=False))

# Joint covariance for public/private/regret in both matched conditions.
six=main_ref[['graph_id','rep','optimal_fraction','private_correct','regret']].merge(
    main_inter[['graph_id','rep','optimal_fraction','private_correct','regret']],on=['graph_id','rep'],suffixes=('_ring','_broadcast'),validate='one_to_one')
columns=[m+'_'+condition for condition in ['ring','broadcast'] for m in ['optimal_fraction','private_correct','regret']]
X=six.groupby('graph_id')[columns].mean().to_numpy()
joint_covariance=np.cov(X.T,ddof=1)/len(X)
(OUT/'main_joint_covariance.json').write_text(json.dumps(dict(order=columns,batch_mean_vectors=X.tolist(),covariance_of_means=joint_covariance.tolist(),df=3),indent=2))

# Pointwise curves: smooth every population before batch aggregation.
curves=[];curve_batch=[];curve_pop=[]
def add_curve(curve_id,x,metric,column):
    arrays=[];gids=[]
    for label,zrows in x.groupby('label',sort=True):
        c=json.loads((BASE/(label+'.json')).read_text())
        with np.load(DATA/(label+'.npz')) as z:arr=z['path'][:,:,column]
        smooth=pd.DataFrame(arr).rolling(10,min_periods=1).mean().to_numpy()
        assert arr.shape[1]==len(zrows)
        arrays.append(smooth);gids.extend([c['graph_id']]*c['reps'])
        for j in range(c['reps']):
            curve_pop.extend(dict(curve_id=curve_id,metric=metric,trial=t+1,graph_id=c['graph_id'],rep=j,value=float(v)) for t,v in enumerate(smooth[:,j]))
    matrix=np.concatenate(arrays,axis=1);gids=np.array(gids);ids=np.unique(gids);B=len(ids)
    batches=np.stack([matrix[:,gids==b].mean(1) for b in ids],axis=1)
    means=batches.mean(1);se=batches.std(1,ddof=1)/math.sqrt(B);critical=student_t.ppf(.975,B-1)
    within=sum(matrix[:,gids==b].var(1,ddof=1)/(B*B*(gids==b).sum()) for b in ids)
    for t in range(len(matrix)):
        curves.append(dict(curve_id=curve_id,metric=metric,trial=t+1,mean=float(means[t]),mcse=float(se[t]),
            ci95_low=float(means[t]-critical*se[t]),ci95_high=float(means[t]+critical*se[t]),df=B-1,batches=B,populations=matrix.shape[1],
            conditional_mcse=float(math.sqrt(within[t])),smoothing='10-trial trailing average within population; min_periods=1',band='pointwise'))
        curve_batch.extend(dict(curve_id=curve_id,metric=metric,trial=t+1,graph_id=int(b),value=float(batches[t,j])) for j,b in enumerate(ids))
for gamma in [0.,.15,.5]:
    add_curve('two_gamma_'+str(gamma),df[(df.group=='two')&(df.initial=='misleading_random')&(df.gamma==gamma)],'optimal_fraction',0)
for geometry in ['ring','ws0.1','broadcast']:
    x=df[(df.group=='primary')&(df.n==100)&(df.initial=='misleading_random')&(df.geometry==geometry)]
    for metric,column in [('optimal_fraction',0),('private_prechoice',1)]:add_curve('n100_scattered_'+geometry,x,metric,column)
pd.DataFrame(curves).to_csv(OUT/'trajectory_intervals.csv',index=False)
pd.DataFrame(curve_batch).to_csv(OUT/'trajectory_batch_means.csv',index=False)
pd.DataFrame(curve_pop).to_csv(OUT/'trajectory_population_values.csv',index=False)

# Regret propagation includes every temporal covariance and preserves pairing.
batch_paths={}
for name,x in [('ring',main_ref),('broadcast',main_inter)]:
    y=[]
    for label,r in x.groupby('label',sort=True):
        with np.load(DATA/(label+'.npz')) as z:y.append(z['path'][:,:,0].mean(1))
    batch_paths[name]=np.stack(y)
gap=.55-.45;propagation=[]
for name,Y in [*batch_paths.items(),('broadcast_minus_ring',batch_paths['ring']-batch_paths['broadcast'])]:
    centered=Y-Y.mean(0);B=len(Y)
    direct=(gap*(Y.shape[1]-Y.sum(1))) if name!='broadcast_minus_ring' else gap*Y.sum(1)
    linear_var=float(gap**2*np.sum(centered.sum(1)**2)/(B*(B-1)))
    direct_var=float(direct.var(ddof=1)/B)
    diagonal_only=float(gap**2*np.sum(centered**2)/(B*(B-1)))
    assert abs(linear_var-direct_var)<1e-10
    propagation.append(dict(quantity=name,propagated_variance=linear_var,direct_variance=direct_var,
        variance_if_temporal_covariance_omitted=diagonal_only,
        full_to_diagonal_variance_ratio=linear_var/diagonal_only if diagonal_only else None))
np.savez_compressed(OUT/'propagation_batch_paths.npz',**batch_paths)

# Exact nonlinear withdrawal functional, local Jacobian and precision sensitivity.
functional=[];max_fd_error=0.
for gid in range(4):
    with np.load(DATA/(f'w{gid:04d}.npz')) as z:Q=z['final_Q'];psi=z['final_psi']
    for r in range(4):
        delta=Q[r,:,0]-Q[r,:,1];field=float(psi[r].mean());p0=1/(1+np.exp(-12*delta));pl=1/(1+np.exp(-12*(delta+.6*field)))
        derivative=float(12*field*np.mean(pl*(1-pl)))
        step=1e-6
        finite_difference=float(((1/(1+np.exp(-12*(delta+(.6+step)*field)))).mean()-(1/(1+np.exp(-12*(delta+(.6-step)*field)))).mean())/(2*step))
        max_fd_error=max(max_fd_error,abs(derivative-finite_difference))
        quantised=Q[r].astype(np.float32).astype(np.float64);qd=quantised[:,0]-quantised[:,1]
        p0q=(1/(1+np.exp(-12*qd))).mean();plq=(1/(1+np.exp(-12*(qd+.6*field)))).mean()
        functional.append(dict(graph_id=gid,rep=r,retained_probability=float(pl.mean()),withdrawn_probability=float(p0.mean()),
            exact_change=float(p0.mean()-pl.mean()),local_derivative_lambda=derivative,
            first_order_prediction_for_lambda_change=-.6*derivative,
            score_gradient_l1_retained=float(24*np.mean(pl*(1-pl))),score_gradient_l1_withdrawn=float(24*np.mean(p0*(1-p0))),
            signal_gradient_l1_retained=float(12*.6*np.mean(pl*(1-pl))),
            actual_float32_readout_change_retained=float(plq-pl.mean()),actual_float32_readout_change_withdrawn=float(p0q-p0.mean())))
pd.DataFrame(functional).to_csv(OUT/'withdrawal_functional_sensitivity.csv',index=False)
precision_bound=12/4*2*2**-25
assert max(abs(x['actual_float32_readout_change_retained']) for x in functional)<precision_bound
assert max(abs(x['actual_float32_readout_change_withdrawn']) for x in functional)<precision_bound
propagation_record=dict(regret=propagation,withdrawal_derivative_finite_difference_max_error=max_fd_error,
    worst_case_probability_error_from_float32_score_rounding=precision_bound,
    quantisation_scope='Readout-only bound assuming per-coordinate nearest float32 rounding; not a global simulator forward-error bound.',
    no_parameter_uncertainty='Parameters are fixed design constants, not fitted random estimates. No parameter posterior is assumed.')
(OUT/'propagation_checks.json').write_text(json.dumps(clean_json(propagation_record),indent=2,allow_nan=False))

many=end[(end.group=='M10000')&(end.metric=='optimal_fraction')]
summary=dict(method='Equal-weight independent seed-batch mean covariance; approximate Student-t 95% Monte Carlo intervals.',
    confidence_level=.95,endpoint_rows=len(end),paired_contrasts=len(contrasts),curve_rows=len(curves),
    main_batches=4,expanded_batches=16,many_batches=2,
    probability_intervals_untruncated=True,tests_or_p_values=False,
    many_probability_ci_min=float(many.ci95_low.min()),many_probability_ci_max=float(many.ci95_high.max()),
    propagation=propagation_record,seed_collision_checks_passed=True)
(OUT/'Statistics_Summary.json').write_text(json.dumps(clean_json(summary),indent=2,allow_nan=False))
print(json.dumps(clean_json(summary),indent=2))
