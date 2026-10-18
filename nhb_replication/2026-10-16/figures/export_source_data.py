"""Export the actual figure series and their population/batch inputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import rebuild_figures as r

OUT=r.HERE/'source_exports';OUT.mkdir(exist_ok=True)
KEYS=['group','N','n','M','geometry','initial','mode','lam','theta','gamma','k','T']
manifest={'version':'2026-10-16','figure_numbering':{'main01_concept':'Fig.1','main06_population_limits':'Fig.2','main02_two_communities':'Fig.3','main03_size_geometry':'Fig.4','main04_private_withdrawal':'Fig.5','main05_many_communities':'Fig.6','supp06_two_initialisations':'Supplementary Fig.1','supp01_graph_diagnostics':'Supplementary Fig.2','supp03_expanded_geometry':'Supplementary Fig.3','supp04_choice_degree':'Supplementary Fig.4','supp05_scaling':'Supplementary Fig.5','supp02_controls':'Supplementary Fig.6','supp07_many_detail':'Supplementary Fig.7','supp08_score_response':'Supplementary Fig.8','supp09_signal_anatomy':'Supplementary Fig.9','supp10_noise_propagation':'Supplementary Fig.10'},'figures':{}}

def put(key,name,df,description):
 p=OUT/(key+'_'+name+'.csv');df.to_csv(p,index=False)
 manifest['figures'].setdefault(key,{'files':[]})['files'].append({'path':str(p.relative_to(r.ROOT)),'description':description,'rows':len(df)})

def populations(key,specs):
 rows=[]
 for panel,d,metrics in specs:
  d=d.copy();d['design_stage']=d.group
  if set(d.group)=={'primary','followup'}:d['group']='expanded'
  z=d.melt(id_vars=[c for c in d.columns if c not in metrics],value_vars=metrics,var_name='metric',value_name='value')
  z['panel']=panel;rows.append(z[['panel']+KEYS+['design_stage','label','graph_id','rep','metric','value']])
 pop=pd.concat(rows,ignore_index=True);put(key,'population_values',pop,'Whole-population outcomes used in the panels; never independent agent observations.')
 batch=pop.groupby(['panel']+KEYS+['metric','graph_id'],dropna=False).value.mean().reset_index()
 put(key,'batch_means',batch,'Means of the recorded population outcomes within each seed batch.')
 cohorts=pop[['panel']+KEYS+['metric']].drop_duplicates()
 intervals=cohorts.merge(r.endpoint_intervals,on=KEYS+['metric'],how='inner',validate='many_to_one')
 if len(intervals):put(key,'plotted_intervals',intervals,'Approximate two-sided 95% t Monte Carlo intervals across seed-batch means; raw endpoints untruncated.')

def trajectories(key,curveids,panelmap):
 for filename,description in [('trajectory_population_values.csv','Trailing ten-trial population series, with min_periods=1.'),('trajectory_batch_means.csv','Seed-batch means of the smoothed population trajectories.'),('trajectory_intervals.csv','Actual plotted means and pointwise approximate 95% t Monte Carlo intervals.')]:
  z=pd.read_csv(r.STATS/filename);z=z[z.curve_id.isin(curveids)].copy();z['panel']=[panelmap[(a,b)] for a,b in zip(z.curve_id,z.metric)]
  put(key,filename.removesuffix('.csv'),z,description)

def export():
 d=r.df;p=r.primary;m=r.many
 two=d[(d.group=='two')&(d.initial=='misleading_random')]
 populations('main02_two_communities',[('a',two,['optimal_fraction','private_correct'])])
 curves=[f'two_gamma_{float(g)}' for g in [0,.15,.5]];trajectories('main02_two_communities',curves,{(x,'optimal_fraction'):'b' for x in curves})
 g=np.linspace(0,.5,200);z=pd.DataFrame({'panel':'c','gamma':g,'common_multiplier':1.8,'difference_multiplier':1.8*(1-2*g)})
 put('main02_two_communities','analytic_multipliers',z,'Exact frozen-score derivatives at h=12, lambda=0.6 and score floor0.5.')
 put('main02_two_communities','annotated_multiplier',pd.DataFrame([dict(panel='c',gamma=.15,difference_multiplier=1.26)]),'The separately annotated exact multiplier.')
 populations('main03_size_geometry',[(a,p[p.initial==i],['optimal_fraction']) for a,i in [('a','misleading_random'),('b','misleading_contiguous')]]+[('c',p[(p.initial=='misleading_contiguous')&p.geometry.isin(r.GEOS[:-1])],['optimal_fraction']),('d',p[(p.initial=='misleading_random')&(p.geometry=='broadcast')],['optimal_fraction','private_correct'])])
 curves=[f'n100_scattered_{g}' for g in ['ring','ws0.1','broadcast']]
 trajectories('main04_private_withdrawal',curves,{(x,metric):panel for x in curves for metric,panel in [('optimal_fraction','a'),('private_prechoice','b')]})
 w=r.wd.copy();ww=[]
 for panel,metric in [('c','first_step_probability'),('d','optimal_fraction')]:
  z=w[['graph_id','rep','lam',metric]].rename(columns={metric:'value'});z['panel']=panel;z['metric']=metric;ww.append(z)
 z=pd.concat(ww,ignore_index=True);put('main04_private_withdrawal','paired_withdrawal_populations',z,'All16 complete saved entry states continued under retained or withdrawn influence; matched by graph_id and rep.')
 put('main04_private_withdrawal','withdrawal_batch_means',z.groupby(['panel','lam','metric','graph_id']).value.mean().reset_index(),'Four seed-batch means for each arm of each withdrawal contrast.')
 z=r.withdrawal_intervals[r.withdrawal_intervals.metric.isin(['first_step_probability','optimal_fraction'])].copy();z['panel']=z.metric.map({'first_step_probability':'c','optimal_fraction':'d'})
 put('main04_private_withdrawal','withdrawal_intervals',z,'Actual untruncated marginal interval whiskers; paired differences are analysed separately in statistics/paired_contrasts.csv.')
 populations('main05_many_communities',[(a,m[m.initial==i],['optimal_fraction']) for a,i in [('a','misleading_random'),('b','misleading_contiguous')]]+[('c',m[(m.initial=='misleading_contiguous')&m.geometry.isin(['ring','ws0.01','ws0.1'])],['wrong_communities']),('d',m[(m.initial=='misleading_contiguous')&(m.geometry=='broadcast')],['optimal_fraction','private_correct'])])
 graphspec=[]
 for row,z in enumerate([p[(p.n==100)&(p.initial=='misleading_random')&p.geometry.isin(r.GEOS[:4])],m[(m.n==50)&(m.initial=='misleading_contiguous')&m.geometry.isin(r.GEOS[:3])]]):
  for j,metric in enumerate(['clustering','mean_path','degree_sd']):graphspec.append((chr(97+3*row+j),z,[metric]))
 populations('supp01_graph_diagnostics',graphspec)
 z=d[d.group=='controls'];populations('supp02_controls',[('a',z[z['mode']==0],['optimal_fraction']),('b',z[z['mode']!=0],['optimal_fraction'])])
 z=d[d.group.isin(['primary','followup'])&(d.initial=='misleading_contiguous')&d.n.isin([50,100,200])&d.geometry.isin(['ring','ws0.01','ws0.1'])]
 populations('supp03_expanded_geometry',[(chr(97+i),z[z.n==n],['optimal_fraction']) for i,n in enumerate([50,100,200])])
 specs=[]
 for i,n in enumerate([50,100]):specs.append((chr(97+i),d[d.group.isin(['primary','choiceonly'])&(d.n==n)&(d.initial=='misleading_contiguous')&d.geometry.isin(['ws0.01','ws0.1','ws1','regular'])],['optimal_fraction']))
 for i,n in enumerate([20,100]):specs.append((chr(99+i),d[d.group.isin(['primary','degree'])&(d.n==n)&(d.initial=='misleading_contiguous')&d.geometry.isin(r.GEOS[:4])],['optimal_fraction']))
 populations('supp04_choice_degree',specs)
 z=d[d.group=='scaling'];specs=[]
 for row,sel in enumerate([z.M==20,z.n==20]):
  for j,metric in enumerate(['optimal_fraction','private_correct']):specs.append((chr(97+2*row+j),z[sel],[metric]))
 populations('supp05_scaling',specs)
 z=d[d.group=='two'];populations('supp06_two_initialisations',[('a',z,['optimal_fraction']),('b',z,['private_correct'])])
 bins=np.linspace(0,1,26);hist=[]
 for j,g in enumerate(['ring','ws0.01','ws0.1']):
  for row,n in enumerate([20,50]):
   configs=[c for c in r.meta.values() if c['group']=='M10000' and c['n']==n and c['geometry']==g and c['initial']=='misleading_contiguous']
   for c in configs:
    with np.load(r.DATA/(c['label']+'.npz')) as data:tail=data['tail']
    for rep,values in enumerate(tail):
     counts=np.histogram(values,bins=bins)[0]
     for k,count in enumerate(counts):hist.append(dict(panel=chr(97+3*row+j),geometry=g,n=n,label=c['label'],graph_id=c['graph_id'],rep=rep,bin_left=bins[k],bin_right=bins[k+1],bin_centre=(bins[k]+bins[k+1])/2,community_count=int(count),fraction=count/len(values)))
 hist=pd.DataFrame(hist);put('supp07_many_detail','population_bin_counts',hist,'Exact counts and normalised fractions for each25-bin within-population community histogram.')
 z=hist.groupby(['panel','geometry','n','graph_id','bin_left','bin_right','bin_centre']).fraction.mean().reset_index();put('supp07_many_detail','batch_histograms',z,'Bin heights averaged over two populations per seed batch.')
 put('supp07_many_detail','plotted_mean_histograms',z.groupby(['panel','geometry','n','bin_left','bin_right','bin_centre']).fraction.mean().reset_index(),'Actual plotted bar heights, averaging the two seed-batch histograms.')
 pp=np.linspace(0,1,1001);put('supp08_score_response','score_grid',pd.DataFrame({'panel':'a','p':pp,'primary':np.maximum(pp,1-pp),'constant_half':.5,'quadratic_floor_half':.5+.5*(2*pp-1)**2,'zero_floor':np.abs(2*pp-1)}),'Exact analytic score functions, not new simulation experiments.')
 x=np.linspace(-.6,.6,1001);prob=1/(1+np.exp(-12*x));u=2*prob-1
 put('supp08_score_response','response_grid',pd.DataFrame({'panel':'b','x':x,'p':prob,'primary':np.maximum(prob,1-prob)*u,'constant_half':.5*u,'zero_floor':np.abs(u)*u}),'Exact analytic frozen-score cue responses at h=12.')
 dd=np.linspace(0,1,200);put('supp08_score_response','gain_grid',pd.DataFrame({'panel':'c','d':dd,'alpha_plus':.4-.35*dd,'alpha_minus':.05+.35*dd}),'Prescribed learning-rate functions; primary score domain starts at0.5.')
 # The noise comparison fixes sender p,d at each actual broadcast entry state.
 noise=np.load(r.STATS/'conditional_noise_covariances.npz');heat=[]
 for kind,panel in [('ring','b'),('broadcast','c')]:
  cov=noise[kind+'_covariance'];sd=np.sqrt(np.diagonal(cov,axis1=1,axis2=2));corr=cov/(sd[:,:,None]*sd[:,None,:]);mean=corr.mean(0)
  for i in range(mean.shape[0]):
   for j in range(mean.shape[1]):heat.append(dict(panel=panel,routing=kind,receiving_community=i+1,other_community=j+1,mean_conditional_correlation=mean[i,j],entry_states=16))
 put('supp10_noise_propagation','plotted_mean_correlations',pd.DataFrame(heat),'Entrywise means of exact conditional correlation matrices across all16 fixed actual sender states; not correlations estimated from16 observations.')
 for name in ['conditional_noise_population.csv','conditional_noise_batch.csv','conditional_noise_ensemble_summary.csv']:
  z=pd.read_csv(r.STATS/name);put('supp10_noise_propagation',name.removesuffix('.csv'),z,'Analytic conditional-noise summaries and Monte Carlo intervals across the saved entry-state ensemble; paneld uses ring_field_rms_sd and broadcast_field_sd.')
 for key,title in [('main01_concept','Decision-generated credibility and exposure channels'),('main06_population_limits','Two distinct population-limit routes'),('supp09_signal_anatomy','Sender/receiver anatomy and immediate withdrawal')]:
  manifest['figures'][key]={'files':[],'type':'Analytic schematic','description':title,'source_tex':str((r.HERE/(key+'.tex')).relative_to(r.ROOT))}
 for key,entry in manifest['figures'].items():entry['display_number']=manifest['figure_numbering'][key]
 (r.HERE/'Source_Data_Manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'figures':len(manifest['figures']),'csv_files':sum(len(x['files']) for x in manifest['figures'].values())}))

if __name__=='__main__':export()
