import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from study import ROOT,RESULT

FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180,'font.family':'DejaVu Sans'})
def save(fig,name):
 fig.savefig(FIG/(name+'.png'),bbox_inches='tight');fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def analyse():
 rows=[]
 for f in sorted(RESULT.glob('*_rows.json')):
  if f.name[0] in 'cfx':rows.extend(json.loads(f.read_text()))
 df=pd.DataFrame(rows)
 df.to_csv(RESULT/'replications.csv',index=False)
 keys=['group','N','n','M','geometry','initial','mode','lam','theta']
 # Gamma must distinguish the two-community experiment.
 meta={p.stem:json.loads(p.read_text()) for prefix in ['c','f','x'] for p in RESULT.glob(prefix+'[0-9][0-9][0-9][0-9].json')}
 df['gamma']=df.label.map(lambda x:meta[x]['gamma']);df['k']=df.label.map(lambda x:meta[x].get('k',4));keys+=['gamma','k']
 stats=df.groupby(keys,dropna=False).agg(replications=('wrong','size'),wrong_count=('wrong','sum'),optimal_fraction=('optimal_fraction','mean'),private_correct=('private_correct','mean'),private_gap=('private_gap','mean'),regret=('regret','mean'),wrong_communities=('wrong_communities','mean'),community_variance=('community_variance','mean'),removed_probability=('mean_removed_probability','mean'),clustering=('clustering','mean'),mean_path=('mean_path','mean'),degree_sd=('degree_sd','mean')).reset_index()
 stats['wrong_rate']=stats.wrong_count/stats.replications;stats.to_csv(RESULT/'summary.csv',index=False)
 graphmeans=df.groupby(keys+['graph_id'],dropna=False).mean(numeric_only=True).reset_index();graphmeans.to_csv(RESULT/'graph_seed_means.csv',index=False)
 primary=stats[stats.group=='primary'];geos=['ring','ws0.01','ws0.1','ws1','regular','broadcast'];sizes=[20,50,100,200,500]
 fig,axes=plt.subplots(1,2,figsize=(12,4.7),sharey=True)
 for ax,initial,title in zip(axes,['misleading_random','misleading_contiguous'],['Initial errors scattered','Initial errors clustered']):
  tab=primary[primary.initial==initial].pivot(index='n',columns='geometry',values='optimal_fraction').reindex(index=sizes,columns=geos)
  im=ax.imshow(tab,vmin=0,vmax=1,cmap='RdYlBu',aspect='auto')
  for i in range(5):
   for j in range(6):
    val=tab.iloc[i,j];ax.text(j,i,f'{val:.3f}',ha='center',va='center',fontsize=8,color='white' if val>.8 or val<.2 else 'black')
  ax.set_xticks(range(6),['Ring','p=.01','p=.1','p=1','Regular\nrandom','Broadcast'],rotation=25,ha='right');ax.set_yticks(range(5),[f'{n} ({10000//n})' for n in sizes]);ax.set_title(title);ax.set_ylabel('Community size (number of communities)')
 fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.025,pad=.025,label='Better-arm choice fraction, final 100 trials')
 fig.suptitle('10,000 agents: geometry, community size and initial error placement',y=1.03);save(fig,'02_size_geometry')
 # Graph metrics separate shortened distances from loss of clustering.
 fig,axes=plt.subplots(1,2,figsize=(10,3.5))
 sub=stats[(stats.group=='primary')&(stats.n==100)&(stats.initial=='misleading_random')&stats.geometry.isin(geos[:4])].set_index('geometry').reindex(geos[:4])
 x=np.array([0,.01,.1,1])
 axes[0].plot(range(4),sub.clustering,'o-',color='#137c8b');axes[0].set_ylabel('Mean local clustering')
 axes[1].plot(range(4),sub.mean_path,'o-',color='#c85d31');axes[1].set_ylabel('Mean shortest-path length')
 for ax in axes:ax.set_xticks(range(4),['0','.01','.1','1']);ax.set_xlabel('Rewiring probability p')
 fig.suptitle('Community graph diagnostics (100 communities, initial degree 4)');fig.tight_layout();save(fig,'03_graph_diagnostics')
 # Mean trajectories, using independent population replications within graph seeds.
 fig,axes=plt.subplots(1,2,figsize=(10,4),sharex=True)
 for geo,col in [('ring','#137c8b'),('ws0.1','#9b6ab1'),('broadcast','#c85d31')]:
  entries=[c for c in meta.values() if c['group']=='primary' and c['n']==100 and c['initial']=='misleading_random' and c['geometry']==geo]
  paths=np.concatenate([np.load(RESULT/(c['label']+'.npz'))['path'] for c in entries],axis=1)
  axes[0].plot(np.arange(1,paths.shape[0]+1),paths[:,:,0].mean(1),label=geo,color=col)
  axes[1].plot(np.arange(1,paths.shape[0]+1),paths[:,:,1].mean(1),label=geo,color=col)
 axes[0].set_ylabel('Better-arm choice fraction');axes[1].set_ylabel('Privately favourable ranking fraction')
 for ax in axes:ax.axhline(.5,color='#999',ls=':',lw=1);ax.set_xlabel('Trial');ax.set_ylim(0,1);ax.legend(frameon=False)
 fig.tight_layout();save(fig,'04_private_public_paths')
 # Controls show negative results as well as failures.
 controls=stats[(stats.group=='controls')&(stats.n==100)]
 labels=[('neutral',0,.6,'Neutral'),('correct',0,.6,'Initially correct'),('reversed',0,.6,'Stronger correct'),('misleading_random',1,.6,'Constant, λ=.6'),('misleading_random',2,.6,'Strong constant'),('misleading_random',1,1.2,'Constant, λ=1.2')]
 fig,ax=plt.subplots(figsize=(10,4.2));width=.2
 for j,geo in enumerate(['ring','ws0.1','regular','broadcast']):
  vals=[float(controls[(controls.initial==a)&(controls['mode']==m)&(controls.lam==l)&(controls.geometry==geo)].optimal_fraction.iloc[0]) for a,m,l,_ in labels]
  ax.bar(np.arange(6)+(j-1.5)*width,vals,width,label=geo)
 ax.set_xticks(range(6),[x[3] for x in labels],rotation=20,ha='right');ax.set_ylim(0,1);ax.set_ylabel('Better-arm fraction');ax.legend(ncol=4,frameon=False);fig.tight_layout();save(fig,'05_controls')
 # Small-world regular-vs-rewired contrasts with graph-seed points.
 fig,axes=plt.subplots(1,2,figsize=(10,3.7),sharey=True)
 for ax,initial in zip(axes,['misleading_random','misleading_contiguous']):
  for geo,col in [('ring','#137c8b'),('ws0.1','#9b6ab1'),('regular','#777'),('broadcast','#c85d31')]:
   z=primary[(primary.initial==initial)&(primary.geometry==geo)].sort_values('n');ax.plot(z.n,z.optimal_fraction,'o-',label=geo,color=col)
  ax.set_xscale('log');ax.set_xticks(sizes,[str(x) for x in sizes]);ax.set_ylim(0,1);ax.set_xlabel('Community size');ax.set_title(initial.replace('misleading_',''));ax.legend(frameon=False,fontsize=8)
 axes[0].set_ylabel('Better-arm choice fraction');fig.tight_layout();save(fig,'06_geometry_lines')
 # Conceptual figure: modelled scores are distinguished from human beliefs.
 fig,ax=plt.subplots(figsize=(12,5.2));ax.set_xlim(0,12);ax.set_ylim(0,5.2);ax.axis('off')
 def box(x,y,w,h,txt,color):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',fc=color,ec='none'));ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=10)
 def arrow(a,b,color='#444',txt=None):
  ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,color=color,lw=1.7));
  if txt:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.13,txt,ha='center',fontsize=9,color=color)
 box(.2,2.3,2.3,1.2,'Private score pair\nQ(0), Q(1)','#d9edf0');box(3.5,2.3,2.4,1.2,'Stochastic choice\n+ expressed strength','#eee3f4');box(7,2.3,2.5,1.2,'Community signal\nmean signed strength','#f7e2cf');box(7,.2,2.5,1.1,'Community graph\nring → small world','#f7e2cf');box(.2,.2,2.3,1.1,'Own / peer rewards\nupdate private scores','#d9edf0');box(9.9,2.3,1.8,1.2,'Common\nbroadcast','#f7e2cf')
 arrow((2.6,2.9),(3.4,2.9));arrow((6,2.9),(6.9,2.9));arrow((8.2,2.2),(8.2,1.4),'#c85d31');arrow((7,.8),(5,.8),'#c85d31');arrow((5,.8),(5,2.2),'#c85d31','lagged field');arrow((4,2.2),(2.6,1),'#137c8b');arrow((1.3,1.4),(1.3,2.2),'#137c8b');arrow((9.6,2.9),(9.8,2.9),'#c85d31');arrow((10.8,2.2),(10.8,1.8),'#c85d31');arrow((10.8,1.8),(5,1.8),'#c85d31');ax.text(9.7,1.93,'common input',ha='center',fontsize=9,color='#c85d31')
 ax.text(.2,4.5,'Private learning, public choice and the route of social exposure',fontsize=15,weight='bold');ax.text(.2,3.9,'Each community retains the distribution of individual score pairs.',fontsize=11);ax.text(.2,-.1,'Geometry changes reward and choice exposure jointly; broadcast reallocates choice exposure only.',fontsize=10)
 save(fig,'01_concept')
 # Two-community bridge.
 two=stats[(stats.group=='two')&(stats.initial=='misleading_random')].sort_values('gamma')
 fig,ax=plt.subplots(figsize=(6.8,3.6));ax.plot(two.gamma,two.optimal_fraction,'o-',color='#137c8b',label='Public choices');ax.plot(two.gamma,two.private_correct,'s--',color='#c85d31',label='Private rankings');ax.set_ylim(0,1);ax.set_xlabel('Between-community exposure Γ');ax.set_ylabel('Fraction favouring better arm');ax.legend(frameon=False);ax.set_title('Two communities, 5,000 agents each');fig.tight_layout();save(fig,'07_two_communities')
 # Expanded follow-up shown separately, not hidden in the initial design.
 expanded=df[df.group.isin(['primary','followup'])&(df.initial=='misleading_contiguous')&df.n.isin([50,100,200])&df.geometry.isin(['ring','ws0.01','ws0.1'])]
 expanded.groupby(['n','geometry']).agg(replications=('wrong','size'),optimal_fraction=('optimal_fraction','mean'),wrong_count=('wrong','sum')).reset_index().to_csv(RESULT/'expanded_clustered_summary.csv',index=False)
 fig,ax=plt.subplots(figsize=(7.8,4));gm=expanded.groupby(['n','geometry','graph_id']).optimal_fraction.mean().reset_index()
 for j,geo in enumerate(['ring','ws0.01','ws0.1']):
  z=gm[gm.geometry==geo];xx=np.array([list([50,100,200]).index(n) for n in z.n])+(j-1)*.2
  ax.scatter(xx,z.optimal_fraction,s=20,alpha=.6,label=geo)
  ax.plot(np.arange(3)+(j-1)*.2,z.groupby('n').optimal_fraction.mean().reindex([50,100,200]),lw=2)
 ax.set_xticks(range(3),['50','100','200']);ax.set_ylim(0,1);ax.set_xlabel('Community size');ax.set_ylabel('Better-arm fraction: graph/seed batch means');ax.legend(frameon=False);fig.tight_layout();save(fig,'08_expanded_geometry')
 # Matched continuation, no entry-state outcome selection.
 wd=pd.DataFrame(json.loads((RESULT/'withdrawal.json').read_text()));wd.to_csv(RESULT/'withdrawal.csv',index=False)
 fig,ax=plt.subplots(figsize=(6.4,3.5));zz=wd.groupby('lam').optimal_fraction.mean();ax.bar(['Influence retained','Influence withdrawn'],[zz.loc[.6],zz.loc[0]],color=['#c85d31','#137c8b']);ax.set_ylim(0,1);ax.set_ylabel('Better-arm fraction after 1,000 further trials');fig.tight_layout();save(fig,'09_withdrawal')
 audit=dict(configurations=len(meta),replications=len(df),N10000_replications=int((df.N==10000).sum()),total_agent_trials=int(sum(c['N']*c['T']*c['reps'] for c in meta.values())),clips=int(sum(c['audit']['clips'] for c in meta.values())),gain_max=max(c['audit']['gain_max'] for c in meta.values()),withdrawal_entry_populations=16,paired_continuations=32)
 (RESULT/'analysis_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)
 return stats
if __name__=='__main__':analyse()
