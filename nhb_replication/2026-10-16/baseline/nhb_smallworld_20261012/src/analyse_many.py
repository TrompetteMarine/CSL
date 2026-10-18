"""Independent review and reporting for exactly 10,000 communities."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study import ROOT,RESULT
rows=[];meta=[];err=0.
for p in sorted(RESULT.glob('m[0-9][0-9][0-9][0-9].json')):
 c=json.loads(p.read_text());meta.append(c);rr=json.loads((RESULT/(p.stem+'_rows.json')).read_text());rows.extend(rr);z=np.load(RESULT/(p.stem+'.npz'))
 assert c['M']==10000 and c['N']==10000*c['n'] and c['audit']['clips']==0
 adj=[]
 for i,row in enumerate(c['B']):
  assert abs(sum(v for _,v in row)-1)<1e-12 and all(v>=0 for _,v in row)
  adj.append({j for j,v in row if j!=i and v>0})
 assert sum(map(len,adj))==40000 and all(i in adj[j] for i,nbs in enumerate(adj) for j in nbs)
 assert z['final_Q'].dtype==np.float64 and 0<=z['final_Q'].min() and z['final_Q'].max()<=1
 for r in rr:
  j=r['rep'];v=z['path'][-100:,j,0].mean();q=z['final_Q'][j];err=max(err,abs(v-r['optimal_fraction']),abs((q[:,0]>q[:,1]).mean()-r['private_correct']),abs((q[:,0]-q[:,1]).mean()-r['private_gap']))
 assert err<1e-12
assert len(meta)==48
df=pd.DataFrame(rows);df.to_csv(RESULT/'many_communities_replications.csv',index=False)
s=df.groupby(['N','n','M','geometry','initial']).agg(replications=('wrong','size'),wrong_count=('wrong','sum'),optimal_fraction=('optimal_fraction','mean'),private_correct=('private_correct','mean'),private_gap=('private_gap','mean'),wrong_communities=('wrong_communities','mean'),regret=('regret','mean'),clustering=('clustering','mean'),mean_path=('mean_path','mean')).reset_index();s.to_csv(RESULT/'many_communities_summary.csv',index=False)
geos=['ring','ws0.01','ws0.1','broadcast'];sizes=[10,20,50];fig,axes=plt.subplots(1,2,figsize=(10.5,4),sharey=True)
for ax,initial,title in zip(axes,['misleading_random','misleading_contiguous'],['Initial errors scattered','Initial errors clustered']):
 a=s[s.initial==initial].pivot(index='n',columns='geometry',values='optimal_fraction').reindex(index=sizes,columns=geos)
 im=ax.imshow(a,aspect='auto',vmin=0,vmax=1,cmap='RdYlBu')
 for i in range(3):
  for j in range(4):v=a.iloc[i,j];ax.text(j,i,f'{v:.3f}',ha='center',va='center',color='white' if v>.8 or v<.2 else 'black')
 ax.set_xticks(range(4),['Ring','p=.01','p=.1','Broadcast']);ax.set_yticks(range(3),['10 / 100,000','20 / 200,000','50 / 500,000']);ax.set_title(title);ax.set_ylabel('Agents per community / total agents')
fig.colorbar(im,ax=list(axes),fraction=.025,pad=.03,label='Better-arm fraction, final 100 trials');fig.suptitle('Exactly 10,000 communities: 500-trial computational experiment',y=1.02)
for ext in ['png','pdf']:fig.savefig(ROOT/'figures'/('10_many_communities.'+ext),bbox_inches='tight',dpi=180)
plt.close(fig)
audit=dict(configurations=48,replications=len(df),communities_per_population=10000,total_agent_trials=sum(c['N']*c['T']*c['reps'] for c in meta),clips=sum(c['audit']['clips'] for c in meta),max_summary_error=err,gain_max=max(c['audit']['gain_max'] for c in meta),structural_batches_per_cell=2,innovations_per_batch=2,horizon=500,matrices_graph_invariants_checked=len(meta))
(RESULT/'Many_Communities_Review.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True);print(s.to_string(index=False),flush=True)
