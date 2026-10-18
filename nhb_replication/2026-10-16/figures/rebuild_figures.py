"""Publication figures from saved population trajectories, without new simulations.

Run from any directory. Displayed uncertainty comes from the independently
audited approximate t Monte Carlo intervals across seed-batch means. Trajectory
bands are pointwise. Very small batch counts limit calibration; raw endpoints
are not truncated. Batch points and descriptive ranges remain in the ledger.
"""
from pathlib import Path
import json, shutil, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, PercentFormatter
from matplotlib.text import Text

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BASE=ROOT/'baseline'/'nhb_smallworld_20261012'/'results'
DATA=ROOT/'data'/'results'
STATS=ROOT/'statistics'
WIDTH=180/25.4
plt.rcParams.update({
 'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':7,
 'axes.labelsize':7,'xtick.labelsize':6.8,'ytick.labelsize':6.8,
 'legend.fontsize':6.8,'axes.linewidth':.55,'lines.linewidth':1.2,
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#55616B',
 'text.color':'#25313B','axes.labelcolor':'#25313B','xtick.color':'#25313B',
 'ytick.color':'#25313B','savefig.dpi':300,'pdf.fonttype':42,'ps.fonttype':42,
 'figure.constrained_layout.use':True,'figure.constrained_layout.h_pad':.06,
 'figure.constrained_layout.w_pad':.075,'figure.constrained_layout.hspace':.06,
 'figure.constrained_layout.wspace':.055,
})
C={'ring':'#0072B2','ws0.01':'#56B4E9','ws0.1':'#009E73',
   'ws1':'#8C5694','regular':'#6C737A','broadcast':'#D55E00',
   'public':'#D55E00','private':'#0072B2','withdrawn':'#009E73'}
L={'ring':'Ring','ws0.01':r'WS $p=0.01$','ws0.1':r'WS $p=0.1$',
   'ws1':r'WS $p=1$','regular':'Regular random','broadcast':'Broadcast'}
MARK={'ring':'o','ws0.01':'s','ws0.1':'^','ws1':'D','regular':'v','broadcast':'P'}
GEOS=list(L)
SIZES=[20,50,100,200,500]
ledger=[]
collision_audit=[]
meta={p.stem:json.loads(p.read_text()) for pref in 'cfxwm' for p in BASE.glob(pref+'[0-9][0-9][0-9][0-9].json')}
df=pd.read_csv(BASE/'replications.csv')
for field,default in [('gamma',.15),('k',4)]:
 df[field]=df.label.map(lambda lab:meta[lab].get(field,default))
df['T']=df.label.map(lambda lab:meta[lab]['T'])
many=pd.read_csv(BASE/'many_communities_replications.csv')
for field,default in [('gamma',.15),('k',4),('T',500)]:
 many[field]=many.label.map(lambda lab:meta[lab].get(field,default))
wd=pd.read_csv(BASE/'withdrawal.csv')
primary=df[df.group=='primary']
endpoint_intervals=pd.read_csv(STATS/'endpoint_intervals.csv')
withdrawal_intervals=pd.read_csv(STATS/'withdrawal_intervals.csv')
trajectory_intervals=pd.read_csv(STATS/'trajectory_intervals.csv')

def interval(d,col='optimal_fraction'):
 """Exact cohort lookup; never silently substitute an interval for another estimand."""
 if 'group' not in d:
  ss=withdrawal_intervals[(withdrawal_intervals.metric==col)&np.isclose(withdrawal_intervals.lam,float(d.lam.iloc[0]))]
 else:
  groups=set(d.group.unique());group='expanded' if groups=={'primary','followup'} else next(iter(groups))
  assert len(groups)==1 or group=='expanded',groups
  ss=endpoint_intervals[(endpoint_intervals.group==group)&(endpoint_intervals.metric==col)]
  for key in ['N','n','M','geometry','initial','mode','lam','theta','gamma','k','T']:
   vals=d[key].unique();assert len(vals)==1,(key,vals)
   ss=ss[np.isclose(ss[key],vals[0])] if np.issubdtype(ss[key].dtype,np.number) else ss[ss[key]==vals[0]]
 assert len(ss)==1,(col,len(d),ss)
 row=ss.iloc[0];assert abs(float(row['mean'])-d[col].mean())<2e-12,(col,row['mean'],d[col].mean())
 return row

def ci_mark(ax,x,d,col,color,capsize=2,lw=.75):
 z=interval(d,col)
 ax.errorbar([x],[z['mean']],yerr=[[z['mean']-z.ci95_low],[z.ci95_high-z['mean']]],fmt='none',ecolor=color,capsize=capsize,lw=lw,zorder=3)
 return z

def ci_band(ax,x,d,key,col,color):
 zz=[interval(d[d[key]==v],col) for v in sorted(d[key].unique())]
 ax.fill_between(x,[z.ci95_low for z in zz],[z.ci95_high for z in zz],color=color,alpha=.1,linewidth=0,zorder=1)

def panel(ax,letter,title):
 ax.set_title(title,loc='left',pad=7,fontweight='medium')
 ax.text(-.14,1.045,letter,transform=ax.transAxes,fontweight='bold',fontsize=7,va='bottom')

def probability(ax,ylabel='Better-arm choice fraction',ylim=(-.025,1.17)):
 ax.set_ylim(*ylim);ax.set_yticks([0,.25,.5,.75,1]);ax.set_ylabel(ylabel)
 ax.axhline(.5,color='#AEB6BC',lw=.65,ls=(0,(3,3)),zorder=0)

def save(fig,name):
 fig.set_size_inches(WIDTH,fig.get_size_inches()[1],forward=True)
 fig.canvas.draw()
 renderer=fig.canvas.get_renderer();texts=[]
 tick_labels=set();drawn_tick_labels=set()
 for ax in fig.axes:
  for axis in [ax.xaxis,ax.yaxis]:
   for tick in axis.get_major_ticks()+axis.get_minor_ticks():tick_labels.update([tick.label1,tick.label2])
   for tick in axis._update_ticks():drawn_tick_labels.update([tick.label1,tick.label2])
 for t in fig.findobj(match=Text):
  if not t.get_visible() or not t.get_text().strip():continue
  if t in tick_labels and t not in drawn_tick_labels:continue
  # An Annotation's override includes its leader arrow. Audit the text itself;
  # leaders and data marks are reviewed separately in the rendered figures.
  bb=Text.get_window_extent(t,renderer)
  if bb.width>0 and bb.height>0:texts.append((t.get_text(),bb))
 clashes=[]
 for i,(ta,a) in enumerate(texts):
  for tb,b in texts[i+1:]:
   intersection=a.intersection(a,b)
   if intersection is not None and intersection.width>.8*fig.dpi/72 and intersection.height>.8*fig.dpi/72:
    clashes.append({'text_a':ta,'text_b':tb,'intersection_px':[intersection.width,intersection.height]})
 collision_audit.append({'figure':name,'text_collision_candidates':clashes})
 for ext in ['pdf','png','svg']:
  tmp=HERE/('.'+name+'.building.'+ext)
  with plt.rc_context({'svg.fonttype':'none'}):fig.savefig(tmp,facecolor='white',dpi=600 if ext in ['pdf','svg'] else 300)
  tmp.replace(HERE/(name+'.'+ext))
 plt.close(fig)

def batches(d,col='optimal_fraction',keys=None):
 return d.groupby((keys or [])+['graph_id'],dropna=False)[col].mean().reset_index()

def record(name,d,col='optimal_fraction',keys=None):
 keys=keys or ['geometry','n','initial']
 g=d.groupby(keys,dropna=False)
 for key,sub in g:
  if not isinstance(key,tuple):key=(key,)
  b=batches(sub,col)
  ci=interval(sub,col)
  ledger.append(dict(figure=name,metric=col,**dict(zip(keys,key)),trajectories=len(sub),
    seed_batches=len(b),mean=float(sub[col].mean()),batch_min=float(b[col].min()),batch_max=float(b[col].max()),
    mcse=ci.mcse,ci95_low=ci.ci95_low,ci95_high=ci.ci95_high,df=ci['df']))

def lines(ax,d,x='n',geos=GEOS,col='optimal_fraction',xvalues=None,points=True,band=False):
 xv=xvalues or sorted(d[x].unique());xx=np.arange(len(xv))
 for j,geo in enumerate(geos):
  ss=d[d.geometry==geo];b=batches(ss,col,[x]);a=b.groupby(x)[col].agg(['mean','min','max']).reindex(xv)
  if len(ss)==0:continue
  ax.plot(xx,a['mean'],marker=MARK[geo],ms=4,color=C[geo],label=L[geo],zorder=4)
  if band:ci_band(ax,xx,ss,x,col,C[geo])
  else:
   for k,v in enumerate(xv):ci_mark(ax,k,ss[ss[x]==v],col,C[geo])
  if points:
   for k,v in enumerate(xv):
    z=b[b[x]==v].sort_values('graph_id');offset=np.linspace(-.045,.045,len(z))
    ax.scatter(k+offset,z[col],s=9,color=C[geo],alpha=.45,linewidths=0,zorder=3)
 ax.set_xticks(xx,[str(v) for v in xv]);ax.set_xlabel('Agents per community, $n$')

def path_batches(configs,col=0):
 # One line per batch: first average population replications within each config.
 out=[]
 for conf in sorted(configs,key=lambda c:c['graph_id']):
  with np.load(DATA/(conf['label']+'.npz')) as z:out.append(z['path'][:,:,col].mean(1))
 return np.asarray(out)

def pathplot(ax,curve_id,col,color,label):
 z=trajectory_intervals[(trajectory_intervals.curve_id==curve_id)&(trajectory_intervals.metric==col)].sort_values('trial')
 assert len(z)==1000,(curve_id,col,len(z))
 ax.fill_between(z.trial,z.ci95_low,z.ci95_high,color=color,alpha=.14,linewidth=0)
 ax.plot(z.trial,z['mean'],color=color,label=label,lw=1.25)
 ax.set_xlabel('Trial');ax.set_xlim(0,1000)

def main_two():
 d=df[(df.group=='two')&(df.initial=='misleading_random')]
 fig,axs=plt.subplots(1,3,figsize=(7.25,2.8),gridspec_kw={'width_ratios':[1,1.15,1]})
 ax=axs[0];panel(ax,'a','Decisions and rankings')
 for col,color,marker,label in [('optimal_fraction',C['public'],'o','Public choices'),('private_correct',C['private'],'s','Private rankings')]:
  b=batches(d,col,['gamma']);a=b.groupby('gamma')[col].agg(['mean','min','max'])
  ax.plot(a.index,a['mean'],marker+'-',color=color,ms=3.5,label=label,lw=1.1)
  for i,g in enumerate(a.index):
   ci_mark(ax,g,d[d.gamma==g],col,color)
   z=b[b.gamma==g];ax.scatter(g+np.linspace(-.008,.008,len(z)),z[col],s=8,color=color,alpha=.5,zorder=4)
 ax.set_xticks([0,.15,.5]);ax.set_xlabel('Between-community\nweight, $\\Gamma$');probability(ax,'Fraction favouring better arm');ax.legend(loc='upper right',frameon=False)
 ax=axs[1];panel(ax,'b','Decision trajectories')
 for gamma,color in [(0,C['ring']),(.15,C['ws0.1']),(.5,C['public'])]:
  pathplot(ax,f'two_gamma_{float(gamma)}','optimal_fraction',color,rf'$\Gamma={gamma:g}$')
 probability(ax);ax.legend(loc='upper right',frameon=False);ax.set_xticks([0,500,1000])
 ax=axs[2];panel(ax,'c','Frozen local multipliers')
 g=np.linspace(0,.5,200);ax.plot(g,np.full_like(g,1.8),color=C['private'],label='Common mode')
 ax.plot(g,1.8*(1-2*g),color=C['public'],label='Difference mode')
 ax.axhline(1,color='#7A858E',ls='--',lw=.8);ax.scatter([.15],[1.26],color=C['public'],s=20,zorder=4)
 ax.annotate('1.26',(.15,1.26),(.02,1.1),fontsize=7,arrowprops={'arrowstyle':'-','lw':.5,'color':C['public']})
 ax.set_xlabel('Between-community\nweight, $\\Gamma$');ax.set_ylabel('Derivative multiplier');ax.set_xticks([0,.15,.5]);ax.set_ylim(-.05,2.05);ax.legend(frameon=False,loc='center right',bbox_to_anchor=(1,.75))
 save(fig,'main02_two_communities');record('main02_two_communities',d,keys=['gamma']);record('main02_two_communities',d,'private_correct',['gamma'])

def heat(ax,d,initial,title,letter):
 tab=d[d.initial==initial].groupby(['n','geometry']).optimal_fraction.mean().unstack().reindex(index=SIZES,columns=GEOS)
 cmap=LinearSegmentedColormap.from_list('accuracy',['#F4E2D7','#F6F6F4','#1A6D91'])
 im=ax.pcolormesh(np.arange(7)-.5,np.arange(6)-.5,tab.to_numpy(),vmin=0,vmax=1,cmap=cmap,shading='flat',edgecolors='face',linewidth=.05)
 ax.set_ylim(4.5,-.5);ax.set_xlim(-.5,5.5)
 for i in range(len(SIZES)):
  for j in range(len(GEOS)):
   v=tab.iloc[i,j];ax.text(j,i,f'{v:.3f}'.removeprefix('0'),ha='center',va='center',fontsize=6.8,color='white' if v>.76 else '#25313B')
 ax.set_xticks(range(6),['Ring','WS .01','WS .1','WS 1','Regular','Broadcast'],rotation=45,ha='right')
 ax.set_yticks(range(5),[str(n) for n in SIZES]);ax.set_ylabel('Agents per community, $n$');panel(ax,letter,title)
 ax.spines[['left','bottom']].set_visible(False);ax.tick_params(length=0)
 return im

def main_size():
 fig,axs=plt.subplots(2,2,figsize=(7.25,5.7),gridspec_kw={'height_ratios':[1,1.1]})
 heat(axs[0,0],primary,'misleading_random','Scattered initial errors','a')
 im=heat(axs[0,1],primary,'misleading_contiguous','Clustered initial errors','b')
 cb=fig.colorbar(im,ax=list(axs[0,:]),fraction=.025,pad=.015);cb.set_label('Better-arm choices');cb.set_ticks([0,.5,1])
 ax=axs[1,0];panel(ax,'c','Clustered errors: local exposure')
 z=primary[primary.initial=='misleading_contiguous'];lines(ax,z,geos=GEOS[:-1],band=False)
 ax.set_ylim(.4,1.17);ax.set_yticks([.5,.75,1]);ax.set_ylabel('Better-arm choice fraction')
 ax.legend(frameon=False,ncol=2,loc='upper left',bbox_to_anchor=(0,-.23),columnspacing=.8,handlelength=1.7)
 ax=axs[1,1];panel(ax,'d','Broadcast: decisions and rankings')
 z=primary[(primary.initial=='misleading_random')&(primary.geometry=='broadcast')]
 for col,color,marker,label in [('optimal_fraction',C['public'],'o','Public choices'),('private_correct',C['private'],'s','Private rankings')]:
  b=batches(z,col,['n']);a=b.groupby('n')[col].agg(['mean','min','max']).reindex(SIZES);x=np.arange(5)
  ax.plot(x,a['mean'],marker=marker,ms=4,color=color,label=label)
  ci_band(ax,x,z,'n',col,color)
  for i,n in enumerate(SIZES):
   zz=b[b.n==n];ax.scatter(i+np.linspace(-.04,.04,len(zz)),zz[col],s=8,color=color,alpha=.5)
 probability(ax,'Fraction favouring better arm');ax.set_xticks(range(5),SIZES);ax.set_xlabel('Agents per community, $n$');ax.legend(frameon=False,loc='center right',bbox_to_anchor=(1,.28))
 save(fig,'main03_size_geometry');record('main03_size_geometry',primary);record('main03_size_geometry',z,'private_correct')

def main_private():
 fig,axs=plt.subplots(2,2,figsize=(7.25,5.45))
 for ax,col,letter,title in [(axs[0,0],0,'a','Public decisions'),(axs[0,1],1,'b','Private rankings')]:
  panel(ax,letter,title)
  for geo in ['ring','ws0.1','broadcast']:
   pathplot(ax,f'n100_scattered_{geo}','optimal_fraction' if col==0 else 'private_prechoice',C[geo],L[geo])
  probability(ax,'Better-arm choice fraction' if col==0 else r'Fraction with $Q_i(0)>Q_i(1)$');ax.set_xticks([0,500,1000])
  ax.legend(frameon=False,loc='lower right' if col==1 else 'center right',bbox_to_anchor=(1,.12 if col==1 else .67))
 for ax,col,letter,title in [(axs[1,0],'first_step_probability','c','Immediate withdrawal effect'),(axs[1,1],'optimal_fraction','d','After 1,000 further trials')]:
  panel(ax,letter,title)
  pv=wd.pivot(index=['graph_id','rep'],columns='lam',values=col)
  for _,row in pv.iterrows():ax.plot([0,1],[row.loc[.6],row.loc[0]],color='#A8B2BA',lw=.65,alpha=.65,zorder=1)
  for x,lam,color in [(0,.6,C['public']),(1,0,C['withdrawn'])]:
   vals=wd[wd.lam==lam];b=batches(vals,col)
   ci_mark(ax,x,vals,col,color,capsize=3,lw=1)
   ax.scatter(x+np.linspace(-.065,.065,len(b)),b[col],s=22,color=color,edgecolor='white',lw=.5,zorder=5)
   avg=vals[col].mean();ax.plot([x-.13,x+.13],[avg,avg],color=color,lw=2.4,zorder=6)
   ax.text(x,avg+.055,f'{avg:.3f}',ha='center',color=color,fontweight='bold',fontsize=7)
  ax.set_xlim(-.38,1.38);ax.set_xticks([0,1],['Influence retained','Influence withdrawn']);probability(ax,'Better-arm choice probability' if col=='first_step_probability' else 'Better-arm choice fraction')
 save(fig,'main04_private_withdrawal')
 record('main04_private_withdrawal',wd,'first_step_probability',['lam']);record('main04_private_withdrawal',wd,'optimal_fraction',['lam'])

def main_many():
 fig,axs=plt.subplots(2,2,figsize=(7.25,5.4));sizes=[10,20,50]
 for ax,init,letter,title in [(axs[0,0],'misleading_random','a','Scattered initial errors'),(axs[0,1],'misleading_contiguous','b','Clustered initial errors')]:
  panel(ax,letter,title);lines(ax,many[many.initial==init],geos=['ring','ws0.01','ws0.1','broadcast'],xvalues=sizes,band=True);probability(ax)
  ax.set_xticklabels(['10\n(100,000)','20\n(200,000)','50\n(500,000)']);ax.set_xlabel('Agents per community, $n$\n(total agents, $N$)')
  ax.legend(frameon=False,ncol=2,loc='center left' if letter=='a' else 'lower left',bbox_to_anchor=(0,.67) if letter=='a' else None,columnspacing=.8,handlelength=1.5)
 ax=axs[1,0];panel(ax,'c','Clustered errors: wrong communities')
 lines(ax,many[many.initial=='misleading_contiguous'],geos=['ring','ws0.01','ws0.1'],xvalues=sizes,col='wrong_communities',band=True)
 ax.set_ylabel('Wrong-community fraction');ax.set_ylim(-.065,.56);ax.set_yticks([0,.25,.5]);ax.legend(frameon=False,loc='upper left')
 ax=axs[1,1];panel(ax,'d','Broadcast: decisions and rankings')
 d=many[(many.initial=='misleading_contiguous')&(many.geometry=='broadcast')]
 for col,color,marker,label in [('optimal_fraction',C['public'],'o','Public choices'),('private_correct',C['private'],'s','Private rankings')]:
  b=batches(d,col,['n']);a=b.groupby('n')[col].agg(['mean','min','max']).reindex(sizes);x=np.arange(3)
  ax.plot(x,a['mean'],marker=marker,color=color,label=label,ms=4)
  ci_band(ax,x,d,'n',col,color)
  for i,n in enumerate(sizes):
   z=b[b.n==n];ax.scatter(i+np.linspace(-.035,.035,len(z)),z[col],s=12,color=color,zorder=5)
 probability(ax,'Fraction favouring better arm');ax.set_xticks(range(3),sizes);ax.set_xlabel('Agents per community, $n$');ax.legend(frameon=False,loc='center right',bbox_to_anchor=(1,.28))
 save(fig,'main05_many_communities');record('main05_many_communities',many);record('main05_many_communities',many,'wrong_communities');record('main05_many_communities',d,'private_correct')

def graph_diagnostics():
 fig,axs=plt.subplots(2,3,figsize=(7.25,4.75))
 for row,(d,title) in enumerate([(primary[(primary.n==100)&(primary.initial=='misleading_random')],'100 communities'),(many[(many.n==50)&(many.initial=='misleading_contiguous')],'10,000 communities')]):
  geos=['ring','ws0.01','ws0.1','ws1'] if row==0 else ['ring','ws0.01','ws0.1']
  for j,(col,ylabel) in enumerate([('clustering','Mean local clustering'),('mean_path','Mean shortest-path length'),('degree_sd','Degree standard deviation')]):
   ax=axs[row,j];panel(ax,chr(97+3*row+j),title)
   for k,g in enumerate(geos):
    z=batches(d[d.geometry==g],col);ax.scatter(k+np.linspace(-.04,.04,len(z)),z[col],s=18,color=C[g],alpha=.75);ax.plot([k-.15,k+.15],[z[col].mean()]*2,color=C[g],lw=2)
   ax.set_xticks(range(len(geos)),['0','.01','.1','1'][:len(geos)]);ax.set_xlabel('WS rewiring probability, $p$');ax.set_ylabel(ylabel);ax.set_ylim(bottom=-.015 if col=='degree_sd' else 0)
 save(fig,'supp01_graph_diagnostics')

def controls():
 controls=df[df.group=='controls'];specs=[('neutral',0,.6,'Neutral'),('correct',0,.6,'Consistently correct'),('reversed',0,.6,'Reversed strength'),('misleading_random',1,.6,'Constant 0.5'),('misleading_random',2,.6,'Strong constant'),('misleading_random',1,1.2,r'Constant 0.5; $\lambda=1.2$')]
 fig,axs=plt.subplots(1,2,figsize=(7.25,3.55),sharey=True)
 for ax,ss,letter,title in [(axs[0],specs[:3],'a','Initial-score controls'),(axs[1],specs[3:],'b','Strength and constant-score controls')]:
  panel(ax,letter,title)
  for j,g in enumerate(['ring','ws0.1','regular','broadcast']):
   means=[]
   for i,(init,mode,lam,lab) in enumerate(ss):
    d=controls[(controls.initial==init)&(controls['mode']==mode)&(controls.lam==lam)&(controls.geometry==g)];b=batches(d);x=i+(j-1.5)*.115
    ax.scatter(x+np.linspace(-.02,.02,len(b)),b.optimal_fraction,s=9,color=C[g],alpha=.55);means.append(d.optimal_fraction.mean());ci_mark(ax,x,d,'optimal_fraction',C[g])
   ax.plot(np.arange(len(ss))+(j-1.5)*.115,means,marker=MARK[g],color=C[g],ms=4,label=L[g],lw=1)
  ax.set_xticks(range(3),[x[3] for x in ss],rotation=22,ha='right');probability(ax)
 axs[0].legend(frameon=False,ncol=2,loc='lower left');save(fig,'supp02_controls');record('supp02_controls',controls,keys=['geometry','initial','mode','lam'])

def expanded():
 d=df[df.group.isin(['primary','followup'])&(df.initial=='misleading_contiguous')&df.n.isin([50,100,200])&df.geometry.isin(['ring','ws0.01','ws0.1'])]
 fig,axs=plt.subplots(1,3,figsize=(7.25,2.95),sharey=True)
 for ax,n,letter in zip(axs,[50,100,200],'abc'):
  panel(ax,letter,rf'$n={n}$; $M={10000//n}$');z=d[d.n==n]
  for j,g in enumerate(['ring','ws0.01','ws0.1']):
   b=batches(z[z.geometry==g]);offset=np.linspace(-.13,.13,len(b));ax.scatter(j+offset,b.optimal_fraction,s=14,color=C[g],alpha=.6)
   ax.plot([j-.23,j+.23],[b.optimal_fraction.mean()]*2,color=C[g],lw=2.0);ci_mark(ax,j,z[z.geometry==g],'optimal_fraction',C[g])
  ax.set_xticks(range(3),['Ring','WS .01','WS .1']);probability(ax);ax.set_ylim(.45,1.04);ax.set_yticks([.5,.75,1]);ax.set_xlabel('Community graph')
 save(fig,'supp03_expanded_geometry');record('supp03_expanded_geometry',d)

def choice_degree():
 fig,axs=plt.subplots(2,2,figsize=(7.25,5.0),sharey=True)
 geos=['ring','ws0.01','ws0.1','ws1']
 for ax,n,letter in zip(axs[0],[50,100],'ab'):
  panel(ax,letter,rf'Choice exposure only: $n={n}$')
  for j,(group,lab,color,off) in enumerate([('primary','Choice and reward exposure',C['private'],-.08),('choiceonly','Choice exposure only',C['public'],.08)]):
   z=df[(df.group==group)&(df.n==n)&(df.initial=='misleading_contiguous')&df.geometry.isin(geos[1:]+['regular'])];xs=['ws0.01','ws0.1','ws1','regular'];means=[]
   for k,g in enumerate(xs):
    b=batches(z[z.geometry==g]);means.append(b.optimal_fraction.mean());ax.scatter(k+off+np.linspace(-.025,.025,len(b)),b.optimal_fraction,s=10,color=color,alpha=.55);ci_mark(ax,k+off,z[z.geometry==g],'optimal_fraction',color)
   ax.plot(np.arange(4)+off,means,'o-',ms=3.8,color=color,label=lab,lw=1.2)
  ax.set_xticks(range(4),['WS .01','WS .1','WS 1','Regular']);probability(ax);ax.set_xlabel('Choice-exposure geometry')
  if n==50:ax.legend(frameon=False,loc='lower left',fontsize=6.8)
 for ax,n,letter in zip(axs[1],[20,100],'cd'):
  panel(ax,letter,rf'Initial ring degree: $n={n}$')
  for group,lab,col,off in [('primary','$k=4$',C['private'],-.06),('degree','$k=8$',C['ws0.1'],.06)]:
   z=df[(df.group==group)&(df.n==n)&(df.initial=='misleading_contiguous')];means=[]
   for j,g in enumerate(geos):
    b=batches(z[z.geometry==g]);means.append(b.optimal_fraction.mean());ax.scatter(j+off+np.linspace(-.025,.025,len(b)),b.optimal_fraction,s=10,color=col,alpha=.6);ci_mark(ax,j+off,z[z.geometry==g],'optimal_fraction',col)
   ax.plot(np.arange(4)+off,means,'o-',color=col,ms=3.8,label=lab,lw=1.2)
  ax.set_xticks(range(4),['Ring','WS .01','WS .1','WS 1']);probability(ax);ax.set_xlabel('Community graph');ax.legend(frameon=False,loc='lower left')
 save(fig,'supp04_choice_degree');record('supp04_choice_degree',df[df.group.isin(['choiceonly','degree'])],keys=['group','geometry','n'])

def scaling():
 d=df[df.group=='scaling'];fig,axs=plt.subplots(2,2,figsize=(7.25,4.8))
 for row,(sel,title) in enumerate([(d.M==20,'Fixed number of communities, $M=20$'),(d.n==20,'Fixed community size, $n=20$')]):
  z=d[sel]
  for j,(col,ylabel) in enumerate([('optimal_fraction','Better-arm choice fraction'),('private_correct',r'Fraction with $Q_i(0)>Q_i(1)$')]):
   ax=axs[row,j];panel(ax,chr(97+2*row+j),title)
   for geo in ['ring','ws0.1']:
    b=batches(z[z.geometry==geo],col,['N']);a=b.groupby('N')[col].agg(['mean','min','max']);x=np.arange(3)
    ax.plot(x,a['mean'],marker=MARK[geo],color=C[geo],label=L[geo],ms=4);ci_band(ax,x,z[z.geometry==geo],'N',col,C[geo])
    for k,N in enumerate(a.index):
     zz=b[b.N==N];ax.scatter(k+np.linspace(-.025,.025,len(zz)),zz[col],s=9,color=C[geo],alpha=.65)
   probability(ax,ylabel);ax.set_xticks(range(3),['1,000','10,000','40,000']);ax.set_xlabel('Total agents, $N$');ax.legend(frameon=False,loc='lower left' if j==0 else 'lower right')
 save(fig,'supp05_scaling');record('supp05_scaling',d,keys=['N','n','M','geometry'])

def two_initialisations():
 d=df[df.group=='two'];fig,axs=plt.subplots(1,2,figsize=(7.25,3.05));initlabels={'misleading_random':'Asymmetrically misleading','neutral':'Neutral','correct':'Consistently correct','reversed':'Reversed strength'}
 for ax,col,letter,title in [(axs[0],'optimal_fraction','a','Public decisions'),(axs[1],'private_correct','b','Private rankings')]:
  panel(ax,letter,title)
  for j,(init,lab) in enumerate(initlabels.items()):
   b=batches(d[d.initial==init],col,['gamma']);a=b.groupby('gamma')[col].agg(['mean','min','max']);x=np.arange(3)+(j-1.5)*.055;color=[C['public'],C['regular'],C['ring'],C['ws0.1']][j]
   ax.plot(x,a['mean'],marker=['o','s','^','D'][j],color=color,label=lab,ms=4,lw=1.1)
   for k,g in enumerate(a.index):
    ci_mark(ax,x[k],d[(d.initial==init)&(d.gamma==g)],col,color)
    zz=b[b.gamma==g];ax.scatter(x[k]+np.linspace(-.015,.015,len(zz)),zz[col],s=9,color=color,alpha=.55)
  probability(ax,'Fraction favouring better arm');ax.set_xticks(range(3),['0','.15','.5']);ax.set_xlabel('Between-community\nweight, $\\Gamma$')
 axs[0].legend(frameon=False,loc='center right');save(fig,'supp06_two_initialisations');record('supp06_two_initialisations',d,keys=['gamma','initial'])

def many_detail():
 # Distribution over communities is a descriptive within-population outcome.
 fig,axs=plt.subplots(2,3,figsize=(7.25,4.9));geos=['ring','ws0.01','ws0.1']
 for j,g in enumerate(geos):
  for row,n in enumerate([20,50]):
   ax=axs[row,j];panel(ax,chr(97+row*3+j),rf'{L[g]}; $n={n}$')
   configs=[c for c in meta.values() if c['group']=='M10000' and c['n']==n and c['geometry']==g and c['initial']=='misleading_contiguous'];bins=np.linspace(0,1,26);hist=[]
   for c in sorted(configs,key=lambda c:c['graph_id']):
    with np.load(DATA/(c['label']+'.npz')) as z:
     hist.append(np.histogram(z['tail'].ravel(),bins=bins)[0]/z['tail'].size)
   h=np.asarray(hist);centres=(bins[:-1]+bins[1:])/2
   ax.bar(centres,h.mean(0),width=.037,color=C[g],alpha=.7,edgecolor='none')
   ax.axvline(.5,color='#9CA7B0',ls='--',lw=.65);ax.set_xlabel('Community choice fraction\n(better arm)');ax.set_ylabel('Fraction of communities');ax.set_ylim(0,1.02);ax.set_xlim(-.02,1.02);ax.set_xticks([0,.5,1])
 save(fig,'supp07_many_detail')

def score_response():
 fig,axs=plt.subplots(1,3,figsize=(7.25,3.45));p=np.linspace(0,1,1001);d=np.maximum(p,1-p)
 ax=axs[0];panel(ax,'a','Decisiveness scores')
 ax.plot(p,d,color=C['ring'],label='Primary score');ax.plot(p,np.full_like(p,.5),color=C['regular'],ls='--',label='Constant 0.5');ax.plot(p,.5+.5*(2*p-1)**2,color=C['ws0.1'],ls='-.',label='Quadratic (floor 0.5)');ax.plot(p,np.abs(2*p-1),color=C['public'],ls=':',label='Zero floor')
 ax.set_xlabel('Better-arm choice probability, $p$');ax.set_ylabel('Score, $s(p)$');ax.set_ylim(-.03,1.03);ax.set_xticks([0,.5,1]);ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.26),fontsize=6.8)
 ax=axs[1];panel(ax,'b','Frozen cue response')
 x=np.linspace(-.6,.6,1001);pp=1/(1+np.exp(-12*x))
 for s,col,lab in [(np.maximum(pp,1-pp),C['ring'],'Primary score'),(np.full_like(pp,.5),C['regular'],'Constant 0.5'),(np.abs(2*pp-1),C['public'],'Zero floor')]:
  ax.plot(x,s*(2*pp-1),color=col,label=lab)
 ax.axhline(0,color='#ADB6BC',lw=.6);ax.axvline(0,color='#ADB6BC',lw=.6);ax.set_xlabel('Score gap plus social input, $x$');ax.set_ylabel(r'$s(\sigma(12x))[2\sigma(12x)-1]$');ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.26),fontsize=6.8)
 ax=axs[2];panel(ax,'c','Private update gains')
 dd=np.linspace(0,1,200);ax.plot(dd,.4-.35*dd,color=C['ws0.1'],label=r'$R=1$: $\alpha_+$');ax.plot(dd,.05+.35*dd,color=C['public'],label=r'$R=0$: $\alpha_-$');ax.axvspan(.5,1,color=C['ring'],alpha=.055);ax.set_xlabel('Decisiveness, $d$');ax.set_ylabel('Learning rate');ax.set_ylim(0,.43);ax.set_xticks([0,.5,1]);ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.26),fontsize=6.8)
 save(fig,'supp08_score_response')

def noise_panels():
 """Exact conditional calculations at saved states; no new simulation."""
 z=np.load(STATS/'conditional_noise_covariances.npz')
 fig,axs=plt.subplots(1,3,figsize=(WIDTH,2.9))
 cmap=LinearSegmentedColormap.from_list('correlation',['#FFFFFF','#C2DFED','#0072B2'])
 for ax,kind,letter,title in zip(axs[:2],['ring','broadcast'],'bc',['Ring routing','Common broadcast']):
  cov=z[kind+'_covariance'];sd=np.sqrt(np.diagonal(cov,axis1=1,axis2=2));corr=cov/(sd[:,:,None]*sd[:,None,:]);mean=corr.mean(0)
  # Only the dense matrix is raster artwork. Labels, axes and all other panels
  # stay editable vector elements; 600-dpi export avoids PDF cell stitching.
  im=ax.imshow(mean,extent=(.5,100.5,100.5,.5),interpolation='nearest',vmin=0,vmax=1,cmap=cmap,rasterized=True)
  ax.set_xlim(.5,100.5);ax.set_ylim(100.5,.5);ax.set_xticks([1,50,100]);ax.set_yticks([1,50,100]);ax.set_xlabel('Receiving community');ax.set_ylabel('Receiving community');ax.set_box_aspect(1);panel(ax,letter,title)
 cb=fig.colorbar(im,ax=list(axs[:2]),fraction=.028,pad=.025,shrink=.75);cb.set_label('Conditional correlation');cb.set_ticks([0,.5,1])
 ax=axs[2];panel(ax,'d','Field noise magnitude')
 pop=pd.read_csv(STATS/'conditional_noise_population.csv');batch=pd.read_csv(STATS/'conditional_noise_batch.csv');ci=pd.read_csv(STATS/'conditional_noise_ensemble_summary.csv')
 for _,row in pop.iterrows():ax.plot([0,1],1000*np.array([row.ring_field_rms_sd,row.broadcast_field_sd]),color='#AEB7BE',alpha=.5,lw=.6,zorder=1)
 for x,col,color in [(0,'ring_field_rms_sd',C['ring']),(1,'broadcast_field_sd',C['broadcast'])]:
  a=ci[ci.metric==col].iloc[0];ax.scatter(x+np.linspace(-.055,.055,4),1000*batch[col],s=14,color=color,zorder=4)
  ax.errorbar([x],[1000*a['mean']],yerr=[[1000*(a['mean']-a.ci95_low)],[1000*(a.ci95_high-a['mean'])]],color=color,fmt='s',ms=4,capsize=3,lw=.9,zorder=5)
 ax.set_xlim(-.3,1.3);ax.set_ylim(0,11);ax.set_yticks([0,5,10]);ax.set_xticks([0,1],['Ring','Broadcast']);ax.set_ylabel(r'Field RMS s.d. ($\times10^{-3}$)');ax.set_box_aspect(1)
 save(fig,'supp10_noise_panels')

def legacy_aliases():
 mapping={'01_concept':'main01_concept','02_size_geometry':'main03_size_geometry','03_graph_diagnostics':'supp01_graph_diagnostics','04_private_public_paths':'main04_private_withdrawal','05_controls':'supp02_controls','06_geometry_lines':'main03_size_geometry','07_two_communities':'main02_two_communities','08_expanded_geometry':'supp03_expanded_geometry','09_withdrawal':'main04_private_withdrawal','10_many_communities':'main05_many_communities'}
 for old,new in mapping.items():
  for ext in ['pdf','png']:
   if (HERE/(new+'.'+ext)).exists():shutil.copyfile(HERE/(new+'.'+ext),HERE/(old+'.'+ext))
 (HERE/'legacy_figure_mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')

if __name__=='__main__':
 main_two();main_size();main_private();main_many();graph_diagnostics();controls();expanded();choice_degree();scaling();two_initialisations();many_detail();score_response();noise_panels();legacy_aliases()
 pd.DataFrame(ledger).to_csv(HERE/'Figure_Numerical_Ledger.csv',index=False)
 (HERE/'Text_Collision_Audit.json').write_text(json.dumps(collision_audit,indent=2)+'\n')
 print(json.dumps({'main_plot_components':4,'supplementary_plot_components':9,'published_figures_after_TikZ_assembly':16,'ledger_rows':len(ledger),'data_path':str(DATA),'uncertainty':'Approximate95% t Monte Carlo intervals across seed-batch means; never agent-level intervals'},indent=2))
