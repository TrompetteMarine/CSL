"""Direct dense-equation reference, independent of the archived simulation code.

Uses unequal community sizes, explicit per-agent weights and externally supplied
uniform draws. Compiles the archived engine without changing the baseline.
"""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'baseline/nhb_smallworld_20261012/src/sim.cpp'
EXE=OUT/'sim_audit'
subprocess.run(['g++','-O2','-std=c++17',str(SOURCE),'-o',str(EXE)],check=True)
sizes=np.array([2,3,4,5,3,2]);M=len(sizes);N=sum(sizes);groups=np.repeat(np.arange(M),sizes)
T,R,window=35,3,10
rng=np.random.default_rng(16102026)
U=rng.random((T,R,N));V=rng.random((T,R,N))
np.concatenate([U.ravel(),V.ravel()]).tofile(OUT/'audit_tape.bin')
qinit=np.array([[.52,.5],[.42,.62],[.5,.5],[.9,.1],[0.,1.],[.3,.7]])
B=np.eye(M)*.75
for c in range(M):B[c,(c-1)%M]+=.125;B[c,(c+1)%M]+=.125
different=np.eye(M)*.2+np.ones((M,M))*.8/M
W=B[groups[:,None],groups[None,:]]/sizes[groups][None,:]
np.fill_diagonal(W,0.)
checks=[]
for mode in [0,1,2]:
  for theta in [0.,.35,1.]:
    for separate in [False,True]:
      choice=different if separate else B
      C=(1-theta)*choice+theta*np.broadcast_to(sizes/N,(M,M))
      text=[' '.join(map(str,[M,T,R,mode,window,123,.6,theta,12.,.15,.05,.4,.001,.55,.45])),
            ' '.join(map(str,sizes)),' '.join(map(str,qinit.ravel()))]
      for matrix in [B]+([choice] if separate else []):
        if matrix is choice and separate:text.append(str(M))
        for c in range(M):
          entries=[(d,matrix[c,d]) for d in range(M) if matrix[c,d]]
          text.append(' '.join(map(str,[len(entries),*[v for e in entries for v in e]])))
      (OUT/'audit_input.txt').write_text('\n'.join(text)+'\n')
      result=subprocess.run([str(EXE),str(OUT/'audit_input.txt'),str(OUT/'audit_output.bin'),str(OUT/'audit_tape.bin')],check=True,capture_output=True,text=True)
      engine=np.fromfile(OUT/'audit_output.bin',dtype=np.float64);offset=0
      def take(shape):
        global offset
        n=int(np.prod(shape));a=engine[offset:offset+n].reshape(shape);offset+=n;return a
      observed=[take((T,R,6)),take((R,M)),take((R,N,2)),take((R,M))]
      paths=np.zeros((T,R,6));tails=np.zeros((R,M));finalq=np.zeros((R,N,2));finalpsi=np.zeros((R,M))
      for rep in range(R):
        q=qinit[groups].copy();psi=np.zeros(M)
        for t in range(T):
          delta=q[:,0]-q[:,1]
          p=1/(1+np.exp(-12*(delta+.6*(C@psi)[groups])))
          d=np.maximum(p,1-p);priv=d if mode==0 else np.full(N,.5);peer=priv
          emitted=d if mode==0 else np.full(N,.5 if mode==1 else 1.)
          action=(U[t,rep]>=p).astype(int);reward=(V[t,rep]<np.where(action==0,.55,.45)).astype(float)
          qchosen=q[np.arange(N),action]
          alpha=np.where(reward-qchosen<0,.05+.35*priv,.4-.35*priv)
          new=q.copy()
          for a in [0,1]:
            selected=(action==a).astype(float)
            mass=W@selected;rewardmass=W@(selected*reward);scoremass=W@(selected*peer)
            new[:,a]+=selected*alpha*(reward-q[:,a])+.15*scoremass/(mass+.001)*(rewardmass-mass*q[:,a])
          m=(action==0).mean()
          paths[t,rep]=[m,(delta>0).mean(),delta.mean(),(1/(1+np.exp(-12*delta))).mean(),p.mean(),(.55-.45)*(1-m)]
          psi=np.array([np.mean(emitted[groups==c]*(1-2*action[groups==c])) for c in range(M)])
          if t>=T-window:
            tails[rep]+=np.array([(action[groups==c]==0).mean() for c in range(M)])/window
          q=new
        finalq[rep]=q;finalpsi[rep]=psi
      expected=[paths,tails,finalq,finalpsi]
      errors={name:float(np.max(abs(a-b))) for name,a,b in zip(['path','tail','Q','psi'],observed,expected)}
      assert max(errors.values())<5e-14
      assert json.loads(result.stdout)['clips']==0
      checks.append(dict(mode=mode,theta=theta,separate_choice_weights=separate,max_abs_errors=errors))
out=dict(comparisons=len(checks),unequal_sizes=sizes.tolist(),horizon=T,replicates=R,
    maximum_absolute_error=max(max(c['max_abs_errors'].values()) for c in checks),
    all_passed=True,checks=checks,engine_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
(OUT/'engine_checks.json').write_text(json.dumps(out,indent=2))
for name in ['audit_tape.bin','audit_input.txt','audit_output.bin']:(OUT/name).unlink()
print(json.dumps({k:v for k,v in out.items() if k!='checks'},indent=2))
