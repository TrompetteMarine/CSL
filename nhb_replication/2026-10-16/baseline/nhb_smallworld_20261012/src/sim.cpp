#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
using namespace std;
int main(int argc,char**argv){
 if(argc<3) return 2;
 ifstream in(argv[1]); string out=argv[2], tape=argc>3?argv[3]:"-", state=argc>4?argv[4]:"-";
 int M,T,reps,mode,window; uint64_t seed; double lam,theta,h,b,amin,amax,eps,mu0,mu1;
 in>>M>>T>>reps>>mode>>window>>seed>>lam>>theta>>h>>b>>amin>>amax>>eps>>mu0>>mu1;
 vector<int> sizes(M),start(M+1),g; vector<double> init(2*M), diag(M); vector<vector<pair<int,double>>> B(M);
 for(int c=0;c<M;c++){in>>sizes[c];start[c+1]=start[c]+sizes[c];}
 int N=start[M]; for(int c=0;c<M;c++)for(int i=0;i<sizes[c];i++)g.push_back(c);
 for(auto&v:init)in>>v;
 for(int c=0;c<M;c++){int k;in>>k;for(int j=0;j<k;j++){int d;double v;in>>d>>v;B[c].push_back({d,v});if(c==d)diag[c]=v;}}
 if(!in || N<1) return 3;
 vector<vector<pair<int,double>>> Choice=B;int rows=0;
 if(in>>rows){if(rows!=M)return 6;Choice.assign(M,{});for(int c=0;c<M;c++){int k;in>>k;for(int j=0;j<k;j++){int d;double v;in>>d>>v;Choice[c].push_back({d,v});}}if(!in)return 6;}
 ifstream tap(tape,ios::binary),st(state,ios::binary);
 vector<double> U,V;
 if(tape!="-"){U.resize((size_t)T*reps*N);V.resize(U.size());tap.read((char*)U.data(),U.size()*8);tap.read((char*)V.data(),V.size()*8);if(!tap)return 4;}
 vector<double> q0(N),q1(N),p(N),d(N),ds(N),dp(N),dr(N),R(N),alpha(N);
 vector<int>A(N); vector<double> psi(M),field(M),agg(6*M),pool(6*M),path((size_t)T*reps*6),tail((size_t)reps*M),fq((size_t)reps*N*2),fp((size_t)reps*M);
 long long clips=0;double gainmax=0;
 for(int rep=0;rep<reps;rep++){
  mt19937_64 cr(seed+0x9e3779b97f4a7c15ULL*(rep+1)),rr(seed+0xd1b54a32d192ed03ULL*(rep+1));
  fill(psi.begin(),psi.end(),0.);
  for(int i=0;i<N;i++){q0[i]=init[2*g[i]];q1[i]=init[2*g[i]+1];}
  if(state!="-") {vector<double>q(2*N);st.read((char*)q.data(),2*N*8);st.read((char*)psi.data(),M*8);if(!st)return 5;for(int i=0;i<N;i++){q0[i]=q[2*i];q1[i]=q[2*i+1];}}
  for(int t=0;t<T;t++){
   double macro=0;for(int c=0;c<M;c++)macro+=sizes[c]*psi[c]/N;
   for(int c=0;c<M;c++){double x=0;for(auto e:Choice[c])x+=e.second*psi[e.first];field[c]=(1-theta)*x+theta*macro;}
   fill(agg.begin(),agg.end(),0.);
   double m=0,privatepre=0,gap=0,probremoved=0,meanp=0;
   for(int i=0;i<N;i++){
    double delta=q0[i]-q1[i];p[i]=1/(1+exp(-h*(delta+lam*field[g[i]])));
    d[i]=max(p[i],1-p[i]); dp[i]=mode==0?d[i]:.5;dr[i]=mode==0?d[i]:.5;ds[i]=mode==0?d[i]:(mode==1?.5:1.);
    double u=tape=="-"?double(cr()>>11)*0x1.0p-53:U[((size_t)t*reps+rep)*N+i];
    double v=tape=="-"?double(rr()>>11)*0x1.0p-53:V[((size_t)t*reps+rep)*N+i];
    A[i]=u>=p[i];R[i]=v<(A[i]?mu1:mu0);double q=A[i]?q1[i]:q0[i];double err=R[i]-q;
    alpha[i]=err<0?amin+(amax-amin)*dp[i]:amax-(amax-amin)*dp[i];
    int c=g[i],a=A[i];agg[6*c+3*a]+=1.;agg[6*c+3*a+1]+=R[i];agg[6*c+3*a+2]+=dr[i];
    m+=1-a;privatepre+=delta>0;gap+=delta;probremoved+=1/(1+exp(-h*delta));meanp+=p[i];
   }
   for(int c=0;c<M;c++)for(int j=0;j<6;j++)agg[6*c+j]/=sizes[c];
   fill(pool.begin(),pool.end(),0.);
   for(int c=0;c<M;c++)for(auto e:B[c])for(int j=0;j<6;j++)pool[6*c+j]+=e.second*agg[6*e.first+j];
   fill(psi.begin(),psi.end(),0.);
   for(int i=0;i<N;i++){
    int c=g[i];double self=diag[c]/sizes[c];double old[2]={q0[i],q1[i]},next[2];
    for(int a=0;a<2;a++){
     double ia=A[i]==a,w=max(0.,pool[6*c+3*a]-self*ia),r=pool[6*c+3*a+1]-self*ia*R[i],cm=max(0.,pool[6*c+3*a+2]-self*ia*dr[i]);
     double gain=b*cm/(w+eps),inc=ia*alpha[i]*(R[i]-old[a])+gain*(r-w*old[a]);
     gainmax=max(gainmax,ia*alpha[i]+gain*w);next[a]=old[a]+inc;
     if(next[a]<-1e-12||next[a]>1+1e-12)clips++;
     next[a]=clamp(next[a],0.,1.);
    }
    q0[i]=next[0];q1[i]=next[1];psi[c]+=(1-2*A[i])*ds[i]/sizes[c];
   }
   size_t idx=((size_t)t*reps+rep)*6;path[idx]=m/N;path[idx+1]=privatepre/N;path[idx+2]=gap/N;path[idx+3]=probremoved/N;path[idx+4]=meanp/N;path[idx+5]=(mu0-mu1)*(1-m/N);
   if(t>=T-window)for(int c=0;c<M;c++)tail[rep*M+c]+=agg[6*c]/window;
  }
  for(int i=0;i<N;i++){fq[((size_t)rep*N+i)*2]=q0[i];fq[((size_t)rep*N+i)*2+1]=q1[i];}
  for(int c=0;c<M;c++)fp[rep*M+c]=psi[c];
 }
 ofstream raw(out,ios::binary);raw.write((char*)path.data(),path.size()*8);raw.write((char*)tail.data(),tail.size()*8);raw.write((char*)fq.data(),fq.size()*8);raw.write((char*)fp.data(),fp.size()*8);
 cout<<"{\"clips\":"<<clips<<",\"gain_max\":"<<gainmax<<",\"N\":"<<N<<",\"M\":"<<M<<"}"<<endl;
 return !raw;
}
