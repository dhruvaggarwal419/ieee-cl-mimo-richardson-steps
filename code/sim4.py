import json,time,os,numpy as np,torch
torch.set_num_threads(12)
import sim2
from sim2 import *
sim2.learn_ms.__defaults__=(500,128,0.0)
_oi=sim2.inits_for
sim2.inits_for=lambda *a: _oi(*a)[:2]
import sim2 as S
R=json.load(open("results2.json")); t0=time.time(); N=128; snr=20; s2=N/10**(snr/10)
def save(): json.dump(R,open("results4.json","w"))
def one(N,K,s2,T,rho=0.0,B=3000):
    beta=K/N; d={"Richardson":const_steps(beta,N,s2,T),"Chebyshev":cheb_steps(beta,N,s2,T)}
    d["MP-opt"],_=mp_opt_steps(beta,N,s2,T); d["Learned"]=learn_ms(N,K,s2,T,inits_for(N,K,s2,T),rho=rho)
    r=evalc(N,K,s2,d,T,rho=rho,B=B); r["theory"]={k:poly_obj(v,beta,N,s2) for k,v in d.items()}
    r["steps"]={k:sorted(map(float,v)) for k,v in d.items()}; return r
R["E2"]={}
for K in (64,96):
    R["E2"][K]={}
    for s in (10,14,18,22,26):
        R["E2"][K][s]=one(N,K,N/10**(s/10),4,B=6000)
    save(); print("E2",K,time.time()-t0,flush=True)
R["E3"]={}
for n in (16,32,64,128,256):
    R["E3"][n]=one(n,n//2,n/100.,4,B=4000)
save(); print("E3",time.time()-t0,flush=True)
R["E5"]={}
for rho in (0.0,0.3,0.6,0.8): R["E5"][rho]=one(N,64,s2,4,rho=rho)
save(); print("E5",time.time()-t0,flush=True)
R["TH"]={}
for K in (32,64,96):
    beta=K/N; R["TH"][K]={T:{"cheb":poly_obj(cheb_steps(beta,N,s2,T),beta,N,s2),"mp":mp_opt_steps(beta,N,s2,T)[1],"const":poly_obj(const_steps(beta,N,s2,T),beta,N,s2)} for T in range(1,16)}
save(); print("done",time.time()-t0)
