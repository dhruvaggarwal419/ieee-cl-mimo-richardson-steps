import json,time,numpy as np,torch
torch.set_num_threads(8)
import sim2
from sim2 import *
sim2.learn_ms.__defaults__=(500,128,0.0)
_oi=sim2.inits_for
sim2.inits_for=lambda *a: _oi(*a)[:2]
t0=time.time(); R={}
def save(): json.dump(R,open("results5.json","w"))
def one(N,K,s2,T,rho=0.0,B=3000):
    beta=K/N; d={"Richardson":const_steps(beta,N,s2,T),"Chebyshev":cheb_steps(beta,N,s2,T)}
    d["MP-opt"],_=mp_opt_steps(beta,N,s2,T); d["Learned"]=learn_ms(N,K,s2,T,inits_for(N,K,s2,T),rho=rho)
    r=evalc(N,K,s2,d,T,rho=rho,B=B); r["theory"]={k:poly_obj(v,beta,N,s2) for k,v in d.items()}
    return r
N=128; s2=N/100.
R["K96"]=one(N,96,s2,4); save(); print("K96",time.time()-t0,flush=True)
R["E5"]={}
for rho in (0.3,0.6,0.8): R["E5"][rho]=one(N,64,s2,4,rho=rho); save(); print("E5",rho,time.time()-t0,flush=True)
R["E3"]={}
for n in (16,32,64): R["E3"][n]=one(n,n//2,n/100.,4,B=3000); save(); print("E3",n,time.time()-t0,flush=True)
print("done",flush=True)
