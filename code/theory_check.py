import numpy as np, json
import mpopt
from mpopt import *
mpopt.mp_nodes.__defaults__=(1200,)
N=128; s2=N/100.; R=json.load(open("results2.json")); out={}
for K in (32,64):
    beta=K/N; x,wq,a,b=mp_nodes(beta); g=N*x; w=(g**2+s2*g)/(g+s2)**2; J1=float(np.sum(wq*w))
    for T in (2,4,6,8):
        st,mp=mp_opt_steps(beta,N,s2,T); ch=poly_obj(cheb_steps(beta,N,s2,T),beta,N,s2); co=poly_obj(const_steps(beta,N,s2,T),beta,N,s2)
        e=R["E1"][str(K)][str(T)]
        out[f"{K}_{T}"]=dict(th=[mp/J1,ch/J1,co/J1],emp=[e["MP-opt"]["dmmse"],e["Chebyshev"]["dmmse"],e["Richardson"]["dmmse"]])
        print(K,T,"theory mp/cheb/const %.3e %.3e %.3e | empirical %.3e %.3e %.3e"%tuple(out[f"{K}_{T}"]["th"]+out[f"{K}_{T}"]["emp"]),flush=True)
json.dump(out,open("theory_check.json","w"))
