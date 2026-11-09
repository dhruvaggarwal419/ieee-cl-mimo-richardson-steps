import json,time,numpy as np,torch
import sim
from sim import *
def gen_c(B,N,K,s2,g=None,rho=0.0):
    x,A,b,ir,iq=None,None,None,None,None
    H=(torch.randn(B,N,K,dtype=RD,device=dev,generator=g)+1j*torch.randn(B,N,K,dtype=RD,device=dev,generator=g))/np.sqrt(2)
    if rho>0:
        idx=torch.arange(N,device=dev); Rr=rho**((idx[:,None]-idx[None,:]).abs().double())
        L=torch.linalg.cholesky(Rr).to(CD); H=torch.einsum('nm,bmk->bnk',L,H)
    ir=torch.randint(0,4,(B,K),device=dev,generator=g); iq=torch.randint(0,4,(B,K),device=dev,generator=g)
    x=(LV[ir]+1j*LV[iq]).to(CD)
    n=(torch.randn(B,N,dtype=RD,device=dev,generator=g)+1j*torch.randn(B,N,dtype=RD,device=dev,generator=g))*np.sqrt(s2/2)
    y=torch.einsum('bnk,bk->bn',H,x)+n
    A=torch.einsum('bnk,bnl->bkl',H.conj(),H)+s2*torch.eye(K,dtype=CD,device=dev)
    b=torch.einsum('bnk,bn->bk',H.conj(),y)
    return x,A,b,ir,iq
def learn_ms(N,K,s2,T,inits,iters=3000,B=256,rho=0.0):
    best=None
    gv=torch.Generator(device=dev); gv.manual_seed(999); xv,Av,bv,_,_=gen_c(2000,N,K,s2,gv,rho)
    for si,init in enumerate(inits):
        g=torch.Generator(device=dev); g.manual_seed(si)
        th=torch.tensor(np.log(init),dtype=RD,device=dev,requires_grad=True)
        opt=torch.optim.Adam([th],lr=0.03); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,iters)
        for it in range(iters):
            x,A,b,_,_=gen_c(B,N,K,s2,g,rho); loss=((rich(A,b,torch.exp(th))-x).abs()**2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        with torch.no_grad(): v=((rich(Av,bv,torch.exp(th))-xv).abs()**2).mean().item()
        if best is None or v<best[0]: best=(v,torch.exp(th).detach().cpu().numpy())
    return best[1]
def inits_for(N,K,s2,T):
    beta=K/N; lo,hi=edges(beta,N,s2); rng=np.random.default_rng(0)
    return [const_steps(beta,N,s2,T),cheb_steps(beta,N,s2,T),np.exp(rng.uniform(np.log(1/hi),np.log(1/lo),T))]
def evalc(N,K,s2,sd,T,rho=0.0,B=3000,seed=123):
    g=torch.Generator(device=dev); g.manual_seed(seed)
    x,A,b,ir,iq=gen_c(B,N,K,s2,g,rho); xm=mmse(A,b); out={}
    def rec(n,xh): out[n]={"dmmse":(((xh-xm).abs()**2).sum(1)/(xm.abs()**2).sum(1)).mean().item(),"ber":ber(xh,ir,iq),
        "dsd":(((xh-xm).abs()**2).sum(1)/(xm.abs()**2).sum(1)).std().item()/np.sqrt(B)}
    rec("MMSE",xm)
    for k,v in sd.items(): rec(k,rich(A,b,[float(a) for a in v]))
    rec("CG",cg(A,b,T)); return out
if __name__=="__main__":
    t0=time.time(); R={}; N=128; snr=20; s2=N/10**(snr/10)
    R["E1"]={}
    for K in (32,64,96):
        R["E1"][K]={}
        for T in range(1,11):
            beta=K/N; d={"Richardson":const_steps(beta,N,s2,T),"Chebyshev":cheb_steps(beta,N,s2,T)}
            d["MP-opt"],_=mp_opt_steps(beta,N,s2,T); d["Learned"]=learn_ms(N,K,s2,T,inits_for(N,K,s2,T))
            r=evalc(N,K,s2,d,T); r["theory"]={k:poly_obj(v,beta,N,s2) for k,v in d.items()}
            r["steps"]={k:sorted(map(float,v)) for k,v in d.items()}; R["E1"][K][T]=r
        print("E1",K,time.time()-t0,flush=True); json.dump(R,open("results2.json","w"))
    R["E2"]={}
    for K in (64,96):
        R["E2"][K]={}
        for snr_ in (10,14,18,22,26):
            s2_=N/10**(snr_/10); beta=K/N; T=4
            d={"Richardson":const_steps(beta,N,s2_,T),"Chebyshev":cheb_steps(beta,N,s2_,T)}
            d["MP-opt"],_=mp_opt_steps(beta,N,s2_,T); d["Learned"]=learn_ms(N,K,s2_,T,inits_for(N,K,s2_,T))
            R["E2"][K][snr_]=evalc(N,K,s2_,d,T,B=6000)
        print("E2",K,time.time()-t0,flush=True); json.dump(R,open("results2.json","w"))
    R["E3"]={}
    for N_ in (16,32,64,128,256):
        K_=N_//2; s2_=N_/100.; T=4; beta=.5
        d={"Chebyshev":cheb_steps(beta,N_,s2_,T)}; d["MP-opt"],_=mp_opt_steps(beta,N_,s2_,T)
        d["Learned"]=learn_ms(N_,K_,s2_,T,inits_for(N_,K_,s2_,T)); R["E3"][N_]=evalc(N_,K_,s2_,d,T,B=4000)
    print("E3",time.time()-t0,flush=True)
    R["E5"]={}
    for rho in (0.0,0.3,0.6,0.8):
        K=64; beta=.5; T=4
        d={"Chebyshev":cheb_steps(beta,N,s2,T)}; d["MP-opt"],_=mp_opt_steps(beta,N,s2,T)
        d["Learned"]=learn_ms(N,K,s2,T,inits_for(N,K,s2,T),rho=rho); R["E5"][rho]=evalc(N,K,s2,d,T,rho=rho)
    print("E5",time.time()-t0,flush=True)
    # theory: objective ratio vs T (large-system)
    R["TH"]={}
    for K in (32,64,96):
        beta=K/N; R["TH"][K]={}
        for T in range(1,16):
            R["TH"][K][T]={"cheb":poly_obj(cheb_steps(beta,N,s2,T),beta,N,s2),"mp":mp_opt_steps(beta,N,s2,T)[1],"const":poly_obj(const_steps(beta,N,s2,T),beta,N,s2)}
    json.dump(R,open("results2.json","w")); print("done",time.time()-t0)
