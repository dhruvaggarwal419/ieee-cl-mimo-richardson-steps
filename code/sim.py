import json, sys, time, numpy as np, torch
from mpopt import *
dev = "cuda" if torch.cuda.is_available() else "cpu"
CD, RD = torch.complex128, torch.float64
LV = torch.tensor([-3., -1., 1., 3.], dtype=RD, device=dev)/np.sqrt(10)
GRAY = torch.tensor([[0,0],[0,1],[1,1],[1,0]], device=dev)

def gen(B, N, K, s2, g=None):
    H = (torch.randn(B,N,K,dtype=RD,device=dev,generator=g)+1j*torch.randn(B,N,K,dtype=RD,device=dev,generator=g))/np.sqrt(2)
    ir = torch.randint(0,4,(B,K),device=dev,generator=g); iq = torch.randint(0,4,(B,K),device=dev,generator=g)
    x = (LV[ir]+1j*LV[iq]).to(CD)
    n = (torch.randn(B,N,dtype=RD,device=dev,generator=g)+1j*torch.randn(B,N,dtype=RD,device=dev,generator=g))*np.sqrt(s2/2)
    y = torch.einsum('bnk,bk->bn',H,x)+n
    A = torch.einsum('bnk,bnl->bkl',H.conj(),H)+s2*torch.eye(K,dtype=CD,device=dev)
    b = torch.einsum('bnk,bn->bk',H.conj(),y)
    return x, A, b, ir, iq

def rich(A,b,steps):
    x = torch.zeros_like(b)
    for al in steps:
        x = x + al*(b - torch.einsum('bkl,bl->bk',A,x))
    return x

def cg(A,b,T):
    x=torch.zeros_like(b); r=b.clone(); p=r.clone(); rs=(r.conj()*r).sum(1).real
    for _ in range(T):
        Ap=torch.einsum('bkl,bl->bk',A,p); a=rs/(p.conj()*Ap).sum(1).real
        x=x+a[:,None]*p; r=r-a[:,None]*Ap; rn=(r.conj()*r).sum(1).real
        p=r+(rn/rs)[:,None]*p; rs=rn
    return x

def mmse(A,b): return torch.linalg.solve(A,b.unsqueeze(-1)).squeeze(-1)

def ber(xh, ir, iq):
    def dec(v):
        return (v>-2/np.sqrt(10)).long()+(v>0).long()+(v>2/np.sqrt(10)).long()
    dr, di = dec(xh.real), dec(xh.imag)
    e = (GRAY[dr]!=GRAY[ir]).sum()+(GRAY[di]!=GRAY[iq]).sum()
    return e.item()/(ir.numel()*4)

def learn(N,K,s2,T,init,iters=1500,B=256,seed=0):
    g=torch.Generator(device=dev); g.manual_seed(seed)
    th=torch.tensor(np.log(init),dtype=RD,device=dev,requires_grad=True)
    opt=torch.optim.Adam([th],lr=0.03)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,iters)
    for it in range(iters):
        x,A,b,_,_=gen(B,N,K,s2,g)
        xh=rich(A,b,torch.exp(th))
        loss=((xh-x).abs()**2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return torch.exp(th).detach().cpu().numpy()

def evaluate(N,K,s2,steps_dict,T,B=2000,seed=123,with_cg=True):
    g=torch.Generator(device=dev); g.manual_seed(seed)
    x,A,b,ir,iq=gen(B,N,K,s2,g); xm=mmse(A,b)
    out={}
    def rec(name,xh):
        out[name]={"mse":((xh-x).abs()**2).mean().item(),"ber":ber(xh,ir,iq),
                   "dmmse":(((xh-xm).abs()**2).sum(1)/(xm.abs()**2).sum(1)).mean().item()}
    rec("MMSE",xm)
    for k,v in steps_dict.items(): rec(k,rich(A,b,[float(a) for a in v]))
    if with_cg: rec("CG",cg(A,b,T))
    return out

def methods(N,K,s2,T,learned=True,seed=0):
    beta=K/N
    d={"Richardson":const_steps(beta,N,s2,T),"Chebyshev":cheb_steps(beta,N,s2,T)}
    d["MP-opt"],_=mp_opt_steps(beta,N,s2,T)
    if learned: d["Learned"]=learn(N,K,s2,T,const_steps(beta,N,s2,T),seed=seed)
    return d

if __name__=="__main__":
    R={}
    t0=time.time()
    # E1: error vs T, three loads, SNR 20 dB (MF-bound), N=128
    N=128; snr=20
    R["E1"]={}
    for K in (32,64,96):
        s2=N/10**(snr/10); R["E1"][K]={}
        for T in range(1,9):
            m=methods(N,K,s2,T); R["E1"][K][T]=evaluate(N,K,s2,m,T)
            R["E1"][K][T]["theory"]={k:poly_obj(v,K/N,N,s2) for k,v in m.items() if k!="Learned"}
            R["E1"][K][T]["theory"]["Learned"]=poly_obj(m["Learned"],K/N,N,s2)
            if T==4: R["E1"][K]["steps4"]={k:sorted(map(float,v)) for k,v in m.items()}
        print("E1",K,time.time()-t0,flush=True)
    # E2: BER vs SNR, T=4
    R["E2"]={}
    for K in (64,96):
        R["E2"][K]={}
        for snr in (10,14,18,22,26):
            s2=N/10**(snr/10); m=methods(N,K,s2,4); R["E2"][K][snr]=evaluate(N,K,s2,m,4,B=4000)
        print("E2",K,time.time()-t0,flush=True)
    # E3: finite size at beta=0.5, T=4, snr 20
    R["E3"]={}
    for N_ in (16,32,64,128,256):
        K_=N_//2; s2=N_/10**(2.0); m=methods(N_,K_,s2,4); R["E3"][N_]=evaluate(N_,K_,s2,m,4,B=4000)
        R["E3"][N_]["theory"]={k:poly_obj(v,.5,N_,s2) for k,v in m.items()}
    print("E3",time.time()-t0,flush=True)
    json.dump(R,open("results.json","w"),indent=1)
    print("done",time.time()-t0)
