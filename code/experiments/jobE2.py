"""(E) learned-step robustness: SNR mismatch, train-set size, init, load mismatch.

Training uses a cached pool of channel realisations stored as (G, x, w) with
G = H^H H, w = H^H n0 (n0 ~ CN(0,I)).  For any sigma^2 this reproduces exactly
A = G + s2 I and b = H^H(Hx + sqrt(s2) n0) = G x + sqrt(s2) w, so one pool
serves every SNR.  Cycling the pool = a finite training set of that many channels.
"""
import json, time, numpy as np, torch
torch.set_num_threads(8)
import r1lib as L

t0 = time.time()
N = 128; T = 4; BT = 32
R = {"meta": {"N": N, "T": T, "batch": BT, "opt": "Adam lr 0.03, cosine decay",
              "note": "training pool stored as (G,x,w); batch b uses pool entry (it mod nbatch)"}}

_pool = {}
def pool(N, K, nbatch, seed=11, M=16):
    key = (N, K, nbatch, seed, M)
    if key in _pool: return _pool[key]
    g = torch.Generator(); g.manual_seed(seed)
    lv, bits, m = L.qam(M)[0], None, None
    lvt = torch.tensor(L.qam(M)[0], dtype=L.RD); mm = len(lvt)
    Gs, xs, ws = [], [], []
    for i in range(nbatch):
        H = (torch.randn(BT, N, K, dtype=L.RD, generator=g) +
             1j*torch.randn(BT, N, K, dtype=L.RD, generator=g))/np.sqrt(2)
        ir = torch.randint(0, mm, (BT, K), generator=g); iq = torch.randint(0, mm, (BT, K), generator=g)
        x = (lvt[ir] + 1j*lvt[iq]).to(L.CD)
        n0 = (torch.randn(BT, N, dtype=L.RD, generator=g) +
              1j*torch.randn(BT, N, dtype=L.RD, generator=g))/np.sqrt(2)
        Hc = H.conj().transpose(-2, -1).resolve_conj()
        Gs.append(torch.matmul(Hc, H)); xs.append(x)
        ws.append(torch.matmul(Hc, n0.unsqueeze(-1)).squeeze(-1))
    _pool[key] = (Gs, xs, ws)
    return _pool[key]

def batch(P, i, s2, K):
    G, x, w = P[0][i], P[1][i], P[2][i]
    A = G + s2*torch.eye(K, dtype=L.CD)
    b = L.mv(G, x) + np.sqrt(s2)*w
    return x, A, b

def learn(N, K, s2, init, iters=1500, seed=0, nbatch=1024, npool=1024):
    P = pool(N, K, npool)
    th = torch.tensor(np.log(np.asarray(init, float)), dtype=L.RD, requires_grad=True)
    opt = torch.optim.Adam([th], lr=0.03)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    off = seed*7919
    for it in range(iters):
        x, A, b = batch(P, (off + it) % nbatch, s2, K)
        loss = ((L.rich(A, b, torch.exp(th))-x).abs()**2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return torch.exp(th).detach().numpy()

_tc = {}
def testset(N, K, s2, B=2000, seed=4242):
    k = (N, K, s2, B, seed)
    if k not in _tc:
        g = torch.Generator(); g.manual_seed(seed)
        x, A, b, ir, iq = L.gen(B, N, K, s2, g, M=16)
        xm = L.mmse(A, b); _tc[k] = (x, A, b, xm, (xm.abs()**2).sum(1))
    return _tc[k]

def ev(N, K, s2, st):
    x, A, b, xm, den = testset(N, K, s2)
    xh = L.rich(A, b, [float(a) for a in st])
    return float((((xh-xm).abs()**2).sum(1)/den).mean().item())

def dist(a, b):
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    return {"rel_l2": float(np.linalg.norm(a-b)/np.linalg.norm(b)),
            "max_rel": float(np.abs(a/b-1).max())}

def save(): json.dump(R, open("r1_E_learned.json", "w"), indent=1)

# ---------- E1: SNR mismatch -------------------------------------------------
K = 64; beta = 0.5
SNRS = [5, 10, 15, 20, 25, 30]
pool(N, K, 1024)
print("pool built %.1fs" % (time.time()-t0), flush=True)
learned = {}; mpst = {}
for sx in SNRS:
    s2x = N/10**(sx/10.0)
    mpst[sx] = L.mp_opt_steps(beta, N, s2x, T)[0]
    learned[sx] = learn(N, K, s2x, L.const_steps(beta, N, s2x, T), iters=1500, seed=0)
    print("E1 train", sx, "%.1fs" % (time.time()-t0), flush=True)
R["E1_snr_mismatch"] = {"snrs": SNRS, "train_channels": 1024*BT,
    "steps_learned": {s: [float(a) for a in learned[s]] for s in SNRS},
    "steps_mp": {s: [float(a) for a in mpst[s]] for s in SNRS},
    "dist_learned_vs_mp_same_snr": {s: dist(learned[s], mpst[s]) for s in SNRS}, "grid": {}}
for sx in SNRS:
    for sy in SNRS:
        s2y = N/10**(sy/10.0)
        R["E1_snr_mismatch"]["grid"][f"{sx}->{sy}"] = {
            "learned": ev(N, K, s2y, learned[sx]),
            "mp_trainSNR": ev(N, K, s2y, mpst[sx]),
            "mp_testSNR": ev(N, K, s2y, mpst[sy])}
save(); print("E1 done %.1fs" % (time.time()-t0), flush=True)

# ---------- E2: training set size x iterations --------------------------------
s2 = N/100.0; mp20 = L.mp_opt_steps(beta, N, s2, T)[0]
R["E2_data"] = {"mp": [float(a) for a in mp20], "mp_dmmse": ev(N, K, s2, mp20), "runs": {}}
for nb in (8, 64, 256, 1024):
    for it in (200, 1000, 3000):
        st = learn(N, K, s2, L.const_steps(beta, N, s2, T), iters=it, seed=1, nbatch=nb)
        R["E2_data"]["runs"][f"nbatch{nb}_iters{it}"] = {
            "steps": [float(a) for a in st], "dmmse": ev(N, K, s2, st),
            "dist_to_mp": dist(st, mp20), "train_channels": nb*BT}
        print("E2", nb, it, "%.1fs" % (time.time()-t0), flush=True)
save()

# ---------- E3: initialisation ------------------------------------------------
lo, hi = L.edges(beta, N, s2)
inits = {"const": L.const_steps(beta, N, s2, T), "cheb": L.cheb_steps(beta, N, s2, T)}
for s in range(3):
    inits[f"rand{s}"] = np.exp(np.random.default_rng(s).uniform(np.log(1/hi), np.log(1/lo), T))
inits["tiny"] = np.full(T, 0.2/hi)
R["E3_init"] = {"mp": [float(a) for a in mp20], "mp_dmmse": ev(N, K, s2, mp20), "runs": {}}
for nm, iv in inits.items():
    for sd in (0, 1):
        st = learn(N, K, s2, iv, iters=1500, seed=sd)
        R["E3_init"]["runs"][f"{nm}_seed{sd}"] = {
            "init": [float(a) for a in iv], "steps": [float(a) for a in st],
            "dmmse": ev(N, K, s2, st), "dist_to_mp": dist(st, mp20)}
        print("E3", nm, sd, "%.1fs" % (time.time()-t0), flush=True)
save()

# ---------- E4: load mismatch -------------------------------------------------
BET = [0.25, 0.5, 0.75, 0.9]
lrn = {}; mps = {}
for bx in BET:
    Kx = int(round(bx*N)); s2 = N/100.0
    np_ = 1024 if Kx <= 64 else 256
    mps[bx] = L.mp_opt_steps(bx, N, s2, T)[0]
    lrn[bx] = learn(N, Kx, s2, L.const_steps(bx, N, s2, T), iters=1500, seed=0,
                    nbatch=np_, npool=np_)
    print("E4 train", bx, "%.1fs" % (time.time()-t0), flush=True)
R["E4_load_mismatch"] = {"betas": BET,
    "steps_learned": {b: [float(a) for a in lrn[b]] for b in BET},
    "steps_mp": {b: [float(a) for a in mps[b]] for b in BET},
    "dist_same_beta": {b: dist(lrn[b], mps[b]) for b in BET}, "grid": {}}
for bx in BET:
    for by in BET:
        Ky = int(round(by*N)); s2 = N/100.0
        R["E4_load_mismatch"]["grid"][f"{bx}->{by}"] = {
            "learned": ev(N, Ky, s2, lrn[bx]),
            "mp_trainLoad": ev(N, Ky, s2, mps[bx]),
            "mp_testLoad": ev(N, Ky, s2, mps[by])}
save(); print("DONE E", time.time()-t0, flush=True)
