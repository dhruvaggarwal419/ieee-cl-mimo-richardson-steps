"""(D) numerical stability: step ordering, float32 vs float64, step quantisation."""
import json, time, numpy as np, torch
torch.set_num_threads(8)
import r1lib as L

t0 = time.time(); R = {"meta": {"N": 128, "snr_dB": 20}}
N = 128; s2 = N/100.0; B = 1000
rng = np.random.default_rng(0)

def run_track(A, b, steps, dtype):
    """Richardson with residual tracking; returns (xT, max rel residual, max rel iterate norm)."""
    cdt = torch.complex128 if dtype == torch.float64 else torch.complex64
    Ad, bd = A.to(cdt), b.to(cdt)
    x = torch.zeros_like(bd); nb = bd.norm(dim=1)
    mr = torch.zeros(len(bd), dtype=torch.float64); mx = torch.zeros_like(mr)
    for al in steps:
        r = bd - torch.einsum('bkl,bl->bk', Ad, x)
        mr = torch.maximum(mr, (r.norm(dim=1)/nb).double())
        x = x + float(al)*r
        mx = torch.maximum(mx, x.norm(dim=1).double())
    return x, mr, mx

def quant(st, nbits, mode="lin"):
    st = np.asarray(st, float)
    if mode == "linp2":                               # shared power-of-2 exponent (hardware fixed point)
        sc = 2.0**np.ceil(np.log2(st.max()))
        q = np.round(st/sc*(2**nbits-1))/(2**nbits-1)*sc
    elif mode == "lin":                               # full-scale = max step (best case uniform)
        sc = st.max()
        q = np.round(st/sc*(2**nbits-1))/(2**nbits-1)*sc
    else:
        lo, hi = np.log2(st.min()), np.log2(st.max())
        if hi <= lo: return st.copy()
        q = 2.0**(lo + np.round((np.log2(st)-lo)/(hi-lo)*(2**nbits-1))/(2**nbits-1)*(hi-lo))
    return q

for beta in (0.5, 0.9):
    K = int(beta*N)
    g = torch.Generator(); g.manual_seed(77)
    x, A, b, ir, iq = L.gen(B, N, K, s2, g, M=16)
    xm = L.mmse(A, b); den = (xm.abs()**2).sum(1)
    lo, hi = L.edges(beta, N, s2)
    key = f"beta{beta}"
    R[key] = {"K": K, "lam_lo": lo, "lam_hi": hi, "cond": hi/lo, "T": {}}
    for T in (4, 8, 16, 30):
        lamq, wqq, aa, bb = L.mp_nodes(beta)
        st0, objq, rtq, imq = L.opt_steps_measure(N*lamq+s2, wqq, s2, T, N*aa+s2, N*bb+s2)
        roots = np.sort(1.0/st0)
        orders = {"leja": st0,
                  "ascending": np.sort(st0),
                  "descending": np.sort(st0)[::-1].copy()}
        for s in range(5):
            p = np.random.default_rng(s).permutation(T)
            orders[f"random{s}"] = st0[p]
        out = {"roots": [float(r) for r in roots], "design_maxImRoot": imq,
               "design_J": objq, "orderings": {}}
        for nm, st in orders.items():
            e = {}
            for dn, dt in (("f64", torch.float64), ("f32", torch.float32)):
                xh, mr, mx = run_track(A, b, st, dt)
                d = ((xh.to(L.CD)-xm).abs()**2).sum(1)/den
                e[dn] = {"dmmse": float(d.mean().item()),
                         "max_rel_residual": float(mr.max().item()),
                         "mean_max_rel_residual": float(mr.mean().item()),
                         "max_iterate_norm": float(mx.max().item())}
            out["orderings"][nm] = e
        # quantisation (leja order)
        qd = {}
        for mode in ("lin", "linp2", "log"):
            for nb in (8, 6, 4):
                sq = quant(st0, nb, mode)
                xh, mr, mx = run_track(A, b, sq, torch.float64)
                d = ((xh-xm).abs()**2).sum(1)/den
                qd[f"{mode}{nb}"] = {"dmmse": float(d.mean().item()),
                                     "max_rel_step_err": float(np.abs(sq/st0-1).max()),
                                     "steps": [float(v) for v in sq]}
        out["quant"] = qd
        out["dmmse_fp_ref"] = out["orderings"]["leja"]["f64"]["dmmse"]
        R[key]["T"][T] = out
        print(key, T, "%.1fs" % (time.time()-t0), flush=True)
    json.dump(R, open("r1_D_stability.json", "w"), indent=1)
json.dump(R, open("r1_D_stability.json", "w"), indent=1)
print("DONE D", time.time()-t0, flush=True)
