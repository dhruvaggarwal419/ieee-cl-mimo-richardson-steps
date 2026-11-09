"""(B) equal computational budget: error vs matvec count / flops, offline + online cost."""
import json, time, numpy as np, torch
torch.set_num_threads(8)
import r1lib as L

t0 = time.time(); R = {}
N = 128; s2 = N/100.0; B = 3000
TS = list(range(1, 13))

# ---- flop / op model (real flops, complex arithmetic) -------------------------
def ops(method, K, T):
    mv = 8*K*K                      # complex mat-vec
    ax = 4*K                        # complex axpy (real scalar)
    sub = 2*K
    dot = 4*K                       # |r|^2 real reduction
    dot2 = 8*K                      # Re(p^H A p)
    if method == "CG":
        per = mv + dot2 + dot + 3*ax
        red = 2
    else:
        per = mv + sub + ax
        red = 0
    return {"matvec": T, "flops": per*T, "reductions": red*T,
            "axpy": (3 if method == "CG" else 1)*T,
            "mem_vectors": 4 if method == "CG" else 2}

R["op_model"] = {}
for K in (32, 64, 96, 128):
    R["op_model"][K] = {m: ops(m, K, 1) for m in ("Richardson", "CG")}
    R["op_model"][K]["flop_ratio_CG_over_fixed"] = ops("CG", K, 1)["flops"]/ops("R", K, 1)["flops"]

# ---- error vs T (and hence vs matvec count) ----------------------------------
R["curves"] = {}
for K in (32, 64, 96):
    beta = K/N
    g = torch.Generator(); g.manual_seed(1234)
    x, A, bv, ir, iq = L.gen(B, N, K, s2, g, M=16)
    xm = L.mmse(A, bv); den = (xm.abs()**2).sum(1)
    cur = {}
    for T in TS:
        d = {"Richardson": L.const_steps(beta, N, s2, T), "Chebyshev": L.cheb_steps(beta, N, s2, T)}
        d["MP-opt"] = L.mp_opt_steps(beta, N, s2, T)[0]
        row = {}
        for nm, st in d.items():
            xh = L.rich(A, bv, [float(a) for a in st])
            r = ((xh-xm).abs()**2).sum(1)/den
            row[nm] = {"dmmse": float(r.mean().item()), "se": float(r.std().item()/np.sqrt(B)),
                       "matvec": T, "flops": ops("R", K, T)["flops"], "reductions": 0}
        xh = L.cg(A, bv, T); r = ((xh-xm).abs()**2).sum(1)/den
        row["CG"] = {"dmmse": float(r.mean().item()), "se": float(r.std().item()/np.sqrt(B)),
                     "matvec": T, "flops": ops("CG", K, T)["flops"], "reductions": 2*T}
        cur[T] = row
    R["curves"][K] = cur
    print("curves K=%d %.1fs" % (K, time.time()-t0), flush=True)
json.dump(R, open("r1_B_budget.json", "w"), indent=1)

# ---- offline design cost ------------------------------------------------------
def tm(f, rep=20):
    f(); ts = []
    for _ in range(rep):
        a = time.perf_counter(); f(); ts.append(time.perf_counter()-a)
    return float(np.median(ts))

R["offline_s"] = {}
for K in (64, 96):
    beta = K/N; T = 4
    o = {"const": tm(lambda: L.const_steps(beta, N, s2, T)),
         "cheb": tm(lambda: L.cheb_steps(beta, N, s2, T)),
         "MP-opt": tm(lambda: L.mp_opt_steps(beta, N, s2, T)),
         "MP-opt_M=400nodes": tm(lambda: L.mp_opt_steps(beta, N, s2, T, M=400))}
    a = time.perf_counter(); L.emp_eigs(N, K, nreal=int(np.ceil(300000/K)), seed=7)
    o["finiteN_MC_eigs_once"] = time.perf_counter()-a
    R["offline_s"][K] = o
print("offline %.1fs" % (time.time()-t0), flush=True)

# ---- online latency: measured wall clock -------------------------------------
R["online_s"] = {}
for K in (32, 64, 96, 128):
    g = torch.Generator(); g.manual_seed(9)
    x1, A1, b1, _, _ = L.gen(1, N, K, s2, g, M=16)
    x2, A2, b2, _, _ = L.gen(1000, N, K, s2, g, M=16)
    st = [float(a) for a in L.mp_opt_steps(K/N, N, s2, 4)[0]]
    o = {}
    for T in (4, 8):
        o[f"rich_T{T}_b1"] = tm(lambda: L.rich(A1, b1, st*(T//4)), 200)
        o[f"cg_T{T}_b1"] = tm(lambda: L.cg(A1, b1, T), 200)
        o[f"rich_T{T}_b1000"] = tm(lambda: L.rich(A2, b2, st*(T//4)), 10)
        o[f"cg_T{T}_b1000"] = tm(lambda: L.cg(A2, b2, T), 10)
    o["solve_b1"] = tm(lambda: L.mmse(A1, b1), 200)
    o["solve_b1000"] = tm(lambda: L.mmse(A2, b2), 10)
    # primitive timings for a latency model
    v1 = b1.clone()
    o["prim_matvec_b1"] = tm(lambda: torch.einsum('bkl,bl->bk', A1, v1), 500)
    o["prim_dot_b1"] = tm(lambda: (v1.conj()*v1).sum(1).real, 500)
    o["prim_axpy_b1"] = tm(lambda: v1 + 0.5*v1, 500)
    R["online_s"][K] = o
    print("online K=%d %.1fs" % (K, time.time()-t0), flush=True)
json.dump(R, open("r1_B_budget.json", "w"), indent=1)
print("DONE B", time.time()-t0, flush=True)
