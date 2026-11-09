"""Missing cell of (A-ext): N=256, beta=0.9. Chunked linear algebra to avoid the
MKL ZLASWP failure seen with a 1000-matrix batch at K=230."""
import json, time, numpy as np, torch
torch.set_num_threads(8)
import r1lib as L

t0 = time.time(); B = 400; CH = 50
N = 256; beta = 0.9; K = int(round(beta*N)); s2 = N/100.0; b_eff = K/N
nreal = int(np.ceil(300000/K))
eig = L.emp_eigs(N, K, nreal=nreal, seed=7, chunk=25)
print("pooled", eig.shape, "%.1fs" % (time.time()-t0), flush=True)
lam_e = eig + s2; wq_e = np.full(lam_e.shape, float(K)/len(lam_e))
xm_, wq_m, a, bb = L.mp_nodes(b_eff); lam_m = N*xm_ + s2
J0_mp = L.obj_measure([], lam_m, wq_m, s2); J0_emp = L.obj_measure([], lam_e, wq_e, s2)/K

g = torch.Generator(); g.manual_seed(1234)
As, bs, xms, evs = [], [], [], []
for c in range(B//CH):
    x, A, bv, ir, iq = L.gen(CH, N, K, s2, g, M=16)
    As.append(A); bs.append(bv); xms.append(L.mmse(A, bv))
    evs.append(torch.linalg.eigvalsh(A - s2*torch.eye(K, dtype=L.CD)).numpy())
ev = np.concatenate(evs)
print("test set built %.1fs" % (time.time()-t0), flush=True)

def run(steps_list, per_channel=False):
    out = []
    i = 0
    for A, bv, xm in zip(As, bs, xms):
        if per_channel:
            xh = L.rich_batched(A, bv, torch.tensor(steps_list[i:i+CH]))
        else:
            xh = L.rich(A, bv, [float(v) for v in steps_list])
        out.append((((xh-xm).abs()**2).sum(1)/(xm.abs()**2).sum(1)).numpy()); i += CH
    r = np.concatenate(out)
    return float(r.mean()), float(r.std()/np.sqrt(len(r)))

R = {f"N{N}_K{K}": {"N": N, "K": K, "beta": b_eff, "J0_mp": J0_mp, "J0_emp": J0_emp,
      "mp_lo": float(N*(1-np.sqrt(b_eff))**2+s2), "mp_hi": float(N*(1+np.sqrt(b_eff))**2+s2),
      "emp_min": float(lam_e.min()), "emp_max": float(lam_e.max()),
      "frac_outside_mp": float(np.mean((lam_e < N*(1-np.sqrt(b_eff))**2+s2) |
                                       (lam_e > N*(1+np.sqrt(b_eff))**2+s2))),
      "Btest": B, "T": {}}}
key = f"N{N}_K{K}"
for T in (2, 4, 6, 8, 10, 12, 16, 20, 24, 30):
    stm = L.mp_opt_steps(b_eff, N, s2, T)[0]
    stf = L.opt_steps_measure(lam_e, wq_e, s2, T)[0]
    stc = L.cheb_steps(b_eff, N, s2, T)
    out = {}
    for nm, st in (("MP-opt", stm), ("Finite-N", stf), ("Chebyshev", stc)):
        m, se = run(st)
        out[nm] = {"dmmse": m, "se": se,
                   "J_mp_over_J0": L.obj_measure(st, lam_m, wq_m, s2)/J0_mp,
                   "J_emp_over_J0": (L.obj_measure(st, lam_e, wq_e, s2)/K)/J0_emp,
                   "max_abs_r_on_emp_support": float(np.abs(np.prod(
                       [1-v*np.array([lam_e.min(), lam_e.max()]) for v in st], axis=0)).max())}
    S = np.zeros((B, T))
    for i in range(B):
        S[i] = L.chan_steps(ev[i], s2, T)[0]
    m, se = run(S, True); out["Oracle-chan"] = {"dmmse": m, "se": se}
    oc = []
    for A, bv, xm in zip(As, bs, xms):
        xh = L.cg(A, bv, T)
        oc.append((((xh-xm).abs()**2).sum(1)/(xm.abs()**2).sum(1)).numpy())
    oc = np.concatenate(oc)
    out["CG"] = {"dmmse": float(oc.mean()), "se": float(oc.std()/np.sqrt(len(oc)))}
    R[key]["T"][T] = out
    print(key, T, "%.1fs" % (time.time()-t0), flush=True)
    json.dump(R, open("r1_G_largeT256.json", "w"), indent=1)
print("DONE G", time.time()-t0, flush=True)
