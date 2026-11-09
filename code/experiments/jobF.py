"""(A-extension) large-T behaviour: asymptotic design vs finite-N design vs oracle,
and asymptotic theory J_T/J_0 against the measured finite-N error."""
import json, time, numpy as np, torch
torch.set_num_threads(8)
import r1lib as L

t0 = time.time(); R = {"meta": {"snr_dB": 20, "Btest": 1000}}
B = 1000
for N in (64, 128, 256):
    for beta in (0.5, 0.9):
        K = int(round(beta*N)); s2 = N/100.0; b_eff = K/N
        key = f"N{N}_K{K}"
        nreal = int(np.ceil(400000/K))
        eig = L.emp_eigs(N, K, nreal=nreal, seed=7)
        lam_e = eig + s2; wq_e = np.full(lam_e.shape, float(K)/len(lam_e))
        xm_, wq_m, a, bb = L.mp_nodes(b_eff); lam_m = N*xm_ + s2
        J0_mp = L.obj_measure([], lam_m, wq_m, s2)
        J0_emp = L.obj_measure([], lam_e, wq_e, s2)/K
        g = torch.Generator(); g.manual_seed(1234)
        x, A, bv, ir, iq = L.gen(B, N, K, s2, g, M=16)
        xm = L.mmse(A, bv); den = (xm.abs()**2).sum(1)
        ev = torch.linalg.eigvalsh(A - s2*torch.eye(K, dtype=L.CD)).numpy()
        R[key] = {"N": N, "K": K, "beta": b_eff, "J0_mp": J0_mp, "J0_emp": J0_emp,
                  "mp_lo": float(N*(1-np.sqrt(b_eff))**2+s2), "mp_hi": float(N*(1+np.sqrt(b_eff))**2+s2),
                  "emp_min": float(lam_e.min()), "emp_max": float(lam_e.max()),
                  "frac_outside_mp": float(np.mean((lam_e < N*(1-np.sqrt(b_eff))**2+s2) |
                                                   (lam_e > N*(1+np.sqrt(b_eff))**2+s2))),
                  "T": {}}
        for T in (2, 4, 6, 8, 10, 12, 16, 20, 24, 30):
            if T >= K: continue
            stm = L.mp_opt_steps(b_eff, N, s2, T)[0]
            stf = L.opt_steps_measure(lam_e, wq_e, s2, T)[0]
            stc = L.cheb_steps(b_eff, N, s2, T)
            out = {}
            for nm, st in (("MP-opt", stm), ("Finite-N", stf), ("Chebyshev", stc)):
                xh = L.rich(A, bv, [float(aa) for aa in st])
                r = ((xh-xm).abs()**2).sum(1)/den
                out[nm] = {"dmmse": float(r.mean().item()), "se": float(r.std().item()/np.sqrt(B)),
                           "J_mp_over_J0": L.obj_measure(st, lam_m, wq_m, s2)/J0_mp,
                           "J_emp_over_J0": (L.obj_measure(st, lam_e, wq_e, s2)/K)/J0_emp,
                           "max_abs_r_on_emp_support": float(np.abs(np.prod(
                               [1-aa*np.array([lam_e.min(), lam_e.max()]) for aa in st], axis=0)).max())}
            S = np.zeros((B, T))
            for i in range(B):
                S[i] = L.chan_steps(ev[i], s2, T)[0]
            xh = L.rich_batched(A, bv, torch.tensor(S))
            r = ((xh-xm).abs()**2).sum(1)/den
            out["Oracle-chan"] = {"dmmse": float(r.mean().item()), "se": float(r.std().item()/np.sqrt(B))}
            xh = L.cg(A, bv, T); r = ((xh-xm).abs()**2).sum(1)/den
            out["CG"] = {"dmmse": float(r.mean().item()), "se": float(r.std().item()/np.sqrt(B))}
            R[key]["T"][T] = out
            print(key, T, "%.1fs" % (time.time()-t0), flush=True)
        json.dump(R, open("r1_F_largeT.json", "w"), indent=1)
json.dump(R, open("r1_F_largeT.json", "w"), indent=1)
print("DONE F", time.time()-t0, flush=True)
