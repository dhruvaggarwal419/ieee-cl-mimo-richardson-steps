"""(A) finite-dimensional vs asymptotic optimality."""
import json, time, numpy as np, torch
torch.set_num_threads(8)
import r1lib as L

t0 = time.time(); R = {"meta": {"snr_dB": 20, "s2": "N/100", "Btest": 2000}}
NS = [16, 32, 64, 128]; BETAS = [0.25, 0.5, 0.75, 0.9]; TS = [2, 4, 6, 8, 10]
B = 2000

def se(v): return float(v.std().item()/np.sqrt(len(v)))

for N in NS:
    for beta in BETAS:
        K = max(2, int(round(beta*N))); b_eff = K/N
        s2 = N/100.0
        key = f"N{N}_K{K}"
        R[key] = {"N": N, "K": K, "beta": b_eff, "s2": s2, "T": {}}
        # pooled finite-N spectrum
        nreal = int(np.ceil(300000/K))
        te = time.time(); eig_pool = L.emp_eigs(N, K, nreal=nreal, seed=7)
        R[key]["emp"] = {"nreal": nreal, "neig": len(eig_pool),
                         "mean": float(eig_pool.mean()), "min": float(eig_pool.min()),
                         "max": float(eig_pool.max()),
                         "mp_lo": float(N*(1-np.sqrt(b_eff))**2), "mp_hi": float(N*(1+np.sqrt(b_eff))**2),
                         "t_pool_s": time.time()-te}
        lam_e = eig_pool + s2; wq_e = np.full(lam_e.shape, float(K)/len(lam_e))
        lam_m, wq_m, a, bb = L.mp_nodes(b_eff); lam_m = N*lam_m + s2
        # test channels (shared across all designs -> paired comparison)
        g = torch.Generator(); g.manual_seed(1234)
        x, A, bv, ir, iq = L.gen(B, N, K, s2, g, M=16)
        xm = L.mmse(A, bv); den = (xm.abs()**2).sum(1)
        ev = torch.linalg.eigvalsh(A - s2*torch.eye(K, dtype=L.CD)).numpy()  # Gram eigs per channel
        for T in TS:
            if T >= K: continue
            d = {}
            d["MP-opt"] = L.mp_opt_steps(b_eff, N, s2, T)[0]
            stf, objf, imf = L.finite_steps(N, K, s2, T, nreal=nreal, seed=7)
            d["Finite-N"] = stf
            d["Chebyshev"] = L.cheb_steps(b_eff, N, s2, T)
            out = {}
            for nm, st in d.items():
                xh = L.rich(A, bv, [float(aa) for aa in st])
                r = ((xh-xm).abs()**2).sum(1)/den
                out[nm] = {"dmmse": float(r.mean().item()), "se": se(r),
                           "J_mp": L.obj_measure(st, lam_m, wq_m, s2),
                           "J_emp": L.obj_measure(st, lam_e, wq_e, s2)/K,
                           "steps": sorted(float(a_) for a_ in st)}
            # channel-specific (oracle): design per test channel
            tc = time.time(); S = np.zeros((B, T)); nim = 0
            for i in range(B):
                stc, _, im = L.chan_steps(ev[i], s2, T)
                S[i] = stc
                if im > 1e-8: nim += 1
            xh = L.rich_batched(A, bv, torch.tensor(S))
            r = ((xh-xm).abs()**2).sum(1)/den
            out["Oracle-chan"] = {"dmmse": float(r.mean().item()), "se": se(r),
                                  "n_complex_roots": nim, "t_design_total_s": time.time()-tc}
            # CG reference on same channels
            xh = L.cg(A, bv, T); r = ((xh-xm).abs()**2).sum(1)/den
            out["CG"] = {"dmmse": float(r.mean().item()), "se": se(r)}
            out["maxIm_finite"] = imf
            R[key]["T"][T] = out
        print(key, "%.1fs" % (time.time()-t0), flush=True)
        json.dump(R, open("r1_A_finite.json", "w"), indent=1)
json.dump(R, open("r1_A_finite.json", "w"), indent=1)
print("DONE A", time.time()-t0, flush=True)
