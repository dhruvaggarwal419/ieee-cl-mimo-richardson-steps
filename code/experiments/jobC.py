"""(C) BER vs SNR: 16/64-QAM, loads, N, perfect+imperfect CSI, with CIs.
usage: python jobC.py <N>"""
import json, sys, time, numpy as np, torch
torch.set_num_threads(8)
import r1lib as L

N = int(sys.argv[1]); t0 = time.time()
s2_of = lambda snr: N/10**(snr/10.0)
SNR = {16: [8, 12, 16, 20, 24, 28], 64: [14, 18, 22, 26, 30, 34]}
LOADS = [0.25, 0.5, 0.75]
TAU = [0.0, 0.001, 0.01, 0.05]
TS = [4, 8]
NG = {64: 16, 128: 16, 256: 16}[N]                 # independent seed groups
GSZ = {64: 500, 128: 375, 256: 250}[N]             # frames per group
rng = np.random.default_rng(0)

def boot(ev, nv, nb=4000):
    """percentile bootstrap 95% CI of pooled BER over independent groups"""
    ev = np.asarray(ev, float); nv = np.asarray(nv, float); G = len(ev)
    idx = rng.integers(0, G, (nb, G))
    bs = ev[idx].sum(1)/nv[idx].sum(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

R = {"meta": {"N": N, "groups": NG, "frames_per_group": GSZ, "T": TS,
              "tau2": TAU, "snr_grid": SNR, "metric": "SNR = N/sigma^2 (per-user MF SNR, dB)"}}
for load in LOADS:
    K = int(round(load*N)); beta = K/N
    for M in (16, 64):
        for snr in SNR[M]:
            s2 = s2_of(snr)
            for tau2 in TAU:
                Neff = N*(1+tau2)
                stp = {}
                for T in TS:
                    stp[T] = {"Richardson": L.const_steps(beta, Neff, s2, T),
                              "Chebyshev": L.cheb_steps(beta, Neff, s2, T),
                              "MP-opt": L.mp_opt_steps(beta, Neff, s2, T)[0]}
                names = ["LMMSE"] + [f"{m}-T{T}" for T in TS for m in ("Richardson", "Chebyshev", "MP-opt", "CG")]
                E = {n: [] for n in names}; NB = {n: [] for n in names}
                DM = {n: [] for n in names}
                for gi in range(NG):
                    g = torch.Generator(); g.manual_seed(100000 + gi)
                    x, A, b, ir, iq = L.gen(GSZ, N, K, s2, g, M=M, tau2=tau2)
                    xm = L.mmse(A, b); den = (xm.abs()**2).sum(1)
                    def rec(n, xh):
                        e, nb_ = L.bit_errors(xh, ir, iq, M); E[n].append(e); NB[n].append(nb_)
                        DM[n].append(float((((xh-xm).abs()**2).sum(1)/den).mean().item()))
                    rec("LMMSE", xm)
                    for T in TS:
                        for m, st in stp[T].items():
                            rec(f"{m}-T{T}", L.rich(A, b, [float(a) for a in st]))
                        rec(f"CG-T{T}", L.cg(A, b, T))
                row = {}
                for n in names:
                    tot_e, tot_n = int(sum(E[n])), int(sum(NB[n]))
                    lo, hi = boot(E[n], NB[n])
                    row[n] = {"ber": tot_e/tot_n, "ci95": [lo, hi], "errors": tot_e, "bits": tot_n,
                              "dmmse": float(np.mean(DM[n]))}
                R[f"L{load}_M{M}_snr{snr}_tau{tau2}"] = row
            print(N, load, M, snr, "%.1fs" % (time.time()-t0), flush=True)
            json.dump(R, open(f"r1_C_ber_N{N}.json", "w"), indent=1)
json.dump(R, open(f"r1_C_ber_N{N}.json", "w"), indent=1)
print("DONE C N=%d" % N, time.time()-t0, flush=True)
