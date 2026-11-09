"""(C) BER vs SNR, restructured: one channel draw per (load, group) is reused for
every SNR, every CSI error level and both constellations.  Exact, because
A = Hh^H Hh + s2 I   and   b = (Hh^H H) x + sqrt(s2) (Hh^H n0).
usage: python jobC3.py <N>"""
import json, sys, time, numpy as np, torch
torch.set_num_threads(8)
import r1lib as L

N = int(sys.argv[1]); t0 = time.time()
SNR = [10, 14, 18, 22, 26, 30, 34]
LOADS = [0.25, 0.5, 0.75]
TAU = [0.0, 0.002, 0.01, 0.05]
TS = [4, 8]
NG = 16
GSZ = {64: 500, 128: 375, 256: 250}[N]
RD, CD = torch.float32, torch.complex64
rng = np.random.default_rng(0)
METH = ("Richardson", "Chebyshev", "MP-opt")

def boot(ev, nv, nb=4000):
    ev = np.asarray(ev, float); nv = np.asarray(nv, float); G = len(ev)
    idx = rng.integers(0, G, (nb, G))
    bs = ev[idx].sum(1)/nv[idx].sum(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

R = {"meta": {"N": N, "groups": NG, "frames_per_group": GSZ, "T": TS, "tau2": TAU,
              "snr_grid": SNR, "dtype": "complex64 (checked bit-identical to complex128)",
              "metric": "SNR = N/sigma^2 (per-user matched-filter SNR, dB)",
              "csi": "Hhat = H + e, e ~ CN(0, tau2) per entry; receiver uses Hhat only",
              "note": "one H per (load,group) reused across SNR, tau and both QAM orders"}}

names = ["LMMSE"] + [f"{m}-T{T}" for T in TS for m in METH + ("CG",)]

for load in LOADS:
    K = int(round(load*N)); beta = K/N
    stp = {}
    for snr in SNR:
        s2 = N/10**(snr/10.0)
        for tau2 in TAU:
            Ne = N*(1+tau2)
            stp[(snr, tau2)] = {T: {"Richardson": L.const_steps(beta, Ne, s2, T),
                                    "Chebyshev": L.cheb_steps(beta, Ne, s2, T),
                                    "MP-opt": L.mp_opt_steps(beta, Ne, s2, T)[0]} for T in TS}
    acc = {(M, snr, tau2): {n: {"e": [], "n": [], "d": []} for n in names}
           for M in (16, 64) for snr in SNR for tau2 in TAU}
    for gi in range(NG):
        g = torch.Generator(); g.manual_seed(100000+gi)
        H = (torch.randn(GSZ, N, K, dtype=RD, generator=g) +
             1j*torch.randn(GSZ, N, K, dtype=RD, generator=g))/np.sqrt(2)
        e0 = (torch.randn(GSZ, N, K, dtype=RD, generator=g) +
              1j*torch.randn(GSZ, N, K, dtype=RD, generator=g))/np.sqrt(2)
        n0 = (torch.randn(GSZ, N, dtype=RD, generator=g) +
              1j*torch.randn(GSZ, N, dtype=RD, generator=g))/np.sqrt(2)
        sym = {}
        for M in (16, 64):
            lv = torch.tensor(L.qam(M)[0], dtype=RD); m = len(lv)
            ir = torch.randint(0, m, (GSZ, K), generator=g)
            iq = torch.randint(0, m, (GSZ, K), generator=g)
            sym[M] = ((lv[ir] + 1j*lv[iq]).to(CD), ir, iq)
        for tau2 in TAU:
            Hh = H if tau2 == 0 else H + np.sqrt(tau2)*e0
            Hc = Hh.conj().transpose(-2, -1).resolve_conj()
            G = torch.matmul(Hc, Hh)
            P = G if tau2 == 0 else torch.matmul(Hc, H)
            q = torch.matmul(Hc, n0.unsqueeze(-1)).squeeze(-1)
            I = torch.eye(K, dtype=CD)
            for snr in SNR:
                s2 = N/10**(snr/10.0)
                A = G + s2*I
                for M in (16, 64):
                    x, ir, iq = sym[M]
                    b = L.mv(P, x) + np.sqrt(s2)*q
                    xm = L.mmse(A, b); den = (xm.abs()**2).sum(1)
                    a = acc[(M, snr, tau2)]
                    def rec(n, xh):
                        e, nb_ = L.bit_errors(xh, ir, iq, M)
                        a[n]["e"].append(e); a[n]["n"].append(nb_)
                        a[n]["d"].append(float((((xh-xm).abs()**2).sum(1)/den).mean().item()))
                    rec("LMMSE", xm)
                    for T in TS:
                        for mm in METH:
                            rec(f"{mm}-T{T}", L.rich(A, b, [float(v) for v in stp[(snr, tau2)][T][mm]]))
                        rec(f"CG-T{T}", L.cg(A, b, T))
        print(N, load, "group", gi, "%.1fs" % (time.time()-t0), flush=True)
    for (M, snr, tau2), a in acc.items():
        row = {}
        for n in names:
            te, tn = int(sum(a[n]["e"])), int(sum(a[n]["n"]))
            lo, hi = boot(a[n]["e"], a[n]["n"])
            row[n] = {"ber": te/tn, "ci95": [lo, hi], "errors": te, "bits": tn,
                      "dmmse": float(np.mean(a[n]["d"]))}
        R[f"L{load}_M{M}_snr{snr}_tau{tau2}"] = row
    json.dump(R, open(f"r1_C_ber_N{N}.json", "w"), indent=1)
    print("load done", load, "%.1fs" % (time.time()-t0), flush=True)
json.dump(R, open(f"r1_C_ber_N{N}.json", "w"), indent=1)
print("DONE C N=%d" % N, time.time()-t0, flush=True)
