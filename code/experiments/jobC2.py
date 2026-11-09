"""(C) BER vs SNR: 16/64-QAM, loads, N, perfect+imperfect CSI, with CIs.
One channel draw serves both constellations.  Single precision (checked against
float64 at the start of the run).   usage: python jobC2.py <N>"""
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

def boot(ev, nv, nb=4000):
    ev = np.asarray(ev, float); nv = np.asarray(nv, float); G = len(ev)
    idx = rng.integers(0, G, (nb, G))
    bs = ev[idx].sum(1)/nv[idx].sum(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def draw(B, N, K, s2, tau2, seed, M, rd, cd):
    g = torch.Generator(); g.manual_seed(seed)
    lv = torch.tensor(L.qam(M)[0], dtype=rd); m = len(lv)
    H = (torch.randn(B, N, K, dtype=rd, generator=g) +
         1j*torch.randn(B, N, K, dtype=rd, generator=g))/np.sqrt(2)
    if tau2 > 0:
        e = (torch.randn(B, N, K, dtype=rd, generator=g) +
             1j*torch.randn(B, N, K, dtype=rd, generator=g))*np.sqrt(tau2/2)
        Hh = H + e
    else:
        Hh = H
    Hc = Hh.conj().transpose(-2, -1).resolve_conj()
    A = torch.matmul(Hc, Hh) + s2*torch.eye(K, dtype=cd)
    return H, Hh, Hc, A, g, lv, m

def syms(B, K, lv, m, g, rd, cd):
    ir = torch.randint(0, m, (B, K), generator=g); iq = torch.randint(0, m, (B, K), generator=g)
    return (lv[ir] + 1j*lv[iq]).to(cd), ir, iq

R = {"meta": {"N": N, "groups": NG, "frames_per_group": GSZ, "T": TS, "tau2": TAU,
              "snr_grid": SNR, "dtype": "complex64",
              "metric": "SNR = N/sigma^2 (per-user matched-filter SNR, dB)",
              "note": "H drawn once per (load,snr,tau,group) and reused for 16- and 64-QAM"}}

# ---- precision self-check ----------------------------------------------------
chk = {}
K = int(round(0.5*N)); s2 = N/10**(2.2)
for nm, (rd, cd) in (("f32", (torch.float32, torch.complex64)),
                     ("f64", (torch.float64, torch.complex128))):
    H, Hh, Hc, A, g, lv, m = draw(GSZ, N, K, s2, 0.01, 777, 16, rd, cd)
    x, ir, iq = syms(GSZ, K, lv, m, g, rd, cd)
    y = L.mv(H, x) + (torch.randn(GSZ, N, dtype=rd, generator=g) +
                      1j*torch.randn(GSZ, N, dtype=rd, generator=g))*np.sqrt(s2/2)
    b = torch.matmul(Hc, y.unsqueeze(-1)).squeeze(-1)
    st = L.mp_opt_steps(0.5, N*1.01, s2, 4)[0]
    xm = L.mmse(A, b)
    e1, n1 = L.bit_errors(L.rich(A, b, [float(a) for a in st]), ir, iq, 16)
    e2, n2 = L.bit_errors(xm, ir, iq, 16)
    chk[nm] = {"mpopt_ber": e1/n1, "lmmse_ber": e2/n2, "errors": [e1, e2]}
R["precision_check"] = chk
print("precision check", chk, flush=True)
json.dump(R, open(f"r1_C_ber_N{N}.json", "w"), indent=1)

# ---- main sweep ---------------------------------------------------------------
for load in LOADS:
    K = int(round(load*N)); beta = K/N
    for snr in SNR:
        s2 = N/10**(snr/10.0)
        for tau2 in TAU:
            Neff = N*(1+tau2)
            stp = {T: {"Richardson": L.const_steps(beta, Neff, s2, T),
                       "Chebyshev": L.cheb_steps(beta, Neff, s2, T),
                       "MP-opt": L.mp_opt_steps(beta, Neff, s2, T)[0]} for T in TS}
            names = ["LMMSE"] + [f"{m}-T{T}" for T in TS
                                 for m in ("Richardson", "Chebyshev", "MP-opt", "CG")]
            acc = {M: {n: {"e": [], "n": [], "d": []} for n in names} for M in (16, 64)}
            for gi in range(NG):
                H, Hh, Hc, A, g, _, _ = draw(GSZ, N, K, s2, tau2, 100000+gi, 16, RD, CD)
                nz = (torch.randn(GSZ, N, dtype=RD, generator=g) +
                      1j*torch.randn(GSZ, N, dtype=RD, generator=g))*np.sqrt(s2/2)
                for M in (16, 64):
                    lv = torch.tensor(L.qam(M)[0], dtype=RD); m = len(lv)
                    x, ir, iq = syms(GSZ, K, lv, m, g, RD, CD)
                    y = L.mv(H, x) + nz
                    b = torch.matmul(Hc, y.unsqueeze(-1)).squeeze(-1)
                    xm = L.mmse(A, b); den = (xm.abs()**2).sum(1)
                    def rec(n, xh):
                        e, nb_ = L.bit_errors(xh, ir, iq, M)
                        acc[M][n]["e"].append(e); acc[M][n]["n"].append(nb_)
                        acc[M][n]["d"].append(float((((xh-xm).abs()**2).sum(1)/den).mean().item()))
                    rec("LMMSE", xm)
                    for T in TS:
                        for mm, st in stp[T].items():
                            rec(f"{mm}-T{T}", L.rich(A, b, [float(a) for a in st]))
                        rec(f"CG-T{T}", L.cg(A, b, T))
            for M in (16, 64):
                row = {}
                for n in names:
                    te, tn = int(sum(acc[M][n]["e"])), int(sum(acc[M][n]["n"]))
                    lo, hi = boot(acc[M][n]["e"], acc[M][n]["n"])
                    row[n] = {"ber": te/tn, "ci95": [lo, hi], "errors": te, "bits": tn,
                              "dmmse": float(np.mean(acc[M][n]["d"]))}
                R[f"L{load}_M{M}_snr{snr}_tau{tau2}"] = row
            print(N, load, snr, tau2, "%.1fs" % (time.time()-t0), flush=True)
        json.dump(R, open(f"r1_C_ber_N{N}.json", "w"), indent=1)
json.dump(R, open(f"r1_C_ber_N{N}.json", "w"), indent=1)
print("DONE C N=%d" % N, time.time()-t0, flush=True)
