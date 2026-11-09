"""Shared machinery for the Comm-Letters reviewer experiments (r1mimo)."""
import numpy as np, torch, time
from numpy.polynomial import chebyshev as C

dev = "cpu"
CD, RD = torch.complex128, torch.float64

# ---------------------------------------------------------------- step design
def leja(z):
    z = [float(v) for v in np.real(np.asarray(z))]
    out = [max(z, key=abs)]; z.remove(out[0])
    while z:
        n = max(z, key=lambda v: float(np.sum(np.log(np.abs(np.array(out) - v) + 1e-300))))
        out.append(n); z.remove(n)
    return np.array(out)

_lg = {}
def _leggauss(M):
    if M not in _lg: _lg[M] = np.polynomial.legendre.leggauss(M)
    return _lg[M]

_mpn = {}
def mp_nodes(beta, M=4000):
    if (beta, M) in _mpn: return _mpn[(beta, M)]
    r = _mp_nodes(beta, M); _mpn[(beta, M)] = r; return r

def _mp_nodes(beta, M=4000):
    a, b = (1-np.sqrt(beta))**2, (1+np.sqrt(beta))**2
    t, wt = _leggauss(M)
    th = (t+1)*np.pi/2; w = wt*np.pi/2
    x = (a+b)/2 - (b-a)/2*np.cos(th)
    dens = np.sqrt(np.maximum((b-x)*(x-a), 0))/(2*np.pi*beta*x)
    jac = (b-a)/2*np.sin(th)
    return x, w*dens*jac, a, b

def edges(beta, N, s2):
    a, b = (1-np.sqrt(beta))**2, (1+np.sqrt(beta))**2
    return N*a+s2, N*b+s2

def opt_steps_measure(lam, wq, s2, T, lo=None, hi=None):
    """Minimise  sum_i wq_i w_i r(lam_i)^2  over deg-T r with r(0)=1.
    lam_i are eigenvalues of A = H^H H + s2 I ; w_i = g/(g+s2) with g = lam-s2.
    Returns (steps leja-ordered, objective, roots, max |Im root|)."""
    lam = np.asarray(lam, float); wq = np.asarray(wq, float)
    g = np.maximum(lam - s2, 0.0)
    w = g/lam
    if lo is None: lo = float(lam.min())
    if hi is None: hi = float(lam.max())
    if hi <= lo: hi = lo*1.0001 + 1e-12
    sc = lambda l: (2*l-(lo+hi))/(hi-lo)
    T0 = np.array([C.chebval(sc(0.0), [0]*k+[1]) for k in range(T+1)])
    cols = [C.chebval(sc(lam), [0]*k+[1]) - T0[k] for k in range(1, T+1)]
    P = np.stack(cols, 1); sw = np.sqrt(wq*w)
    c, *_ = np.linalg.lstsq(P*sw[:, None], -sw, rcond=None)
    coef = np.zeros(T+1); coef[0] = 1.0
    for k in range(1, T+1):
        coef[k] += c[k-1]; coef[0] -= c[k-1]*T0[k]
    rts = C.chebroots(coef)
    imax = float(np.max(np.abs(np.imag(rts)))) if len(rts) else 0.0
    lr = (np.real(rts)*(hi-lo)+(lo+hi))/2
    obj = float(np.sum(wq*w*C.chebval(sc(lam), coef)**2))
    return 1.0/leja(lr), obj, np.sort(lr), imax

def mp_opt_steps(beta, N, s2, T, M=4000):
    x, wq, a, b = mp_nodes(beta, M)
    lam = N*x + s2
    lo, hi = N*a+s2, N*b+s2
    st, obj, rts, im = opt_steps_measure(lam, wq, s2, T, lo, hi)
    return st, obj

def cheb_steps(beta, N, s2, T):
    lo, hi = edges(beta, N, s2)
    k = np.arange(1, T+1)
    roots = (hi+lo)/2 - (hi-lo)/2*np.cos((2*k-1)*np.pi/(2*T))
    return 1.0/leja(roots)

def const_steps(beta, N, s2, T):
    lo, hi = edges(beta, N, s2)
    return np.full(T, 2/(lo+hi))

def poly_obj(steps, beta, N, s2, M=4000):
    x, wq, a, b = mp_nodes(beta, M); g = N*x; lam = g+s2
    w = g/lam
    r = np.prod([1-al*lam for al in steps], axis=0)
    return float(np.sum(wq*w*r**2))

def obj_measure(steps, lam, wq, s2):
    lam = np.asarray(lam, float); wq = np.asarray(wq, float)
    g = np.maximum(lam-s2, 0.0); w = g/lam
    r = np.prod([1-al*lam for al in steps], axis=0)
    return float(np.sum(wq*w*r**2))

# ------------------------------------------------------- finite-N empirical law
def emp_eigs(N, K, nreal=400, seed=7, dtype=torch.float64, chunk=100):
    """Pooled eigenvalues of H^H H, H: N x K iid CN(0,1)."""
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    out = []
    done = 0
    while done < nreal:
        B = min(chunk, nreal-done); done += B
        H = (torch.randn(B, N, K, dtype=dtype, generator=g) +
             1j*torch.randn(B, N, K, dtype=dtype, generator=g))/np.sqrt(2)
        G = torch.matmul(H.conj().transpose(-2, -1).resolve_conj(), H)
        out.append(torch.linalg.eigvalsh(G).reshape(-1).numpy())
    return np.concatenate(out)

def finite_steps(N, K, s2, T, nreal=400, seed=7, cache={}):
    key = (N, K, nreal, seed)
    if key not in cache:
        cache[key] = emp_eigs(N, K, nreal, seed)
    e = cache[key]
    lam = e + s2
    wq = np.full(lam.shape, float(K)/len(lam))   # total mass K -> obj/K is per-user
    st, obj, rts, im = opt_steps_measure(lam, wq, s2, T)
    return st, obj/K, im          # per-user objective

def chan_steps(eigs, s2, T):
    """Steps optimal for ONE realization, given its exact Gram eigenvalues."""
    lam = np.asarray(eigs, float) + s2
    wq = np.full(lam.shape, 1.0/len(lam))
    st, obj, rts, im = opt_steps_measure(lam, wq, s2, T)
    return st, obj, im

# ------------------------------------------------------------------ simulation
def qam(M):
    m = int(round(np.sqrt(M))); nb = int(round(np.log2(m)))
    lv = (2*np.arange(m) - (m-1)).astype(float)
    lv = lv/np.sqrt((lv**2).mean()*2)
    gi = np.arange(m) ^ (np.arange(m) >> 1)
    bits = ((gi[:, None] >> np.arange(nb-1, -1, -1)[None, :]) & 1)
    return lv, bits, nb

def gen(B, N, K, s2, g=None, M=16, tau2=0.0, dtype=RD):
    cdt = CD if dtype == RD else torch.complex64
    lv, bits, nb = qam(M); m = len(lv)
    LV = torch.tensor(lv, dtype=dtype)
    H = (torch.randn(B, N, K, dtype=dtype, generator=g) +
         1j*torch.randn(B, N, K, dtype=dtype, generator=g))/np.sqrt(2)
    ir = torch.randint(0, m, (B, K), generator=g); iq = torch.randint(0, m, (B, K), generator=g)
    x = (LV[ir] + 1j*LV[iq]).to(cdt)
    n = (torch.randn(B, N, dtype=dtype, generator=g) +
         1j*torch.randn(B, N, dtype=dtype, generator=g))*np.sqrt(s2/2)
    y = torch.matmul(H, x.unsqueeze(-1)).squeeze(-1) + n
    if tau2 > 0:
        e = (torch.randn(B, N, K, dtype=dtype, generator=g) +
             1j*torch.randn(B, N, K, dtype=dtype, generator=g))*np.sqrt(tau2/2)
        Hh = H + e
    else:
        Hh = H
    Hc = Hh.conj().transpose(-2, -1).resolve_conj()
    A = torch.matmul(Hc, Hh) + s2*torch.eye(K, dtype=cdt)
    b = torch.matmul(Hc, y.unsqueeze(-1)).squeeze(-1)
    return x, A, b, ir, iq

def mv(A, v):
    return torch.matmul(A, v.unsqueeze(-1)).squeeze(-1)

def gram(H):
    return torch.matmul(H.conj().transpose(-2, -1).resolve_conj(), H)

def rich(A, b, steps):
    x = torch.zeros_like(b)
    for al in steps:
        x = x + al*(b - mv(A, x))
    return x

def rich_batched(A, b, steps):
    """steps: (B,T) tensor, one step list per channel."""
    x = torch.zeros_like(b)
    for t in range(steps.shape[1]):
        x = x + steps[:, t, None].to(x.dtype)*(b - mv(A, x))
    return x

def cg(A, b, T):
    x = torch.zeros_like(b); r = b.clone(); p = r.clone(); rs = (r.conj()*r).sum(1).real
    for _ in range(T):
        Ap = mv(A, p); a = rs/(p.conj()*Ap).sum(1).real
        x = x + a[:, None]*p; r = r - a[:, None]*Ap; rn = (r.conj()*r).sum(1).real
        p = r + (rn/rs)[:, None]*p; rs = rn
    return x

def mmse(A, b):
    # A = H^H H + s2 I is Hermitian positive definite -> Cholesky.
    # (MKL's batched complex LU, zgetrf/ZLASWP, fails on this machine for K >= ~230.)
    Lc = torch.linalg.cholesky(A)
    return torch.cholesky_solve(b.unsqueeze(-1), Lc).squeeze(-1)

def bit_errors(xh, ir, iq, M=16):
    lv, bits, nb = qam(M); m = len(lv)
    d = lv[1]-lv[0]
    BT = torch.tensor(bits, dtype=torch.long)
    def dec(v):
        return torch.clamp(torch.round(v/d + (m-1)/2.0).long(), 0, m-1)
    dr, di = dec(xh.real.double()), dec(xh.imag.double())
    e = (BT[dr] != BT[ir]).sum() + (BT[di] != BT[iq]).sum()
    return int(e.item()), int(ir.numel()*2*nb)
