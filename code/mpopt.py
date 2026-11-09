import numpy as np
from numpy.polynomial import chebyshev as C

def mp_nodes(beta, M=4000):
    # Marchenko-Pastur law of eig(H^H H / N) for H: N x K, beta=K/N<=1
    a, b = (1-np.sqrt(beta))**2, (1+np.sqrt(beta))**2
    t, wt = np.polynomial.legendre.leggauss(M)
    # substitution x = (a+b)/2 - (b-a)/2 cos(theta) removes edge singularity
    th = (t+1)*np.pi/2; w = wt*np.pi/2
    x = (a+b)/2 - (b-a)/2*np.cos(th)
    dens = np.sqrt(np.maximum((b-x)*(x-a),0))/(2*np.pi*beta*x)
    jac = (b-a)/2*np.sin(th)
    return x, w*dens*jac, a, b

def edges(beta, N, s2):
    a, b = (1-np.sqrt(beta))**2, (1+np.sqrt(beta))**2
    return N*a+s2, N*b+s2

def leja(z):
    z = list(z); out = [max(z, key=abs)]; z.remove(out[0])
    while z:
        n = max(z, key=lambda v: np.prod([abs(v-o) for o in out])); out.append(n); z.remove(n)
    return np.array(out)

def steps_from_roots(roots):
    return 1.0/leja(np.real(roots))

def mp_opt_steps(beta, N, s2, T):
    """Residual polynomial r(0)=1 minimising  E_MP[ w(g) r(g+s2)^2 ],
    w(g)=(g^2+s2 g)/(g+s2)^2  (error to the MMSE solution)."""
    x, wq, a, b = mp_nodes(beta)
    g = N*x; lam = g + s2
    w = (g**2 + s2*g)/lam**2
    lo, hi = N*a+s2, N*b+s2
    s = lambda l: (2*l-(lo+hi))/(hi-lo)
    T0 = [C.chebval(s(0.0), [0]*k+[1]) for k in range(T+1)]
    cols = [C.chebval(s(lam), [0]*k+[1]) - T0[k] for k in range(1, T+1)]
    P = np.stack(cols, 1); sw = np.sqrt(wq*w)
    c, *_ = np.linalg.lstsq(P*sw[:,None], -sw*1.0, rcond=None)
    coef = np.zeros(T+1); coef[0] = 1.0
    for k in range(1, T+1):
        e = np.zeros(k+1); e[k] = 1; coef[:k+1] += c[k-1]*(e) ; coef[0] -= c[k-1]*T0[k]
    # coef are Chebyshev coefficients of r in variable s
    rts = C.chebroots(coef)
    lam_roots = (rts*(hi-lo) + (lo+hi))/2
    obj = float(np.sum(wq*w*C.chebval(s(lam), coef)**2))
    return steps_from_roots(lam_roots), obj

def cheb_steps(beta, N, s2, T):
    lo, hi = edges(beta, N, s2)
    k = np.arange(1, T+1)
    roots = (hi+lo)/2 - (hi-lo)/2*np.cos((2*k-1)*np.pi/(2*T))
    return steps_from_roots(roots)

def const_steps(beta, N, s2, T):
    lo, hi = edges(beta, N, s2)
    return np.full(T, 2/(lo+hi))

def poly_obj(steps, beta, N, s2):
    x, wq, a, b = mp_nodes(beta); g=N*x; lam=g+s2
    w=(g**2+s2*g)/lam**2
    r=np.prod([1-al*lam for al in steps],axis=0)
    return float(np.sum(wq*w*r**2))

if __name__=="__main__":
    N,beta,s2,T=128,0.5,0.1,4
    for nm,f in [("const",const_steps),("cheb",cheb_steps)]:
        st=f(beta,N,s2,T); print(nm, poly_obj(st,beta,N,s2))
    st,ob=mp_opt_steps(beta,N,s2,T); print("mp",ob,poly_obj(st,beta,N,s2),st)
