import json,numpy as np,sys
sys.path.insert(0,'.')
import mpopt; mpopt.mp_nodes.__defaults__=(1200,)
from mpopt import *
from diagram_kit import *
rng=np.random.default_rng(1)
N,K=128,64; s2=N/100.; beta=K/N; T=4
mp,_=mp_opt_steps(beta,N,s2,T); ch=cheb_steps(beta,N,s2,T)
lo,hi=edges(beta,N,s2)
# eigenvalues of A=H^H H+s2 I from simulated channels
ev=[]
for _ in range(40):
    Hm=(rng.standard_normal((N,K))+1j*rng.standard_normal((N,K)))/np.sqrt(2); ev.append(np.linalg.eigvalsh(Hm.conj().T@Hm)+s2)
ev=np.concatenate(ev)
xs=np.linspace(lo,hi,300); a_=(1-np.sqrt(beta))**2; b_=(1+np.sqrt(beta))**2
x=(xs-s2)/N; dens=np.sqrt(np.clip((b_-x)*(x-a_),0,None))/(2*np.pi*beta*x)/N
lam=np.linspace(0,hi*1.08,400); r=lambda st: np.prod([1-a*lam for a in st],axis=0)
F=json.load(open('r1_F_largeT.json'))['N64_K32']['T']; Ts=sorted(int(t) for t in F)
D=Diagram(7.16,3.35,100,46)
Y0,YH=10.5,32
D.group(0.8,Y0,13.2,YH,'1. Known',c='in',fs=6.4)
D.group(15.5,Y0,46,YH,'2. Design against the spectrum (offline, no data)',c='warn',fs=6.0)
D.group(63.3,Y0,15.2,YH,'3. Detector (online)',c='stat',fs=6.0)
D.group(80.3,Y0,18.9,YH,'4. Result',c='out',fs=6.4)
# inputs
D.box(2.3,31.5,10.2,5.5,'Antennas $N$\nusers $K$','white',5.6); D.box(2.3,23.0,10.2,5.5,'Noise\nvariance $\\sigma^2$','white',5.6); D.box(2.3,14.5,10.2,5.5,'Iterations\n$T$','white',5.6)
# design: two routes
ax=D.inset(17.2,28.8,12.2,8.5); ax.hist(ev,bins=40,density=True,color='#9DB9D8',ec='none'); ax.plot(xs,dens,color='#B85450',lw=1); ax.set_xticks([]); ax.set_yticks([]); ax.set_title('closed form: MP law',fontsize=5,pad=1)
ax=D.inset(17.2,14.2,12.2,10.3); ax.hist(ev,bins=40,density=True,color='#82B366',ec='none'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title('simulated: pooled eigenvalues',fontsize=4.8,pad=1)
D.box(32.3,22.0,12.5,11.5,'Kernel polynomial\nweight $g/(g+\\sigma^2)$\nroots $z_t$','white',5.3)
D.box(47.4,30.0,12.8,6.0,'Steps\n$\\alpha_t=1/z_t$','white',5.4)
ax=D.inset(46.8,14.2,13.4,11.5); ax.axhline(0,color='#555',lw=0.4); ax.plot(lam,r(ch),color='#1565C0',lw=0.9,label='Cheb.'); ax.plot(lam,r(mp),color='#B85450',lw=1.2,label='design'); ax.set_xticks([]); ax.set_yticks([0,1]); ax.set_ylim(-0.9,1.05); ax.legend(fontsize=4.2,frameon=False,loc='upper right',handlelength=1.1,borderpad=0.1); ax.set_title('residual polynomial $r_T(\\lambda)$',fontsize=5,pad=1)
D.arrow(12.7,34.0,17.0,34.0); D.arrow(12.7,23.8,17.0,23.8)
D.arrow(29.5,31.0,32.1,31.0); D.arrow(29.5,23.5,32.1,23.5)
D.arrow(44.9,32.4,47.3,32.4); D.arrow(53.8,29.9,53.8,27.6)
# detector
D.box(64.8,28.2,12.2,8.2,'$x_t=x_{t-1}+\\alpha_t$\n$(b-A\\,x_{t-1})$','white',5.2)
D.text(70.9,24.2,'$t=1,\\dots,T$',fs=5.2)
D.text(70.9,18.4,'one product with $A$\nper step, no inner\nproducts, no inverse',fs=4.9)
D.arrow(60.4,33.0,64.7,33.0)
D.text(70.9,13.2,'Leja order of steps',fs=4.8,style='italic')
# results
ax=D.inset(83.2,18.6,14.4,17.4)
for k,l,c,ls in [('MP-opt','MP design','#B85450','-'),('Finite-N','finite-size','#2E7D32','-'),('Chebyshev','Chebyshev','#1565C0','--')]:
    ax.semilogy(Ts,[F[str(t)][k]['dmmse'] for t in Ts],ls,color=c,lw=1.0,label=l)
ax.set_title('error to LMMSE vs $T$',fontsize=5,pad=1); ax.legend(fontsize=4.0,frameon=False,loc='lower left',handlelength=1.1,borderpad=0.1,labelspacing=0.15); ax.set_xticks([10,20,30]); ax.set_yticks([1e-9,1e-5,1]); ax.set_ylim(1e-10,30)
D.text(89.8,13.6,'MP design diverges\nbeyond $T\\approx10$;\nsimulated spectrum fixes it',fs=4.5,color='#B85450',bold=True)
D.arrow(78.0,31.0,83.0,31.0,dashed=True)
D.legend([('in','Known inputs'),('warn','Offline design'),('stat','Online detector'),('out','Outcome')],y=5.0,x0=3,dx=24)
D.text(50,1.6,'At $T=4$ the error to LMMSE is 2.6 to 3.3 times below Chebyshev steps. The finite-size design stays stable for large $T$.',fs=5.4,style='italic')
D.save('../fig_overview.pdf','/tmp/l1v2_ov.png')
