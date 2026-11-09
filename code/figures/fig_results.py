import json,numpy as np,matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family']='serif'; plt.rcParams['font.size']=7.5
F=json.load(open('r1_F_largeT.json'))['N64_K32']['T']
B=json.load(open('r1_C_ber_N128.json'))
fig,ax=plt.subplots(1,2,figsize=(7.16,2.45))
a=ax[0]; Ts=[int(t) for t in F.keys()]
for k,l,c,ls in [('MP-opt','Marchenko-Pastur design','#C62828','-'),('Finite-N','Simulated-spectrum design','#2E7D32','-'),('Oracle-chan','Per-channel genie','#6A1B9A',':'),('Chebyshev','Chebyshev','#1565C0','--')]:
    y=[F[str(t)][k]['dmmse'] for t in Ts]; a.semilogy(Ts,y,ls,color=c,lw=1.3,marker='o',ms=2.5,label=l)
a.set_xlabel('Iterations $T$'); a.set_ylabel('Relative error to LMMSE'); a.grid(alpha=0.25,lw=0.4); a.legend(fontsize=6,frameon=False,loc='lower left'); a.set_title('(a) $N=64$, $K=32$, 20 dB',fontsize=8)
b=ax[1]; snr=B['meta']['snr_grid']; fl=1e-5
for k,l,c,ls in [('Chebyshev-T4','Chebyshev $T{=}4$','#1565C0','--'),('CG-T4','CG $T{=}4$','#F9A825','-.'),('MP-opt-T4','Proposed $T{=}4$','#C62828','-'),('MP-opt-T8','Proposed $T{=}8$','#C62828',':'),('LMMSE','LMMSE','#333','-')]:
    y=np.array([B[f'L0.5_M16_snr{s}_tau0.0'][k]['ber'] for s in snr]); y=np.maximum(y,fl)
    lo=np.array([B[f'L0.5_M16_snr{s}_tau0.0'][k]['ci95'][0] for s in snr]); hi=np.array([B[f'L0.5_M16_snr{s}_tau0.0'][k]['ci95'][1] for s in snr])
    b.semilogy(snr,y,ls,color=c,lw=1.3,marker='o',ms=2.5,label=l); b.fill_between(snr,np.maximum(lo,fl),np.maximum(hi,fl),color=c,alpha=0.2,lw=0)
b.set_xlabel('SNR (dB)'); b.set_ylabel('Bit error rate'); b.grid(alpha=0.25,lw=0.4); b.legend(fontsize=6,frameon=False,loc='lower left'); b.set_title('(b) $N=128$, $K=64$, 16-QAM',fontsize=8)
b.set_ylim(5e-6,0.3)
fig.tight_layout(pad=0.5,w_pad=1.2); fig.savefig('../fig_results.pdf',bbox_inches='tight',pad_inches=0.02); fig.savefig('/tmp/l1v2_res.png',dpi=120)
