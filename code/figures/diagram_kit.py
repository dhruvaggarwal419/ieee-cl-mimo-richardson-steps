import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
plt.rcParams['font.family']='serif'; plt.rcParams['mathtext.fontset']='dejavuserif'
PAL={'in':'#DAE8FC','stat':'#D5E8D4','theory':'#FFE6CC','est':'#E1D5E7','out':'#F8CECC','warn':'#FFF2CC','grp':'#F5F5F5','dec':'#DAE8FC'}
class Diagram:
    def __init__(s,w=7.16,h=3.1,xmax=100,ymax=44):
        s.fig=plt.figure(figsize=(w,h)); s.ax=s.fig.add_axes([0,0,1,1]); s.ax.set_xlim(0,xmax); s.ax.set_ylim(0,ymax); s.ax.axis('off'); s.xmax=xmax; s.ymax=ymax
    def group(s,x,y,w,h,title,c='grp',tc='#222',fs=6.8,ls='-'):
        s.ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=1.2',fc=PAL.get(c,c),ec='#333',lw=0.9,ls=ls))
        s.ax.text(x+w/2,y+h-0.9,title,ha='center',va='top',fontsize=fs,fontweight='bold',color=tc)
    def box(s,x,y,w,h,t,c='stat',fs=6.0,bold=False,ec='#333'):
        s.ax.add_patch(Rectangle((x,y),w,h,fc=PAL.get(c,c),ec=ec,lw=0.8)); s.ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,fontweight='bold' if bold else 'normal',linespacing=1.15)
    def arrow(s,x1,y1,x2,y2,dashed=False,lw=0.9,color='#111'):
        assert abs(x1-x2)<1e-9 or abs(y1-y2)<1e-9,f'diagonal arrow {(x1,y1,x2,y2)}'
        s.ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',lw=lw,color=color,ls='--' if dashed else '-',shrinkA=0,shrinkB=0,mutation_scale=7))
    def poly(s,pts,dashed=False,lw=0.9,color='#111'):
        for (a,b),(c,d) in zip(pts[:-2],pts[1:-1]): assert a==c or b==d,'non-axis segment'
        for (x1,y1),(x2,y2) in zip(pts[:-2],pts[1:-1]): s.line([x1,x2],[y1,y2],dashed,lw,color)
        (x1,y1),(x2,y2)=pts[-2],pts[-1]; assert x1==x2 or y1==y2,'non-axis arrow'; s.arrow(x1,y1,x2,y2,dashed,lw,color)
    def line(s,xs,ys,dashed=False,lw=0.9,color='#111'):
        s.ax.plot(xs,ys,color=color,lw=lw,ls='--' if dashed else '-',solid_capstyle='butt')
    def text(s,x,y,t,fs=6,ha='center',va='center',color='#111',bold=False,style='normal'):
        s.ax.text(x,y,t,ha=ha,va=va,fontsize=fs,color=color,fontweight='bold' if bold else 'normal',fontstyle=style,linespacing=1.15)
    def inset(s,x,y,w,h):
        W,Hh=s.fig.get_size_inches(); ax=s.fig.add_axes([x/s.xmax,y/s.ymax,w/s.xmax,h/s.ymax])
        ax.tick_params(which='both',labelsize=4.5,length=1.5,pad=1,width=0.4); [sp.set_linewidth(0.5) for sp in ax.spines.values()]
        ax.patch.set_facecolor('white'); return ax
    def legend(s,items,y=1.6,x0=4,dx=None):
        n=len(items); dx=dx or (s.xmax-2*x0)/n
        for i,(c,l) in enumerate(items):
            x=x0+i*dx; s.ax.add_patch(Rectangle((x,y-0.9),2.2,1.8,fc=PAL.get(c,c),ec='#333',lw=0.6)); s.ax.text(x+3,y,l,va='center',fontsize=5.8)
    def save(s,path,preview=None):
        s.fig.savefig(path,bbox_inches='tight',pad_inches=0.02)
        if preview: s.fig.savefig(preview,dpi=130)
