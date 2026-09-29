"""Diagnose whether a fixed linear filter leaves predictable structure."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs'/'diagnostics'
FS=48000

def frames(x,n=8192,hop=4096):
    return np.lib.stride_tricks.sliding_window_view(x,n)[::hop]

def ratio_curve(x,y,selection):
    f=frames(x)[selection]*np.hanning(8192)
    g=frames(y)[selection]*np.hanning(8192)
    sx=np.median(abs(np.fft.rfft(f,axis=1))**2,axis=0)
    sy=np.median(abs(np.fft.rfft(g,axis=1))**2,axis=0)
    hz=np.fft.rfftfreq(8192,1/FS)
    db=10*np.log10(np.maximum(sy,1e-20)/np.maximum(sx,1e-20))
    db-=np.median(db[(hz>=500)&(hz<=2000)])
    return hz,db

def summary_for(x,y,name):
    f,coh=signal.coherence(x,y,fs=FS,nperseg=8192,noverlap=4096,window='hann')
    _,h=signal.csd(y,x,fs=FS,nperseg=8192,noverlap=4096,window='hann')
    _,px=signal.welch(x,fs=FS,nperseg=8192,noverlap=4096,window='hann')
    H=h/np.maximum(px,1e-20)
    bands={}
    for key,lo,hi in [('70_250',70,250),('250_500',250,500),('500_2000',500,2000),('2000_4000',2000,4000),('4000_8000',4000,8000)]:
        use=(f>=lo)&(f<hi)
        bands[key]=dict(coherence=float(np.median(coh[use])), magnitude_db=float(np.median(20*np.log10(abs(H[use])+1e-20))), phase_deg=float(np.median(np.unwrap(np.angle(H[use])))*180/np.pi))
    return f,coh,H,bands

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=np.load(ROOT/'data/aligned.npz');x,y=d['x'],d['y']
    info=json.loads((ROOT/'data/alignment.json').read_text())
    splits={k:tuple(v) for k,v in info['splits'].items()}
    result={}
    curves={}
    for name,(a,b) in splits.items():
        f,coh,H,bands=summary_for(x[a:b],y[a:b],name)
        result[name]=dict(bands=bands)
        # RMS tiers are based on iPhone frames; silence is excluded first.
        xx=frames(x[a:b]); rms=np.sqrt(np.mean(xx**2,axis=1)); active=rms>max(np.quantile(rms,.15),1e-9)
        q=np.quantile(rms[active],[1/3,2/3])
        tiers={'low':active&(rms<=q[0]),'mid':active&(rms>q[0])&(rms<=q[1]),'high':active&(rms>q[1])}
        tier_data={}
        for tier,sel in tiers.items():
            hz,db=ratio_curve(x[a:b],y[a:b],sel)
            tier_data[tier]=dict(frames=int(sel.sum()),relative_curve_db=db.tolist())
        result[name]['level_tiers']=tier_data
        curves[name]=(f,coh,H,hz,tier_data)
    # plots use validation only so model decisions remain independent of test.
    f,coh,H,hz,tiers=curves['validation']
    fig,ax=plt.subplots(2,1,figsize=(11,7),sharex=True,constrained_layout=True)
    ax[0].semilogx(f,coh,color='#3973ac');ax[0].axhline(.8,color='gray',ls='--',lw=.8);ax[0].set(ylabel='Magnitude-squared coherence',ylim=(0,1.02),title='Validation 51–60 s: iPhone ↔ KM184 linear relationship')
    for name,color in [('low','#6baed6'),('mid','#31a354'),('high','#e6550d')]: ax[1].semilogx(hz,tiers[name]['relative_curve_db'],label=f'{name} RMS ({tiers[name]["frames"]} frames)',color=color)
    ax[1].axhline(0,color='black',lw=.7);ax[1].set(xlim=(40,18000),ylim=(-20,20),xlabel='Hz',ylabel='KM184/iPhone relative dB',title='Level-tier relative spectra (validation)');ax[1].legend()
    fig.savefig(OUT/'coherence-level-validation.png',dpi=160)
    (OUT/'transfer-diagnostics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v['bands'] for k,v in result.items()},indent=2))
if __name__=='__main__':main()
