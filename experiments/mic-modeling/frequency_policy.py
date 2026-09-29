"""Derive per-recording frequency-of-interest policy before neural training.

The output is not a microphone confidence interval. It describes whether the
*present paired recording* contains enough stable evidence for a linear or
neural correction in each 1/6-octave band.
"""
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

def smooth_log(f, v, centers):
    return np.array([np.median(v[(f>=c*2**(-1/12))&(f<=c*2**(1/12))]) for c in centers])

def active_support(x):
    n,hop=8192,4096
    frames=np.lib.stride_tricks.sliding_window_view(x,n)[::hop]
    p=abs(np.fft.rfft(frames*np.hanning(n),axis=1))**2
    level=np.mean(frames**2,axis=1)
    active=level>max(np.quantile(level,.3),1e-9)
    live=np.median(p[active],axis=0); quiet=np.median(p[~active],axis=0)
    f=np.fft.rfftfreq(n,1/FS)
    contrast=10*np.log10(np.maximum(live,1e-20)/np.maximum(quiet,1e-20))
    prevalence=np.mean(p[active]>4*quiet,axis=0)
    support=np.clip((contrast-3)/15,0,1)*np.clip((prevalence-.2)/.6,0,1)
    return f,np.sqrt(support)

def tier_spread(x,y):
    n,hop=8192,4096
    xf=np.lib.stride_tricks.sliding_window_view(x,n)[::hop]; yf=np.lib.stride_tricks.sliding_window_view(y,n)[::hop]
    rms=np.sqrt(np.mean(xf**2,axis=1)); active=rms>max(np.quantile(rms,.15),1e-9)
    q=np.quantile(rms[active],[1/3,2/3]); groups=[active&(rms<=q[0]),active&(rms>q[0])&(rms<=q[1]),active&(rms>q[1])]
    curves=[]; f=np.fft.rfftfreq(n,1/FS)
    for g in groups:
        a=np.median(abs(np.fft.rfft(xf[g]*np.hanning(n),axis=1))**2,axis=0)
        b=np.median(abs(np.fft.rfft(yf[g]*np.hanning(n),axis=1))**2,axis=0)
        r=10*np.log10(np.maximum(b,1e-20)/np.maximum(a,1e-20)); r-=np.median(r[(f>=500)&(f<=2000)])
        curves.append(r)
    return f,np.std(np.stack(curves),axis=0)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=np.load(ROOT/'data/aligned.npz'); info=json.loads((ROOT/'data/alignment.json').read_text())
    a,b=info['splits']['validation']; x,y=d['x'][a:b],d['y'][a:b]
    f,support=active_support(x)
    fc,coh=signal.coherence(x,y,fs=FS,nperseg=8192,noverlap=4096,window='hann')
    fs,spread=tier_spread(x,y)
    centers=np.geomspace(40,18000,96)
    supp=smooth_log(f,support,centers); coh=smooth_log(fc,coh,centers); spread=smooth_log(fs,spread,centers)
    # Neural residual is permitted only where all evidence is strong. Linear
    # correction can use a wider band but keeps the existing FIR gain limits.
    policy=np.where((supp>=.45)&(coh>=.75)&(spread<=1.5),'neural_residual',
            np.where(supp>=.25,'bounded_linear','conservative_reference'))
    payload=dict(method='Validation-only, 1/6-octave bands. support is active-vs-quiet programme evidence; coherence is Welch magnitude-squared coherence; tier_spread is RMS low/mid/high transfer-curve standard deviation in dB. These are recording evidence measures, not microphone specifications.',
        thresholds=dict(neural_residual=dict(min_support=.45,min_coherence=.75,max_tier_spread_db=1.5),bounded_linear=dict(min_support=.25)),
        bands=[dict(center_hz=float(c),support=float(s),coherence=float(q),tier_spread_db=float(t),policy=p) for c,s,q,t,p in zip(centers,supp,coh,spread,policy)])
    (OUT/'frequency-policy.json').write_text(json.dumps(payload,indent=2))
    fig,ax=plt.subplots(2,1,figsize=(11,6),sharex=True,constrained_layout=True)
    ax[0].semilogx(centers,supp,label='iPhone programme support'); ax[0].semilogx(centers,coh,label='iPhone↔KM184 coherence'); ax[0].axhline(.75,color='gray',ls='--',lw=.8); ax[0].set(ylim=(0,1.05),ylabel='0–1 evidence',title='Validation evidence for frequency-specific processing'); ax[0].legend()
    ax[1].semilogx(centers,spread,color='#c44e52',label='RMS-tier transfer spread');ax[1].axhline(1.5,color='gray',ls='--',lw=.8);ax[1].set(xlim=(40,18000),ylim=(0,8),xlabel='Hz',ylabel='dB',title='Level stability: lower supports a fixed mapping');ax[1].legend()
    for axis in ax:
        for p,color in [('neural_residual','#8bc34a'),('bounded_linear','#ffc107'),('conservative_reference','#90a4ae')]:
            mask=policy==p
            if mask.any(): axis.scatter(centers[mask],np.full(mask.sum(),axis.get_ylim()[0]),s=12,color=color,marker='s',label=None)
    fig.savefig(OUT/'frequency-policy-validation.png',dpi=160)
    print(json.dumps({p:int(np.sum(policy==p)) for p in np.unique(policy)},indent=2))
if __name__=='__main__':main()
