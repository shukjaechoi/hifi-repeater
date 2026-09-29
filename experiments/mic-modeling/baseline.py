"""Select a causal FIR on fit/validation only; no test scoring."""
import json
from pathlib import Path
import numpy as np
from scipy import signal, linalg
ROOT=Path(__file__).resolve().parent

def main():
    info=json.loads((ROOT/'data/alignment.json').read_text())
    d=np.load(ROOT/'data/aligned.npz')
    end=info['splits']['fit'][1]
    x,y=d['x'][:end].astype('float64'),d['y'][:end].astype('float64')
    auto=signal.correlate(x,x,method='fft',mode='full')[len(x)-1:]
    cross=signal.correlate(y,x,method='fft',mode='full')[len(x)-1:]
    a,b=info['splits']['validation']
    candidates=[]
    best=float('inf')
    for size in [64,256,1024,4096]:
        for ridge in [1e-5,1e-4,1e-3,1e-2]:
            r=auto[:size].copy()
            r[0]+=ridge*auto[0]
            h=linalg.solve_toeplitz(r,cross[:size])
            pred=signal.fftconvolve(d['x'][:b],h)[:b][a:b]
            esr=float(np.mean((pred-d['y'][a:b])**2)/np.mean(d['y'][a:b]**2))
            candidates.append(dict(taps=size,ridge=ridge,val_esr=esr))
            if esr<best:
                best=esr
                np.save(ROOT/'runs/fir.npy',h)
    gain=float(x@y/(x@x))
    result=dict(candidates=candidates,best=min(candidates,key=lambda c:c['val_esr']),least_squares_gain=gain)
    (ROOT/'runs/baseline.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
