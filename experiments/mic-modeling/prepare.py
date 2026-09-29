"""Training-only synchronization; original WAV files are never changed."""
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal, stats

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data'
OUT.mkdir(exist_ok=True)
FS = 48000

def read(name):
    x, sr = sf.read(Path('/Users/sjchoi/dev/mic-fr') / f'{name}.wav', always_2d=True)
    assert sr == FS
    print(name, x.shape, 'duration', len(x)/sr, 'channel_difference', np.max(np.abs(x[:, 0]-x[:, -1])))
    return x.mean(axis=1)

def lag(y, x, sec, center=0, reach=48000, polarity=None):
    start, size = int(sec*FS), 2*FS
    a = y[start:start+size]
    lo = start + int(round(center)) - reach
    b = x[lo:lo+size+2*reach]
    c = signal.correlate(b, a, mode='valid', method='fft')
    v = abs(c) if polarity is None else polarity*c
    k = np.argmax(v)
    frac = .5*(v[k-1]-v[k+1])/(v[k-1]-2*v[k]+v[k+1]) if 0 < k < len(c)-1 else 0
    return float(lo-start+k+frac), float(np.corrcoef(a, b[k:k+size])[0,1])

def main():
    x, y = read('iphone'), read('km184')
    # All calibration observations end before 50 s. Test is untouched.
    coarse, corr = lag(y, x, 10, reach=2*FS)
    polarity = int(np.sign(corr))
    observations = [(s+1, *lag(y, x, s, coarse, 100, polarity)) for s in range(3, 48, 3)]
    t, shifts, correlations = np.array(observations).T
    keep = (polarity*correlations > .65) & (np.abs(shifts-coarse)<98)
    t, shifts = t[keep], shifts[keep]
    slope, intercept, _, _ = stats.theilslopes(shifts, t)
    # Do not resample for a negligible, uncertain sub-sample slope.
    drift = slope if abs(slope*60) > .5 else 0.0
    offset = float(np.median(shifts-drift*t))
    indices = np.arange(len(y), dtype=np.float64)
    positions = indices + offset + drift*indices/FS
    valid = (positions >= 32) & (positions < len(x)-33)
    first, last = np.flatnonzero(valid)[[0,-1]]
    first, last = int(first), int(last)
    positions = positions[first:last+1]
    # Windowed sinc interpolation preserves fractional-sample phase better than linear interpolation.
    aligned = np.zeros(len(positions))
    norm = np.zeros(len(positions))
    base = np.floor(positions).astype(int)
    for k in range(-31, 33):
        d = positions-(base+k)
        w = np.sinc(d)*np.sinc(d/32)
        aligned += w*x[base+k]
        norm += w
    aligned = polarity*aligned/norm
    target = y[first:last+1]
    # Fixed gain and DC estimates use only the fitting region.
    fit_end = int(50*FS)-first
    xm, ym = aligned[:fit_end].mean(), target[:fit_end].mean()
    aligned -= xm
    target -= ym
    gain = np.sqrt(np.mean(target[:fit_end]**2)/np.mean(aligned[:fit_end]**2))
    aligned *= gain
    scale = .1/np.sqrt(np.mean(target[:fit_end]**2))
    aligned *= scale
    target *= scale
    # Times are referenced to original KM184, including the exact final 15 seconds.
    test_start = len(y)-15*FS-first
    splits = {'fit':[0,fit_end], 'validation':[int(51*FS)-first,int(60*FS)-first], 'test':[test_start,len(target)]}
    assert splits['validation'][1] < test_start
    info = dict(sample_rate=FS, original_duration=len(y)/FS, reference_start_sample=int(first),
                duration=len(target)/FS, lag_samples=offset, drift_samples_per_second=float(drift),
                drift_ppm=float(drift/FS*1e6), polarity=polarity, input_gain=float(gain), shared_scale=float(scale),
                input_dc=float(xm), target_dc=float(ym), observations=observations, splits=splits,
                note='Calibration fitted before 50s; validation 51-60s; last 15s test; 50-51s guard. Fractional sinc synchronization is offline preprocessing.')
    np.savez(OUT/'aligned.npz', x=aligned.astype('float32'), y=target.astype('float32'))
    (OUT/'alignment.json').write_text(json.dumps(info,indent=2))
    for name, (a,b) in splits.items():
        sf.write(OUT/f'{name}_iphone.wav', aligned[a:b]/scale, FS, subtype='FLOAT')
        sf.write(OUT/f'{name}_km184.wav', target[a:b]/scale, FS, subtype='FLOAT')
    print(json.dumps(info,indent=2))

if __name__ == '__main__': main()
