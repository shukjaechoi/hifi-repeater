"""Deterministic wide-band excerpt selection and five-insertion detection fixtures."""
from pathlib import Path
import hashlib
import json
import numpy as np
import soundfile as sf
from scipy.signal import stft

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
source = Path('/Users/sjchoi/dev/mic-fr/iphone.wav')
x, rate = sf.read(source, always_2d=True)
x = x.mean(axis=1)
# Selection uses spectral coverage, never the detector's scores or outcomes.
f, t, z = stft(x, rate, nperseg=4096, noverlap=2048)
edges = np.geomspace(80, 4000, 13)
bands = np.array([np.mean(np.abs(z[(f >= a) & (f < b)]) ** 2, axis=0) for a, b in zip(edges[:-1], edges[1:])])
candidates = []
for start in np.arange(2, len(x)/rate-2, .1):
    duration = 1.2
    power = bands[:, (t >= start) & (t < start+duration)].mean(axis=1)
    probability = power / power.sum()
    entropy = -np.sum(probability * np.log(np.maximum(probability, 1e-15)))
    coverage = int(np.sum(power > power.max() * .01))
    segment = x[round(start*rate):round((start+duration)*rate)]
    rms = float(np.sqrt(np.mean(segment**2)))
    if rms < .015: continue
    candidates.append((coverage + entropy / np.log(12), start, power, rms, coverage))
score, start, power, rms, coverage = max(candidates, key=lambda row:row[0])
trigger = x[round(start*rate):round((start+1.2)*rate)].copy()
# Tiny edge fades avoid insertion clicks becoming the distinguishing feature.
fade = round(.01*rate)
trigger[:fade] *= np.linspace(0, 1, fade)
trigger[-fade:] *= np.linspace(1, 0, fade)
# Background excludes the excerpt and its neighbourhood, so exact original replays
# are not counted as background false alarms. Retain ordinary music, not just silence.
background_source = np.concatenate([x[:max(0,round((start-2)*rate))], x[min(len(x),round((start+3.2)*rate)):]])
length = round(35*rate)
background = np.tile(background_source, int(np.ceil(length/len(background_source))))[:length].copy()
positions = [3., 9., 15., 21., 28.]
def write(name, data):
    sf.write(OUT / f'{name}.wav', data, rate, subtype='FLOAT')
    np.asarray(data, dtype='<f4').tofile(OUT / f'{name}.f32')
write('trigger', trigger)
write('background', background)
conditions = []
for name, gaps, gains in [('gapped',True,[1]*5), ('gapped_levels',True,[.5,.75,1,.6,.9]), ('continuous_replace',False,[1]*5), ('overlay',False,[1]*5)]:
    data = background.copy()
    events = []
    for at, gain in zip(positions,gains):
        a = round(at*rate); b = a+len(trigger)
        if gaps:
            data[a-round(.45*rate):b+round(.45*rate)] = 0
            data[a:b] = trigger * gain
        elif name == 'continuous_replace':
            data[a:b] = trigger * gain
        else:
            data[a:b] += trigger * gain
        events.append({'start':at,'end':at+len(trigger)/rate,'gain':gain})
    write(name,data)
    conditions.append({'name':name,'events':events,'peak':float(np.max(np.abs(data)))})
manifest = {'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'rate':rate,
 'source_trigger_start':float(start),'trigger_duration':len(trigger)/rate,'rms':rms,
 'band_edges_hz':edges.tolist(),'band_relative_db':(10*np.log10(power/power.max())).tolist(),
 'bands_within_20db':coverage,'duration':35,'conditions':conditions,
 'selection':'Maximum count of 12 log-spaced 80–4000 Hz bands within 20 dB of strongest band, then spectral entropy; 1.2 s windows, 0.1 s stride; no detector-based selection.',
 'matching':'One-to-one match: detection delivered between insertion end - 0.1 s and end + 0.65 s; other detections are false positives.'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'source_start':start,'bands':coverage,'rms':rms,'band_db':manifest['band_relative_db']},indent=2))
