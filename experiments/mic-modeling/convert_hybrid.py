"""Apply the selected train-only hybrid FIR processor on an original iPhone WAV."""
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from train_hybrid import fir_apply

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/hybrid-fir-residual-16ch-relative-active'

def main():
    source=Path('/Users/sjchoi/dev/mic-fr/iphone.wav')
    out=ROOT/'runs/comparison/iphone_hybrid_fir_train_only_timeline.wav'
    info=json.loads((ROOT/'data/alignment.json').read_text())
    audio,sr=sf.read(source,always_2d=True)
    if sr!=48000: raise ValueError('48 kHz required')
    x=((audio.mean(axis=1)*info['polarity']-info['input_dc'])*info['input_gain']*info['shared_scale']).astype('float32')
    y=fir_apply(x,np.load(RUN/'fir_70_4k_train_only.npy'))/info['shared_scale']+info['target_dc']
    sf.write(out,y,sr,subtype='FLOAT')
    print(out)
if __name__=='__main__': main()
