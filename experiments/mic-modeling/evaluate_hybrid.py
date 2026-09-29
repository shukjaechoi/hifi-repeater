"""Evaluate the selected FIR + residual experiment once on the held-out tail."""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from evaluate import metrics
from train_hybrid import fir_apply

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/hybrid-fir-residual-16ch-relative-active'
OUT=ROOT/'runs/comparison'

def main():
    info=json.loads((ROOT/'data/alignment.json').read_text())
    checkpoint=__import__('torch').load(RUN/'best.pt',map_location='cpu',weights_only=False)
    data=np.load(ROOT/'data/aligned.npz')
    x,y=data['x'],data['y']
    taps=np.load(RUN/'fir_70_4k_train_only.npy')
    # The selected checkpoint is epoch zero: intentional zero residual means
    # FIR-only is the validated hybrid output.
    prediction=fir_apply(x,taps)
    a,b=info['splits']['test']
    OUT.mkdir(exist_ok=True)
    for prefix,z in [('test',prediction[a:b]),('complete',prediction)]:
        sf.write(OUT/f'{prefix}_hybrid_fir_residual.wav',z/info['shared_scale'],48000,subtype='PCM_24')
    result=json.loads((OUT/'metrics.json').read_text())
    result['metrics']['hybrid_fir_residual']=metrics(prediction[a:b],y[a:b])
    result['hybrid_training']=dict(run=RUN.name,selected_epoch=checkpoint['epoch'],
        selected_validation_esr=checkpoint['val_esr'],
        selected_validation_relative_spectral_mae_db=checkpoint['val_relative_spectral_mae_db'],
        stopped_after_epoch=19,
        note='Residual epochs reduced waveform ESR but did not improve validation relative-spectrum MAE; epoch 0 FIR-only checkpoint selected.')
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'hybrid':result['metrics']['hybrid_fir_residual'],'training':result['hybrid_training']},indent=2))
if __name__=='__main__': main()
