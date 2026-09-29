"""Convert a new 48 kHz phone WAV using saved calibration and a trained model.

Preserves the phone's timeline; paired-reference synchronization is not applied.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from nam.models.wavenet import WaveNet
from train import ROOT, predict

def main():
    p=argparse.ArgumentParser()
    p.add_argument('input',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--model',choices=['lite','full'],default='full')
    p.add_argument('--run', help='Directory under runs/ containing the selected checkpoint; defaults to --model.')
    args=p.parse_args()
    if args.input.resolve()==args.output.resolve(): raise ValueError('Choose a different output path')
    info=json.loads((ROOT/'data/alignment.json').read_text())
    audio,sr=sf.read(args.input,always_2d=True)
    if sr!=48000: raise ValueError('Model requires 48000 Hz input')
    audio=audio.mean(axis=1)
    x=((audio*info['polarity']-info['input_dc'])*info['input_gain']*info['shared_scale']).astype('float32')
    run=args.run or args.model
    checkpoint=torch.load(ROOT/f'runs/{run}/best.pt',map_location='cpu',weights_only=False)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    model=WaveNet.init_from_config(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['state'])
    output=predict(model,x,device=device)/info['shared_scale']+info['target_dc']
    sf.write(args.output,output,sr,subtype='FLOAT')
    print(f'Saved {args.output}; {len(output)/sr:.3f}s, peak {np.max(abs(output)):.5f}, device {device}')
if __name__=='__main__': main()
