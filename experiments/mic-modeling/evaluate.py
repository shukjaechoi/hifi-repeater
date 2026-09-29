"""One final held-out evaluation after all checkpoint and FIR selection."""
import json
import os
import argparse
os.environ.setdefault('MPLCONFIGDIR','/tmp/mic-mpl')
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
from scipy import signal
from nam.models.wavenet import WaveNet
from train import predict
ROOT=Path(__file__).resolve().parent

def metrics(pred,y):
    error=float(np.mean((pred-y)**2)/np.mean(y**2))
    resolutions=[]
    for size in [512,1024,2048]:
        _,_,a=signal.stft(pred,nperseg=size,noverlap=size*3//4)
        _,_,b=signal.stft(y,nperseg=size,noverlap=size*3//4)
        a,b=abs(a),abs(b)
        resolutions.append(float(np.linalg.norm(a-b)/np.linalg.norm(b)))
    return dict(esr=error,snr_db=float(-10*np.log10(error)),mr_spectral_convergence=float(np.mean(resolutions)),
                correlation=float(np.corrcoef(pred,y)[0,1]),peak=float(np.max(abs(pred))),rms=float(np.sqrt(np.mean(pred**2))))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-run', default='full',
                        help='Directory under runs/ that contains the selected Full checkpoint.')
    args = parser.parse_args()
    torch.set_num_threads(4)
    info=json.loads((ROOT/'data/alignment.json').read_text())
    d=np.load(ROOT/'data/aligned.npz')
    x,y=d['x'],d['y']
    a,b=info['splits']['test']
    out=ROOT/'runs/comparison'
    out.mkdir(exist_ok=True)
    baseline=json.loads((ROOT/'runs/baseline.json').read_text())
    tracks={'iphone':x,'gain_only':x*baseline['least_squares_gain'],
            'fir':signal.fftconvolve(x,np.load(ROOT/'runs/fir.npy'))[:len(x)]}
    models={}
    model_runs={'lite':'lite', 'full':args.full_run}
    for name, run in model_runs.items():
        ckpt=torch.load(ROOT/f'runs/{run}/best.pt',map_location='cpu',weights_only=False)
        m=WaveNet.init_from_config(ckpt['config']).to('mps')
        m.load_state_dict(ckpt['state'])
        tracks[name]=predict(m,x)
        models[name]={'epoch':ckpt['epoch'],'validation_esr':ckpt['val_esr']}
        # Verify exported .nam weights reconstruct the trained network.
        export=json.loads((ROOT/f'runs/{run}/iphone_to_km184_a2_{name}.nam').read_text())
        # NAM export schema is for the C++ reader, not Python init_from_config.
        # Rebuild from the saved training architecture and import only serialized weights.
        restored=WaveNet.init_from_config(ckpt['config'])
        restored.import_weights(export['weights'])
        expected=predict(m,x[a:a+48000])
        actual=predict(restored,x[a:a+48000],device='cpu')
        models[name]['export_max_absolute_error']=float(np.max(abs(actual-expected)))
        assert np.allclose(actual,expected,atol=2e-5,rtol=2e-4)
    result={'test_seconds':(b-a)/48000,'reference_start_seconds':a/48000,
            'full_run':args.full_run,'models':models,
            'metrics':{k:metrics(v[a:b],y[a:b]) for k,v in tracks.items()},
            'note':'No per-test gain fitting or time alignment. All tracks share original reference level. Lower ESR and spectral convergence are better.'}
    for name,v in {**tracks,'km184':y}.items():
        sf.write(out/f'test_{name}.wav',v[a:b]/info['shared_scale'],48000,subtype='PCM_24')
        sf.write(out/f'complete_{name}.wav',v/info['shared_scale'],48000,subtype='PCM_24')
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
