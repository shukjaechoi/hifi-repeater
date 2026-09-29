"""Official NAM A2 WaveNets, trained separately on Apple MPS.

Only fit/validation are read here. Test evaluation is a separate command.
"""
import argparse
import json
import os
import time
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mic-mpl')
import numpy as np
import torch
from scipy import signal
from nam.models.wavenet import WaveNet
from nam._dependencies.auraloss.freq import MultiResolutionSTFTLoss

ROOT = Path(__file__).resolve().parent

def config_for(name):
    source = ROOT/'vendor/neural-amp-modeler/nam/train/_resources/config_model_packed.json'
    config = json.loads(source.read_text())['net']['config']['submodels'][0 if name=='lite' else 1]['config']
    config['sample_rate'] = 48000
    return config


def bandpass(x, sample_rate=48000, low_hz=70.0, high_hz=4000.0):
    """Differentiable offline band selection used only while training.

    The model remains a normal full-band NAM WaveNet. This loss view isolates
    the range supported by the programme material and by the experiment goal.
    """
    n=x.shape[-1]
    frequencies=torch.fft.rfftfreq(n, 1/sample_rate, device=x.device)
    # 20 Hz / 250 Hz cosine transitions avoid a sharp FFT-bin objective.
    low=torch.clamp((frequencies-(low_hz-20))/20,0,1)
    high=torch.clamp(((high_hz+250)-frequencies)/250,0,1)
    ramp=lambda z: .5-.5*torch.cos(torch.pi*z)
    response=ramp(low)*ramp(high)
    return torch.fft.irfft(torch.fft.rfft(x,dim=-1)*response,n=n,dim=-1)


def relative_spectral_mae_db(pred, target, sample_rate=48000):
    """mic-fr-compatible validation criterion, restricted to 70 Hz–4 kHz."""
    nfft, hop=8192,4096
    win=np.hanning(nfft)
    def median_power(x):
        frames=np.lib.stride_tricks.sliding_window_view(x,nfft)[::hop]
        power=np.abs(np.fft.rfft(frames*win,axis=1))**2
        level=np.mean(frames**2,axis=1)
        active=level>max(np.quantile(level,.3),1e-9)
        return np.median(power[active],axis=0)
    p,t=median_power(pred),median_power(target)
    f=np.fft.rfftfreq(nfft,1/sample_rate)
    ratio=10*np.log10(np.maximum(p,1e-20)/np.maximum(t,1e-20))
    ratio-=np.median(ratio[(f>=500)&(f<=2000)])
    use=(f>=70)&(f<=4000)
    return float(np.mean(np.abs(ratio[use])))

@torch.no_grad()
def predict(model, x, device='mps', chunk=16384):
    model.eval()
    rf = model.receptive_field
    padded = np.pad(x, (rf-1,0))
    result = []
    for a in range(0,len(x),chunk):
        z = torch.from_numpy(padded[a:a+chunk+rf-1].copy()).to(device)
        result.append(model(z, pad_start=False).cpu().numpy())
    return np.concatenate(result)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['lite','full'], required=True)
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--steps', type=int, default=40)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--length', type=int, default=8192)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--run-name', help='Store this run under runs/<name>; preserves prior runs.')
    p.add_argument('--learning-rate', type=float, default=.004)
    p.add_argument('--gamma', type=float, default=.98)
    p.add_argument('--early-stop-patience', type=int, default=25,
                   help='Set 0 to run every requested epoch.')
    p.add_argument('--checkpoint-every', type=int, default=25)
    p.add_argument('--loss-profile', choices=['nam_default','band_spectral'], default='nam_default')
    args = p.parse_args()
    assert torch.backends.mps.is_available(), 'MPS required; refusing silent CPU training'
    torch.set_num_threads(4)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    run_name = args.run_name or args.model
    out = ROOT/'runs'/run_name
    out.mkdir(parents=True,exist_ok=True)
    info = json.loads((ROOT/'data/alignment.json').read_text())
    data = np.load(ROOT/'data/aligned.npz')
    end = info['splits']['fit'][1]
    x, y = data['x'][:end], data['y'][:end]
    va,vb = info['splits']['validation']
    vx,vy = data['x'][va:vb],data['y'][va:vb]
    config = config_for(args.model)
    model = WaveNet.init_from_config(config).to('mps')
    rf = model.receptive_field
    # Include preceding context for validation without fitting any validation targets.
    vx_context = data['x'][va-rf+1:vb]
    optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate,weight_decay=3.17e-7)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=args.gamma)
    stft = MultiResolutionSTFTLoss()
    metadata = dict(model=args.model,config=config,device='mps',torch=torch.__version__,
                    receptive_field=rf,parameters=sum(p.numel() for p in model.parameters()),
                    args=vars(args),loss='MSE + 0.0005 * official MRSTFT; differentiable STFT on CPU, network/backprop on MPS',
                    selection=('minimum validation ESR' if args.loss_profile=='nam_default' else
                               'minimum 70 Hz–4 kHz relative-spectrum MAE in dB; no test access'),seed=42)
    (out/'config.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata),flush=True)
    best, start, stale = float('inf'), 0, 0
    if args.resume:
        checkpoint = torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['state'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best,start = checkpoint['best'],checkpoint['epoch']
        rng.bit_generator.state = checkpoint['rng']
    began = time.time()
    for epoch in range(start,args.epochs):
        model.train()
        losses=[]
        for step in range(args.steps):
            starts = rng.integers(rf-1,len(x)-args.length,size=args.batch)
            xx = torch.from_numpy(np.stack([x[s-rf+1:s+args.length] for s in starts])).to('mps')
            yy = torch.from_numpy(np.stack([y[s:s+args.length] for s in starts])).to('mps')
            optimizer.zero_grad(set_to_none=True)
            pred = model(xx,pad_start=False)
            mse = (pred-yy).square().mean()
            # MPS FFT support differs by PyTorch version. Explicit CPU loss preserves gradients.
            spectral = stft(pred[:,None].cpu(),yy[:,None].cpu()).to('mps')
            if args.loss_profile=='nam_default':
                loss = mse + .0005*spectral
            else:
                # Direct waveform fit plus the trusted-band terms.  The scale
                # factors were chosen from an initial train-window measurement
                # so all three terms affect optimisation at comparable scale.
                bp,by=bandpass(pred.cpu()),bandpass(yy.cpu())
                band_esr=((bp-by).square().mean(dim=1)/by.square().mean(dim=1).clamp_min(1e-12)).mean().to('mps')
                band_spectral=stft(bp[:,None],by[:,None]).to('mps')
                loss = mse + .005*spectral + .005*band_esr + .01*band_spectral
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        prediction = predict(model,vx_context)[rf-1:]
        esr = float(np.mean((prediction-vy)**2)/np.mean(vy**2))
        spectral_mae=relative_spectral_mae_db(prediction,vy)
        selection_value=esr if args.loss_profile=='nam_default' else spectral_mae
        improved = selection_value < best
        if improved:
            best = selection_value
            stale=0
            torch.save({'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'config':config,'epoch':epoch+1,'val_esr':esr,'val_relative_spectral_mae_db':spectral_mae},out/'best.pt')
        else: stale+=1
        scheduler.step()
        row = dict(epoch=epoch+1,loss=float(np.mean(losses)),val_esr=esr,
                   val_relative_spectral_mae_db=spectral_mae,best_selection=best,
                   seconds=round(time.time()-began,1),lr=scheduler.get_last_lr()[0])
        print(json.dumps(row),flush=True)
        with (out/'history.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        torch.save(dict(state=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),best=best,epoch=epoch+1,rng=rng.bit_generator.state),out/'last.pt')
        if args.checkpoint_every and (epoch + 1) % args.checkpoint_every == 0:
            torch.save({'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'config':config,'epoch':epoch+1,'val_esr':esr}, out/f'epoch_{epoch+1:04d}.pt')
        if args.early_stop_patience and stale >= args.early_stop_patience and epoch >= 49:
            print('Early stopping: 25 epochs without validation improvement',flush=True)
            break
    saved = torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    model.cpu().load_state_dict(saved['state'])
    model.eval()
    model.export(out,basename=f'iphone_to_km184_a2_{args.model}',other_metadata={'training':metadata,'validation_esr':saved['val_esr']})
    print('COMPLETE',args.model,'best epoch',saved['epoch'],'ESR',saved['val_esr'],flush=True)

if __name__=='__main__': main()
