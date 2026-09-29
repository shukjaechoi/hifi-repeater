import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { AudioEngine } from './audio';
import { allTakes, putTake, type Take } from './storage';
import { leadingSoundOffset, validRange, wav } from './audio-utils';
import { Waveform } from './Waveform';
import './style.css';
type State = 'idle' | 'preparing' | 'recording' | 'finishing' | 'playing';
const slotStatus = (current: boolean, take: Take | undefined, recording: boolean, elapsed: number) => {
  if (current) return `작업중 · ${time(recording ? elapsed : take ? take.samples.length/take.rate : elapsed)}`;
  if (take) return `${time(take.samples.length/take.rate)} · ${take.saved?'보관됨':'작업중'}`;
  return '대기중';
};
const time = (s: number) => `${Math.floor(s/60).toString().padStart(2,'0')}:${Math.floor(s%60).toString().padStart(2,'0')}.${Math.floor(s%1*10)}`;
function App() {
  const engine = useRef(new AudioEngine()).current;
  const [takes, setTakes] = useState<Take[]>([]), [slot, setSlot] = useState(0), [selected, setSelected] = useState<Record<number,string>>({});
  const [state, setState] = useState<State>('idle'), [loaded, setLoaded] = useState(false), [loop, setLoop] = useState(false);
  const [autoTrimSilence, setAutoTrimSilence] = useState(() => localStorage.getItem('hifi-auto-trim-silence') === 'true');
  const [zoom, setZoom] = useState<[number, number] | null>(null);
  const [position, setPosition] = useState(0), [elapsed, setElapsed] = useState(0), [level, setLevel] = useState(0), [clipped, setClipped] = useState(false);
  const [live, setLive] = useState<Float32Array>(new Float32Array()), [notice, setNotice] = useState(''), [error, setError] = useState('');
  const busy = useRef(false), settings = useRef<MediaTrackSettings>({}), frames = useRef(0), liveChunks = useRef<number[]>([]);
  const active = takes.find(t => t.id === selected[slot]) ?? takes.find(t => t.slot === slot);
  const duration = active ? active.samples.length/active.rate : 0;
  const viewStart = zoom?.[0] ?? 0, viewEnd = zoom?.[1] ?? duration;
  const visibleSamples = useMemo(() => active && zoom ? active.samples.subarray(Math.floor(viewStart * active.rate), Math.ceil(viewEnd * active.rate)) : active?.samples, [active?.samples, active?.rate, zoom, viewStart, viewEnd]);
  const recording = state === 'recording', locked = recording || state === 'preparing' || state === 'finishing';
  useEffect(() => { allTakes().then(setTakes).catch(() => setError('작업 저장소를 열 수 없습니다. 브라우저의 저장 권한을 확인하세요.')).finally(() => setLoaded(true)); return () => {engine.stopPlayback(); engine.release();}; }, [engine]);
  useEffect(() => { if (state !== 'playing') return; let id: number; const tick = () => {setPosition(engine.position()); id = requestAnimationFrame(tick);}; id = requestAnimationFrame(tick); return () => cancelAnimationFrame(id); }, [state, engine]);
  const stop = () => { engine.stopPlayback(); setState('idle'); };
  async function play(take: Take) {
    await engine.play(take.samples, take.rate, take.start, take.end, loop, () => {setState('idle'); setPosition(take.end);});
    setPosition(take.start); setState('playing');
  }
  async function record() {
    if (busy.current || !loaded) return;
    busy.current = true; setError(''); setNotice('');
    try {
      if (recording) {
        setState('finishing');
        const data = await engine.finish();
        if (data.samples.length < data.rate * .05) throw new Error('녹음이 너무 짧습니다. 조금 더 길게 연주해 주세요.');
        const duration = data.samples.length/data.rate;
        const start = autoTrimSilence ? leadingSoundOffset(data.samples, data.rate) ?? 0 : 0;
        const take: Take = { ...data, id: crypto.randomUUID(), slot, created: Date.now(), saved: false, start, end: duration, settings: settings.current };
        setTakes(previous => [take, ...previous]); setSelected(previous => ({...previous, [slot]: take.id}));
        // Playback must not wait for disk persistence.
        void putTake(take).catch(() => setError('내부 저장에 실패했습니다. 이 녹음은 현재 화면에서 WAV로 내보낼 수 있습니다.'));
        await play(take);
        if (start > 0) setNotice(`앞 무음 ${time(start)}을 자동으로 제외했습니다.`);
      } else {
        setZoom(null); setState('preparing'); setElapsed(0); setLevel(0); setClipped(false); setLive(new Float32Array()); frames.current = 0; liveChunks.current = [];
        if (!navigator.mediaDevices?.getUserMedia) throw new Error('마이크를 사용하려면 HTTPS 또는 localhost에서 열어 주세요.');
        settings.current = await engine.record(chunk => {
          frames.current += chunk.length; setElapsed(frames.current/(engine.context?.sampleRate ?? 48000));
          let peak = 0; for (const sample of chunk) peak = Math.max(peak, Math.abs(sample));
          setLevel(peak); if (peak >= .99) setClipped(true);
          // Keep a bounded visual history; the engine retains full-resolution PCM.
          liveChunks.current.push(peak, -peak); if (liveChunks.current.length > 1200) liveChunks.current.splice(0, 2);
          setLive(new Float32Array(liveChunks.current));
        }, () => {setError('마이크 연결이 중단되었습니다. 녹음 버튼을 눌러 수집된 소리를 마무리하세요.');});
        setState('recording');
      }
    } catch (e) { setError(e instanceof Error ? e.message : '오디오를 처리할 수 없습니다.'); setState('idle'); }
    finally {busy.current = false;}
  }
  async function update(take: Take) {
    setTakes(previous => previous.map(t => t.id === take.id ? take : t));
    try { await putTake(take); } catch { setError('변경 내용을 저장하지 못했습니다. WAV로 내보내 주세요.'); }
  }
  function range(a: number, b: number) { if (!active || locked) return; stop(); const [start,end] = validRange(a,b,duration); setPosition(start); void update({...active,start,end}); }
  function skipLeadingSilence() {
    if (!active || locked) return;
    const start = leadingSoundOffset(active.samples, active.rate);
    if (start === null) { setNotice('연주 시작점을 찾지 못했습니다. A 슬라이더로 직접 조정해 주세요.'); return; }
    stop(); setPosition(start); setZoom(null); void update({...active, start, end: active.end < start + .02 ? duration : active.end});
    setNotice(`앞 무음 ${time(start)}을 재생 범위에서 제외했습니다.`);
  }
  async function togglePlay() { if (!active || busy.current) return; if (state === 'playing') {stop(); return;} busy.current = true; try {await play(active);} catch {setError('재생하지 못했습니다. 다시 시도해 주세요.'); setState('idle');} finally {busy.current = false;} }
  function exportTake() { if (!active) return; const url = URL.createObjectURL(new Blob([wav(active.samples,active.rate)],{type:'audio/wav'})); const a = document.createElement('a'); a.href = url; a.download = `practice-${slot+1}-${new Date(active.created).toISOString().replaceAll(':','-')}.wav`; a.click(); setTimeout(() => URL.revokeObjectURL(url),10000); setNotice('원본 WAV를 내보냈습니다.'); }
  const status = {idle:'',preparing:'마이크 연결 중',recording:'녹음 중',finishing:'재생 준비 중',playing: loop ? '구간 반복 중' : '재생 중'}[state];
  return <div className="app">
    <header><a className="brand" href="/" aria-label="hifi repeater 홈"><span className="brand-mark">≋</span> hifi<span> / repeater</span></a><span className="local"><i/> LOCAL PRACTICE SPACE</span></header>
    <main>
      <nav className="slots" aria-label="연습 공간">{[0,1,2].map(i => {const take = takes.find(t => t.id === selected[i]) ?? takes.find(t => t.slot === i); return <button key={i} className={`slot ${slot===i?'active':''}`} disabled={locked} aria-pressed={slot===i} onClick={() => {stop();setZoom(null);setSlot(i);setPosition(0);}}><strong>{slotStatus(slot===i,take,recording,elapsed)}</strong></button>;})}</nav>
      <section className="studio" aria-label="녹음 및 반복 재생">
        {status && <div className="studio-top"><span className={`status ${recording?'red':''}`}><i/>{status}</span></div>}
        <div className="time-row"><div className="timer">{time(recording?elapsed:position)}<small> / {time(recording?elapsed:duration)}</small></div><span className="format">MONO <span>·</span> {active ? `${(active.rate/1000).toFixed(1)} kHz` : '48 kHz'}</span></div>
        <div className="wave-wrap"><Waveform samples={recording?live:visibleSamples} duration={recording?elapsed:viewEnd-viewStart} start={recording?0:Math.max(0,(active?.start??0)-viewStart)} end={recording?elapsed:Math.min(viewEnd-viewStart,(active?.end??0)-viewStart)} position={recording?elapsed:position-viewStart} onRange={(a,b)=>range(a+viewStart,b+viewStart)}/>{!active && !recording && <div className="empty-wave"><span>⌁</span><p>한 번의 연주부터 시작하세요</p><small>녹음을 멈추면 자동으로 들려드릴게요.</small></div>}</div>
        <div className="wave-caption"><span>{recording?'연주를 듣고 있어요': '파형을 드래그해 반복할 구간을 선택하세요'}</span><span>{time(recording?elapsed:duration)}</span></div>
        <div className="zoom-row"><span>{zoom ? `${time(viewStart)} — ${time(viewEnd)}` : "전체 파형"}</span><div><button disabled={!active||locked} onClick={skipLeadingSilence}>시작 무음 건너뛰기</button><button disabled={!active||locked} onClick={()=>setZoom(zoom?null:[active!.start,active!.end])}>{zoom?"전체 보기":"선택 구간 확대"}</button></div></div><div className="range-controls"><label>A <input aria-label="구간 시작" type="range" min="0" max={duration||1} step="0.01" value={active?.start??0} disabled={!active||locked} onChange={e=>range(Number(e.target.value),active!.end)}/><span>{time(active?.start??0)}</span></label><label>B <input aria-label="구간 끝" type="range" min="0" max={duration||1} step="0.01" value={active?.end??0} disabled={!active||locked} onChange={e=>range(active!.start,Number(e.target.value))}/><span>{time(active?.end??0)}</span></label></div>
        <div className="silence-setting"><label><input type="checkbox" checked={autoTrimSilence} onChange={e=>{setAutoTrimSilence(e.target.checked);localStorage.setItem('hifi-auto-trim-silence',String(e.target.checked));}}/><span>시작 무음 건너뛰기</span></label><small>{autoTrimSilence?'켜짐 · 연주 시작점부터 자동 재생':'꺼짐 · 전체 녹음을 재생'}</small></div>
        <div className="transport"><div className="transport-side"><button className={`chip ${loop?'enabled':''}`} aria-pressed={loop} disabled={locked} onClick={()=>{stop();setLoop(!loop);}}>↻ 구간 반복 {loop?'ON':'OFF'}</button><button className="text-button" disabled={!active||locked} onClick={()=>range(0,duration)}>전체 구간</button></div><div className="main-controls"><button className="play-button" aria-label={state==='playing'?'재생 정지':'선택 구간 재생'} disabled={!active||locked} onClick={()=>void togglePlay()}>{state==='playing'?'■':'▶'}</button><button className={`record-button ${recording?'recording':''}`} disabled={!loaded||state==='preparing'||state==='finishing'} onClick={()=>void record()}><span className={recording?'square':'circle'}/>{recording?'정지하고 듣기':state==='preparing'?'마이크 연결 중':state==='finishing'?'재생 준비 중':'녹음 시작'}</button></div><div className="transport-side right"><span className="auto-dot"/> 정지하면 자동 재생</div></div>
        <div className="studio-footer"><div className="input-meter"><span>INPUT</span><div><i style={{width:`${Math.min(100,level*100)}%`,background:clipped?'#eb8a77':undefined}}/></div><span>{clipped?'피크 주의':recording?'LIVE':'대기'}</span></div><span>파일 이름 없이, 연습에만 집중하세요.</span></div>
      </section>
      {error && <div role="alert" className="message error">{error}</div>}{notice && <div role="status" className="message">{notice}</div>}
      <section className="takes"><div className="takes-heading"><div><h2>이 공간의 연주 <span>{takes.filter(t=>t.slot===slot).length.toString().padStart(2,'0')}</span></h2><p>이전 연주를 선택해 지금의 소리와 비교해 보세요.</p></div><div><button disabled={!active||locked} onClick={()=>{if(active) void update({...active,saved:!active.saved});}}>{active?.saved?'★ 보관됨':'☆ 따로 보관'}</button><button disabled={!active||locked} onClick={exportTake}>↗ WAV 내보내기</button></div></div>
      <div className="take-list">{takes.filter(t=>t.slot===slot).map((t,i,arr)=><button key={t.id} disabled={locked} className={`take ${active?.id===t.id?'selected':''}`} onClick={()=>{stop();setZoom(null);setSelected(previous=>({...previous,[slot]:t.id}));setPosition(t.start);}}><span className="take-icon">{active?.id===t.id?'≋':'▷'}</span><span><strong>Take {String(arr.length-i).padStart(2,'0')} {t.saved?'★':''}</strong><small>{new Date(t.created).toLocaleTimeString('ko-KR',{hour:'2-digit',minute:'2-digit'})}</small></span><span className="take-duration">{time(t.samples.length/t.rate)}</span></button>)}{!takes.some(t=>t.slot===slot)&&<div className="no-takes">아직 녹음된 연주가 없습니다. 새 녹음은 이 공간에 자동으로 남아요.</div>}</div></section>
      <footer><span>내 소리를 듣는 가장 짧은 거리.</span><span>기기 내 작업 저장 · 외부 업로드 없음</span></footer>
    </main>
  </div>;
}
createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>);
