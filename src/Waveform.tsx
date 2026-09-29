import { useMemo, useRef } from 'react';
export function Waveform({samples, duration, start, end, position, onRange}: {samples?: Float32Array; duration: number; start: number; end: number; position: number; onRange: (a: number, b: number) => void}) {
  const anchor = useRef<number | null>(null);
  const d = duration || 1;
  const x = (event: React.PointerEvent<SVGSVGElement>) => Math.max(0, Math.min(d, (event.clientX - event.currentTarget.getBoundingClientRect().left) / event.currentTarget.getBoundingClientRect().width * d));
  const path = useMemo(() => {
  let path = '';
  if (samples?.length) {
    for (let i = 0; i < 600; i++) {
      const a = Math.floor(i * samples.length / 600), b = Math.max(a + 1, Math.floor((i + 1) * samples.length / 600));
      let lo = 0, hi = 0;
      for (let j = a; j < b; j++) { lo = Math.min(lo, samples[j] ?? 0); hi = Math.max(hi, samples[j] ?? 0); }
      path += `M${i * 2},${100 - hi * 88}V${100 - lo * 88} `;
    }
  }
  return path;
  }, [samples]);
  return <svg className="waveform" viewBox="0 0 1200 200" preserveAspectRatio="none" role="img" aria-label="녹음 파형. 아래 구간 슬라이더로도 반복 범위를 조정할 수 있습니다."
    onPointerDown={e => { if (!samples?.length) return; anchor.current = x(e); e.currentTarget.setPointerCapture(e.pointerId); }}
    onPointerUp={e => { if (anchor.current === null) return; const a = anchor.current, b = x(e); anchor.current = null; if (Math.abs(a-b) > .02) onRange(Math.min(a,b), Math.max(a,b)); }}
    onPointerCancel={() => {anchor.current = null;}}>
    {[25, 50, 75, 100, 125, 150, 175].map(y => <line key={y} x1="0" x2="1200" y1={y} y2={y} className="grid-line"/>)}
    <rect x={start/d*1200} y="0" width={Math.max(0, (end-start)/d*1200)} height="200" className="selection"/>
    <path d={path} className="wave-path"/>
    {samples?.length ? <><line x1={start/d*1200} x2={start/d*1200} y1="0" y2="200" className="range-line"/><line x1={end/d*1200} x2={end/d*1200} y1="0" y2="200" className="range-line"/><line x1={position/d*1200} x2={position/d*1200} y1="0" y2="200" className="playhead"/></> : null}
  </svg>;
}
