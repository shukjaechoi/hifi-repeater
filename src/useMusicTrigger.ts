import { useEffect, useRef, useState } from 'react';
import type { AudioEngine } from './audio';
import { concatenate } from './audio-utils';
import { validTemplate, type MusicTemplate, type MusicSegment, type Calibration } from './music-trigger';
const KEY = 'hifi-music-trigger-v2', LEGACY_KEY = 'hifi-music-trigger-v1';
function storedTemplate(): MusicTemplate | undefined {
  try { const value = JSON.parse(localStorage.getItem(KEY) ?? localStorage.getItem(LEGACY_KEY) ?? 'null'); return validTemplate(value) ? value : undefined; } catch { return undefined; }
}
type Review = {segments: MusicSegment[]; calibration: Calibration; samples: Float32Array<ArrayBuffer>; rate: number};
export function useMusicTrigger(engine: AudioEngine, eligible: boolean, onMatch: (seconds: number) => void) {
  const [template, setTemplate] = useState(storedTemplate);
  const [enabled, setEnabled] = useState(false), [enrolling, setEnrolling] = useState(false), [connecting, setConnecting] = useState(false);
  const [analyzing, setAnalyzing] = useState(false), [count, setCount] = useState(0), [review, setReview] = useState<Review>();
  const [testing, setTesting] = useState(false), [testCount, setTestCount] = useState(0), [previewing, setPreviewing] = useState<number>();
  const [error, setError] = useState('');
  const worker = useRef<Worker | null>(null), generation = useRef(0), pending = useRef(0), timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const operation = useRef(0), callback = useRef(onMatch); callback.current = onMatch;
  const testingRef = useRef(false), capturing = useRef(false), chunks = useRef<Float32Array[]>([]), captured = useRef(0), rateRef = useRef(48000);
  const mode = enrolling && !analyzing ? 'enroll' : testing || enabled && eligible ? 'detect' : 'off';
  const modeRef = useRef(mode);
  function reset(nextMode: string) {
    generation.current++; pending.current = 0; modeRef.current = nextMode;
    worker.current?.postMessage({type:'reset', generation:generation.current, mode:nextMode, template});
  }
  function detach() { engine.monitor = undefined; if (!engine.capturing) engine.release(); }
  function stopPreview() {engine.stopPlayback(); setPreviewing(undefined);}
  function finishEnrollment() {
    if (!capturing.current) return;
    clearTimeout(timer.current); capturing.current = false; modeRef.current = 'off';
    // Keep generation unchanged until the ordered PCM messages and finish are processed.
    setAnalyzing(true);
    worker.current?.postMessage({type:'finish', generation:generation.current});
    detach();
  }
  function cancel() {
    operation.current++; clearTimeout(timer.current); capturing.current = false; testingRef.current = false;
    setEnrolling(false); setEnabled(false); setConnecting(false); setAnalyzing(false); setTesting(false);
    setReview(undefined); chunks.current = []; captured.current = 0; reset('off'); detach(); stopPreview();
  }
  useEffect(() => {
    const current = new Worker(new URL('./trigger-worker.ts', import.meta.url), {type:'module'}); worker.current = current;
    current.onmessage = ({data}) => {
      if (data.generation !== generation.current) return;
      if (data.type === 'ack') pending.current = Math.max(0, pending.current - 1);
      if (data.type === 'count') setCount(data.count);
      if (data.type === 'match') {if (testingRef.current) setTestCount(n=>n+1); else callback.current(data.seconds);}
      if (data.type === 'failure') {cancel(); setError('입력 장치가 변경되었습니다. 다시 시작해 주세요.');}
      if (data.type === 'review') {
        setReview({segments:data.segments, calibration:data.calibration, samples:concatenate(chunks.current), rate:rateRef.current});
        chunks.current = []; setEnrolling(false); setAnalyzing(false); setError('');
      }
    };
    current.onerror = () => {cancel(); setError('음악 감지를 실행하지 못했습니다. 다시 활성화해 주세요.');};
    return () => { operation.current++; clearTimeout(timer.current); current.terminate(); worker.current = null; engine.monitor = undefined; };
  }, [engine]);
  useEffect(() => { if (!analyzing) reset(mode); }, [mode, template, analyzing]);
  useEffect(() => {
    if ((!enabled && !enrolling && !testing) || analyzing) return;
    const id = setInterval(() => {
      if (!engine.stream?.active || engine.context?.state !== 'running') {cancel(); setError('마이크가 중단되었습니다. 화면을 열고 다시 활성화해 주세요.');}
    }, 500);
    return () => clearInterval(id);
  }, [enabled, enrolling, analyzing, testing]);
  async function open(enroll: boolean, test = false) {
    cancel(); const id = ++operation.current; setConnecting(true); setError(''); setCount(0); setTestCount(0);
    try {
      await engine.listen();
      if (operation.current !== id) return;
      engine.monitor = (samples, rate) => {
        if (modeRef.current === 'off') return;
        if (pending.current >= 4) {cancel(); setError('음악 분석이 지연되었습니다. 다시 시도해 주세요.'); return;}
        if (capturing.current) {
          rateRef.current = rate; chunks.current.push(samples); captured.current += samples.length;
        }
        pending.current++;
        worker.current?.postMessage({type:'pcm', generation:generation.current, samples, rate});
        if (capturing.current && captured.current >= rate * 30) finishEnrollment();
      };
      if (enroll) {capturing.current = true; setEnrolling(true); timer.current = setTimeout(finishEnrollment,30000);}
      else if (test) {testingRef.current = true; setTesting(true); timer.current = setTimeout(cancel,30000);}
      else setEnabled(true);
    } catch (e) { if (operation.current === id) setError(e instanceof Error ? e.message : '마이크를 열지 못했습니다.'); }
    finally {if (operation.current === id) setConnecting(false);}
  }
  function confirm() {
    const next = review?.calibration.template;
    if (!next || !validTemplate(next)) return;
    try {
      localStorage.setItem(KEY,JSON.stringify(next)); localStorage.removeItem(LEGACY_KEY);
      operation.current++; setTemplate(next); setReview(undefined); stopPreview(); setError('');
    } catch {setError('트리거를 저장하지 못했습니다. 저장 공간을 확인해 주세요.');}
  }
  async function preview(index: number) {
    if (!review) return;
    const id = ++operation.current; stopPreview();
    const segment = review.segments[index];
    try {
      await engine.play(review.samples,review.rate,segment.start,Math.min(segment.end,review.samples.length/review.rate),false,()=>setPreviewing(undefined));
      if (operation.current !== id) {engine.stopPlayback(); return;} setPreviewing(index);
    } catch {setError('등록 구간을 재생하지 못했습니다.');}
  }
  function remove() {
    cancel();
    try {localStorage.removeItem(KEY); localStorage.removeItem(LEGACY_KEY); setTemplate(undefined); setError('');} catch {setError('트리거를 삭제하지 못했습니다.');}
  }
  return {template, enabled, enrolling, connecting, analyzing, count, review, testing, testCount, previewing, error, open, cancel, remove, finishEnrollment, confirm, preview};
}
