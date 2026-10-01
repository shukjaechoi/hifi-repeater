import { useEffect, useRef, useState } from 'react';
import type { AudioEngine } from './audio';
import { validTemplate, type MusicTemplate } from './music-trigger';
const KEY = 'hifi-music-trigger-v1';
function storedTemplate(): MusicTemplate | undefined {
  try { const value = JSON.parse(localStorage.getItem(KEY) ?? 'null'); return validTemplate(value) ? value : undefined; } catch { return undefined; }
}
export function useMusicTrigger(engine: AudioEngine, eligible: boolean, onMatch: (seconds: number) => void) {
  const [template, setTemplate] = useState(storedTemplate);
  const [enabled, setEnabled] = useState(false), [enrolling, setEnrolling] = useState(false), [connecting, setConnecting] = useState(false);
  const [error, setError] = useState('');
  const worker = useRef<Worker | null>(null), generation = useRef(0), pending = useRef(0), timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const operation = useRef(0), callback = useRef(onMatch); callback.current = onMatch;
  const enabledRef = useRef(false), enrollingRef = useRef(false);
  const mode = enrolling ? 'enroll' : enabled && eligible ? 'detect' : 'off';
  const modeRef = useRef(mode); modeRef.current = mode;
  function reset(nextMode: string) {
    generation.current++; pending.current = 0;
    worker.current?.postMessage({type:'reset', generation:generation.current, mode:nextMode, template});
  }
  function detach() { engine.monitor = undefined; if (!engine.capturing) engine.release(); }
  function finishEnrollment() {
    clearTimeout(timer.current);
    worker.current?.postMessage({type:'finish', generation:generation.current});
  }
  useEffect(() => {
    const current = new Worker(new URL('./trigger-worker.ts', import.meta.url), {type:'module'}); worker.current = current;
    current.onmessage = ({data}) => {
      if (data.generation !== generation.current) return;
      if (data.type === 'ack') pending.current = Math.max(0, pending.current - 1);
      if (data.type === 'match') callback.current(data.seconds);
      if (data.type === 'template') {
        clearTimeout(timer.current); enrollingRef.current = false; setEnrolling(false);
        if (!validTemplate(data.template)) setError('0.2~4초의 음악 패턴을 녹음해 주세요. 무음이거나 너무 긴 소리는 등록할 수 없습니다.');
        else {
          try { localStorage.setItem(KEY, JSON.stringify(data.template)); setTemplate(data.template); setError(''); }
          catch { setError('트리거를 저장하지 못했습니다. 저장 공간을 확인해 주세요.'); }
        }
        if (!enabledRef.current) detach();
      }
    };
    current.onerror = () => { setError('음악 감지를 실행하지 못했습니다. 다시 활성화해 주세요.'); cancel(); };
    return () => { operation.current++; clearTimeout(timer.current); current.terminate(); worker.current = null; engine.monitor = undefined; };
  }, [engine]);
  useEffect(() => { reset(mode); }, [mode, template]);
  useEffect(() => {
    if (!enabled && !enrolling) return;
    const id = setInterval(() => {
      if (!engine.stream?.active || engine.context?.state !== 'running') {
        setError('마이크가 중단되었습니다. 화면을 열고 다시 활성화해 주세요.'); cancel();
      }
    }, 500);
    return () => clearInterval(id);
  }, [enabled, enrolling]);
  async function open(enroll: boolean) {
    const id = ++operation.current; setConnecting(true); setError('');
    engine.stopPlayback();
    try {
      await engine.listen();
      if (operation.current !== id) return;
      engine.monitor = (samples, rate) => {
        if (modeRef.current === 'off') return;
        // Bound queued PCM to two blocks; discard stale detection if processing falls behind.
        if (pending.current >= 2) { reset(enrollingRef.current ? 'enroll' : 'off'); setError('음악 분석이 지연되었습니다. 다시 시도해 주세요.'); cancel(); return; }
        pending.current++;
        worker.current?.postMessage({type:'pcm', generation:generation.current, samples, rate});
      };
      if (enroll) { enrollingRef.current = true; setEnrolling(true); timer.current = setTimeout(finishEnrollment, 4500); }
      else { enabledRef.current = true; setEnabled(true); }
    } catch (e) { if (operation.current === id) setError(e instanceof Error ? e.message : '마이크를 열지 못했습니다.'); }
    finally { if (operation.current === id) setConnecting(false); }
  }
  function cancel() {
    operation.current++; clearTimeout(timer.current); enrollingRef.current = false; enabledRef.current = false;
    setEnrolling(false); setEnabled(false); setConnecting(false); reset('off'); detach();
  }
  function remove() {
    cancel();
    try { localStorage.removeItem(KEY); setTemplate(undefined); setError(''); } catch { setError('트리거를 삭제하지 못했습니다.'); }
  }
  return {template, enabled, enrolling, connecting, error, open, cancel, remove, finishEnrollment};
}
