import { MusicFeatures, MusicSegmenter, audibleSegments, audibleMatch, calibrate, templateDistance, MAX_EXAMPLES, type MusicSegment, type MusicTemplate } from './music-trigger';
let features = new MusicFeatures(), segmenter = new MusicSegmenter(), template: MusicTemplate | undefined;
let segments: MusicSegment[] = [], generation = 0, mode = 'off', rate = 0, cooldown = 0, overflow = false;
self.onmessage = ({data}) => {
  if (data.type === 'reset') {
    generation = data.generation; mode = data.mode; template = data.template;
    features = new MusicFeatures(); segmenter = new MusicSegmenter(); segments = []; cooldown = 0; rate = 0; overflow = false;
  } else if (data.type === 'finish' && data.generation === generation && mode === 'enroll') {
    const final = segmenter.finish();
    if (final) segments.push(final);
    const kept = audibleSegments(segments);
    const quiet = segments.length - kept.length;
    segments = kept;
    overflow ||= segments.length > MAX_EXAMPLES;
    const calibration = calibrate(segments.map(segment => segment.frames));
    if (segmenter.rejected || quiet) calibration.warning = `${segmenter.rejected + quiet}개 구간은 너무 약하거나 짧거나 길어 제외했습니다. 아래 횟수와 소리를 확인해 주세요.`;
    if (overflow) {
      calibration.template = undefined;
      calibration.error = '8회를 초과했습니다. 5회 정도로 다시 등록해 주세요.';
    }
    self.postMessage({type:'review', generation, segments, calibration});
    mode = 'off';
  } else if (data.type === 'pcm' && data.generation === generation && mode !== 'off') {
    if (rate && rate !== data.rate) { self.postMessage({type:'failure',generation}); mode = 'off'; return; }
    rate = data.rate;
    for (const frame of features.push(data.samples, rate)) {
      const segment = segmenter.push(frame);
      cooldown = Math.max(0, cooldown - 1);
      if (!segment) continue;
      if (mode === 'enroll') {
        if (segments.length < 128) segments.push(segment); else overflow = true;
        self.postMessage({type:'count', generation, count:audibleSegments(segments).length, overflow:overflow || audibleSegments(segments).length > MAX_EXAMPLES});
      } else if (!cooldown && template && audibleMatch(template, segment.frames) && templateDistance(template, segment.frames) < (template.version === 1 ? .12 : template.threshold)) {
        self.postMessage({type:'match', generation, seconds:segment.end - segment.start + .224});
        cooldown = 10;
      }
    }
    self.postMessage({type:'ack', generation});
  }
};
