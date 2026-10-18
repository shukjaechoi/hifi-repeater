import { MusicFeatures, MusicSegmenter, calibrate, templateDistance, MAX_EXAMPLES, type MusicSegment, type MusicTemplate } from './music-trigger';
let features = new MusicFeatures(), segmenter = new MusicSegmenter(), template: MusicTemplate | undefined;
let segments: MusicSegment[] = [], generation = 0, mode = 'off', rate = 0, cooldown = 0, overflow = false;
self.onmessage = ({data}) => {
  if (data.type === 'reset') {
    generation = data.generation; mode = data.mode; template = data.template;
    features = new MusicFeatures(); segmenter = new MusicSegmenter(); segments = []; cooldown = 0; rate = 0; overflow = false;
  } else if (data.type === 'finish' && data.generation === generation && mode === 'enroll') {
    segmenter.finish();
    const calibration = calibrate(segments.map(segment => segment.frames));
    if (segmenter.rejected || overflow) {
      calibration.template = undefined;
      calibration.error = overflow ? '8회를 초과했습니다. 5회 정도로 다시 등록해 주세요.' : `${segmenter.rejected}개 구간이 너무 짧거나 길거나 끝나지 않았습니다. 각 패턴 뒤에 쉬고 다시 등록해 주세요.`;
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
        if (segments.length < MAX_EXAMPLES) segments.push(segment); else overflow = true;
        self.postMessage({type:'count', generation, count:segments.length, overflow});
      } else if (!cooldown && template && templateDistance(template, segment.frames) < (template.version === 1 ? .12 : template.threshold)) {
        self.postMessage({type:'match', generation, seconds:segment.end - segment.start + .224});
        cooldown = 32;
      }
    }
    self.postMessage({type:'ack', generation});
  }
};
