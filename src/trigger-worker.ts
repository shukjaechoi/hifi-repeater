import { MusicFeatures, musicDistance, trimMusic, HOP_SECONDS, type MusicFrame, type MusicTemplate } from './music-trigger';
let features = new MusicFeatures(), template: MusicTemplate | undefined;
let frames: MusicFrame[] = [], silent = 0, armed = false, generation = 0, mode = 'off', rate = 0;
let cooldown = 0, noise = .001;
self.onmessage = ({data}) => {
  if (data.type === 'reset') {
    generation = data.generation; mode = data.mode; template = data.template;
    features = new MusicFeatures(); frames = []; silent = 0; armed = false; cooldown = 0; rate = 0; noise = .001;
  } else if (data.type === 'finish' && data.generation === generation && mode === 'enroll') {
    const trimmed = trimMusic(frames);
    self.postMessage({type:'template', generation, template: {version:1, frames:trimmed}});
    mode = 'off'; frames = [];
  } else if (data.type === 'pcm' && data.generation === generation && mode !== 'off') {
    if (rate && rate !== data.rate) { features = new MusicFeatures(); frames = []; armed = false; }
    rate = data.rate;
    for (const frame of features.push(data.samples, rate)) {
      if (mode === 'enroll') { if (frames.length < 155) frames.push(frame); continue; }
      const quiet = frame.rms < Math.max(.004, noise * 3);
      if (quiet) noise = noise * .98 + frame.rms * .02;
      if (cooldown > 0) { cooldown--; frames = []; silent = 0; continue; }
      if (!armed) { silent = quiet ? silent + 1 : 0; if (silent >= 7) { armed = true; silent = 0; } continue; }
      if (!frames.length && quiet) continue;
      frames.push(frame); silent = quiet ? silent + 1 : 0;
      if (frames.length > 150) { frames = []; armed = false; silent = 0; continue; }
      if (silent >= 7) {
        const candidate = trimMusic(frames);
        if (template && candidate.length >= 6 && musicDistance(template.frames, candidate) < .12) {
          self.postMessage({type:'match', generation, seconds: (frames.length + 1) * HOP_SECONDS});
          cooldown = 32;
        }
        frames = []; silent = 0;
      }
    }
    self.postMessage({type:'ack', generation});
  }
};
