import {test} from 'node:test';
import assert from 'node:assert/strict';
import { MusicFeatures, musicDistance, trimMusic, validTemplate } from '../src/music-trigger.ts';
function pattern(notes: number[], speed = 1, volume = .3, rate = 16000) {
  const duration = .24 * speed, samples = new Float32Array(Math.ceil(notes.length * duration * rate));
  for (let i = 0; i < samples.length; i++) {
    const local = i / rate % duration, frequency = notes[Math.min(notes.length-1,Math.floor(i / rate / duration))];
    samples[i] = volume * Math.min(1, local / .01) * Math.max(0, 1-local/duration) * (Math.sin(2*Math.PI*frequency*i/rate)+.2*Math.sin(4*Math.PI*frequency*i/rate));
  }
  return samples;
}
function extract(samples: Float32Array, rate = 16000) { return trimMusic(new MusicFeatures().push(samples,rate)); }
test('music matcher tolerates level and moderate tempo changes but rejects unrelated notes',()=>{
  const reference = extract(pattern([330,392,523]));
  const repeat = extract(pattern([330,392,523],1.15,.08));
  const different = extract(pattern([277,415,466]));
  assert.ok(musicDistance(reference,repeat)<.12);
  assert.ok(musicDistance(reference,different)>.12);
  assert.ok(musicDistance(reference,extract(pattern([523,392,330])))>.12);
  assert.ok(validTemplate({version:1,frames:reference}));
  assert.equal(validTemplate({version:1,frames:[]}),false);
  assert.equal(musicDistance(reference,[]),Infinity);
});
test('chunk boundaries and native sample rates preserve music features',()=>{
  const samples=pattern([330,392,523]), stream=new MusicFeatures();
  const split=[];
  for(let i=0;i<samples.length;i+=777) split.push(...stream.push(samples.slice(i,i+777),16000));
  assert.deepEqual(trimMusic(split),extract(samples));
  assert.ok(musicDistance(extract(samples),extract(pattern([330,392,523],1,.3,48000),48000))<.12);
  assert.equal(extract(new Float32Array(16000)).length,0);
});
