import {test} from 'node:test';
import assert from 'node:assert/strict';
import { MusicFeatures, MusicSegmenter, calibrate, templateDistance, musicDistance, trimMusic, validTemplate } from '../src/music-trigger.ts';
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

test('leave-one-out calibration accepts repeats and rejects isolated wrong patterns',()=>{
  const examples=[.8,.9,1,1.1,1.2].map(speed=>extract(pattern([330,392,523],speed,.2)));
  const result=calibrate(examples);
  assert.ok(result.template); assert.equal(result.scores.length,5);
  assert.ok(validTemplate(result.template));
  assert.equal(result.template.version,2);
  if(result.template.version!==2)throw Error('Wrong version');
  assert.ok(result.scores.every(score=>score<(result.template as {threshold:number}).threshold));
  assert.ok(templateDistance(result.template,extract(pattern([330,392,523],1.05,.1)))<result.template.threshold);
  assert.ok(templateDistance(result.template,extract(pattern([277,415,466])))>result.template.threshold);
  assert.ok(calibrate(examples.slice(0,3)).template);
  assert.ok(calibrate(examples.slice(0,2)).error);
  assert.ok(calibrate([...examples.slice(0,4),extract(pattern([277,415,466]))]).error);
  assert.equal(validTemplate({...result.template,threshold:1}),false);
});
test('shared pause segmentation counts repeats and rejects incomplete or oversized events',()=>{
  const sound=pattern([330,392,523]),gap=new Float32Array(16000*.5);
  const pcm=new Float32Array(gap.length*6+sound.length*5);let offset=gap.length;
  for(let i=0;i<5;i++){pcm.set(sound,offset);offset+=sound.length+gap.length;}
  const segmenter=new MusicSegmenter(),segments=[];
  for(const frame of new MusicFeatures().push(pcm,16000)){const segment=segmenter.push(frame);if(segment)segments.push(segment);}
  segmenter.finish();assert.equal(segments.length,5);assert.equal(segmenter.rejected,0);
  assert.ok(segments.every(s=>s.end>s.start&&s.end<pcm.length/16000));
  const unfinished=new MusicSegmenter();
  for(const frame of new MusicFeatures().push(pcm.slice(0,gap.length+sound.length/2),16000))unfinished.push(frame);
  unfinished.finish();assert.equal(unfinished.rejected,1);
});

test('threshold grows with observed variation while staying bounded',()=>{
  const examples=[0,.3,.6,.9,1.2].map(angle=>Array.from({length:10},()=>({
    pitch:[Math.cos(angle),Math.sin(angle),...Array(46).fill(0)],
    chroma:[Math.cos(angle),Math.sin(angle),...Array(10).fill(0)],rms:.2
  })));
  const result=calibrate(examples);
  assert.ok(result.template?.version===2);
  assert.ok(result.template.threshold>.10&&result.template.threshold<=.24);
  assert.ok(result.scores.every(score=>score<result.template.threshold));
  assert.equal(validTemplate({version:1,frames:Array.from({length:140},()=>examples[0][0])}),true);
});
test('a fading string tail counts as a pause without requiring absolute silence',()=>{
  const frame=extract(pattern([330]))[0];const segmenter=new MusicSegmenter();let found;
  for(let i=0;i<10;i++)segmenter.push({...frame,rms:.001});
  for(let i=0;i<12;i++)segmenter.push({...frame,rms:.2});
  for(let i=0;i<8;i++)found=segmenter.push({...frame,rms:.006})??found;
  assert.ok(found);assert.equal(found.frames.length,12);
});
