import { test } from 'node:test';
import assert from 'node:assert/strict';
import { concatenate, validRange, wav } from '../src/audio-utils.ts';
test('PCM chunks preserve the final partial block and sample order',()=>{
  assert.deepEqual([...concatenate([new Float32Array([1,.5]),new Float32Array([-.5])])],[1,.5,-.5]);
});
test('WAV encodes clipped signed PCM and sample rate correctly',()=>{
  const buffer=wav(new Float32Array([-2,-1,0,1,2]),48000), v=new DataView(buffer);
  assert.equal(buffer.byteLength,54); assert.equal(v.getUint32(24,true),48000);
  assert.equal(v.getUint32(40,true),10);
  assert.deepEqual(Array.from({length:5},(_,i)=>v.getInt16(44+i*2,true)),[-32768,-32768,0,32767,32767]);
});
test('loop selection remains ordered and within recording',()=>{
  assert.deepEqual(validRange(-1,20,3),[0,3]);
  const [start,end]=validRange(5,0,3); assert.ok(start<end); assert.equal(end,3);
});

test('recorder worklet flushes partial PCM before completion without monitoring input', async()=>{
  const { readFile } = await import('node:fs/promises');
  const { runInNewContext } = await import('node:vm');
  let Recorder: any; const sent: any[]=[];
  class Processor { port={postMessage:(data: any)=>sent.push(data),onmessage:null as any}; }
  runInNewContext(await readFile(new URL('../public/recorder-worklet.js',import.meta.url),'utf8'),{AudioWorkletProcessor:Processor,registerProcessor:(_:string,ctor:any)=>{Recorder=ctor;},Float32Array});
  const recorder=new Recorder();
  recorder.process([[new Float32Array(4096).fill(.25)]]);
  recorder.process([[new Float32Array([.5,-.5])]]);
  recorder.port.onmessage({data:'stop'});
  assert.equal(sent[0].samples.length,4096);
  assert.deepEqual([...sent[1].samples],[.5,-.5]);
  assert.deepEqual(sent[2].done,true);
  recorder.process([[new Float32Array([1])]]);
  assert.equal(sent.length,3);
});
