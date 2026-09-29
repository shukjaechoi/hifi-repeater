import { test } from 'node:test';
import assert from 'node:assert/strict';
import { reduceStringSqueak } from '../src/de-esser.ts';
const rms = (samples: Float32Array) => Math.sqrt(samples.reduce((sum, x) => sum + x*x, 0) / samples.length);
for (const rate of [22050, 44100, 48000, 96000]) {
  test(`de-esser reduces prominent squeak while preserving low notes at ${rate} Hz`, () => {
    const tone = (frequency: number) => Float32Array.from({length: rate}, (_, i) => .3 * Math.sin(2*Math.PI*frequency*i/rate));
    const squeak = tone(4500), original = squeak.slice();
    const output = reduceStringSqueak(squeak, rate);
    const ratio = rms(output.subarray(rate/2)) / rms(squeak.subarray(rate/2));
    assert.ok(ratio < .65 && ratio > .45, `squeak ratio ${ratio}`);
    assert.deepEqual(squeak, original);
    assert.equal(output.length, squeak.length);
    const note = tone(220);
    assert.deepEqual(reduceStringSqueak(note, rate), note);
  });
}
test('silence, empty input, and quiet high frequencies remain unchanged', () => {
  for (const input of [new Float32Array(), new Float32Array(1000), Float32Array.from({length:4800}, (_,i)=>.001*Math.sin(2*Math.PI*4500*i/48000))]) {
    assert.deepEqual(reduceStringSqueak(input, 48000), input);
  }
});
test('burst recovery is finite and returns to unprocessed low notes', () => {
  const rate = 48000;
  const input = Float32Array.from({length:rate}, (_,i)=>.2*Math.sin(2*Math.PI*220*i/rate) + (i>rate*.2 && i<rate*.4 ? .6*Math.sin(2*Math.PI*4500*i/rate):0));
  const output = reduceStringSqueak(input,rate);
  assert.ok(output.every(x=>Number.isFinite(x) && Math.abs(x)<=1));
  const tailError = output.slice(rate*.9).map((x,i)=>x-input[rate*.9+i]);
  assert.ok(rms(tailError)<.001);
});
