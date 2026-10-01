// Execute the actual app Worker in a VM, changing only TS compilation/module loading.
import fs from 'node:fs';
import vm from 'node:vm';
import {createHash} from 'node:crypto';
import ts from 'typescript';
import {fileURLToPath} from 'node:url';
const root = fileURLToPath(new URL('../../',import.meta.url));
const out = fileURLToPath(new URL('./output/',import.meta.url));
const manifest = JSON.parse(fs.readFileSync(out+'manifest.json','utf8'));
function compile(path) {return ts.transpileModule(fs.readFileSync(root+path,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;}
const exports = {};
vm.runInNewContext(compile('src/music-trigger.ts'),{exports,Float32Array,Float64Array});
let deliveredAt=0, events=[], template;
const self={postMessage(data){if(data.type==='template')template=data.template;if(data.type==='match')events.push({...data,detectedAt:deliveredAt});}};
vm.runInNewContext(compile('src/trigger-worker.ts'),{exports:{},require:()=>exports,self,Float32Array,Float64Array});
function read(name) {const b=fs.readFileSync(out+name+'.f32');return new Float32Array(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));}
let generation=0;
function reset(mode) {self.onmessage({data:{type:'reset',generation:++generation,mode,template}});events=[];}
function feed(samples) {for(let i=0;i<samples.length;i+=4096){deliveredAt=Math.min(samples.length,i+4096)/manifest.rate;self.onmessage({data:{type:'pcm',generation,samples:samples.slice(i,i+4096),rate:manifest.rate}});}}
const trigger=read('trigger'), enrollment=new Float32Array(trigger.length+Math.round(manifest.rate*.8));
enrollment.set(trigger,Math.round(manifest.rate*.4));reset('enroll');feed(enrollment);self.onmessage({data:{type:'finish',generation}});
if(!exports.validTemplate(template))throw Error('Enrollment rejected');
const results=[];
for(const condition of [{name:'background',events:[]},...manifest.conditions]){
  reset('detect');const input=read(condition.name);const started=performance.now();feed(input);const runtimeMs=performance.now()-started;
  const unmatched=new Set(events.map((_,i)=>i));
  const expected=condition.events.map(original=>{
    // Diagnostic only: known ground-truth boundaries, never used by the streaming detector.
    const candidate=exports.trimMusic(new exports.MusicFeatures().push(input.slice(Math.round(original.start*manifest.rate),Math.round(original.end*manifest.rate)),manifest.rate));
    const event={...original,oracleBoundaryDistance:exports.musicDistance(template.frames,candidate)};
    const i=[...unmatched].find(i=>events[i].detectedAt>=event.end-.1&&events[i].detectedAt<=event.end+.65);
    if(i===undefined)return {...event,detectedAt:null,latency:null};
    unmatched.delete(i);return {...event,detectedAt:events[i].detectedAt,latency:events[i].detectedAt-event.end};
  });
  results.push({name:condition.name,insertions:expected.length,detections:events.length,truePositives:expected.filter(e=>e.detectedAt!==null).length,
    falsePositives:unmatched.size,misses:expected.filter(e=>e.detectedAt===null).length,expected,falsePositiveTimes:[...unmatched].map(i=>events[i].detectedAt),runtimeMs});
}
const sourceHashes=Object.fromEntries(['src/music-trigger.ts','src/trigger-worker.ts'].map(path=>[path,createHash('sha256').update(fs.readFileSync(root+path)).digest('hex')]));
const report={manifest,sourceHashes,templateFrames:template.frames.length,results};
fs.writeFileSync(out+'results.json',JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(results,null,2));

for(const result of results.filter(r=>['background','gapped','gapped_levels'].includes(r.name))) {if(result.truePositives!==result.insertions||result.falsePositives)process.exitCode=1;}
