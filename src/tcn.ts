type TcnManifest = {
  floatCount: number;
  weights: Record<string, {offset:number; length:number; shape:number[]}>;
  residualBand: {offset:number; length:number};
};

const manifestUrl = new URL('filters/iphone-hybrid-tcn128.json', document.baseURI);
const weightsUrl = new URL('filters/iphone-hybrid-tcn128.f32le', document.baseURI);
let assets: Promise<{manifest:TcnManifest; weights:Float32Array<ArrayBuffer>}> | undefined;
let devicePromise: Promise<any> | undefined;

const shader = /* wgsl */`
struct Params { n:u32, cin:u32, cout:u32, dilation:u32, weightOffset:u32, biasOffset:u32, mode:u32, _pad:u32 };
@group(0) @binding(0) var<storage, read> xbuf: array<f32>;
@group(0) @binding(1) var<storage, read> skipbuf: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
@group(0) @binding(4) var<storage, read_write> ybuf: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
  let index=gid.x;
  let total=p.n*p.cout;
  if(index>=total){return;}
  let channel=index/p.n;
  let t=index-channel*p.n;
  var sum=weights[p.biasOffset+channel];
  if(p.mode==0u){
    for(var c=0u;c<p.cin;c=c+1u){sum+=weights[p.weightOffset+channel*p.cin+c]*xbuf[c*p.n+t];}
    ybuf[index]=tanh(sum);
  } else if(p.mode==1u){
    for(var c=0u;c<p.cin;c=c+1u){
      for(var k=0u;k<3u;k=k+1u){
        let delay=(2u-k)*p.dilation;
        if(t>=delay){sum+=weights[p.weightOffset+(channel*p.cin+c)*3u+k]*xbuf[c*p.n+t-delay];}
      }
    }
    ybuf[index]=tanh(sum);
  } else if(p.mode==2u){
    for(var c=0u;c<p.cin;c=c+1u){sum+=weights[p.weightOffset+channel*p.cin+c]*xbuf[c*p.n+t];}
    ybuf[index]=skipbuf[index]+tanh(sum);
  } else {
    for(var c=0u;c<p.cin;c=c+1u){sum+=weights[p.weightOffset+c]*xbuf[c*p.n+t];}
    ybuf[t]=sum;
  }
}`;

export function tcnGpuAvailable() {
  return typeof navigator !== 'undefined' && Boolean((navigator as Navigator & {gpu?:unknown}).gpu);
}

async function loadAssets() {
  assets ??= Promise.all([fetch(manifestUrl),fetch(weightsUrl)]).then(async ([mr,wr])=>{
    if(!mr.ok||!wr.ok) throw new Error('TCN128 필터 파일을 불러오지 못했습니다.');
    const manifest=await mr.json() as TcnManifest;
    const bytes=await wr.arrayBuffer();
    if(bytes.byteLength!==manifest.floatCount*4) throw new Error('TCN128 필터 파일 크기가 올바르지 않습니다.');
    return {manifest,weights:new Float32Array(bytes)};
  }).catch(error=>{assets=undefined;throw error;});
  return assets;
}

async function getDevice() {
  if(!tcnGpuAvailable()) throw new Error('TCN128 재생에는 WebGPU가 필요합니다. 이 브라우저에서는 사용할 수 없습니다.');
  devicePromise ??= (async()=>{
    const adapter=await (navigator as Navigator & {gpu:any}).gpu.requestAdapter();
    if(!adapter) throw new Error('이 기기에서 사용할 수 있는 GPU를 찾지 못했습니다.');
    return adapter.requestDevice();
  })().catch(error=>{devicePromise=undefined;throw error;});
  return devicePromise;
}

function gpuBuffer(device:any, size:number, usage:number) {
  return device.createBuffer({size:Math.max(4,Math.ceil(size/4)*4),usage});
}

/** Run the exported 48 kHz causal TCN on overlapped chunks. */
export async function processHybridTcn(input:Float32Array,base:Float32Array,envelope:Float32Array) {
  const [{manifest,weights},device]=await Promise.all([loadAssets(),getDevice()]);
  const GPU=(globalThis as any).GPUBufferUsage;
  const shaderModule=device.createShaderModule({code:shader});
  const pipeline=device.createComputePipeline({layout:'auto',compute:{module:shaderModule,entryPoint:'main'}});
  const chunkSize=8192, context=2046, residual=new Float32Array(input.length);
  for(let start=0;start<input.length;start+=chunkSize){
    const count=Math.min(chunkSize,input.length-start), n=context+count;
    const packed=new Float32Array(n*3);
    const absStart=start-context;
    for(let i=0;i<n;i++){
      const source=absStart+i;
      if(source>=0&&source<input.length){packed[i]=input[source];packed[n+i]=base[source];packed[2*n+i]=envelope[source];}
    }
    const inputBuffer=gpuBuffer(device,packed.byteLength,GPU.STORAGE|GPU.COPY_DST);
    const a=gpuBuffer(device,n*128*4,GPU.STORAGE), b=gpuBuffer(device,n*128*4,GPU.STORAGE), c=gpuBuffer(device,n*128*4,GPU.STORAGE);
    const result=gpuBuffer(device,n*4,GPU.STORAGE|GPU.COPY_SRC);
    const readback=gpuBuffer(device,n*4,GPU.MAP_READ|GPU.COPY_DST);
    device.queue.writeBuffer(inputBuffer,0,packed);
    const encoder=device.createCommandEncoder();
    const run=(mode:number,source:any,skip:any,target:any,cin:number,cout:number,dilation:number,weightKey:string,biasKey:string)=>{
      const weight=manifest.weights[weightKey], bias=manifest.weights[biasKey];
      const params=new Uint32Array([n,cin,cout,dilation,weight.offset,bias.offset,mode,0]);
      const pbuf=gpuBuffer(device,32,GPU.UNIFORM|GPU.COPY_DST); device.queue.writeBuffer(pbuf,0,params);
      const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[
        {binding:0,resource:{buffer:source}}, {binding:1,resource:{buffer:skip}},
        {binding:2,resource:{buffer:weightsBuffer}}, {binding:3,resource:{buffer:pbuf}}, {binding:4,resource:{buffer:target}},
      ]});
      const pass=encoder.beginComputePass(); pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(n*cout/64));pass.end();
      return pbuf;
    };
    // Weight upload is shared across chunks; created lazily on the device.
    const weightsBuffer=getWeightsBuffer(device,weights,GPU);
    const paramsBuffers:any[]=[];
    paramsBuffers.push(run(0,inputBuffer,inputBuffer,a,3,128,0,'input.weight','input.bias'));
    let current=a, other=b, spare=c;
    for(let layer=0;layer<10;layer++){
      paramsBuffers.push(run(1,current,current,other,128,128,2**layer,`blocks.${layer}.conv.weight`,`blocks.${layer}.conv.bias`));
      paramsBuffers.push(run(2,other,current,spare,128,128,0,`blocks.${layer}.mix.weight`,`blocks.${layer}.mix.bias`));
      const old=current;current=spare;spare=old;
    }
    paramsBuffers.push(run(3,current,current,result,128,1,0,'head.weight','head.bias'));
    encoder.copyBufferToBuffer(result,0,readback,0,n*4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPU.MAP_READ);
    const values=new Float32Array(readback.getMappedRange().slice(0));
    residual.set(values.subarray(context,context+count),start);
    readback.unmap();
    for(const buffer of [inputBuffer,a,b,c,result,readback,...paramsBuffers]) buffer.destroy();
  }
  for(let i=0;i<residual.length;i++)residual[i]*=.1;
  const band=manifest.residualBand;
  return {residual,bandTaps:weights.subarray(band.offset,band.offset+band.length)};
}

const weightBuffers=new WeakMap<object,any>();
function getWeightsBuffer(device:any,weights:Float32Array,gpuUsage:any) {
  let buffer=weightBuffers.get(device);
  if(!buffer){buffer=gpuBuffer(device,weights.byteLength,gpuUsage.STORAGE|gpuUsage.COPY_DST);device.queue.writeBuffer(buffer,0,weights);weightBuffers.set(device,buffer);}
  return buffer;
}
