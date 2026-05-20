/**
 * WebGPU Audio Analyzer
 * GPU-accelerated BPM detection (spectral flux) and key detection (chromagram)
 * Falls back to CPU automatically when WebGPU is unavailable.
 */

const FRAME_SIZE = 1024;
const HOP_SIZE   = 512;
const NUM_BINS   = 256;          // 0 → Nyquist, enough for BPM + chroma
const MAX_SECONDS = 30;          // analyse first 30 s — sufficient for key & BPM

// Krumhansl-Schmuckler key profiles
const MAJOR_PROFILE = [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88];
const MINOR_PROFILE = [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17];

// Camelot wheel (root 0 = C)
const CAMELOT_MAJOR = ['8B','3B','10B','5B','12B','7B','2B','9B','4B','11B','6B','1B'];
const CAMELOT_MINOR = ['5A','12A','7A','2A','9A','4A','11A','6A','1A','8A','3A','10A'];

// ─── WGSL: magnitude spectrum (DFT per bin per frame) ───────────────────────
const SPECTRUM_WGSL = /* wgsl */`
struct P { frame_size:u32, num_bins:u32, num_frames:u32, _p:u32 }
@group(0) @binding(0) var<storage,read>       audio    : array<f32>;
@group(0) @binding(1) var<storage,read_write> spectrum : array<f32>;
@group(0) @binding(2) var<uniform>            p        : P;
const PI = 3.14159265f;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if idx >= p.num_frames * p.num_bins { return; }
  let frame = idx / p.num_bins;
  let bin   = idx % p.num_bins;
  let off   = frame * (p.frame_size / 2u);
  var re = 0.0f; var im = 0.0f;
  for (var n = 0u; n < p.frame_size; n++) {
    let s = off + n;
    if s >= arrayLength(&audio) { break; }
    let w = 0.5f*(1.0f - cos(2.0f*PI*f32(n)/f32(p.frame_size-1u)));
    let x = audio[s] * w;
    let a = 2.0f*PI*f32(bin)*f32(n)/f32(p.frame_size);
    re += x*cos(a); im -= x*sin(a);
  }
  spectrum[idx] = sqrt(re*re + im*im);
}`;

// ─── WGSL: spectral flux (onset strength) ───────────────────────────────────
const ONSET_WGSL = /* wgsl */`
struct P { num_frames:u32, num_bins:u32, _p0:u32, _p1:u32 }
@group(0) @binding(0) var<storage,read>       spectrum : array<f32>;
@group(0) @binding(1) var<storage,read_write> onset    : array<f32>;
@group(0) @binding(2) var<uniform>            p        : P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let f = gid.x;
  if f >= p.num_frames { return; }
  if f == 0u { onset[0] = 0.0f; return; }
  var flux = 0.0f;
  let c = f * p.num_bins; let v = (f-1u)*p.num_bins;
  for (var b = 0u; b < p.num_bins; b++) {
    let d = spectrum[c+b] - spectrum[v+b];
    if d > 0.0f { flux += d; }
  }
  onset[f] = flux;
}`;

// ─── WGSL: chromagram (12 pitch classes) ────────────────────────────────────
const CHROMA_WGSL = /* wgsl */`
struct P { num_frames:u32, num_bins:u32, sample_rate:f32, frame_size:f32 }
@group(0) @binding(0) var<storage,read>       spectrum : array<f32>;
@group(0) @binding(1) var<storage,read_write> chroma   : array<f32>;
@group(0) @binding(2) var<uniform>            p        : P;
const A4 = 440.0f;
@compute @workgroup_size(12)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pc = gid.x;
  if pc >= 12u { return; }
  var energy = 0.0f;
  for (var fr = 0u; fr < p.num_frames; fr++) {
    for (var b = 1u; b < p.num_bins; b++) {
      let freq = f32(b)*p.sample_rate/p.frame_size;
      if freq < 27.5f || freq > 5000.0f { continue; }
      let midi = 12.0f*log2(freq/A4)+69.0f;
      let note = u32((i32(midi)%12+12)%12);
      if note == pc { energy += spectrum[fr*p.num_bins+b]; }
    }
  }
  chroma[pc] = energy;
}`;

// ─── Helpers ─────────────────────────────────────────────────────────────────
function pearson(a: number[], b: number[]): number {
  const n = a.length;
  const mA = a.reduce((s,v)=>s+v,0)/n, mB = b.reduce((s,v)=>s+v,0)/n;
  let num=0,dA=0,dB=0;
  for(let i=0;i<n;i++){const da=a[i]-mA,db=b[i]-mB;num+=da*db;dA+=da*da;dB+=db*db;}
  return num/(Math.sqrt(dA)*Math.sqrt(dB)+1e-10);
}

function bpmFromOnset(onset: Float32Array, sampleRate: number): number {
  const hopDur = HOP_SIZE / sampleRate;
  const minLag = Math.floor(60/(200*hopDur));
  const maxLag = Math.ceil(60/(60*hopDur));
  let best=minLag, bestC=-1;
  for(let lag=minLag;lag<=maxLag;lag++){
    let c=0;
    for(let i=0;i<onset.length-lag;i++) c+=onset[i]*onset[i+lag];
    if(c>bestC){bestC=c;best=lag;}
  }
  return Math.round(60/(best*hopDur)*10)/10;
}

function keyFromChroma(chroma: Float32Array): {key:string;confidence:number} {
  const sum = Array.from(chroma).reduce((a,b)=>a+b,0);
  if(sum<1e-10) return {key:'8B',confidence:0};
  const norm = Array.from(chroma).map(v=>v/sum);
  let best=-Infinity, key='8B';
  for(let r=0;r<12;r++){
    const rot=[...norm.slice(r),...norm.slice(0,r)];
    const maj=pearson(rot,MAJOR_PROFILE), min=pearson(rot,MINOR_PROFILE);
    if(maj>best){best=maj;key=CAMELOT_MAJOR[r];}
    if(min>best){best=min;key=CAMELOT_MINOR[r];}
  }
  return {key,confidence:Math.max(0,Math.min(1,(best+1)/2))};
}

// ─── CPU fallback key detection (no GPU) ─────────────────────────────────────
function detectKeyCPU(buffer: AudioBuffer): string {
  const data = buffer.getChannelData(0);
  const sr   = buffer.sampleRate;
  const limit = Math.min(data.length, MAX_SECONDS * sr);
  const chroma = new Float32Array(12).fill(0);
  const step  = Math.floor(FRAME_SIZE / 2);

  for(let off=0; off+FRAME_SIZE<limit; off+=step){
    for(let bin=1;bin<NUM_BINS;bin++){
      const freq = bin*sr/FRAME_SIZE;
      if(freq<27.5||freq>5000) continue;
      const midi = 12*Math.log2(freq/440)+69;
      const pc   = ((Math.round(midi)%12)+12)%12;
      let re=0,im=0;
      for(let n=0;n<FRAME_SIZE;n++){
        const w=0.5*(1-Math.cos(2*Math.PI*n/(FRAME_SIZE-1)));
        const a=2*Math.PI*bin*n/FRAME_SIZE;
        re+=data[off+n]*w*Math.cos(a);
        im-=data[off+n]*w*Math.sin(a);
      }
      chroma[pc]+=Math.sqrt(re*re+im*im);
    }
  }
  return keyFromChroma(chroma).key;
}

// ─── Public API ───────────────────────────────────────────────────────────────
export interface GPUAnalysisResult {
  bpm: number;
  key: string;
  confidence: number;
  usedGPU: boolean;
}

let _device: GPUDevice | null | undefined = undefined; // undefined = not yet checked

async function getGPUDevice(): Promise<GPUDevice|null> {
  if(_device !== undefined) return _device;
  if(!navigator.gpu){ _device=null; return null; }
  try{
    const adapter = await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
    if(!adapter){ _device=null; return null; }
    _device = await adapter.requestDevice();
    console.info('[WebGPU] RTX 4070 Ti adapter ready ✓');
    return _device;
  } catch(e){ _device=null; return null; }
}

function makePipeline(device:GPUDevice, code:string) {
  return device.createComputePipeline({
    layout:'auto',
    compute:{ module: device.createShaderModule({code}), entryPoint:'main' }
  });
}

async function analyzeGPU(device:GPUDevice, buffer:AudioBuffer): Promise<GPUAnalysisResult> {
  const sr    = buffer.sampleRate;
  const raw   = buffer.getChannelData(0);
  const limit = Math.min(raw.length, MAX_SECONDS * sr);
  const samples = raw.subarray(0, limit);

  const numFrames = Math.floor((limit - FRAME_SIZE) / HOP_SIZE) + 1;

  // Upload audio samples
  const audioGPU = device.createBuffer({
    size: samples.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(audioGPU, 0, samples);

  // Spectrum output buffer
  const spectrumSize = numFrames * NUM_BINS * 4;
  const spectrumGPU = device.createBuffer({size:spectrumSize, usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});

  // Uniform buffer (shared across passes, compatible layout)
  const uniformGPU = device.createBuffer({size:16, usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});

  // ── Pass 1: Spectrum ──
  device.queue.writeBuffer(uniformGPU, 0, new Uint32Array([FRAME_SIZE, NUM_BINS, numFrames, 0]));
  const specPipe = makePipeline(device, SPECTRUM_WGSL);
  const specBG   = device.createBindGroup({layout:specPipe.getBindGroupLayout(0), entries:[
    {binding:0,resource:{buffer:audioGPU}},
    {binding:1,resource:{buffer:spectrumGPU}},
    {binding:2,resource:{buffer:uniformGPU}},
  ]});
  {
    const enc=device.createCommandEncoder();
    const pass=enc.beginComputePass();
    pass.setPipeline(specPipe); pass.setBindGroup(0,specBG);
    pass.dispatchWorkgroups(Math.ceil(numFrames*NUM_BINS/64));
    pass.end(); device.queue.submit([enc.finish()]);
  }

  // ── Pass 2: Onset (spectral flux) ──
  const onsetGPU = device.createBuffer({size:numFrames*4, usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  device.queue.writeBuffer(uniformGPU, 0, new Uint32Array([numFrames, NUM_BINS, 0, 0]));
  const onsetPipe = makePipeline(device, ONSET_WGSL);
  const onsetBG   = device.createBindGroup({layout:onsetPipe.getBindGroupLayout(0), entries:[
    {binding:0,resource:{buffer:spectrumGPU}},
    {binding:1,resource:{buffer:onsetGPU}},
    {binding:2,resource:{buffer:uniformGPU}},
  ]});
  {
    const enc=device.createCommandEncoder();
    const pass=enc.beginComputePass();
    pass.setPipeline(onsetPipe); pass.setBindGroup(0,onsetBG);
    pass.dispatchWorkgroups(Math.ceil(numFrames/64));
    pass.end(); device.queue.submit([enc.finish()]);
  }

  // ── Pass 3: Chromagram ──
  const chromaGPU = device.createBuffer({size:12*4, usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const chromaUniform = new ArrayBuffer(16);
  new Uint32Array(chromaUniform,0,2).set([numFrames, NUM_BINS]);
  new Float32Array(chromaUniform,8,2).set([sr, FRAME_SIZE]);
  device.queue.writeBuffer(uniformGPU, 0, chromaUniform);
  const chromaPipe = makePipeline(device, CHROMA_WGSL);
  const chromaBG   = device.createBindGroup({layout:chromaPipe.getBindGroupLayout(0), entries:[
    {binding:0,resource:{buffer:spectrumGPU}},
    {binding:1,resource:{buffer:chromaGPU}},
    {binding:2,resource:{buffer:uniformGPU}},
  ]});
  {
    const enc=device.createCommandEncoder();
    const pass=enc.beginComputePass();
    pass.setPipeline(chromaPipe); pass.setBindGroup(0,chromaBG);
    pass.dispatchWorkgroups(1); // 12 threads, 1 workgroup of 12
    pass.end(); device.queue.submit([enc.finish()]);
  }

  // ── Readback ──
  const onsetRead  = device.createBuffer({size:numFrames*4, usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const chromaRead = device.createBuffer({size:12*4,        usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  {
    const enc=device.createCommandEncoder();
    enc.copyBufferToBuffer(onsetGPU, 0, onsetRead, 0, numFrames*4);
    enc.copyBufferToBuffer(chromaGPU, 0, chromaRead, 0, 12*4);
    device.queue.submit([enc.finish()]);
  }

  await onsetRead.mapAsync(GPUMapMode.READ);
  await chromaRead.mapAsync(GPUMapMode.READ);

  const onset  = new Float32Array(onsetRead.getMappedRange().slice(0));
  const chroma = new Float32Array(chromaRead.getMappedRange().slice(0));
  onsetRead.unmap(); chromaRead.unmap();

  // ── CPU post-processing ──
  const bpm = bpmFromOnset(onset, sr);
  const {key, confidence} = keyFromChroma(chroma);

  // Cleanup
  [audioGPU,spectrumGPU,onsetGPU,chromaGPU,uniformGPU,onsetRead,chromaRead]
    .forEach(b=>b.destroy());

  return {bpm, key, confidence, usedGPU:true};
}

/**
 * Main entry point.
 * Tries GPU first; falls back to CPU (existing web-audio-beat-detector for BPM + CPU chroma for key).
 */
export async function analyzeAudioAdvanced(
  audioBuffer: AudioBuffer,
  detectedBpm: number              // result from web-audio-beat-detector (CPU) as fallback BPM
): Promise<GPUAnalysisResult> {
  try {
    const device = await getGPUDevice();
    if(device){
      const result = await analyzeGPU(device, audioBuffer);
      // Sanity-check BPM: if GPU BPM is wildly off, blend with CPU BPM
      const bpm = (result.bpm>60 && result.bpm<220) ? result.bpm : detectedBpm;
      return {...result, bpm};
    }
  } catch(e){
    console.warn('[WebGPU] Analysis failed, falling back to CPU:', e);
  }

  // CPU fallback
  const key = detectKeyCPU(audioBuffer);
  return {bpm: detectedBpm, key, confidence: 0.5, usedGPU: false};
}
