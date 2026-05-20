/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import * as React from 'react';
import { GoogleGenAI } from "@google/genai";
import { useState, useEffect, useRef, useCallback } from 'react';
import { Play, Pause, Square, Radio, SkipBack, Download, Headphones, Mic2, Save } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { analyze } from 'web-audio-beat-detector';
import { analyzeAudioAdvanced, analyzeChordMap, findBestHarmonicMixPoints, type ChordSegment, type HarmonicMixPoint } from './webgpu-analyzer';

// AI client created once at module level (avoids re-instantiation on every render)
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || '' });

type MixPoint = 'outro' | 'mid_break' | 'post_drop' | 'early_cut';

interface AiTransitionPlan {
  technique: 'filter_sweep' | 'echo_out' | 'cut' | 'blend';
  transitionDuration: number;
  bassSwapBeat: number;
  advice: string;
  energy: 'maintain' | 'boost' | 'drop';
  mixPoint?: MixPoint;
  warning: string | null;
}

// Where in the OUTGOING track the autopilot should fire the mix.
// Maps mixPoint → fraction of track duration consumed before triggering.
const MIX_POINT_RATIO: Record<MixPoint, number> = {
  early_cut: 0.40,
  mid_break: 0.50,
  post_drop: 0.65,
  outro:     0.70,
};

// Multimodal track profile — built once when track is loaded by sending
// audio snippets to Gemini. Cached on the deck and reused for every transition.
interface AiTrackProfile {
  genre: string;
  mood: string;
  instruments: string[];
  vocalPresence: 'none' | 'background' | 'lead';
  energyArc: 'building' | 'sustained' | 'descending' | 'wave';
  bassWeight: 'sub' | 'punchy' | 'light';
  mixInTechnique: 'cut' | 'blend' | 'filter_sweep' | 'echo_out';
  mixOutTechnique: 'cut' | 'blend' | 'filter_sweep' | 'echo_out';
  phase: 1 | 2 | 3 | 4;
  realBpm: number; // AI's estimate of the REAL BPM after listening — overrides the detector if mismatched
  recommendedMixPoint: MixPoint; // when the autopilot should fire the OUTGOING mix
  notes: string;
}

// Ground-truth BPM ranges per phase (provided by the curator).
// Used to detect halftime/doubletime errors from the audio analyser.
const PHASE_BPM_RANGE: Record<1 | 2 | 3 | 4, [number, number]> = {
  1: [85, 110],   // Halftime/Neurohop
  2: [125, 140],  // Breakbeat/Cinematic
  3: [168, 174],  // Liquid D&B
  4: [172, 178],  // Rollers/Peak D&B
};

// Convert an AudioBuffer slice → mono 16kHz PCM WAV → base64 (Gemini-friendly, ~256KB per 12s)
function audioBufferToWavBase64(buffer: AudioBuffer, startSec: number, durationSec: number): string {
  const targetRate = 16000;
  const sourceRate = buffer.sampleRate;
  const ratio = sourceRate / targetRate;
  const startSample = Math.max(0, Math.floor(startSec * sourceRate));
  const sourceLen = buffer.length - startSample;
  const outSamples = Math.min(
    Math.floor(durationSec * targetRate),
    Math.floor(sourceLen / ratio)
  );
  if (outSamples <= 0) return '';

  // Mono mix
  const numCh = buffer.numberOfChannels;
  const ch0 = buffer.getChannelData(0);
  const ch1 = numCh > 1 ? buffer.getChannelData(1) : null;

  const pcm = new Int16Array(outSamples);
  for (let i = 0; i < outSamples; i++) {
    const srcIdx = Math.floor(startSample + i * ratio);
    let s = ch0[srcIdx] || 0;
    if (ch1) s = (s + (ch1[srcIdx] || 0)) * 0.5;
    s = Math.max(-1, Math.min(1, s));
    pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }

  // WAV header (44 bytes) + PCM data
  const wav = new ArrayBuffer(44 + pcm.length * 2);
  const v = new DataView(wav);
  const ws = (off: number, str: string) => { for (let i = 0; i < str.length; i++) v.setUint8(off + i, str.charCodeAt(i)); };
  ws(0, 'RIFF');
  v.setUint32(4, 36 + pcm.length * 2, true);
  ws(8, 'WAVE'); ws(12, 'fmt ');
  v.setUint32(16, 16, true);   // fmt chunk size
  v.setUint16(20, 1, true);    // PCM
  v.setUint16(22, 1, true);    // mono
  v.setUint32(24, targetRate, true);
  v.setUint32(28, targetRate * 2, true); // byte rate
  v.setUint16(32, 2, true);    // block align
  v.setUint16(34, 16, true);   // bits per sample
  ws(36, 'data');
  v.setUint32(40, pcm.length * 2, true);
  for (let i = 0; i < pcm.length; i++) v.setInt16(44 + i * 2, pcm[i], true);

  // ArrayBuffer → base64 (chunked to avoid call-stack issues)
  const bytes = new Uint8Array(wav);
  let bin = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK) as any);
  }
  return btoa(bin);
}



function BeatSyncOverlay({ timeA, timeB, bpmA, bpmB }: { timeA: number, timeB: number, bpmA: number, bpmB: number }) {
  const hasA = bpmA > 0;
  const hasB = bpmB > 0;
  if (!hasA && !hasB) return null;

  // Per-deck beat phase 0..1
  const phaseA = hasA ? (timeA % (60 / bpmA)) / (60 / bpmA) : 0;
  const phaseB = hasB ? (timeB % (60 / bpmB)) / (60 / bpmB) : 0;

  // Sync quality (circular distance between phases)
  let diff = Math.abs(phaseA - phaseB);
  if (diff > 0.5) diff = 1 - diff;
  const syncQuality = (hasA && hasB) ? Math.max(0, 1 - diff * 2) : 0;
  const bpmDiff = (hasA && hasB) ? Math.abs(bpmA - bpmB) : 99;
  const isLocked = syncQuality > 0.92 && bpmDiff < 0.5;

  // Beat-tick pulse on each downbeat for visual emphasis at center
  const pulseA = hasA ? Math.max(0, 1 - phaseA * 4) : 0;
  const pulseB = hasB ? Math.max(0, 1 - phaseB * 4) : 0;

  return (
    <>
      {/* Centered vertical NOW reference line — bright green when locked */}
      <div
        className="absolute left-1/2 top-0 bottom-0 w-[2px] -translate-x-1/2 z-30 pointer-events-none"
        style={{
          background: '#39ff14',
          opacity: isLocked ? 0.95 : 0.4,
          boxShadow: isLocked
            ? '0 0 16px #39ff14, 0 0 32px rgba(57,255,20,0.6)'
            : '0 0 6px rgba(57,255,20,0.5)',
          transition: 'opacity 120ms, box-shadow 120ms',
        }}
      />

      {/* Pulse rings on the center line at each deck downbeat */}
      {hasA && (
        <div
          className="absolute left-1/2 top-[25%] -translate-x-1/2 -translate-y-1/2 z-30 pointer-events-none rounded-full"
          style={{
            width: `${8 + pulseA * 14}px`,
            height: `${8 + pulseA * 14}px`,
            border: `1.5px solid #39ff14`,
            opacity: pulseA * 0.8,
          }}
        />
      )}
      {hasB && (
        <div
          className="absolute left-1/2 top-[75%] -translate-x-1/2 -translate-y-1/2 z-30 pointer-events-none rounded-full"
          style={{
            width: `${8 + pulseB * 14}px`,
            height: `${8 + pulseB * 14}px`,
            border: `1.5px solid #39ff14`,
            opacity: pulseB * 0.8,
          }}
        />
      )}

      {/* SYNC status badge — top-center */}
      {hasA && hasB && (
        <div
          className="absolute top-1 left-1/2 -translate-x-1/2 z-40 pointer-events-none text-[8px] font-black tracking-[0.2em] uppercase px-2 py-0.5 rounded-sm"
          style={{
            color: isLocked ? '#000' : '#39ff14',
            background: isLocked ? '#39ff14' : 'rgba(0,0,0,0.6)',
            border: `1px solid #39ff14${isLocked ? '' : '60'}`,
            boxShadow: isLocked ? '0 0 10px #39ff14' : 'none',
          }}
        >
          {isLocked ? '✓ BEATS LOCKED' : `SYNC ${Math.round(syncQuality * 100)}%`}
        </div>
      )}

      {/* BPM diff indicator — bottom-center */}
      {hasA && hasB && !isLocked && (
        <div
          className="absolute bottom-1 left-1/2 -translate-x-1/2 z-40 pointer-events-none text-[7px] font-mono tracking-wider px-1.5 py-0.5 rounded-sm"
          style={{
            color: bpmDiff < 0.5 ? '#39ff14' : '#ffcc00',
            background: 'rgba(0,0,0,0.6)',
            border: `1px solid ${bpmDiff < 0.5 ? '#39ff14' : '#ffcc00'}40`,
          }}
        >
          ΔBPM {bpmDiff.toFixed(2)}
        </div>
      )}
    </>
  );
}

// Old PhaseMeter kept as no-op stub (still referenced elsewhere? no — safe to remove)
function PhaseMeter({ timeA, timeB, bpmA, bpmB }: { timeA: number, timeB: number, bpmA: number, bpmB: number }) {
  const hasA = bpmA > 0;
  const hasB = bpmB > 0;

  // Each row shows a 4-beat bar that scrolls left at the deck's BPM
  // The center vertical line is the "now" reference. When both decks' beat ticks
  // cross the center at the same time → tracks are perfectly empatados.
  const BEATS_VISIBLE = 8; // 2 bars
  const TICKS = Array.from({ length: BEATS_VISIBLE }, (_, i) => i);

  const renderRow = (time: number, bpm: number, label: string, active: boolean, intensity: number) => {
    const beatLen = bpm > 0 ? 60 / bpm : 1;
    // Position offset as a fraction (0..1) of one beat. Tracks scroll right→left.
    const phase = bpm > 0 ? ((time % beatLen) / beatLen) : 0;
    // Width of one beat in % of the row (visible beats fill the row)
    const beatPct = 100 / BEATS_VISIBLE;
    const offsetPct = -phase * beatPct;

    return (
      <div className={`relative h-5 rounded-sm border border-white/5 overflow-hidden ${active ? 'bg-[#0a0a0a]' : 'bg-[#080808]'}`}>
        {/* Scrolling beat grid */}
        {active && (
          <div
            className="absolute top-0 bottom-0 flex items-center"
            style={{
              left: `${offsetPct}%`,
              width: `${BEATS_VISIBLE * 2 * beatPct}%`,
            }}
          >
            {TICKS.concat(TICKS).map((i, idx) => {
              const isDownbeat = i % 4 === 0;
              return (
                <div
                  key={idx}
                  className="flex items-center justify-start"
                  style={{ width: `${beatPct}%` }}
                >
                  <div
                    className={`${isDownbeat ? 'w-[2px] h-4' : 'w-[1px] h-2'} bg-[#39ff14]`}
                    style={{
                      opacity: isDownbeat ? 0.95 * intensity : 0.45 * intensity,
                      boxShadow: isDownbeat ? `0 0 6px #39ff14` : 'none',
                    }}
                  />
                </div>
              );
            })}
          </div>
        )}
        {/* Label */}
        <div className="absolute left-1.5 top-1/2 -translate-y-1/2 text-[7px] font-black tracking-widest text-[#39ff14]/60 uppercase pointer-events-none">
          {label} · {bpm > 0 ? bpm.toFixed(1) : '—'}
        </div>
      </div>
    );
  };

  // Sync quality: compare beat phases (only when both have BPM)
  let syncQuality = 0;
  if (hasA && hasB) {
    const beatLenA = 60 / bpmA;
    const beatLenB = 60 / bpmB;
    const pA = (timeA % beatLenA) / beatLenA;
    const pB = (timeB % beatLenB) / beatLenB;
    let diff = Math.abs(pA - pB);
    if (diff > 0.5) diff = 1 - diff;
    syncQuality = Math.max(0, 1 - diff * 2);
  }
  const bpmDiff = hasA && hasB ? Math.abs(bpmA - bpmB) : 0;
  const isLocked = syncQuality > 0.92 && bpmDiff < 0.5;

  return (
    <div className="w-full flex flex-col gap-1 relative">
      {renderRow(timeA, bpmA, 'A', hasA, 1)}
      {renderRow(timeB, bpmB, 'B', hasB, 0.7)}

      {/* Center NOW reference — vertical line through both rows */}
      <div
        className="absolute left-1/2 top-0 bottom-0 w-[2px] -translate-x-1/2 pointer-events-none"
        style={{
          background: '#39ff14',
          opacity: isLocked ? 1 : 0.35,
          boxShadow: isLocked ? '0 0 12px #39ff14, 0 0 24px #39ff14' : '0 0 4px #39ff14',
          transition: 'opacity 120ms, box-shadow 120ms',
        }}
      />

      {/* Sync status badge */}
      {hasA && hasB && (
        <div
          className="absolute right-1 top-1/2 -translate-y-1/2 text-[7px] font-black tracking-widest uppercase pointer-events-none px-1.5 py-0.5 rounded-sm"
          style={{
            color: isLocked ? '#000' : '#39ff14',
            background: isLocked ? '#39ff14' : 'transparent',
            border: `1px solid #39ff14${isLocked ? '' : '40'}`,
          }}
        >
          {isLocked ? 'LOCKED' : `SYNC ${Math.round(syncQuality * 100)}%`}
        </div>
      )}
    </div>
  );
}

const DAW_PANEL = `
  bg-daw-panel rounded-xl
  shadow-[10px_10px_20px_#0b0b0d,-10px_-10px_20px_#212123] text-[#cccccc]
  border border-white/5
`;

const DAW_HEADER_PANEL = `
  bg-daw-bg shadow-[inset_0_-2px_10px_#0b0b0d] flex items-center px-4 gap-4 h-12 select-none
  border-b border-daw-border
`;

const LIQUID_GLASS_BTN = `
  inline-flex items-center justify-center align-middle select-none font-sans font-bold text-center 
  text-[#aaa] text-xs rounded-lg bg-daw-panel
  shadow-[4px_4px_8px_#0b0b0d,-4px_-4px_8px_#252527]
  hover:text-white hover:bg-daw-panel-light active:scale-95 transition-all duration-200 cursor-pointer
  active:shadow-[inset_4px_4px_8px_#0b0b0d,inset_-4px_-4px_8px_#252527]
  border border-white/5 disabled:opacity-20 disabled:cursor-not-allowed
`;

const LED_METER_BG = `bg-[#1a1a1a] border border-[#121212] rounded-sm overflow-hidden`;

const SLIDER_THUMB = `
  appearance-none w-4 h-4 bg-white/80 rounded-full cursor-pointer 
  shadow-[0_0_10px_rgba(255,255,255,0.5)] border border-white/50
`;

// --- TYPES ---
interface DeckState {
  id: 'A' | 'B';
  isPlaying: boolean;
  trackName: string;
  playbackRate: number;
  bend: number;
  gain: number;
  low: number;
  mid: number;
  high: number;
  bpm: number; 
  baseBpm: number;
  cuePoint: number;
  hotCues: (number | null)[];
  isAnalyzing: boolean;
  syncLocked: boolean;
  key?: string; // e.g., "1A", "8B"
  pitchCurve: 'linear' | 'log' | 'exp';
  loopStart?: number | null;
  loopEnd?: number | null;
  isLooping?: boolean;
  isMaster: boolean;
  structureMarkers: { time: number; label: string; type: 'intro' | 'build' | 'drop' | 'break' | 'outro' }[];
  aiProfile?: AiTrackProfile; // built once via Gemini multimodal when track loads
  chordMap?: ChordSegment[];  // harmonic chord analysis per 2-bar segment
  fxReverb: number;  // 0-1 wet amount
  fxDelay: number;   // 0-1 wet amount
}

// Harmonic Compatibility (Camelot Wheel)
const getCamelotDiff = (key1: string, key2: string) => {
  const match1 = key1.match(/(\d+)([AB])/);
  const match2 = key2.match(/(\d+)([AB])/);
  if (!match1 || !match2) return 0;
  
  const val1 = parseInt(match1[1]);
  const type1 = match1[2];
  const val2 = parseInt(match2[1]);
  const type2 = match2[2];

  let diff = 0;
  // Rotation diff
  let rotational = Math.abs(val1 - val2);
  if (rotational > 6) rotational = 12 - rotational;
  
  if (type1 === type2) {
    if (rotational <= 1) return 0; // Compatible
    return val1 > val2 ? -1 : 1; // Suggest shift
  } else {
    if (rotational === 0) return 0; // Relative major/minor compatible
    return 0; // Too complex for simple shift, stick to current
  }
};

function generateImpulse(ctx: AudioContext, duration = 2.5, decay = 2.0): AudioBuffer {
  const sr = ctx.sampleRate;
  const len = Math.floor(sr * duration);
  const buf = ctx.createBuffer(2, len, sr);
  for (let c = 0; c < 2; c++) {
    const d = buf.getChannelData(c);
    for (let i = 0; i < len; i++) {
      d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, decay);
    }
  }
  return buf;
}

function MixerSpectrum({ analyserA, analyserB, isActive }: {
  analyserA: AnalyserNode | null;
  analyserB: AnalyserNode | null;
  isActive: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current || !isActive) return;
    const canvas = canvasRef.current;
    const ctx2d = canvas.getContext('2d');
    if (!ctx2d) return;

    const BARS = 28;
    let raf: number;

    const draw = () => {
      ctx2d.clearRect(0, 0, canvas.width, canvas.height);
      const bufA = new Uint8Array(128);
      const bufB = new Uint8Array(128);
      if (analyserA) analyserA.getByteFrequencyData(bufA);
      if (analyserB) analyserB.getByteFrequencyData(bufB);

      const totalW = canvas.width;
      const barW = Math.floor(totalW / BARS) - 1;

      for (let i = 0; i < BARS; i++) {
        const binIdx = Math.min(127, Math.floor(Math.pow(i / BARS, 1.5) * 80));
        const vA = (bufA[binIdx] || 0) / 255;
        const vB = (bufB[binIdx] || 0) / 255;
        const combined = Math.max(vA, vB);
        const hPx = Math.max(1, Math.floor(combined * canvas.height));

        const x = i * (barW + 1);
        // Color: cyan-green for low-mid, yellow for high
        const t = i / BARS;
        const r = Math.floor(t * 100);
        const g = Math.floor(200 + t * 55);
        const b = Math.floor(255 * (1 - t));

        const grad = ctx2d.createLinearGradient(0, canvas.height, 0, canvas.height - hPx);
        grad.addColorStop(0, `rgba(${r},${g},${b},0.9)`);
        grad.addColorStop(1, `rgba(${r},${g},${b},0.2)`);
        ctx2d.fillStyle = grad;
        ctx2d.fillRect(x, canvas.height - hPx, barW, hPx);
      }
      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [analyserA, analyserB, isActive]);

  return (
    <canvas
      ref={canvasRef}
      width={240}
      height={40}
      className="w-full h-full"
    />
  );
}

export default function App() {
  const [audioStarted, setAudioStarted] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [gpuStatus, setGpuStatus] = useState<'idle'|'analyzing'|'gpu'|'cpu'>('idle');
  const [ramMB,    setRamMB]    = useState(0);
  const [ramMax,   setRamMax]   = useState(1);
  const [gpuLoad,  setGpuLoad]  = useState(0);



  // Audio Refs
  const audioCtx = useRef<AudioContext | null>(null);
  const masterGain = useRef<GainNode | null>(null);
  const destination = useRef<MediaStreamAudioDestinationNode | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const recordingBlobs = useRef<Blob[]>([]);

  // Crossfader state
  const [crossfader, setCrossfader] = useState(0.5); // 0 to 1

  // Decks Management
  const [deckA, setDeckA] = useState<DeckState>({ id: 'A', isPlaying: false, trackName: '', playbackRate: 1, bend: 0, gain: 0.8, low: 0, mid: 0, high: 0, bpm: 128, baseBpm: 128, cuePoint: 0, hotCues: [null, null, null, null], isAnalyzing: false, syncLocked: false, isMaster: false, key: '8A', pitchCurve: 'linear', loopStart: null, loopEnd: null, isLooping: false, structureMarkers: [], fxReverb: 0, fxDelay: 0 });
  const [deckB, setDeckB] = useState<DeckState>({ id: 'B', isPlaying: false, trackName: '', playbackRate: 1, bend: 0, gain: 0.8, low: 0, mid: 0, high: 0, bpm: 128, baseBpm: 128, cuePoint: 0, hotCues: [null, null, null, null], isAnalyzing: false, syncLocked: false, isMaster: false, key: '3B', pitchCurve: 'linear', loopStart: null, loopEnd: null, isLooping: false, structureMarkers: [], fxReverb: 0, fxDelay: 0 });
  const [isAiMixing, setIsAiMixing] = useState(false);

  // Poll RAM (Chrome performance.memory) + estimate GPU load
  useEffect(() => {
    const tick = () => {
      const mem = (performance as any).memory;
      if (mem) {
        const used = Math.round(mem.usedJSHeapSize / 1024 / 1024);
        const total = Math.round(mem.jsHeapSizeLimit / 1024 / 1024);
        setRamMB(used);
        setRamMax(total);
      }
      // GPU load: baseline + boost when analyzing or AI mixing
      const base  = (Math.random() * 6 + 4);
      const extra = gpuStatus === 'analyzing' ? 55 : isAiMixing ? 35 : 0;
      setGpuLoad(Math.min(99, Math.max(1, base + extra)));
    };
    tick();
    const id = setInterval(tick, 800);
    return () => clearInterval(id);
  }, [gpuStatus, isAiMixing]);
  
  // --- AI AUTOPILOT STATE ---
  const [showAutopilot, setShowAutopilot] = useState(false);
  const [playlist, setPlaylist] = useState<File[]>([]);
  const playlistRef = useRef<File[]>([]);
  useEffect(() => { playlistRef.current = playlist; }, [playlist]);
  const [directorPrompt, setDirectorPrompt] = useState(`Eres el "Director Automático", un DJ Headliner experto curando una sesión de Bass Music.
El sistema te enviará el nombre y BPM de la pista actual (DECK A) y la entrante (DECK B). Como las canciones están en orden aleatorio, DEBES clasificar la pista del DECK B usando este Diccionario de Fases y aplicar la técnica correspondiente.
DICCIONARIO DE FASES Y REGLAS DE MEZCLA (Basado en Artista/Track):
FASE 1: Pantano Técnico (Halftime/Neurohop ~ 85-110 BPM)
- Artistas: Ekcle, Poseidon, Proxima, Xsetra, Kursa & Seppa, Noisia, Audeka & Rawtekk, Vorso, mindvacy, COPYCATT, Frequent, RUN DMT, Skope.
- REGLA: Usa "cut" (si es agresivo) o "echo_out". NUNCA "blend" para no chocar sub-bajos densos.
- Params: transitionDuration: 8, bassSwapBeat: 4, energy: "drop".
FASE 2: El Puente Táctico (Breakbeat/Cinematic ~ 130 BPM)
- Artistas: Energy Airforce Soundtrack.
- REGLA: Usa "filter_sweep" para barrer el track anterior.
- Params: transitionDuration: 16, bassSwapBeat: 8, energy: "boost".
FASE 3: Vuelo Eufórico (Liquid/Soulful D&B ~ 170-174 BPM)
- Artistas: Artificial Intelligence, Technimatic, Halogenix, Makoto, 4hero, DJ Marky & SOLAH, MC Conrad.
- REGLA: Usa "blend". Cruza acordes y pads suavemente con igual potencia.
- Params: transitionDuration: 32, bassSwapBeat: 16, energy: "maintain".
FASE 4: Peak Time & Cierre (Rollers/Dancefloor ~ 174-175 BPM)
- Artistas: Influx Datum, High Contrast, Fred V, Inja x Whiney, L-Side & MC Fats, Alibi, DJ Marky, S.P.Y, DJ Fresh, Lenny Fontana.
- REGLA: Usa "filter_sweep" agresivo o "blend". Energía debe subir.
- Params: transitionDuration: 16 a 24, bassSwapBeat: 8, energy: "boost".
CASO EXTREMO: Si saltas de Fase 1 (90 BPM) a Fase 4 (174 BPM) o viceversa, OBLIGATORIO "echo_out" rápido (transitionDuration: 8).
FORMATO ESTRICTO: Responde SOLO con JSON válido, sin markdown.`);

  // CARGAR configuracion al arrancar
  useEffect(() => {
    try {
      const saved = localStorage.getItem('ljdj-settings');
      if (!saved) return;
      const s = JSON.parse(saved);
      if (s.crossfader !== undefined) setCrossfader(s.crossfader);
      if (s.directorPrompt) setDirectorPrompt(s.directorPrompt);
      if (s.deckA) setDeckA(prev => ({ ...prev, gain: s.deckA.gain ?? prev.gain, low: s.deckA.low ?? prev.low, mid: s.deckA.mid ?? prev.mid, high: s.deckA.high ?? prev.high }));
      if (s.deckB) setDeckB(prev => ({ ...prev, gain: s.deckB.gain ?? prev.gain, low: s.deckB.low ?? prev.low, mid: s.deckB.mid ?? prev.mid, high: s.deckB.high ?? prev.high }));
    } catch(e) {}
  }, []);

  // GUARDAR configuracion al cambiar
  useEffect(() => {
    try {
      localStorage.setItem('ljdj-settings', JSON.stringify({
        crossfader,
        directorPrompt,
        deckA: { gain: deckA.gain, low: deckA.low, mid: deckA.mid, high: deckA.high },
        deckB: { gain: deckB.gain, low: deckB.low, mid: deckB.mid, high: deckB.high },
      }));
    } catch(e) {}
  }, [crossfader, directorPrompt, deckA.gain, deckA.low, deckA.mid, deckA.high, deckB.gain, deckB.low, deckB.mid, deckB.high]);

  const moveTrack = useCallback((index: number, direction: -1 | 1) => {
    setPlaylist(prev => {
      const next = [...prev];
      const target = index + direction;
      if (target < 0 || target >= next.length) return prev;
      [next[index], next[target]] = [next[target], next[index]];
      return next;
    });
  }, []);

  const [isAutopilotActive, setIsAutopilotActive] = useState(false);
  const [autopilotActiveDeckUI, setAutopilotActiveDeckUI] = useState<'A' | 'B'>('A');
  // --- AUTOPILOT REFS ---
  const isAutopilotActiveRef = useRef(false);
  useEffect(() => { isAutopilotActiveRef.current = isAutopilotActive; }, [isAutopilotActive]);
  const autopilotActiveDeckRef = useRef<'A' | 'B'>('A');
  const playlistIndexRef = useRef(0);
  const mixTriggeredRef = useRef(false);
  const prevAiMixingRef = useRef(false);
  
  const [aiMixAdvice, setAiMixAdvice] = useState<string | null>(null);
  const isCueingA = useRef(false);
  const isCueingB = useRef(false);
  // --- MASTER TEMPO LOCK LOGIC ---
  useEffect(() => {
    if (deckA.isMaster && deckB.syncLocked) { // Using syncLocked as the 'slave' state if needed, or just enforce if master exists
      const targetBpm = deckA.bpm;
      if (Math.abs(deckB.bpm - targetBpm) > 0.01) {
        setDeckB(prev => ({
          ...prev,
          bpm: targetBpm,
          playbackRate: targetBpm / prev.baseBpm
        }));
      }
    } else if (deckB.isMaster && deckA.syncLocked) {
      const targetBpm = deckB.bpm;
      if (Math.abs(deckA.bpm - targetBpm) > 0.01) {
        setDeckA(prev => ({
          ...prev,
          bpm: targetBpm,
          playbackRate: targetBpm / prev.baseBpm
        }));
      }
    }
  }, [deckA.isMaster, deckB.isMaster, deckA.bpm, deckB.bpm, deckA.syncLocked, deckB.syncLocked]);

  const toggleMaster = (id: 'A' | 'B') => {
    if (id === 'A') {
      setDeckA(prev => ({ ...prev, isMaster: !prev.isMaster }));
      if (!deckA.isMaster) setDeckB(prev => ({ ...prev, isMaster: false }));
    } else {
      setDeckB(prev => ({ ...prev, isMaster: !prev.isMaster }));
      if (!deckB.isMaster) setDeckA(prev => ({ ...prev, isMaster: false }));
    }
  };

  const [mixIntensity, setMixIntensity] = useState(5);

  // Big Number Pulse based on beat
  const [pulse, setPulse] = useState(1);
  useEffect(() => {
    const interval = setInterval(() => {
      const activeBpm = deckA.isPlaying ? deckA.bpm : (deckB.isPlaying ? deckB.bpm : 128);
      const beatDuration = 60000 / activeBpm;
      
      // Simple pulse animation on each beat
      setPulse(1.1);
      setTimeout(() => setPulse(1), 100);
    }, 60000 / (deckA.isPlaying ? deckA.bpm : (deckB.isPlaying ? deckB.bpm : 128)));
    
    return () => clearInterval(interval);
  }, [deckA.isPlaying, deckA.bpm, deckB.isPlaying, deckB.bpm]);

  // Update intensity based on crossfader and ai mix
  useEffect(() => {
    const baseIntensity = Math.floor(Math.abs(0.5 - crossfader) * 10) + 1;
    const aiBonus = isAiMixing ? 3 : 0;
    setMixIntensity(Math.min(10, baseIntensity + aiBonus));
  }, [crossfader, isAiMixing]);

  // AI client is created once outside via module-level ref

  const refineBpmWithAi = async (id: 'A' | 'B', trackName: string, detectedBpm: number) => {
    if (!process.env.GEMINI_API_KEY) return;
    try {
      const response = await ai.models.generateContent({
        model: 'gemini-2.0-flash',
        contents: `You are a DJ assistant. For the track named "${trackName}", what is its standard BPM? The detected BPM is ${detectedBpm.toFixed(1)}. Respond ONLY with the correct BPM number. If unknown, return ${detectedBpm.toFixed(1)}.`,
      });
      const refined = parseFloat(response.text.trim());
      if (!isNaN(refined)) {
        const setter = id === 'A' ? setDeckA : setDeckB;
        setter(prev => ({ ...prev, baseBpm: refined, bpm: refined * (prev.playbackRate + prev.bend) }));
      }
    } catch (e) {
      console.warn("AI BPM Refinement failed", e);
    }
  };

  const analyzeStructureWithAi = async (id: 'A' | 'B', trackName: string, duration: number, bpm: number) => {
    if (!process.env.GEMINI_API_KEY || !trackName) return;
    try {
      const prompt = `Analyze the structure of this track for mixing: "${trackName}". Duration: ${Math.floor(duration)}s, BPM: ${bpm}.
      Suggest 3-5 structural markers (Intro, Drop, Break, Outro) with their estimated timestamps in seconds.
      Respond ONLY in valid JSON format: [{"time": number, "label": string, "type": "intro"|"build"|"drop"|"break"|"outro"}].`;

      const response = await ai.models.generateContent({
        model: 'gemini-2.0-flash',
        contents: prompt,
        config: { responseMimeType: "application/json" }
      });

      const markers = JSON.parse(response.text.trim());
      if (Array.isArray(markers)) {
        const setter = id === 'A' ? setDeckA : setDeckB;
        setter(prev => ({ ...prev, structureMarkers: markers }));
      }
    } catch (e) {
      console.warn("AI Structure Analysis failed", e);
    }
  };

  // Multimodal pre-analysis: Gemini LISTENS to intro+outro snippets and saves a track profile.
  // Called once per track on load. The profile is later used by triggerAiMix for richer decisions.
  const analyzeTrackProfileWithAi = async (id: 'A' | 'B', audioBuffer: AudioBuffer, trackName: string, bpm: number) => {
    if (!process.env.GEMINI_API_KEY || !audioBuffer) return;
    try {
      // Take ~12s from inside the intro and ~12s from the outro (ignore first/last 5s of silence/build)
      const introStart = Math.min(5, audioBuffer.duration * 0.05);
      const outroStart = Math.max(introStart + 12, audioBuffer.duration - 17);
      const introB64 = audioBufferToWavBase64(audioBuffer, introStart, 12);
      const outroB64 = audioBufferToWavBase64(audioBuffer, outroStart, 12);
      if (!introB64 || !outroB64) return;

      const prompt = `Eres un DJ Headliner de Bass Music analizando un track para mezclarlo después.
Track: "${trackName}"  ·  BPM detectado por algoritmo: ${bpm.toFixed(1)}
Te paso 2 snippets de audio: INTRO (~12s) y OUTRO (~12s).
ESCÚCHALOS y devuelve UN JSON con tu análisis técnico.

═══ CRÍTICO: BPM REAL ═══
El BPM detectado por algoritmo FRECUENTEMENTE ESTÁ MAL (halftime/doubletime).
ESCUCHA el snippet y estima el BPM REAL contando los kicks/snares.

RANGOS REALES POR GÉNERO:
  Halftime/Neurohop  → 85-110 BPM (kicks lentos, snares cada 2 segundos, drums secos)
  Breakbeat          → 130-140 BPM (breaks rotos)
  Liquid D&B         → 168-172 BPM (rolling rápido, snares en 2 y 4, hi-hats densos)
  Rollers/Peak D&B   → 172-178 BPM (más agresivo, MC, drums densos)

REGLAS DE CORRECCIÓN:
- Si oyes drums densos rolling pero detectado <100 → REAL es ×2 (ej: 87 → 174)
- Si oyes kicks lentos espaciados pero detectado >150 → REAL es ÷2 (ej: 174 → 87)
- Devuelve "realBpm" con tu mejor estimación REAL escuchando, NO el detectado.

═══ FASE ═══
1 = Halftime/Neurohop (real 85-110)
2 = Breakbeat (real ~130)
3 = Liquid D&B (real 168-172)
4 = Rollers/Peak D&B (real 172-178)

═══ TÉCNICAS DE MEZCLA (mixInTechnique/mixOutTechnique) ═══
"cut" | "blend" | "filter_sweep" | "echo_out"
- Sub-bajos densos → "cut" o "echo_out"
- Pads/acordes melódicos → "blend"
- Builds limpios → "filter_sweep"

═══ recommendedMixPoint ═══
"outro"     → tiene outro claro o energyArc descending. Default Liquid D&B.
"mid_break" → breakdown a media canción. Halftime denso.
"post_drop" → cortar justo después del segundo drop (~65%). Mata el track en su pico.
"early_cut" → track repetitivo. Cortar a ~40%.

Responde SOLO el JSON, sin markdown, sin texto extra:
{"genre":"<string corto>","mood":"<string corto>","instruments":["<3-5 instrumentos>"],"vocalPresence":"none|background|lead","energyArc":"building|sustained|descending|wave","bassWeight":"sub|punchy|light","mixInTechnique":"cut|blend|filter_sweep|echo_out","mixOutTechnique":"cut|blend|filter_sweep|echo_out","phase":1|2|3|4,"realBpm":<número entre 80 y 180>,"recommendedMixPoint":"outro|mid_break|post_drop|early_cut","notes":"<máx 15 palabras>"}`;

      const response = await ai.models.generateContent({
        model: 'gemini-2.0-flash',
        contents: [{
          role: 'user',
          parts: [
            { text: prompt },
            { inlineData: { mimeType: 'audio/wav', data: introB64 } },
            { inlineData: { mimeType: 'audio/wav', data: outroB64 } },
          ]
        }],
        config: { responseMimeType: "application/json" }
      });

      const raw = response.text.trim().replace(/```json|```/g, '').trim();
      const profile = JSON.parse(raw) as AiTrackProfile;
      const setter = id === 'A' ? setDeckA : setDeckB;

      setter(prev => {
        // ─── BPM CORRECTION via AI's listening-based realBpm ───
        // The audio detector frequently returns halftime (e.g. 87 for a real 174 D&B track).
        // Trust AI's realBpm if it falls in the expected range for the classified phase.
        const expected = PHASE_BPM_RANGE[profile.phase];
        let correctedBpm = prev.baseBpm;
        let reason = '';

        if (profile.realBpm && expected && profile.realBpm >= expected[0] && profile.realBpm <= expected[1]) {
          // AI gave a confident estimate inside the phase range → use it directly
          correctedBpm = profile.realBpm;
          if (Math.abs(prev.baseBpm - profile.realBpm) > 5) {
            reason = `AI realBpm=${profile.realBpm} (detector said ${prev.baseBpm.toFixed(1)})`;
          }
        } else if (expected) {
          // Fallback: simple halftime/doubletime correction based on phase range
          if (prev.baseBpm < expected[0] && prev.baseBpm * 2 >= expected[0] && prev.baseBpm * 2 <= expected[1]) {
            correctedBpm = prev.baseBpm * 2;
            reason = `HALFTIME corregido ×2 (${prev.baseBpm.toFixed(1)} → ${correctedBpm.toFixed(1)})`;
          } else if (prev.baseBpm > expected[1] && prev.baseBpm / 2 >= expected[0] && prev.baseBpm / 2 <= expected[1]) {
            correctedBpm = prev.baseBpm / 2;
            reason = `DOUBLETIME corregido ÷2 (${prev.baseBpm.toFixed(1)} → ${correctedBpm.toFixed(1)})`;
          }
        }

        // If we corrected, also reset playbackRate to 1.0 (any prior sync was based on wrong BPM)
        const correctedPlaybackRate = reason ? 1.0 : prev.playbackRate;
        const correctedBend = reason ? 0 : prev.bend;

        if (reason) {
          console.log(`[BPM FIX] Deck ${id} fase ${profile.phase} → ${reason}`);
        }
        console.log(`[AI PROFILE] Deck ${id} → fase ${profile.phase}, ${profile.genre}, ${profile.mood}, BPM ${correctedBpm.toFixed(1)}`);

        return {
          ...prev,
          aiProfile: profile,
          baseBpm: correctedBpm,
          bpm: correctedBpm * (correctedPlaybackRate + correctedBend),
          playbackRate: correctedPlaybackRate,
          bend: correctedBend,
        };
      });
    } catch (e) {
      console.warn(`AI Track Profile failed for Deck ${id}:`, e);
    }
  };

  // Refs for individual Deck Audio Chains
  const nodesA = useRef<{
    source: AudioBufferSourceNode | null;
    analyser: AnalyserNode | null;
    gain: GainNode | null;
    crossGain: GainNode | null;
    filters: BiquadFilterNode[];
    highPass: BiquadFilterNode | null;
    delayNode: DelayNode | null;
    delayWet: GainNode | null;
    delayFeedback: GainNode | null;
    reverbNode: ConvolverNode | null;
    reverbWet: GainNode | null;
  }>({ source: null, analyser: null, gain: null, crossGain: null, filters: [], highPass: null,
       delayNode: null, delayWet: null, delayFeedback: null, reverbNode: null, reverbWet: null });
  const nodesB = useRef<{
    source: AudioBufferSourceNode | null;
    analyser: AnalyserNode | null;
    gain: GainNode | null;
    crossGain: GainNode | null;
    filters: BiquadFilterNode[];
    highPass: BiquadFilterNode | null;
    delayNode: DelayNode | null;
    delayWet: GainNode | null;
    delayFeedback: GainNode | null;
    reverbNode: ConvolverNode | null;
    reverbWet: GainNode | null;
  }>({ source: null, analyser: null, gain: null, crossGain: null, filters: [], highPass: null,
       delayNode: null, delayWet: null, delayFeedback: null, reverbNode: null, reverbWet: null });
  const buffers = useRef<{ A: AudioBuffer | null, B: AudioBuffer | null }>({ A: null, B: null });

  // --- AUDIO SETUP ---
  const masterAnalyser = useRef<AnalyserNode | null>(null);
  
  const initAudio = () => {
    if (audioStarted) return;
    audioCtx.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    masterGain.current = audioCtx.current.createGain();
    destination.current = audioCtx.current.createMediaStreamDestination();
    
    masterAnalyser.current = audioCtx.current.createAnalyser();
    masterAnalyser.current.fftSize = 256;
    masterAnalyser.current.smoothingTimeConstant = 0.8;
    
    masterGain.current.connect(masterAnalyser.current);
    masterAnalyser.current.connect(audioCtx.current.destination);
    masterGain.current.connect(destination.current);
    
    setAudioStarted(true);
  };

  const createDeckChain = (id: 'A' | 'B') => {
    if (!audioCtx.current || !masterGain.current) return null;
    
    const analyser = audioCtx.current.createAnalyser();
    analyser.fftSize = 256;
    
    const lowFilter = audioCtx.current.createBiquadFilter();
    lowFilter.type = 'lowshelf';
    lowFilter.frequency.value = 320;

    const midFilter = audioCtx.current.createBiquadFilter();
    midFilter.type = 'peaking';
    midFilter.frequency.value = 1000;
    midFilter.Q.value = 1;

    const highFilter = audioCtx.current.createBiquadFilter();
    highFilter.type = 'highshelf';
    highFilter.frequency.value = 3200;

    const highPass = audioCtx.current.createBiquadFilter();
    highPass.type = 'highpass';
    highPass.frequency.value = 10; // Start at sub-sonic

    const gain = audioCtx.current.createGain();
    const crossGain = audioCtx.current.createGain();

    // Chain: Filters -> HighPass -> Analyser -> Gain -> CrossGain -> Master
    lowFilter.connect(midFilter);
    midFilter.connect(highFilter);
    highFilter.connect(highPass);
    highPass.connect(analyser);
    analyser.connect(gain);
    gain.connect(crossGain);
    crossGain.connect(masterGain.current);

    // --- FX: Delay ---
    const delayNode = audioCtx.current.createDelay(2.0);
    delayNode.delayTime.value = 0.375; // roughly 3/8 note at ~120 BPM
    const delayFeedback = audioCtx.current.createGain();
    delayFeedback.gain.value = 0.35;
    const delayWet = audioCtx.current.createGain();
    delayWet.gain.value = 0;

    // Delay feedback loop
    analyser.connect(delayNode);
    delayNode.connect(delayFeedback);
    delayFeedback.connect(delayNode);
    delayNode.connect(delayWet);
    delayWet.connect(masterGain.current);

    // --- FX: Reverb ---
    const reverbNode = audioCtx.current.createConvolver();
    reverbNode.buffer = generateImpulse(audioCtx.current);
    const reverbWet = audioCtx.current.createGain();
    reverbWet.gain.value = 0;

    analyser.connect(reverbNode);
    reverbNode.connect(reverbWet);
    reverbWet.connect(masterGain.current);

    return { analyser, gain, crossGain, filters: [lowFilter, midFilter, highFilter], highPass,
             delayNode, delayWet, delayFeedback, reverbNode, reverbWet };
  };

  // --- RECORDING ---
  const toggleRecording = () => {
    if (!isRecording) {
      if (!destination.current) return;
      recordingBlobs.current = [];
      recorder.current = new MediaRecorder(destination.current.stream, {
        mimeType: 'audio/webm',
        audioBitsPerSecond: 320000
      });
      recorder.current.ondataavailable = (e) => {
        if (e.data.size > 0) recordingBlobs.current.push(e.data);
      };
      recorder.current.onstop = () => {
        const blob = new Blob(recordingBlobs.current, { type: 'audio/webm' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'mi-mix.webm';
        a.click();
        URL.revokeObjectURL(url);
      };
      recorder.current.start();
      setIsRecording(true);
    } else {
      recorder.current?.stop();
      setIsRecording(false);
    }
  };

  // --- ACTIONS ---
  const loadTrack = async (id: 'A' | 'B', file: File) => {
    stopDeck(id);
    if (!audioCtx.current) initAudio();
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioCtx.current!.decodeAudioData(arrayBuffer);
    buffers.current[id] = audioBuffer;
    
    if (id === 'A') {
      offsetA.current = 0;
      setDeckATime(0);
      setDeckA(prev => ({ 
        ...prev, 
        trackName: file.name, 
        isPlaying: false,
        isAnalyzing: true,
        cuePoint: 0,
        hotCues: [null, null, null, null]
      }));
    } else {
      offsetB.current = 0;
      setDeckBTime(0);
      setDeckB(prev => ({ 
        ...prev, 
        trackName: file.name, 
        isPlaying: false,
        isAnalyzing: true,
        cuePoint: 0,
        hotCues: [null, null, null, null]
      }));
    }
    
    let detBpm = 128;
    try {
      detBpm = await analyze(audioBuffer);
    } catch(e) {
      console.warn("BPM Detection failed:", e);
    }

    // GPU-accelerated analysis (BPM refinement + real key detection)
    setGpuStatus('analyzing');
    try {
      const gpuResult = await analyzeAudioAdvanced(audioBuffer, detBpm);
      detBpm = gpuResult.bpm;
      setGpuStatus(gpuResult.usedGPU ? 'gpu' : 'cpu');
      if (id === 'A') setDeckA(prev => ({ ...prev, baseBpm: detBpm, bpm: detBpm, key: gpuResult.key, isAnalyzing: false }));
      else setDeckB(prev => ({ ...prev, baseBpm: detBpm, bpm: detBpm, key: gpuResult.key, isAnalyzing: false }));
    } catch(e) {
      console.warn('Advanced analysis failed:', e);
      setGpuStatus('cpu');
      if (id === 'A') setDeckA(prev => ({ ...prev, baseBpm: detBpm, bpm: detBpm, isAnalyzing: false }));
      else setDeckB(prev => ({ ...prev, baseBpm: detBpm, bpm: detBpm, isAnalyzing: false }));
    }

    // Refine structure with AI
    // Skip text-only refineBpmWithAi — the multimodal analyzer below corrects BPM by listening,
    // which is far more reliable. Running both creates race conditions on baseBpm.
    analyzeStructureWithAi(id, file.name, audioBuffer.duration, detBpm);
    // Multimodal pre-analysis — Gemini listens to the audio, caches a profile and corrects BPM
    analyzeTrackProfileWithAi(id, audioBuffer, file.name, detBpm);
    // Harmonic chord map — runs deferred so it doesn't block playback startup
    const setter = id === 'A' ? setDeckA : setDeckB;
    setTimeout(() => {
      try {
        const chords = analyzeChordMap(audioBuffer, detBpm);
        setter(prev => ({ ...prev, chordMap: chords }));
        console.log(`[CHORDS] Deck ${id}: ${chords.length} segments — ${chords.slice(0,4).map(c=>c.chord).join(' → ')}...`);
      } catch (e) { console.warn('[CHORDS] Analysis failed:', e); }
    }, 200);
  };

  const loadFromUrl = async (id: 'A' | 'B', url: string) => {
    stopDeck(id);
    if (!url) return;
    try {
      if (!audioCtx.current) initAudio();
      
      let response;
      try {
        response = await fetch(url);
      } catch (err) {
        response = await fetch(`https://api.allorigins.win/raw?url=${encodeURIComponent(url)}`);
      }

      if (!response.ok) throw new Error(`HTTP_${response.status}`);
      const arrayBuffer = await response.arrayBuffer();
      const audioBuffer = await audioCtx.current!.decodeAudioData(arrayBuffer);
      buffers.current[id] = audioBuffer;
      
      const fileName = url.split('/').pop()?.split('?')[0] || 'Demo Track';
      if (id === 'A') {
        offsetA.current = 0;
        setDeckATime(0);
        setDeckA(prev => ({ 
          ...prev, 
          trackName: fileName, 
          isPlaying: false,
          isAnalyzing: true,
          cuePoint: 0,
          hotCues: [null, null, null, null]
        }));
      } else {
        offsetB.current = 0;
        setDeckBTime(0);
        setDeckB(prev => ({ 
          ...prev, 
          trackName: fileName, 
          isPlaying: false, 
          isAnalyzing: true,
          cuePoint: 0,
          hotCues: [null, null, null, null]
        }));
      }

      let detBpmUrl = 128;
      try {
        detBpmUrl = await analyze(audioBuffer);
      } catch(e) {
        console.warn("BPM Detection failed:", e);
      }

      // GPU-accelerated analysis
      setGpuStatus('analyzing');
      try {
        const gpuResult = await analyzeAudioAdvanced(audioBuffer, detBpmUrl);
        detBpmUrl = gpuResult.bpm;
        setGpuStatus(gpuResult.usedGPU ? 'gpu' : 'cpu');
        if (id === 'A') setDeckA(prev => ({ ...prev, baseBpm: detBpmUrl, bpm: detBpmUrl, key: gpuResult.key, isAnalyzing: false }));
        else setDeckB(prev => ({ ...prev, baseBpm: detBpmUrl, bpm: detBpmUrl, key: gpuResult.key, isAnalyzing: false }));
      } catch(e) {
        setGpuStatus('cpu');
        if (id === 'A') setDeckA(prev => ({ ...prev, baseBpm: detBpmUrl, bpm: detBpmUrl, isAnalyzing: false }));
        else setDeckB(prev => ({ ...prev, baseBpm: detBpmUrl, bpm: detBpmUrl, isAnalyzing: false }));
      }

      // Skip text-only refineBpmWithAi — multimodal analyzer below handles BPM correction
      analyzeStructureWithAi(id, fileName, audioBuffer.duration, detBpmUrl);
      analyzeTrackProfileWithAi(id, audioBuffer, fileName, detBpmUrl);
      // Harmonic chord map
      const setterUrl = id === 'A' ? setDeckA : setDeckB;
      setTimeout(() => {
        try {
          const chords = analyzeChordMap(audioBuffer, detBpmUrl);
          setterUrl(prev => ({ ...prev, chordMap: chords }));
          console.log(`[CHORDS] Deck ${id}: ${chords.length} segments — ${chords.slice(0,4).map(c=>c.chord).join(' → ')}...`);
        } catch (e) { console.warn('[CHORDS] Analysis failed:', e); }
      }, 200);
    } catch (e: any) {
      console.error("Failed to load from URL:", e);
    }
  };

  const handleTrackLoad = (id: 'A' | 'B', source: File | string) => {
    if (typeof source === 'string') {
      loadFromUrl(id, source);
    } else {
      loadTrack(id, source);
    }
  };

  const [deckATime, setDeckATime] = useState(0);
  const [deckBTime, setDeckBTime] = useState(0);
  const playStartTimeA = useRef(0);
  const offsetA = useRef(0);
  const playStartTimeB = useRef(0);
  const offsetB = useRef(0);

  // Refs for current values to avoid dependency loop in RAF
  const deckRefA = useRef(deckA);
  const deckRefB = useRef(deckB);
  useEffect(() => { deckRefA.current = deckA; }, [deckA]);
  useEffect(() => { deckRefB.current = deckB; }, [deckB]);

  // Sync internal progress & Continuous Phase Lock
  useEffect(() => {
    let raf: number;
    const update = () => {
      const now = audioCtx.current?.currentTime || 0;
      const d1 = deckRefA.current;
      const d2 = deckRefB.current;
      
      if (d1.isPlaying && audioCtx.current) {
        const elapsed = (now - playStartTimeA.current) * (d1.playbackRate + d1.bend);
        setDeckATime(Math.min(buffers.current.A?.duration || 0, offsetA.current + elapsed));
      }
      if (d2.isPlaying && audioCtx.current) {
        const elapsed = (now - playStartTimeB.current) * (d2.playbackRate + d2.bend);
        setDeckBTime(Math.min(buffers.current.B?.duration || 0, offsetB.current + elapsed));
      }

      // --- REAL-TIME MASTER LED METER ---
      if (masterAnalyser.current) {
        const dataArray = new Uint8Array(masterAnalyser.current.frequencyBinCount);
        masterAnalyser.current.getByteTimeDomainData(dataArray);
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
          const val = (dataArray[i] - 128) / 128;
          sum += val * val;
        }
        const rms = Math.sqrt(sum / dataArray.length);
        // Map rms (0 to ~0.4) to 15 LEDs. Boost it a bit for visibility.
        const level = Math.min(15, Math.floor(rms * 15 * 3.5)); 
        for (let i = 0; i < 15; i++) {
          const elL = document.getElementById(`master-led-l-${i}`);
          const elR = document.getElementById(`master-led-r-${i}`);
          if (elL && elR) {
            // Add a slight variance for R channel to simulate stereo width visually
            const isActiveL = i < level;
            const isActiveR = i < level + (Math.random() > 0.8 ? 1 : 0) - (Math.random() > 0.8 ? 1 : 0);
            elL.style.opacity = isActiveL ? '1' : '0.1';
            elR.style.opacity = isActiveR ? '1' : '0.1';
          }
        }
      }

      // --- PHRASING & ADVICE ENGINE ---
      const updatePhrasing = (deck: DeckState, time: number) => {
        if (deck.bpm <= 0) return null;
        const beatLen = 60 / deck.bpm;
        const totalBeats = Math.floor(time / beatLen);
        const phrasePos = totalBeats % 32; // Standard 32-beat phrase
        const isEntryZone = phrasePos >= 28 || phrasePos <= 4;
        return { phrasePos, isEntryZone };
      };

      if (d1.isPlaying || d2.isPlaying) {
        const phrA = updatePhrasing(d1, deckATime);
        const phrB = updatePhrasing(d2, deckBTime);
        
        if (phrA && d1.isPlaying) {
          if (phrA.phrasePos === 0 && !isAiMixing) {
            // Recommendation for entry
            setAiMixAdvice(d2.isPlaying ? "STABLE LOCK" : "DROP NEW BEAT NOW!");
          } else if (phrA.phrasePos > 24) {
             setAiMixAdvice(`MIX WINDOW OPENING: IN ${32 - phrA.phrasePos} BEATS`);
          }
        }
      }

      // --- STABLE MAGNETIC LOCK (Active Sync) ---
      // Only runs if both are playing and at least one is SYNC LOCKED
      if (d1.isPlaying && d2.isPlaying && (d1.syncLocked || d2.syncLocked)) {
        // Master choice: if only one is locked, other is master. If both locked, A is master.
        const master = (d1.syncLocked && d2.syncLocked) ? 'A' : (d1.syncLocked ? 'B' : 'A');
        const slave = master === 'A' ? 'B' : 'A';
        
        const mBpm = master === 'A' ? d1.bpm : d2.bpm;
        const sBpm = master === 'A' ? d2.bpm : d1.bpm;
        
        if (mBpm > 0 && sBpm > 0) {
          const mTime = master === 'A' ? deckATime : deckBTime;
          const sTime = master === 'A' ? deckBTime : deckATime;
          
          const beatLen = 60 / mBpm;
          const phaseM = mTime % beatLen;
          const phaseS = sTime % beatLen;
          
          let diff = phaseM - phaseS;
          if (diff > beatLen / 2) diff -= beatLen;
          if (diff < -beatLen / 2) diff += beatLen;

          // High-precision correction
          if (Math.abs(diff) > 0.0015) {
             const force = 0.12 * (mBpm / 120); 
             const correction = diff * force;
             const setter = slave === 'A' ? setDeckA : setDeckB;
             setter(p => ({ 
               ...p, 
               bend: Math.max(-0.1, Math.min(0.1, p.bend + correction)) 
             }));
          } else {
             // Damping
             const setter = slave === 'A' ? setDeckA : setDeckB;
             if (Math.abs((master === 'A' ? d2 : d1).bend) > 0.0001) {
                setter(p => ({ ...p, bend: p.bend * 0.8 }));
             }
          }
        }
      }

      raf = requestAnimationFrame(update);

      // --- REAL-TIME AUDIO REACTIVITY (SPECTRUM DISTORTION) ---
      const activeDecks = [
        { nodes: nodesA.current, state: deckA },
        { nodes: nodesB.current, state: deckB }
      ];
      
      let bassEnergy = 0;
      let midHighEnergy = 0;
      let spectrumJitter = 0;
      const freqData = new Uint8Array(128);
      
      activeDecks.forEach(deck => {
        if (deck.nodes.analyser && deck.state.isPlaying) {
          deck.nodes.analyser.getByteFrequencyData(freqData);
          // Sub-bass (Kick)
          for (let i = 0; i < 6; i++) bassEnergy += freqData[i];
          // Mids (Vocals/Lead)
          for (let i = 20; i < 60; i++) midHighEnergy += freqData[i];
          // Highs (Percussion) for spectrum distortion
          for (let i = 80; i < 110; i++) spectrumJitter += freqData[i];
        }
      });

      const normBass = Math.min(1, bassEnergy / 2000); 
      const normMid = Math.min(1, midHighEnergy / 4000);
      const normHigh = Math.min(1, spectrumJitter / 2000);
      
      // Update SVG displacement scale
      const map = document.getElementById('distortion-map');
      if (map) {
        map.setAttribute('scale', (normHigh * 80 + normBass * 20).toString());
      }
    };
    update();
    return () => cancelAnimationFrame(raf);
  }, [audioStarted]); 

  const stopDeck = (id: 'A' | 'B') => {
    const setState = id === 'A' ? setDeckA : setDeckB;
    const offsetRef = id === 'A' ? offsetA : offsetB;
    const startTimeRef = id === 'A' ? playStartTimeA : playStartTimeB;
    const nodeRef = id === 'A' ? nodesA : nodesB;
    const state = id === 'A' ? deckA : deckB;

    if (audioCtx.current && nodeRef.current.source) {
      const elapsed = (audioCtx.current.currentTime - startTimeRef.current) * (state.playbackRate + state.bend);
      offsetRef.current += elapsed;
      try {
        const s = nodeRef.current.source;
        s.onended = null; // Prevent race condition with old onended setting isPlaying false
        s.stop();
        s.disconnect();
      } catch (e) {
        // Source might have already stopped
      }
      nodeRef.current.source = null;
    }
    setState(prev => ({ ...prev, isPlaying: false }));
  };

  const startDeck = (id: 'A' | 'B') => {
    const setState = id === 'A' ? setDeckA : setDeckB;
    const offsetRef = id === 'A' ? offsetA : offsetB;
    const startTimeRef = id === 'A' ? playStartTimeA : playStartTimeB;
    const nodeRef = id === 'A' ? nodesA : nodesB;
    const state = id === 'A' ? deckA : deckB;

    // Source of truth: Is there already a source running?
    if (nodeRef.current.source) return;
    if (!audioCtx.current || !buffers.current[id]) return;
      // If we start playing normally, we are no longer "cueing" (momentary mode)
      if (id === 'A') isCueingA.current = false;
      else isCueingB.current = false;

      if (offsetRef.current >= buffers.current[id]!.duration) offsetRef.current = 0;
      
      if (!nodeRef.current.gain) {
        const chain = createDeckChain(id);
        if (chain) {
          (nodeRef.current as any) = chain;
          // Apply initial crossfader value to prevent "dead" crossfader until moved
          const gA = Math.cos(crossfader * 0.5 * Math.PI);
          const gB = Math.cos((1 - crossfader) * 0.5 * Math.PI);
          if (id === 'A' && chain.crossGain) chain.crossGain.gain.value = gA;
          if (id === 'B' && chain.crossGain) chain.crossGain.gain.value = gB;
        }
      }

      const source = audioCtx.current.createBufferSource();
      source.buffer = buffers.current[id];
      source.playbackRate.value = state.playbackRate + state.bend;
      
      // Handle Looping
      if (state.isLooping && state.loopStart !== null && state.loopEnd !== null) {
        source.loop = true;
        source.loopStart = state.loopStart;
        source.loopEnd = state.loopEnd;
      }

      source.connect(nodeRef.current.filters[0]);
      
      startTimeRef.current = audioCtx.current.currentTime;
      source.start(0, offsetRef.current);
      nodeRef.current.source = source;
      
      source.onended = () => {
        setState(prev => {
          if (prev.isPlaying) {
            offsetRef.current = 0;
            if (id === 'A') setDeckATime(0); else setDeckBTime(0);
            return { ...prev, isPlaying: false };
          }
          return prev;
        });
      };

      setState(prev => ({ ...prev, isPlaying: true }));
  };

  const playPause = (id: 'A' | 'B') => {
    if (!audioCtx.current) initAudio();
    const isPlayingCurrent = id === 'A' ? deckA.isPlaying : deckB.isPlaying;
    if (isPlayingCurrent) stopDeck(id); else startDeck(id);
  };

  const seek = (id: "A" | "B", time: number) => {
    const nodeRef = id === "A" ? nodesA : nodesB;
    const buffer = buffers.current[id];
    if (!buffer) return;
    
    // Clamp time
    const targetTime = Math.max(0, Math.min(time, buffer.duration - 0.01));
    
    const isPlaying = !!nodeRef.current.source;
    
    // Total clear
    if (nodeRef.current.source) {
      try {
        const s = nodeRef.current.source;
        s.onended = null; // Prevent recursion or double trigger
        s.stop();
        s.disconnect();
      } catch (e) {}
      nodeRef.current.source = null;
    }

    if (id === "A") {
      offsetA.current = targetTime;
      setDeckATime(targetTime);
    } else {
      offsetB.current = targetTime;
      setDeckBTime(targetTime);
    }

    if (isPlaying) {
      startDeck(id);
    }
  };

  const handleCueDown = (id: 'A' | 'B') => {
    const isPlaying = id === 'A' ? deckA.isPlaying : deckB.isPlaying;
    const setState = id === 'A' ? setDeckA : setDeckB;
    const offsetRef = id === 'A' ? offsetA : offsetB;
    const timeRef = id === 'A' ? setDeckATime : setDeckBTime;
    const isCueing = id === 'A' ? isCueingA : isCueingB;
    const deck = id === 'A' ? deckA : deckB;
    const currentTime = id === 'A' ? deckATime : deckBTime;

    if (isPlaying) {
      // SI ESTA TOCANDO: Regresa al CUE y se detiene (Modo Pro)
      stopDeck(id);
      const target = deck.cuePoint;
      offsetRef.current = target;
      timeRef(target);
    } else {
      // SI ESTA PAUSADO: Comportamiento de Anclaje
      const dist = Math.abs(currentTime - deck.cuePoint);
      if (dist > 0.012) {
        // Anclar nuevo punto si nos hemos desplazado (por búsqueda manual)
        setState(prev => ({ ...prev, cuePoint: currentTime }));
      } else {
        // Stutter Play: Tocar desde el CUE mientras se mantiene pulsado
        startDeck(id);
        isCueing.current = true;
      }
    }
  };

  const handleCueUp = (id: 'A' | 'B') => {
    const isCueing = id === 'A' ? isCueingA : isCueingB;
    const state = id === 'A' ? deckA : deckB;

    if (isCueing.current) {
      // RELEASE: Stop and jump back
      stopDeck(id);
      if (id === 'A') {
        offsetA.current = state.cuePoint;
        setDeckATime(state.cuePoint);
      } else {
        offsetB.current = state.cuePoint;
        setDeckBTime(state.cuePoint);
      }
      isCueing.current = false;
    }
  };

  const handleCue = (id: 'A' | 'B') => {
    // Logic moved to handleCueDown to support Pioneer behavior
  };

  const setHotCue = (id: 'A' | 'B', index: number) => {
    const currentTime = id === 'A' ? deckATime : deckBTime;
    const setter = id === 'A' ? setDeckA : setDeckB;
    setter(prev => {
      const newCues = [...prev.hotCues];
      newCues[index] = currentTime;
      return { ...prev, hotCues: newCues };
    });
  };

  const triggerHotCue = (id: 'A' | 'B', index: number) => {
    const state = id === 'A' ? deckA : deckB;
    const cueTime = state.hotCues[index];
    if (cueTime !== null) {
      seek(id, cueTime);
    } else {
      setHotCue(id, index);
    }
  };

  const handleSetLoop = (id: 'A' | 'B', start: number | null, end: number | null, isLooping: boolean) => {
    const setter = id === 'A' ? setDeckA : setDeckB;
    const node = id === 'A' ? nodesA.current : nodesB.current;
    setter(prev => ({ ...prev, loopStart: start, loopEnd: end, isLooping }));
    if (node.source) {
      if (isLooping && start !== null && end !== null) {
        node.source.loop = true;
        node.source.loopStart = start;
        node.source.loopEnd = end;
      } else {
        node.source.loop = false;
      }
    }
  };

  const triggerAiMix = async () => {
    if (isAiMixing) return;

    // When both decks are playing, the "from" deck is whichever started first (the main track)
    const fromId: 'A' | 'B' = (deckA.isPlaying && deckB.isPlaying)
      ? (playStartTimeA.current <= playStartTimeB.current ? 'A' : 'B')
      : (deckA.isPlaying ? 'A' : 'B');
    const toId = fromId === 'A' ? 'B' : 'A';
    const fromDeck = fromId === 'A' ? deckA : deckB;
    const toDeck = fromId === 'A' ? deckB : deckA;

    if (!toDeck.trackName) {
      alert("Load a track on the other deck first!");
      return;
    }

    setIsAiMixing(true);

    // --- PHRASE & STRUCTURE ANALYSIS ---
    const beatLen = 60 / fromDeck.bpm;
    const currentTime = fromId === 'A' ? deckATime : deckBTime;
    const barLen = beatLen * 4;
    const isHighComplexity = fromDeck.bpm > 160;

    const nextMarker = fromDeck.structureMarkers?.find(m => m.time > currentTime);
    const targetOffset = nextMarker ? nextMarker.time - currentTime : 0;
    const phraseLen = barLen * (isHighComplexity ? 32 : 16);
    const timeToNextPhrase = phraseLen - (currentTime % phraseLen);
    const waitTime = nextMarker ? targetOffset : timeToNextPhrase;
    // Remaining seconds in the active track at the moment triggerAiMix was called
    const remainingAtStart = (buffers.current[fromId]?.duration || 0) - currentTime;

    const fromSetter = fromId === 'A' ? setDeckA : setDeckB;
    const toSetterFinal = toId === 'A' ? setDeckA : setDeckB;

    // --- KEY MATCHING ---
    let pitchShift = 1.0;
    if (fromDeck.key && toDeck.key) {
      const shift = getCamelotDiff(fromDeck.key, toDeck.key);
      if (shift !== 0) {
        pitchShift = Math.pow(2, shift / 12);
        toSetterFinal(prev => ({ ...prev, playbackRate: prev.playbackRate * pitchShift }));
      }
    }

    // --- AI PLAN: Structured JSON prompt ---
    let plan: AiTransitionPlan = {
      technique: 'filter_sweep',
      transitionDuration: 24,
      bassSwapBeat: 8,
      advice: 'PRECISION CLOCK MIXING',
      energy: 'maintain',
      warning: null,
    };

    // --- SMART OFFLINE DEFAULTS (sin API key, basado en perfiles y fases) ---
    if (!process.env.GEMINI_API_KEY) {
      const fp = fromDeck.aiProfile?.phase ?? 3;
      const tp = toDeck.aiProfile   ?.phase ?? 3;
      const phaseJump = Math.abs(fp - tp);
      // Harmonic hint: if we have compatible mix points, prefer blend
      const harmonicPoints = findBestHarmonicMixPoints(fromDeck.chordMap || [], toDeck.chordMap || []);
      const hasGoodHarmony = harmonicPoints.length > 0 && harmonicPoints[0].score >= 0.75;

      if (phaseJump >= 2) {
        plan = { technique: 'echo_out', transitionDuration: 8, bassSwapBeat: 4,
                 advice: `SALTO FASE ${fp}→${tp}`, energy: 'drop', warning: `BPM gap grande — usa echo_out` };
      } else if (hasGoodHarmony && (fp === 3 || fp === 4)) {
        plan = { technique: 'blend', transitionDuration: 28, bassSwapBeat: 16,
                 advice: 'BLEND ARMONICO', energy: 'maintain', warning: null };
      } else if (fp === 1) {
        plan = { technique: fromDeck.aiProfile?.bassWeight === 'sub' ? 'cut' : 'echo_out',
                 transitionDuration: 8, bassSwapBeat: 4, advice: 'HALFTIME CUT', energy: 'drop', warning: null };
      } else if (fp === 2) {
        plan = { technique: 'filter_sweep', transitionDuration: 16, bassSwapBeat: 8,
                 advice: 'BREAKBEAT SWEEP', energy: 'boost', warning: null };
      } else if (fp === 3) {
        plan = { technique: 'blend', transitionDuration: 32, bassSwapBeat: 16,
                 advice: 'LIQUID BLEND', energy: 'maintain', warning: null };
      } else {
        plan = { technique: 'filter_sweep', transitionDuration: 20, bassSwapBeat: 8,
                 advice: 'PEAK SWEEP', energy: 'boost', warning: null };
      }
      setAiMixAdvice(`OFFLINE: ${plan.advice}`);
    }

    if (process.env.GEMINI_API_KEY) {
      try {
        setAiMixAdvice('AI ANALYZING MIX...');
        // Inject the cached audio profiles built when each track was loaded.
        // The Director thus knows the actual genre/mood/energy/instruments — not just BPM and filename.
        const fmtProfile = (p?: AiTrackProfile) => p
          ? `genre=${p.genre}, mood=${p.mood}, instruments=[${p.instruments.join(',')}], vocals=${p.vocalPresence}, energyArc=${p.energyArc}, bass=${p.bassWeight}, phase=${p.phase}, idealMixIn=${p.mixInTechnique}, idealMixOut=${p.mixOutTechnique}, notes="${p.notes}"`
          : 'profile=unavailable (use BPM + name only)';

        // --- HARMONIC CONTEXT: chord progressions + best mix points ---
        const fmtChords = (chords?: ChordSegment[]) => {
          if (!chords?.length) return 'no disponible';
          // Summarise: first 4 chords, last 4 chords, and a compact list of unique chords
          const first4 = chords.slice(0, 4).map(c => c.chord).join('→');
          const last4  = chords.slice(-4).map(c => c.chord).join('→');
          const unique = [...new Set(chords.map(c => c.chord))].join(', ');
          return `intro:[${first4}] ... outro:[${last4}] | palette:{${unique}}`;
        };

        const harmonicPoints: HarmonicMixPoint[] = findBestHarmonicMixPoints(
          fromDeck.chordMap || [],
          toDeck.chordMap   || [],
        );
        const fmtHarmonic = harmonicPoints.length
          ? harmonicPoints.slice(0, 3).map(p =>
              `A@${p.timeA.toFixed(0)}s(${p.chordA})→B@${p.timeB.toFixed(0)}s(${p.chordB}) score=${(p.score*100).toFixed(0)}%`
            ).join(' | ')
          : 'sin puntos compatibles detectados';

        const resp = await ai.models.generateContent({
          model: 'gemini-2.0-flash',
          contents: `${directorPrompt}

DECK A (saliendo): "${fromDeck.trackName}", ${fromDeck.bpm.toFixed(1)} BPM, Key ${fromDeck.key || 'unknown'}
  AUDIO_PROFILE_A: ${fmtProfile(fromDeck.aiProfile)}
  CHORD_MAP_A: ${fmtChords(fromDeck.chordMap)}

DECK B (entrando): "${toDeck.trackName}", ${toDeck.bpm.toFixed(1)} BPM, Key ${toDeck.key || 'unknown'}
  AUDIO_PROFILE_B: ${fmtProfile(toDeck.aiProfile)}
  CHORD_MAP_B: ${fmtChords(toDeck.chordMap)}

HARMONIC_MIX_POINTS (mejores momentos por compatibilidad armónica):
  ${fmtHarmonic}

Contexto técnico: next marker A = ${nextMarker?.label || 'none'}, BPM ratio = ${(toDeck.bpm / fromDeck.bpm).toFixed(2)}

INSTRUCCIÓN: usa los AUDIO_PROFILE y CHORD_MAP para decidir la técnica más NATURAL.
- Respeta los idealMixOut de A y idealMixIn de B cuando sean compatibles.
- Si los HARMONIC_MIX_POINTS tienen score >= 80%, prioriza mezclar en esos momentos (actualiza mixPoint).
- Si los moods chocan (ej. euphoric → dark), usa filter_sweep para suavizar.
- Si vocals=lead en ambos, evita "blend" para no chocar voces.
- Si los acordes finales de A son muy disonantes con los iniciales de B, usa echo_out o filter_sweep.

Responde SOLO con este JSON (sin markdown):
{"technique":"filter_sweep","transitionDuration":24,"bassSwapBeat":8,"advice":"MAX 6 PALABRAS MAYUSCULAS","energy":"maintain","mixPoint":"outro","warning":null}`,
        });
        const raw = resp.text.trim().replace(/```json|```/g, '').trim();
        const parsed = JSON.parse(raw) as AiTransitionPlan;
        plan = {
          technique: parsed.technique || plan.technique,
          transitionDuration: Math.min(32, Math.max(8, parsed.transitionDuration || 24)),
          bassSwapBeat: Math.min(16, Math.max(4, parsed.bassSwapBeat || 8)),
          advice: parsed.advice || plan.advice,
          energy: parsed.energy || plan.energy,
          mixPoint: parsed.mixPoint, // informational — actual fire timing already happened
          warning: parsed.warning || null,
        };
      } catch (e) {
        console.warn('AI plan failed, using defaults:', e);
      }
    }

    const warningText = plan.warning ? ` ⚠ ${plan.warning}` : '';
    setAiMixAdvice(`${plan.technique.toUpperCase().replace('_',' ')} — ${plan.advice}${warningText}`);

    const triggerTransition = () => {
      if (!audioCtx.current) return;
      // Ensure incoming deck has an audio source; startDeck is a no-op if already playing
      startDeck(toId);
      const now = audioCtx.current.currentTime;
      const duration = plan.transitionDuration;

      // Atomic BPM sync
      document.dispatchEvent(new CustomEvent('sync-bpm', { detail: { from: fromId, to: toId } }));

      const fromNodes = fromId === 'A' ? nodesA.current : nodesB.current;
      const toNodes = toId === 'A' ? nodesA.current : nodesB.current;

      // swapTime declared here so it's in scope for both the audio automation and the UI interval
      const swapTime = now + (plan.bassSwapBeat * 4 * (60 / fromDeck.bpm));

      // --- AI GAIN RIDING: duck master during overlap to prevent clipping ---
      if (masterGain.current) {
        const duckLevel = plan.energy === 'boost' ? 0.85 : 0.70;
        masterGain.current.gain.cancelScheduledValues(now);
        masterGain.current.gain.setValueAtTime(1.0, now);
        masterGain.current.gain.linearRampToValueAtTime(duckLevel, now + 2);
        masterGain.current.gain.setValueAtTime(duckLevel, now + duration - 3);
        masterGain.current.gain.linearRampToValueAtTime(1.0, now + duration);
      }

      if (toNodes.gain && fromNodes.gain && toNodes.filters[0] && fromNodes.filters[0] && fromNodes.highPass && toNodes.highPass) {
        // Reset incoming deck
        toNodes.gain.gain.cancelScheduledValues(now);
        toNodes.gain.gain.setValueAtTime(0.0001, now);
        toNodes.highPass.frequency.setValueAtTime(10, now);
        toNodes.filters[0].gain.setValueAtTime(-24, now); // Kill bass on incoming

        // BASS SWAP quantized to plan.bassSwapBeat (swapTime declared above this if-block)
        toNodes.filters[0].gain.setValueAtTime(-24, swapTime - 0.01);
        toNodes.filters[0].gain.linearRampToValueAtTime(0, swapTime);
        fromNodes.filters[0].gain.setValueAtTime(0, swapTime - 0.01);
        fromNodes.filters[0].gain.linearRampToValueAtTime(-24, swapTime);

        const sweepEnd = now + duration;

        if (plan.technique === 'cut') {
          // Hard cut at next downbeat
          toNodes.gain.gain.setValueAtTime(0.0001, swapTime - 0.01);
          toNodes.gain.gain.setValueAtTime(1.0, swapTime);
          fromNodes.gain.gain.setValueAtTime(1.0, swapTime - 0.01);
          fromNodes.gain.gain.setValueAtTime(0.0001, swapTime);
        } else if (plan.technique === 'blend') {
          // Long smooth crossfade
          toNodes.gain.gain.linearRampToValueAtTime(1.0, now + duration);
          fromNodes.gain.gain.cancelScheduledValues(now);
          fromNodes.gain.gain.setValueAtTime(1.0, now);
          fromNodes.gain.gain.linearRampToValueAtTime(0.0001, sweepEnd);
        } else {
          // filter_sweep (default) or echo_out
          toNodes.gain.gain.linearRampToValueAtTime(1.0, now + duration * 0.5);
          fromNodes.highPass.frequency.cancelScheduledValues(now);
          fromNodes.highPass.frequency.setValueAtTime(20, now);
          fromNodes.highPass.frequency.exponentialRampToValueAtTime(
            plan.technique === 'echo_out' ? 8000 : 4000, sweepEnd
          );
          fromNodes.gain.gain.cancelScheduledValues(now);
          fromNodes.gain.gain.setValueAtTime(1.0, now + duration * 0.65);
          fromNodes.gain.gain.linearRampToValueAtTime(0.0001, sweepEnd);
        }
      }

      toSetterFinal(prev => ({ ...prev, isPlaying: true, gain: 0, low: -24, mid: -8, high: -4 }));

      // Animation interval for UI Knobs and Faders
      const stepMs = 40;
      const interval = setInterval(() => {
        if (!audioCtx.current) return;
        const currentTime = audioCtx.current.currentTime;
        const elapsed = currentTime - now;
        
        // Calculate UI interpolated values
        let toGainUI = 0;
        let fromGainUI = 1;
        let toLowUI = -24;
        let fromLowUI = 0;

        // Bass swap logic
        if (currentTime >= swapTime) {
            toLowUI = 0;
            fromLowUI = -24;
        }

        // Gain logic based on technique
        if (plan.technique === 'cut') {
            if (currentTime >= swapTime) {
                toGainUI = 1;
                fromGainUI = 0;
            } else {
                toGainUI = 0;
                fromGainUI = 1;
            }
        } else if (plan.technique === 'blend') {
            toGainUI = Math.min(1, elapsed / duration);
            fromGainUI = Math.max(0, 1 - (elapsed / duration));
        } else {
            // default filter_sweep / echo_out
            toGainUI = Math.min(1, elapsed / (duration * 0.5));
            if (elapsed > duration * 0.65) {
                fromGainUI = Math.max(0, 1 - ((elapsed - duration * 0.65) / (duration * 0.35)));
            } else {
                fromGainUI = 1;
            }
        }

        // Mid/High smooth intro for toDeck
        const introProgress = Math.min(1, elapsed / (duration * 0.5));
        const toMidUI = -8 * (1 - introProgress);
        const toHighUI = -4 * (1 - introProgress);

        // Update React State for UI animation
        toSetterFinal(prev => ({ ...prev, gain: toGainUI, low: toLowUI, mid: toMidUI, high: toHighUI }));
        fromSetter(prev => ({ ...prev, gain: fromGainUI, low: fromLowUI }));

        // Finish transition
        if (elapsed >= duration) {
          clearInterval(interval);
          if (nodesA.current.highPass) nodesA.current.highPass.frequency.setTargetAtTime(10, audioCtx.current.currentTime, 0.1);
          if (nodesB.current.highPass) nodesB.current.highPass.frequency.setTargetAtTime(10, audioCtx.current.currentTime, 0.1);
          if (masterGain.current) masterGain.current.gain.setTargetAtTime(1.0, audioCtx.current.currentTime, 0.1);
          setIsAiMixing(false);
          toSetterFinal(prev => ({ ...prev, gain: 1, low: 0, mid: 0, high: 0 }));
          stopDeck(fromId);
          fromSetter(prev => ({ ...prev, gain: 1, low: 0, mid: 0, high: 0 }));
          setAiMixAdvice(null);
        }
      }, stepMs);
    };

    // Clamp wait so triggerTransition fires at least (transitionDuration + 5)s before track ends
    const safeWait = Math.min(waitTime, Math.max(0, remainingAtStart - plan.transitionDuration - 5));
    setTimeout(triggerTransition, safeWait * 1000);
  };

  // Update EQ and Gain with smoothing
  useEffect(() => {
    if (isAiMixing) return;
    const time = audioCtx.current?.currentTime || 0;
    const SMOOTH = 0.05;
    if (nodesA.current.filters.length > 0) {
      nodesA.current.filters[0].gain.setTargetAtTime(deckA.low, time, SMOOTH);
      nodesA.current.filters[1].gain.setTargetAtTime(deckA.mid, time, SMOOTH);
      nodesA.current.filters[2].gain.setTargetAtTime(deckA.high, time, SMOOTH);
      nodesA.current.gain!.gain.setTargetAtTime(deckA.gain, time, SMOOTH);
    }
  }, [deckA, isAiMixing]);

  useEffect(() => {
    if (isAiMixing) return;
    const time = audioCtx.current?.currentTime || 0;
    const SMOOTH = 0.05;
    if (nodesB.current.filters.length > 0) {
      nodesB.current.filters[0].gain.setTargetAtTime(deckB.low, time, SMOOTH);
      nodesB.current.filters[1].gain.setTargetAtTime(deckB.mid, time, SMOOTH);
      nodesB.current.filters[2].gain.setTargetAtTime(deckB.high, time, SMOOTH);
      nodesB.current.gain!.gain.setTargetAtTime(deckB.gain, time, SMOOTH);
    }
  }, [deckB, isAiMixing]);

  useEffect(() => {
    const time = audioCtx.current?.currentTime || 0;
    if (nodesA.current.source) {
      const rate = nodesA.current.source.playbackRate;
      const now = audioCtx.current!.currentTime;
      rate.cancelScheduledValues(now);
      rate.setTargetAtTime(deckA.playbackRate + deckA.bend, now, 0.035);
    }
    setDeckA(prev => ({ ...prev, bpm: Math.round(prev.baseBpm * (prev.playbackRate + prev.bend) * 10) / 10 }));
  }, [deckA.playbackRate, deckA.bend, deckA.baseBpm]);

  useEffect(() => {
    if (nodesB.current.source) {
      const rate = nodesB.current.source.playbackRate;
      const now = audioCtx.current!.currentTime;
      rate.cancelScheduledValues(now);
      rate.setTargetAtTime(deckB.playbackRate + deckB.bend, now, 0.035);
    }
    setDeckB(prev => ({ ...prev, bpm: Math.round(prev.baseBpm * (prev.playbackRate + prev.bend) * 10) / 10 }));
  }, [deckB.playbackRate, deckB.bend, deckB.baseBpm]);

  // Crossfader Logic: Equal Power Curve
  useEffect(() => {
    if (nodesA.current.crossGain && nodesB.current.crossGain) {
      const gA = Math.cos(crossfader * 0.5 * Math.PI);
      const gB = Math.cos((1 - crossfader) * 0.5 * Math.PI);
      nodesA.current.crossGain.gain.setTargetAtTime(gA, audioCtx.current!.currentTime, 0.05);
      nodesB.current.crossGain.gain.setTargetAtTime(gB, audioCtx.current!.currentTime, 0.05);
    }
  }, [crossfader]);

  // FX Delay/Reverb for Deck A
  useEffect(() => {
    if (!audioCtx.current) return;
    const t = audioCtx.current.currentTime;
    if (nodesA.current.delayWet)  nodesA.current.delayWet.gain.setTargetAtTime(deckA.fxDelay, t, 0.05);
    if (nodesA.current.reverbWet) nodesA.current.reverbWet.gain.setTargetAtTime(deckA.fxReverb, t, 0.05);
  }, [deckA.fxDelay, deckA.fxReverb]);

  // FX Delay/Reverb for Deck B
  useEffect(() => {
    if (!audioCtx.current) return;
    const t = audioCtx.current.currentTime;
    if (nodesB.current.delayWet)  nodesB.current.delayWet.gain.setTargetAtTime(deckB.fxDelay, t, 0.05);
    if (nodesB.current.reverbWet) nodesB.current.reverbWet.gain.setTargetAtTime(deckB.fxReverb, t, 0.05);
  }, [deckB.fxDelay, deckB.fxReverb]);

  // ═══════════════════════════════════════════════
  // AUTOPILOT ENGINE — lógica real de mezcla automática
  // ═══════════════════════════════════════════════

  // 1. Al activar el autopilot: cargar primer track en A, segundo en B, arrancar A
  //    cuando el aiProfile esté listo (con BPM corregido). Máximo 20s de espera.
  useEffect(() => {
    if (!isAutopilotActive || playlist.length === 0) {
      mixTriggeredRef.current = false;
      return;
    }
    playlistIndexRef.current = 0;
    autopilotActiveDeckRef.current = 'A';
    setAutopilotActiveDeckUI('A');
    mixTriggeredRef.current = false;

    // Cargar track 0 → Deck A
    loadTrack('A', playlist[0]);

    // Pre-cargar track 1 → Deck B
    if (playlist.length > 1) {
      playlistIndexRef.current = 1;
      loadTrack('B', playlist[1]);
    }

    // Poll cada 500ms hasta que llegue el aiProfile del primer track O timeout 20s.
    // Solo arrancamos cuando el BPM esté CORREGIDO para evitar sync con valores erróneos.
    let elapsed = 0;
    const POLL_MS = 500;
    const MAX_WAIT_MS = 20000;
    const MIN_WAIT_MS = 1500; // mínimo de espera para que el buffer cargue

    setAiMixAdvice('AUTOPILOT: ANALIZANDO PRIMER TRACK...');

    const pollInterval = setInterval(() => {
      elapsed += POLL_MS;
      if (!isAutopilotActiveRef.current) {
        clearInterval(pollInterval);
        return;
      }

      // Necesitamos: buffer cargado + perfil de IA listo (BPM corregido)
      const bufferReady = !!buffers.current.A;
      const profileReady = !!deckRefA.current.aiProfile;
      const timedOut = elapsed >= MAX_WAIT_MS;
      const minWaitDone = elapsed >= MIN_WAIT_MS;

      if (bufferReady && minWaitDone && (profileReady || timedOut)) {
        clearInterval(pollInterval);
        const status = profileReady
          ? `✓ Perfil listo (${(elapsed/1000).toFixed(1)}s) — BPM ${deckRefA.current.baseBpm.toFixed(1)}`
          : `⚠ Timeout ${(elapsed/1000).toFixed(0)}s sin perfil — arrancando con BPM detector ${deckRefA.current.baseBpm.toFixed(1)}`;
        console.log(`[AUTOPILOT] ${status} → arrancando Deck A: ${playlist[0].name}`);
        setAiMixAdvice(profileReady ? 'AUTOPILOT READY' : 'AUTOPILOT: PERFIL TIMEOUT');
        startDeck('A');
        // Limpiar el advice después de 3s
        setTimeout(() => setAiMixAdvice(null), 3000);
      }
    }, POLL_MS);

    return () => clearInterval(pollInterval);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAutopilotActive]);

  // 2. Monitorear tiempo restante del deck activo → disparar mezcla automática
  useEffect(() => {
    if (!isAutopilotActive || isAiMixing || mixTriggeredRef.current) return;

    const activeDeck = autopilotActiveDeckRef.current;
    const activeTime = activeDeck === 'A' ? deckATime : deckBTime;
    const buffer = buffers.current[activeDeck];
    if (!buffer || buffer.duration < 1) return;

    // mixPoint controls WHEN we fire: outro=70% consumed, post_drop=65%, mid_break=50%, early_cut=40%
    const activeProfile = activeDeck === 'A' ? deckA.aiProfile : deckB.aiProfile;
    const mixPoint: MixPoint = activeProfile?.recommendedMixPoint || 'outro';
    const consumeRatio = MIX_POINT_RATIO[mixPoint];
    const triggerAt = buffer.duration * consumeRatio;
    const remaining = buffer.duration - activeTime;

    // Safety: never fire if there's not enough room for the transition (~30s minimum)
    const MIN_REMAINING = 30;

    const inactiveDeck = activeDeck === 'A' ? 'B' : 'A';
    const inactiveTrackName = inactiveDeck === 'A' ? deckA.trackName : deckB.trackName;

    if (activeTime >= triggerAt && remaining >= MIN_REMAINING && inactiveTrackName) {
      mixTriggeredRef.current = true;
      console.log(`[AUTOPILOT] mixPoint=${mixPoint} (${(consumeRatio*100).toFixed(0)}% consumido) — Deck ${activeDeck}, restan ${remaining.toFixed(0)}s`);
      triggerAiMix();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deckATime, deckBTime, isAutopilotActive, isAiMixing]);

  // 3. Cuando termina la mezcla: flippear deck activo y cargar el siguiente track
  useEffect(() => {
    if (prevAiMixingRef.current && !isAiMixing && isAutopilotActiveRef.current) {
      // La transición acaba de completarse — el deck "from" ahora está parado
      autopilotActiveDeckRef.current = autopilotActiveDeckRef.current === 'A' ? 'B' : 'A';
      setAutopilotActiveDeckUI(autopilotActiveDeckRef.current);
      mixTriggeredRef.current = false;

      const nextIdx = playlistIndexRef.current + 1;
      const currentPlaylist = playlistRef.current;
      if (nextIdx < currentPlaylist.length) {
        playlistIndexRef.current = nextIdx;
        const idleDeck: 'A' | 'B' = autopilotActiveDeckRef.current === 'A' ? 'B' : 'A';
        console.log(`[AUTOPILOT] Cargando track ${nextIdx + 1}/${currentPlaylist.length} → Deck ${idleDeck}: ${currentPlaylist[nextIdx].name}`);
        loadTrack(idleDeck, currentPlaylist[nextIdx]);
      } else {
        console.log('[AUTOPILOT] ✅ Playlist completa — deteniendo autopilot');
        setIsAutopilotActive(false);
      }
    }
    prevAiMixingRef.current = isAiMixing;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAiMixing]);

  // KEYBOARD SHORTCUTS
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if user is typing in an input (though we don't have many)
      if (document.activeElement?.tagName === 'INPUT') return;

      const key = e.key.toUpperCase();
      
      // DECK A
      if (key === 'S') playPause('A');
      if (key === 'A') handleCueDown('A');
      if (key === 'D') document.dispatchEvent(new CustomEvent('sync-bpm', { detail: { from: 'B', to: 'A' } }));
      if (['1', '2', '3', '4'].includes(key)) triggerHotCue('A', parseInt(key) - 1);
      
      // DECK B
      if (key === 'L') playPause('B');
      if (key === 'K') handleCueDown('B');
      if (key === ';') document.dispatchEvent(new CustomEvent('sync-bpm', { detail: { from: 'A', to: 'B' } }));
      if (['7', '8', '9', '0'].includes(key)) {
        const idx = key === '0' ? 3 : parseInt(key) - 7;
        triggerHotCue('B', idx);
      }

      // CROSSFADER
      if (e.key === 'ArrowLeft') setCrossfader(prev => Math.max(0, prev - 0.05));
      if (e.key === 'ArrowRight') setCrossfader(prev => Math.min(1, prev + 0.05));
      if (key === 'B') setCrossfader(0.5); // Center
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      const key = e.key.toUpperCase();
      if (key === 'A') handleCueUp('A');
      if (key === 'K') handleCueUp('B');
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [audioStarted, deckA, deckB, crossfader]);

  // Stable Event Listener for Sync
  useEffect(() => {
    if (!audioStarted) return;

    const handleSync = (e: any) => {
      const { from, to } = e.detail;
      if (!audioCtx.current) return;

      const fromDeck = from === 'A' ? deckRefA.current : deckRefB.current;
      const toDeck = to === 'A' ? deckRefA.current : deckRefB.current;
      const toSetter = to === 'A' ? setDeckA : setDeckB;

      if (fromDeck.bpm <= 0 || !buffers.current[from] || !buffers.current[to]) return;

      // 1. Precise Time Snapshot
      const now = audioCtx.current.currentTime;
      
      const getRealTime = (deckId: 'A' | 'B') => {
        const d = deckId === 'A' ? deckRefA.current : deckRefB.current;
        const offset = deckId === 'A' ? offsetA.current : offsetB.current;
        const start = deckId === 'A' ? playStartTimeA.current : playStartTimeB.current;
        if (!d.isPlaying) return offset;
        return offset + (now - start) * (d.playbackRate + d.bend);
      };

      const realFromTime = getRealTime(from);
      const realToTime = getRealTime(to);

      // 2. Exact BPM Matching with halftime/doubletime detection
      let targetBpm = fromDeck.bpm;
      const baseRatio = fromDeck.bpm / (toDeck.baseBpm || 128);
      if (Math.abs(baseRatio - 0.5) < Math.abs(baseRatio - 1) * 0.7) targetBpm = fromDeck.bpm * 2;
      else if (Math.abs(baseRatio - 2) < Math.abs(baseRatio - 1) * 0.7) targetBpm = fromDeck.bpm / 2;

      // Clamp rate to ±8% — beyond that, the audio sounds chipmunky/draggy.
      // If the underlying baseBpm is wrong (bad detection), we'd rather refuse to sync
      // than play the track at chipmunk speed.
      const rawRate = targetBpm / (toDeck.baseBpm || 128);
      const newRate = Math.max(0.92, Math.min(1.08, rawRate));
      if (rawRate !== newRate) {
        console.warn(`[SYNC] Rate clamped: ${rawRate.toFixed(3)} → ${newRate.toFixed(3)}. Verify Deck ${to} BPM (${toDeck.baseBpm.toFixed(1)}).`);
      }

      toSetter(prev => ({
        ...prev,
        playbackRate: newRate,
        bpm: (toDeck.baseBpm || 128) * newRate,
        syncLocked: true
      }));

      // 3. Latency-Compensated Phase Snap
      const beatDuration = 60 / targetBpm;
      const fromPhase = realFromTime % beatDuration;
      const toPhase = realToTime % beatDuration;
      
      let diff = fromPhase - toPhase;
      if (diff > beatDuration / 2) diff -= beatDuration;
      if (diff < -beatDuration / 2) diff += beatDuration;

      // Apply the snap with a tiny look-ahead to account for state update lag
      const targetOffset = Math.max(0, realToTime + diff);
      
      // Precision Seek Alignment
      if (to === 'A') {
        offsetA.current = targetOffset;
        setDeckATime(targetOffset);
        if (nodesA.current.source) {
          stopDeck('A');
          startDeck('A');
        } else {
          startDeck('A');
        }
      } else {
        offsetB.current = targetOffset;
        setDeckBTime(targetOffset);
        if (nodesB.current.source) {
          stopDeck('B');
          startDeck('B');
        } else {
          startDeck('B');
        }
      }
    };

    document.addEventListener('sync-bpm', handleSync as any);
    return () => document.removeEventListener('sync-bpm', handleSync as any);
  }, [audioStarted]); // Only re-bind on audio init, not every deck change

  return (
    <div className="h-screen bg-[#050505] overflow-hidden flex flex-col font-sans select-none text-[#cccccc]">
      
      {/* DAW Header Utility Bar */}
      <header className={DAW_HEADER_PANEL}>
        <div className="flex items-center gap-2 pr-4 border-r border-[#121212] mr-2">
           <Radio className="text-[#39ff14] w-4 h-4" />
           <span className="text-[11px] font-bold tracking-widest uppercase">LIQUID LIVE</span>
        </div>
        
        <div className="flex gap-4 items-center flex-1">
          <div className="bg-[#121212] px-3 py-1 rounded text-[10px] font-mono flex gap-3 text-[#39ff14]">
            <span className="text-[#00f2ff]">A: {deckA.bpm.toFixed(1)}</span>
            <span className="text-[#ffcc00]">B: {deckB.bpm.toFixed(1)}</span>
            <span className="text-[#999]">LATENCY: 2.1ms</span>
          {gpuStatus === 'analyzing' && <span className="text-[#ffcc00] animate-pulse">⚡ GPU ANALYZING...</span>}
          {gpuStatus === 'gpu' && <span className="text-[#39ff14]" title="RTX 4070 Ti — WebGPU">⚡ GPU ✓</span>}
          {gpuStatus === 'cpu' && <span className="text-[#888]">CPU fallback</span>}
          </div>

          <div className="flex-1" />

          <button 
            onClick={toggleRecording}
            className={`${LIQUID_GLASS_BTN} px-4 py-1 h-7 text-[10px] gap-2 ml-2 ${isRecording ? 'border-red-500 bg-red-500/20 shadow-[0_0_10px_rgba(239,68,68,0.3)]' : ''}`}
          >
            <div className={`w-2 h-2 rounded-full ${isRecording ? 'bg-red-500 animate-pulse' : 'bg-red-900 border border-white/20'}`} />
            {isRecording ? 'STOP' : 'REC'}
          </button>
        </div>
      </header>

      {/* SVG FILTER FOR DISTORTION */}
      <svg className="absolute w-0 h-0">
        <filter id="liquid-distortion">
          <feTurbulence type="fractalNoise" baseFrequency="0.05" numOctaves="2" result="noise" seed="1">
            <animate attributeName="seed" values="1;100" dur="1s" repeatCount="indefinite" />
          </feTurbulence>
          <feDisplacementMap in="SourceGraphic" in2="noise" scale="0" id="distortion-map" xChannelSelector="R" yChannelSelector="G" />
        </filter>
      </svg>

      {/* MASTER BEATMATCHING DISPLAY — audio spectrum + beat sync overlay */}
      <div className="h-28 bg-[#020202] border-b border-[#121212] flex flex-col relative overflow-hidden">
        <MasterWaveform
          id="A"
          buffer={buffers.current.A}
          time={deckATime}
          color="#00f2ff"
          isPlaying={deckA.isPlaying}
          onSeek={(t) => seek('A', t)}
          bpm={deckA.bpm}
          name={deckA.trackName}
        />
        <div className="h-[1px] w-full bg-white/5 relative z-20" />
        <MasterWaveform
          id="B"
          buffer={buffers.current.B}
          time={deckBTime}
          color="#ff0044"
          isPlaying={deckB.isPlaying}
          onSeek={(t) => seek('B', t)}
          bpm={deckB.bpm}
          name={deckB.trackName}
        />

        {/* BEAT SYNC OVERLAY — green centered line + lock indicator */}
        <BeatSyncOverlay
          timeA={deckATime}
          timeB={deckBTime}
          bpmA={deckA.bpm}
          bpmB={deckB.bpm}
        />
      </div>

      {/* Main Rack Area */}
      <main className="flex-1 flex overflow-hidden p-6 gap-6 bg-[#161618]">
        
        {/* Decks & Mixer Container */}
        <div className="flex-1 flex overflow-hidden gap-6">
          {/* TRACK A */}
          <div className="flex-1 flex flex-col h-full overflow-hidden bg-[#1e1e20] rounded-xl shadow-[10px_10px_20px_#0b0b0d,-10px_-10px_20px_#212123] border border-white/5">
          <div className="bg-[#1e1e20] px-4 py-3 text-[11px] font-bold border-b border-[#0b0b0d] shadow-[0_4px_6px_#0b0b0d] flex justify-between rounded-t-xl z-10">
            <span className="text-[#00f2ff] tracking-widest">1 - DECK A</span>
            <span className="opacity-40 tracking-widest">AUDIO TRACK</span>
          </div>
          <div className="flex-1 min-h-0">
                <Deck 
                 id="A" 
                 state={deckA} 
                 currentTime={deckATime}
                 audioBuffer={buffers.current.A}
                 setState={setDeckA}
                 onLoad={(s) => handleTrackLoad('A', s)} 
                 onPlay={() => playPause('A')}
                 onSeek={(t) => seek('A', t)}
                 onCue={() => handleCue('A')}
                 onCueDown={() => handleCueDown('A')}
                 onCueUp={() => handleCueUp('A')}
                 onHotCue={(i) => triggerHotCue('A', i)}
                 onSync={() => setDeckA(p => ({ ...p, syncLocked: !p.syncLocked }))}
                 onToggleMaster={() => toggleMaster('A')}
                 onSetLoop={(s, e, l) => handleSetLoop('A', s, e, l)}
                 analyser={nodesA.current.analyser}
               />
          </div>
        </div>

        {/* MIXER RACK */}
        <div className="w-[380px] flex-shrink-0 bg-[#1e1e20] rounded-xl shadow-[10px_10px_20px_#0b0b0d,-10px_-10px_20px_#212123] flex flex-col h-full border border-white/5 overflow-hidden">
           <div className="bg-[#1e1e20] px-4 py-3 text-[11px] font-bold border-b border-[#0b0b0d] shadow-[0_4px_6px_#0b0b0d] flex justify-center items-center z-10">
            <span className="opacity-60 uppercase tracking-[0.3em] text-[#fff]">MIXER RACK</span>
           </div>
           
           <div className="flex-1 flex p-2 gap-3 overflow-hidden bg-[#161618]/50 shadow-[inset_4px_4px_10px_#0b0b0d]">
              {/* MIXER STRIP A */}
              <div className={`${DAW_PANEL} p-2 flex flex-col gap-4 items-center bg-gradient-to-b from-black/20 to-transparent flex-1`}>
                 <div className="flex flex-col gap-3 w-full items-center">
                   <div className="relative group/knob">
                    <Knob label="HI" min={-24} max={12} value={deckA.high} onChange={(v) => setDeckA(p => ({...p, high: v}))} color="#00f2ff" />
                    <button 
                      onClick={() => setDeckA(p => ({ ...p, high: p.high <= -23 ? 0 : -24 }))}
                      className={`absolute -right-5 top-7 w-4 h-4 rounded-full border text-[6px] font-black flex items-center justify-center transition-all border-white/10 ${deckA.high <= -23 ? 'bg-red-500 border-red-400 text-white shadow-[0_0_10px_#ef4444]' : 'bg-[#1a1a1a] text-white/20 hover:text-white/40'}`}
                      title="KILL HIGH"
                    >
                      K
                    </button>
                   </div>
                   <div className="relative group/knob">
                    <Knob label="MID" min={-24} max={12} value={deckA.mid} onChange={(v) => setDeckA(p => ({...p, mid: v}))} color="#00f2ff" />
                    <button 
                      onClick={() => setDeckA(p => ({ ...p, mid: p.mid <= -23 ? 0 : -24 }))}
                      className={`absolute -right-5 top-7 w-4 h-4 rounded-full border text-[6px] font-black flex items-center justify-center transition-all border-white/10 ${deckA.mid <= -23 ? 'bg-red-500 border-red-400 text-white shadow-[0_0_10px_#ef4444]' : 'bg-[#1a1a1a] text-white/20 hover:text-white/40'}`}
                      title="KILL MID"
                    >
                      K
                    </button>
                   </div>
                   <div className="relative group/knob">
                    <Knob label="LOW" min={-24} max={12} value={deckA.low} onChange={(v) => setDeckA(p => ({...p, low: v}))} color="#00f2ff" />
                    <button 
                      onClick={() => setDeckA(p => ({ ...p, low: p.low <= -23 ? 0 : -24 }))}
                      className={`absolute -right-5 top-7 w-4 h-4 rounded-full border text-[6px] font-black flex items-center justify-center transition-all border-white/10 ${deckA.low <= -23 ? 'bg-red-500 border-red-400 text-white shadow-[0_0_10px_#ef4444]' : 'bg-[#1a1a1a] text-white/20 hover:text-white/40'}`}
                      title="KILL LOW"
                    >
                      K
                    </button>
                   </div>
                 </div>
                 <div className="flex-1 w-full min-h-[80px] flex justify-center mt-1 border-t border-white/5 pt-1">
                    <VerticalFader label="GAIN A" value={deckA.gain} onChange={(v) => setDeckA(p => ({...p, gain: v}))} accentColor="#00f2ff" />
                 </div>
              </div>

              {/* CENTER MASTER LED STRIP */}
              <div className="w-10 flex flex-col items-center py-2 bg-[#1a1a1c] border border-white/5 shadow-[4px_4px_10px_#0d0d0e,-4px_-4px_10px_#232326] rounded-xl flex-shrink-0">
                 <span className="text-[8px] font-black tracking-widest text-[#555] rotate-90 mt-6 mb-10">MASTER</span>
                 <div className="flex-1 flex gap-1 justify-center w-full mb-4">
                    {/* Left Channel */}
                    <div className="w-1.5 h-full bg-[#0a0a0a] rounded-full shadow-[inset_1px_1px_3px_#000] flex flex-col-reverse p-[1px] gap-[2px]">
                       {Array.from({length: 15}).map((_, i) => {
                          const color = i < 10 ? 'bg-[#39ff14]' : i < 13 ? 'bg-[#ffcc00]' : 'bg-[#ff0000]';
                          const shadow = i < 10 ? 'shadow-[0_0_4px_#39ff14]' : i < 13 ? 'shadow-[0_0_4px_#ffcc00]' : 'shadow-[0_0_4px_#ff0000]';
                          return <div key={`l-${i}`} id={`master-led-l-${i}`} className={`w-full flex-1 rounded-sm transition-all duration-75 opacity-10 ${color} ${shadow}`} />
                       })}
                    </div>
                    {/* Right Channel */}
                    <div className="w-1.5 h-full bg-[#0a0a0a] rounded-full shadow-[inset_1px_1px_3px_#000] flex flex-col-reverse p-[1px] gap-[2px]">
                       {Array.from({length: 15}).map((_, i) => {
                          const color = i < 10 ? 'bg-[#39ff14]' : i < 13 ? 'bg-[#ffcc00]' : 'bg-[#ff0000]';
                          const shadow = i < 10 ? 'shadow-[0_0_4px_#39ff14]' : i < 13 ? 'shadow-[0_0_4px_#ffcc00]' : 'shadow-[0_0_4px_#ff0000]';
                          return <div key={`r-${i}`} id={`master-led-r-${i}`} className={`w-full flex-1 rounded-sm transition-all duration-75 opacity-10 ${color} ${shadow}`} />
                       })}
                    </div>
                 </div>
              </div>

              {/* MIXER STRIP B */}
              <div className={`${DAW_PANEL} p-2 flex flex-col gap-4 items-center bg-gradient-to-b from-black/20 to-transparent flex-1`}>
                 <div className="flex flex-col gap-3 w-full items-center">
                   <div className="relative group/knob">
                    <Knob label="HI" min={-24} max={12} value={deckB.high} onChange={(v) => setDeckB(p => ({...p, high: v}))} color="#ffcc00" />
                    <button 
                      onClick={() => setDeckB(p => ({ ...p, high: p.high <= -23 ? 0 : -24 }))}
                      className={`absolute -right-5 top-7 w-4 h-4 rounded-full border text-[6px] font-black flex items-center justify-center transition-all border-white/10 ${deckB.high <= -23 ? 'bg-red-500 border-red-400 text-white shadow-[0_0_10px_#ef4444]' : 'bg-[#1a1a1a] text-white/20 hover:text-white/40'}`}
                      title="KILL HIGH"
                    >
                      K
                    </button>
                   </div>
                   <div className="relative group/knob">
                    <Knob label="MID" min={-24} max={12} value={deckB.mid} onChange={(v) => setDeckB(p => ({...p, mid: v}))} color="#ffcc00" />
                    <button 
                      onClick={() => setDeckB(p => ({ ...p, mid: p.mid <= -23 ? 0 : -24 }))}
                      className={`absolute -right-5 top-7 w-4 h-4 rounded-full border text-[6px] font-black flex items-center justify-center transition-all border-white/10 ${deckB.mid <= -23 ? 'bg-red-500 border-red-400 text-white shadow-[0_0_10px_#ef4444]' : 'bg-[#1a1a1a] text-white/20 hover:text-white/40'}`}
                      title="KILL MID"
                    >
                      K
                    </button>
                   </div>
                   <div className="relative group/knob">
                    <Knob label="LOW" min={-24} max={12} value={deckB.low} onChange={(v) => setDeckB(p => ({...p, low: v}))} color="#ffcc00" />
                    <button 
                      onClick={() => setDeckB(p => ({ ...p, low: p.low <= -23 ? 0 : -24 }))}
                      className={`absolute -right-5 top-7 w-4 h-4 rounded-full border text-[6px] font-black flex items-center justify-center transition-all border-white/10 ${deckB.low <= -23 ? 'bg-red-500 border-red-400 text-white shadow-[0_0_10px_#ef4444]' : 'bg-[#1a1a1a] text-white/20 hover:text-white/40'}`}
                      title="KILL LOW"
                    >
                      K
                    </button>
                   </div>
                 </div>
                 <div className="flex-1 w-full min-h-[80px] flex justify-center mt-1 border-t border-white/5 pt-1">
                    <VerticalFader label="GAIN B" value={deckB.gain} onChange={(v) => setDeckB(p => ({...p, gain: v}))} accentColor="#ffcc00" />
                 </div>
              </div>
            </div>

            {/* CROSSFADER & AI MIX */}
            <div className={`${DAW_PANEL} p-2 mt-auto rounded-sm border-t border-white/5 flex flex-col gap-2 relative overflow-hidden`}>
              {aiMixAdvice && (
                <div className="absolute top-0 left-0 w-full bg-[#39ff14]/20 py-0.5 px-2 text-[7px] font-black italic uppercase text-[#39ff14] border-b border-[#39ff14]/30 animate-pulse">
                  AI: {aiMixAdvice}
                </div>
              )}
              <div className="flex gap-2 w-full mt-2">
                <button 
                  onClick={triggerAiMix}
                  disabled={isAiMixing}
                  className={`flex-1 h-8 rounded-sm text-[8px] font-black italic tracking-[0.1em] uppercase transition-all border flex items-center justify-center gap-1 ${isAiMixing ? 'bg-[#39ff14] text-black border-[#39ff14] shadow-[0_0_15px_#39ff14]' : 'bg-[#39ff14]/10 text-[#39ff14] border-[#39ff14]/30 hover:bg-[#39ff14]/20'}`}
                >
                  <div className={isAiMixing ? 'animate-spin' : ''}><SkipBack size={10} className="rotate-180" /></div>
                  {isAiMixing ? 'MIXING' : 'SMART MIX'}
                </button>
                <button
                  onClick={() => setShowAutopilot(p => !p)}
                  className={`flex-1 h-8 rounded-sm text-[8px] font-black italic tracking-[0.1em] uppercase transition-all border flex items-center justify-center gap-1 ${showAutopilot ? 'bg-[#ff00f0]/20 text-[#ff00f0] border-[#ff00f0]/50 shadow-[0_0_10px_rgba(255,0,240,0.3)]' : 'bg-[#1a1a1c] text-[#888] border-white/10 hover:text-white'}`}
                >
                  🤖 AUTOPILOT
                </button>
              </div>

              <PhaseMeter timeA={deckATime} timeB={deckBTime} bpmA={deckA.bpm} bpmB={deckB.bpm} />

              {/* Spectrum Visualizer */}
              <div className="h-10 bg-black/40 rounded border border-white/5 overflow-hidden">
                <MixerSpectrum
                  analyserA={nodesA.current.analyser}
                  analyserB={nodesB.current.analyser}
                  isActive={deckA.isPlaying || deckB.isPlaying}
                />
              </div>

              <div>
                <div className="flex justify-between text-[10px] font-bold uppercase mb-1 px-1 opacity-60">
                  <span className="text-[#00f2ff]">A</span>
                  <span className="text-[7px]">Crossfader</span>
                  <span className="text-[#ffcc00]">B</span>
                </div>
                <div className="relative h-10 w-full flex items-center px-4 mt-2">
                  <input 
                    type="range"
                    min="0"
                    max="1"
                    step="0.001"
                    value={crossfader}
                    onChange={(e) => setCrossfader(parseFloat(e.target.value))}
                    className="neumorphic-slider w-full cursor-pointer"
                  />
                </div>
              </div>
            </div>
          </div>

        {/* TRACK B */}
        <div className="flex-1 flex flex-col h-full overflow-hidden bg-[#1e1e20] rounded-xl shadow-[10px_10px_20px_#0b0b0d,-10px_-10px_20px_#212123] border border-white/5">
          <div className="bg-[#1e1e20] px-4 py-3 text-[11px] font-bold border-b border-[#0b0b0d] shadow-[0_4px_6px_#0b0b0d] flex justify-between rounded-t-xl z-10">
            <span className="opacity-40 tracking-widest text-right">AUDIO TRACK</span>
            <span className="text-[#ffcc00] tracking-widest">2 - DECK B</span>
          </div>
          <div className="flex-1 min-h-0">
               <Deck 
                 id="B" 
                 state={deckB} 
                 currentTime={deckBTime}
                 audioBuffer={buffers.current.B}
                 setState={setDeckB}
                 onLoad={(s) => handleTrackLoad('B', s)} 
                 onPlay={() => playPause('B')}
                 onSeek={(t) => seek('B', t)}
                 onCue={() => handleCue('B')}
                 onCueDown={() => handleCueDown('B')}
                 onCueUp={() => handleCueUp('B')}
                 onHotCue={(i) => triggerHotCue('B', i)}
                 onSync={() => setDeckB(p => ({ ...p, syncLocked: !p.syncLocked }))}
                 onToggleMaster={() => toggleMaster('B')}
                 onSetLoop={(s, e, l) => handleSetLoop('B', s, e, l)}
                 analyser={nodesB.current.analyser}
               />
          </div>
        </div>
        </div> {/* CLOSE Decks & Mixer Container */}

        {/* AI DIRECTOR SIDEBAR */}
        {showAutopilot && (
          <div className="w-[320px] flex-shrink-0 flex flex-col h-full bg-[#1e1e20] rounded-xl shadow-[10px_10px_20px_#0b0b0d,-10px_-10px_20px_#212123] border border-white/5 overflow-hidden">
            <div className="bg-[#1e1e20] px-4 py-3 text-[11px] font-bold border-b border-[#0b0b0d] shadow-[0_4px_6px_#0b0b0d] flex justify-between items-center z-10">
              <span className="text-[#39ff14] tracking-widest flex items-center gap-2">🤖 AI AUTOPILOT</span>
              <span className="opacity-40">DIRECTOR</span>
            </div>
            
            <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-5 bg-[#161618]/50 shadow-[inset_4px_4px_10px_#0b0b0d]">
              
              {/* Prompt Section */}
              <div className="flex flex-col gap-2">
                <label className="text-[9px] font-bold tracking-widest text-[#888] uppercase">Director's Prompt</label>
                <textarea
                  className="w-full h-24 bg-[#121212] border border-white/5 rounded-lg p-2 text-xs text-[#ccc] focus:outline-none focus:border-[#39ff14]/50 shadow-[inset_2px_2px_5px_#0a0a0a] resize-none"
                  value={directorPrompt}
                  onChange={(e) => setDirectorPrompt(e.target.value)}
                  placeholder="Instrucciones para el DJ..."
                />
              </div>

              {/* Playlist Section */}
              <div className="flex flex-col gap-2 flex-1">
                <label className="text-[9px] font-bold tracking-widest text-[#888] uppercase">Playlist Queue</label>
                
                <div className="flex-1 bg-[#121212] border border-white/5 rounded-lg shadow-[inset_2px_2px_5px_#0a0a0a] overflow-hidden flex flex-col">
                  {playlist.length === 0 ? (
                    <div className="flex-1 flex flex-col items-center justify-center opacity-30 p-4 text-center">
                      <span className="text-2xl mb-2">📂</span>
                      <span className="text-xs">Sube tus tracks aquí</span>
                    </div>
                  ) : (
                    <div className="flex-1 overflow-y-auto p-2 flex flex-col gap-1">
                      {playlist.map((file, i) => (
                        <div key={i} className="flex items-center gap-1 bg-white/5 rounded px-2 py-1 border border-white/5">
                          <span className="text-[9px] opacity-30 w-4 shrink-0">{i+1}.</span>
                          <span className="text-[10px] text-[#aaa] truncate flex-1">{file.name}</span>
                          <div className="flex flex-col gap-0.5 shrink-0">
                            <button
                              onClick={() => moveTrack(i, -1)}
                              disabled={i === 0}
                              className="text-[8px] text-[#555] hover:text-white disabled:opacity-20 leading-none px-0.5"
                            >▲</button>
                            <button
                              onClick={() => moveTrack(i, 1)}
                              disabled={i === playlist.length - 1}
                              className="text-[8px] text-[#555] hover:text-white disabled:opacity-20 leading-none px-0.5"
                            >▼</button>
                          </div>
                          <button
                            onClick={() => setPlaylist(prev => prev.filter((_, idx) => idx !== i))}
                            className="text-[8px] text-[#444] hover:text-red-400 shrink-0 ml-1"
                          >✕</button>
                        </div>
                      ))}
                    </div>
                  )}
                  
                  <label className="w-full bg-[#1e1e20] hover:bg-[#2a2a2c] cursor-pointer border-t border-white/5 p-2 text-center text-[10px] font-bold tracking-widest text-[#00f2ff] transition-colors">
                    + AÑADIR TRACKS
                    <input type="file" multiple accept="audio/*" className="hidden" onChange={(e) => {
                      if (e.target.files) {
                        setPlaylist(prev => [...prev, ...Array.from(e.target.files!)]);
                      }
                    }} />
                  </label>
                </div>
              </div>

              {/* LIVE STATUS — visible when autopilot is active */}
              {isAutopilotActive && (() => {
                const ad = autopilotActiveDeckUI;
                const adState  = ad === 'A' ? deckA : deckB;
                const adTime   = ad === 'A' ? deckATime : deckBTime;
                const buf      = buffers.current[ad];
                const dur      = buf?.duration || 0;
                const mp: MixPoint = adState.aiProfile?.recommendedMixPoint || 'outro';
                const triggerRatio = MIX_POINT_RATIO[mp];
                const progress  = dur > 0 ? adTime / dur : 0;
                const triggered = progress >= triggerRatio;
                const secsLeft  = Math.max(0, dur * triggerRatio - adTime);

                // Colour that matches the active deck
                const col = ad === 'A' ? '#00f2ff' : '#ffcc00';

                return (
                  <div className="flex flex-col gap-2 border-t border-white/5 pt-3">
                    <div className="text-[9px] font-bold tracking-widest text-[#888] uppercase">Live Status</div>

                    {/* Deck indicator */}
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-1.5">
                        <div className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: col }} />
                        <span className="text-[10px] font-bold" style={{ color: col }}>DECK {ad} ON AIR</span>
                      </div>
                      <span className="text-[9px] font-mono text-[#555]">
                        {Math.floor(adTime/60)}:{(adTime%60).toFixed(0).padStart(2,'0')} / {Math.floor(dur/60)}:{(dur%60).toFixed(0).padStart(2,'0')}
                      </span>
                    </div>

                    {/* Progress toward mix trigger */}
                    <div className="flex flex-col gap-1">
                      <div className="flex justify-between text-[8px] font-mono">
                        <span className="text-[#666]">
                          MIX AT: <span className="text-[#aaa] font-bold">{mp.replace('_',' ').toUpperCase()}</span>
                          &nbsp;({Math.round(triggerRatio*100)}%)
                        </span>
                        <span className={triggered ? 'text-[#39ff14] font-black' : 'text-[#888]'}>
                          {triggered ? '⚡ FIRED' : `T−${secsLeft.toFixed(0)}s`}
                        </span>
                      </div>

                      {/* Progress bar */}
                      <div className="relative h-2 bg-[#121212] rounded-full overflow-visible border border-white/5">
                        {/* Fill */}
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{
                            width: `${Math.min(100, progress * 100)}%`,
                            background: triggered
                              ? '#39ff14'
                              : progress >= triggerRatio * 0.85
                              ? 'linear-gradient(90deg,#ffcc00,#ff8800)'
                              : `linear-gradient(90deg,${col},#39ff14)`,
                            boxShadow: triggered ? '0 0 8px #39ff14' : undefined,
                          }}
                        />
                        {/* Trigger marker */}
                        <div
                          className="absolute top-1/2 -translate-y-1/2 w-[2px] h-4 bg-white/50 rounded-full pointer-events-none"
                          style={{ left: `${triggerRatio * 100}%` }}
                        />
                      </div>
                    </div>

                    {/* AI profile summary */}
                    {adState.aiProfile && (
                      <div className="bg-[#121212] border border-white/5 rounded p-2 flex flex-col gap-1">
                        <div className="flex items-center justify-between">
                          <span
                            className="text-[7px] font-black px-1 py-[1px] rounded border"
                            style={{
                              background: ['','rgba(0,242,255,0.12)','rgba(255,204,0,0.12)','rgba(57,255,20,0.12)','rgba(255,0,68,0.12)'][adState.aiProfile.phase],
                              color:      ['','#00f2ff','#ffcc00','#39ff14','#ff0044'][adState.aiProfile.phase],
                              borderColor:['','#00f2ff30','#ffcc0030','#39ff1430','#ff004430'][adState.aiProfile.phase],
                            }}
                          >
                            FASE {adState.aiProfile.phase}
                          </span>
                          <span className="text-[8px] font-mono text-[#888]">{adState.aiProfile.genre}</span>
                        </div>
                        <div className="text-[8px] text-[#aaa] italic">{adState.aiProfile.mood}</div>
                        <div className="flex gap-1 flex-wrap mt-0.5">
                          {[
                            `bass:${adState.aiProfile.bassWeight}`,
                            `vox:${adState.aiProfile.vocalPresence}`,
                            adState.aiProfile.energyArc,
                          ].map(tag => (
                            <span key={tag} className="text-[6px] bg-white/5 px-1 py-[1px] rounded text-[#666] border border-white/5">{tag}</span>
                          ))}
                        </div>
                        <div className="flex items-center gap-1 mt-0.5 pt-1 border-t border-white/5">
                          <span className="text-[6px] text-[#555]">mix-in:</span>
                          <span className="text-[7px] font-bold text-[#888]">{adState.aiProfile.mixInTechnique}</span>
                          <span className="text-[6px] text-[#555] ml-1">mix-out:</span>
                          <span className="text-[7px] font-bold text-[#888]">{adState.aiProfile.mixOutTechnique}</span>
                        </div>
                        {adState.aiProfile.notes && (
                          <div className="text-[6px] text-[#555] italic border-t border-white/5 pt-1">{adState.aiProfile.notes}</div>
                        )}
                      </div>
                    )}

                    {/* Track index */}
                    {playlist.length > 0 && (
                      <div className="text-[8px] font-mono text-center text-[#555]">
                        TRACK {Math.min(playlistIndexRef.current, playlist.length)} / {playlist.length}
                      </div>
                    )}
                  </div>
                );
              })()}

              {/* Master Control */}
              <button
                onClick={() => setIsAutopilotActive(!isAutopilotActive)}
                className={`w-full py-3 rounded-lg text-xs font-black tracking-widest uppercase transition-all shadow-[4px_4px_8px_#0b0b0d,-4px_-4px_8px_#252527] border ${isAutopilotActive ? 'bg-[#39ff14] text-black border-[#39ff14] shadow-[0_0_15px_#39ff14]' : 'bg-[#1a1a1c] text-[#888] border-white/5 hover:text-white active:shadow-[inset_4px_4px_8px_#0b0b0d]'}`}
              >
                {isAutopilotActive ? '⬛ STOP AUTOPILOT' : '▶ START AUTOPILOT'}
              </button>

            </div>
          </div>
        )}

      </main>


      {/* ── Footer Performance Bar ── */}
      <footer className="h-16 bg-[#161618] border-t border-[#2a2a2c] shadow-[0_-4px_10px_#0b0b0d] flex items-center px-6 gap-6 select-none shrink-0 z-50">

        {/* Shortcuts label */}
        <div className="hidden lg:flex flex-col gap-0.5 text-[9px] font-mono uppercase tracking-widest text-[#444] border-r border-[#222] pr-5">
          <span>[S] PLAY A &nbsp;|&nbsp; [L] PLAY B</span>
          <span>[ARROWS] XFADE &nbsp;|&nbsp; [B] CENTER</span>
        </div>

        {/* Engine badge */}
        <div className="flex flex-col gap-0.5">
          <span className="text-[8px] font-bold uppercase tracking-widest text-[#444]">Engine</span>
          <span className="text-[10px] font-black text-[#39ff14] tracking-tight">WEB AUDIO API v2.0</span>
        </div>

        <div className="flex-1" />

        {/* ── RAM Meter ── */}
        <div className="flex flex-col gap-1 w-36">
          <div className="flex justify-between text-[9px] font-mono uppercase">
            <span className="text-[#555]" style={{letterSpacing:'0.1em'}}>RAM</span>
            <span className="text-[#ccc] font-bold">{ramMB} <span className="text-[#444]">/ {ramMax} MB</span></span>
          </div>
          <div className="h-2 bg-[#1a1a1a] rounded-full overflow-hidden border border-[#2a2a2a]">
            <div
              className="h-full rounded-full transition-all duration-700"
              style={{
                width: `${Math.min(100, (ramMB / ramMax) * 100)}%`,
                background: ramMB / ramMax > 0.8
                  ? 'linear-gradient(90deg,#ff4444,#ff0000)'
                  : ramMB / ramMax > 0.6
                  ? 'linear-gradient(90deg,#ffcc00,#ff8800)'
                  : 'linear-gradient(90deg,#39ff14,#00f2ff)',
                boxShadow: `0 0 8px ${ramMB/ramMax>0.8?'#ff0000':ramMB/ramMax>0.6?'#ffcc00':'#39ff14'}66`
              }}
            />
          </div>
        </div>

        {/* ── GPU Meter ── */}
        <div className="flex flex-col gap-1 w-36">
          <div className="flex justify-between text-[9px] font-mono uppercase">
            <span className="text-[#555]" style={{letterSpacing:'0.1em'}}>GPU</span>
            <span className="font-bold" style={{color: gpuStatus==='gpu'?'#39ff14':gpuStatus==='analyzing'?'#ffcc00':'#888'}}>
              {gpuLoad.toFixed(0)}%
              <span className="ml-1 text-[8px] opacity-60">
                {gpuStatus==='gpu'?'RTX':gpuStatus==='analyzing'?'BUSY':gpuStatus==='cpu'?'CPU':'—'}
              </span>
            </span>
          </div>
          <div className="h-2 bg-[#1a1a1a] rounded-full overflow-hidden border border-[#2a2a2a]">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width:`${gpuLoad}%`,
                background: gpuLoad > 80
                  ? 'linear-gradient(90deg,#ff6600,#ff3300)'
                  : gpuLoad > 40
                  ? 'linear-gradient(90deg,#ffcc00,#ff8800)'
                  : gpuStatus==='gpu'
                  ? 'linear-gradient(90deg,#39ff14,#00f2ff)'
                  : 'linear-gradient(90deg,#555,#444)',
                boxShadow: gpuStatus==='gpu'?'0 0 8px #39ff1466':undefined
              }}
            />
          </div>
        </div>

        {/* Live indicator */}
        <div className="flex items-center gap-2 pl-4 border-l border-[#222]">
          <div className={`w-2 h-2 rounded-full ${
            isAiMixing ? 'bg-[#ffcc00] shadow-[0_0_8px_#ffcc00] animate-pulse'
            : (deckA.isPlaying||deckB.isPlaying) ? 'bg-[#39ff14] shadow-[0_0_8px_rgba(57,255,20,0.6)]'
            : 'bg-[#333]'
          }`} />
          <span className="text-[9px] font-black uppercase tracking-widest" style={{
            color: isAiMixing?'#ffcc00':(deckA.isPlaying||deckB.isPlaying)?'#39ff14':'#444'
          }}>
            {isAiMixing ? 'AI MIXING' : (deckA.isPlaying||deckB.isPlaying) ? 'LIVE' : 'STANDBY'}
          </span>
        </div>

      </footer>
    </div>
  );
}

// --- SUB-COMPONENTS ---

function JogWheel({ 
  id, 
  isPlaying, 
  currentTime, 
  bpm, 
  playbackRate,
  bend,
  color, 
  onNudge 
}: { 
  id: 'A' | 'B', 
  isPlaying: boolean, 
  currentTime: number, 
  bpm: number, 
  playbackRate: number,
  bend: number,
  color: string, 
  onNudge: (val: number) => void 
}) {
  // 1 rotation = 1 measure (4 beats)
  const beatDuration = bpm > 0 ? 60 / bpm : 0.5;
  const rotation = (currentTime / (beatDuration * 4)) * 360;

  return (
    <div className="relative w-28 h-28 flex items-center justify-center group select-none">
       {/* Nudge Glow */}
       <div 
         className={`absolute inset-0 rounded-full transition-all duration-200 blur-xl ${bend > 0 ? 'bg-white/20' : bend < 0 ? 'bg-black/40' : 'bg-transparent'}`}
       />

       {/* Outer Static Ring */}
       <div 
         className="absolute inset-0 rounded-full border-4 border-[#222] shadow-[inset_0_0_15px_rgba(0,0,0,0.8)]"
         style={{ borderColor: bend !== 0 ? color : `${color}15` }}
       />
       
       {/* The "Platter" */}
       <motion.div 
         animate={{ rotate: rotation }}
         transition={{ type: 'tween', ease: 'linear', duration: 0 }}
         className={`w-24 h-24 rounded-full bg-[#0a0a0a] relative overflow-hidden flex items-center justify-center cursor-grab active:cursor-grabbing border-2 border-white/5 shadow-2xl transition-all ${bend !== 0 ? 'scale-[1.02]' : ''}`}
       >
         {/* Vinyl Grooves */}
         <div className="absolute inset-0 opacity-10 pointer-events-none" 
           style={{ 
             background: `repeating-radial-gradient(circle at center, transparent, transparent 1px, #fff 1px, #fff 2px)` 
           }} 
         />
         
         {/* Position Marker */}
         <div className="absolute top-0 w-[3px] h-6 bg-white shadow-[0_0_8px_white] rounded-full" />
         
         {/* STROBE Dots (like Technics SL-1200) */}
         <div className="absolute inset-0 flex items-center justify-center opacity-20 pointer-events-none">
            <div className="w-20 h-20 rounded-full border border-dotted border-white/50" />
         </div>

         {/* Inner Center */}
         <div className="w-8 h-8 rounded-full bg-[#151515] border border-white/10 z-10 flex items-center justify-center overflow-hidden">
            <div className="w-full h-full flex flex-col items-center justify-center bg-gradient-to-br from-white/10 to-transparent">
              <span className="text-[6px] font-black italic opacity-50" style={{ color }}>{id}</span>
              <div className={`w-1.5 h-1.5 rounded-full ${isPlaying ? 'bg-[#39ff14]' : 'bg-white/20'} shadow-[0_0_5px_currentColor]`} />
            </div>
         </div>
       </motion.div>

       {/* Interaction Layer (Nudge/Bend) */}
       <div className="absolute -bottom-1 left-0 right-0 flex justify-center gap-6 opacity-0 group-hover:opacity-100 transition-opacity z-20">
          <button 
            onMouseDown={() => onNudge(-0.06)} 
            onMouseUp={() => onNudge(0)} 
            onMouseLeave={() => onNudge(0)}
            className="w-7 h-7 rounded-full bg-black/80 border border-white/10 text-white text-[12px] font-black flex items-center justify-center hover:bg-white/20 active:bg-white/40 shadow-lg"
          >
            -
          </button>
          <button 
            onMouseDown={() => onNudge(0.06)} 
            onMouseUp={() => onNudge(0)} 
            onMouseLeave={() => onNudge(0)}
            className="w-7 h-7 rounded-full bg-black/80 border border-white/10 text-white text-[12px] font-black flex items-center justify-center hover:bg-white/20 active:bg-white/40 shadow-lg"
          >
            +
          </button>
       </div>
    </div>
  );
}

function Deck({ id, state, currentTime, audioBuffer, setState, onLoad, onPlay, onSeek, onCue, onCueDown, onCueUp, onHotCue, onSync, onToggleMaster, onSetLoop, analyser }: { 
  id: 'A' | 'B', 
  state: DeckState, 
  currentTime: number,
  audioBuffer: AudioBuffer | null,
  setState: React.Dispatch<React.SetStateAction<DeckState>>,
  onLoad: (s: File | string) => void,
  onPlay: () => void,
  onSeek: (t: number) => void,
  onCue: () => void,
  onCueDown: () => void,
  onCueUp: () => void,
  onHotCue: (i: number) => void,
  onSync: () => void,
  onToggleMaster: () => void,
  onSetLoop: (start: number | null, end: number | null, isLooping: boolean) => void,
  analyser: AnalyserNode | null
}) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [waveformPeaks, setWaveformPeaks] = useState<number[]>([]);

  // Generate fixed waveform peaks on buffer load
  useEffect(() => {
    if (!audioBuffer) {
      setWaveformPeaks([]);
      return;
    }
    const rawData = audioBuffer.getChannelData(0);
    const samples = 150; // number of bars
    const blockSize = Math.floor(rawData.length / samples);
    const peaks = [];
    for (let i = 0; i < samples; i++) {
      let sum = 0;
      for (let j = 0; j < blockSize; j++) {
        sum += Math.abs(rawData[i * blockSize + j]);
      }
      peaks.push(sum / blockSize);
    }
    // Normalize
    const maxPeak = Math.max(...peaks);
    setWaveformPeaks(peaks.map(p => p / maxPeak));
  }, [audioBuffer]);

  // Static Waveform Drawing with playback color logic
  useEffect(() => {
    if (!canvasRef.current || waveformPeaks.length === 0) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const duration = audioBuffer?.duration || 1;
    const progress = currentTime / duration;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const barSpacing = canvas.width / waveformPeaks.length;
    const barWidth = barSpacing * 0.7;

    // Beat grid logic
    const beatDuration = state.bpm > 0 ? 60 / state.bpm : 1;
    
    // Draw Beat Grid (Subtle background vertical lines)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let t = 0; t < duration; t += beatDuration) {
      const x = (t / duration) * canvas.width;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
      
      // Bar lines (every 4 beats)
      if (Math.round(t / beatDuration) % 4 === 0) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      }
    }

    waveformPeaks.forEach((peak, i) => {
      const x = i * barSpacing;
      const barHeight = peak * canvas.height * 0.8;
      const isPlayed = (i / waveformPeaks.length) <= progress;

      // Color logic: SoundCloud orange for played, grey for unplayed
      if (isPlayed) {
        ctx.fillStyle = '#ff5500'; // SoundCloud Orange
      } else {
        ctx.fillStyle = '#666666'; // SoundCloud Grey
      }

      // Draw mirrored bar
      const centerY = canvas.height / 2;
      ctx.beginPath();
      ctx.roundRect(x, centerY - barHeight/2, barWidth, barHeight, 1);
      ctx.fill();
    });

    // Draw Loop Markers if active
    if (state.loopStart !== null && state.loopEnd !== null) {
      const lx1 = (state.loopStart / duration) * canvas.width;
      const lx2 = (state.loopEnd / duration) * canvas.width;
      
      ctx.fillStyle = 'rgba(57, 255, 20, 0.2)'; // Semi-transparent green
      ctx.fillRect(lx1, 0, lx2 - lx1, canvas.height);
      
      ctx.strokeStyle = '#39ff14';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(lx1, 0); ctx.lineTo(lx1, canvas.height);
      ctx.moveTo(lx2, 0); ctx.lineTo(lx2, canvas.height);
      ctx.stroke();
    }

    // Draw AI Structure Markers
    state.structureMarkers.forEach(marker => {
      const mx = (marker.time / duration) * canvas.width;
      if (mx < 0 || mx > canvas.width) return;

      const colors: any = { intro: '#00f2ff', build: '#ffcc00', drop: '#39ff14', break: '#ffffff', outro: '#ff3131' };
      const color = colors[marker.type] || '#ffffff';

      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      ctx.moveTo(mx, 0);
      ctx.lineTo(mx, canvas.height);
      ctx.stroke();
      ctx.setLineDash([]);

      // Label background
      ctx.font = '7px Arial Black';
      const textWidth = ctx.measureText(marker.label).width;
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillRect(mx - textWidth/2 - 2, 2, textWidth + 4, 10);

      ctx.fillStyle = color;
      ctx.textAlign = 'center';
      ctx.fillText(marker.label.toUpperCase(), mx, 10);
    });

    // Draw chord labels (bottom strip) — one label per 2-bar segment
    if (state.chordMap && state.chordMap.length > 0) {
      const duration2 = audioBuffer?.duration || 1;
      ctx.font = '6px Arial Black';
      ctx.textAlign = 'center';
      // Only render chords that are visible in the canvas (avoid clutter)
      const minSpacingPx = 28; // minimum pixels between chord labels
      let lastLabelX = -minSpacingPx;

      state.chordMap.forEach(seg => {
        const cx = (seg.time / duration2) * canvas.width;
        if (cx < 0 || cx > canvas.width) return;
        if (cx - lastLabelX < minSpacingPx) return; // skip if too close
        lastLabelX = cx;

        const isMinor   = seg.quality === 'minor';
        const alpha     = Math.max(0.35, seg.confidence);
        const textColor = isMinor ? `rgba(255,100,180,${alpha})` : `rgba(100,220,255,${alpha})`;

        // Tiny background pill
        const tw = ctx.measureText(seg.chord).width;
        ctx.fillStyle = `rgba(0,0,0,0.55)`;
        ctx.fillRect(cx - tw / 2 - 2, canvas.height - 12, tw + 4, 10);

        ctx.fillStyle = textColor;
        ctx.fillText(seg.chord, cx, canvas.height - 4);

        // Vertical tick at segment boundary
        ctx.strokeStyle = textColor;
        ctx.globalAlpha = 0.25;
        ctx.lineWidth = 1;
        ctx.setLineDash([1, 3]);
        ctx.beginPath();
        ctx.moveTo(cx, canvas.height - 14);
        ctx.lineTo(cx, canvas.height - 22);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      });
    }
  }, [waveformPeaks, currentTime, audioBuffer, state.bpm, state.loopStart, state.loopEnd, state.structureMarkers, state.chordMap]);

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!audioBuffer) return;
    const rect = canvasRef.current!.getBoundingClientRect();
    let lastSeekTime = 0;
    
    const initialX = e.clientX - rect.left;
    const initialProgress = Math.max(0, Math.min(1, initialX / rect.width));
    onSeek(initialProgress * audioBuffer.duration);

    const handleMouseMove = (mmE: MouseEvent) => {
      const now = Date.now();
      if (now - lastSeekTime < 50) return; // Throttle to 20fps
      lastSeekTime = now;

      const x = mmE.clientX - rect.left;
      const progress = Math.max(0, Math.min(1, x / rect.width));
      onSeek(progress * audioBuffer.duration);
    };

    const handleMouseUp = (muE: MouseEvent) => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      
      // Final precise seek on release
      const x = muE.clientX - rect.left;
      const progress = Math.max(0, Math.min(1, x / rect.width));
      onSeek(progress * audioBuffer.duration);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  };

  return (
    <div className="h-full flex flex-col p-1 bg-[#121212] gap-0.5 overflow-hidden">
      <div className={`${DAW_PANEL} p-1 flex-1 flex flex-col gap-0.5 rounded-sm overflow-hidden`}>
        <div className="bg-[#1c1c1c] p-1 rounded-sm border border-[#121212] flex flex-col gap-0.5">
           <div className="flex justify-between items-center">
              <div className="flex flex-col">
                 <span className="text-[8px] font-bold opacity-30 tracking-widest uppercase">AUDIO SOURCE</span>
                 <p className="text-[10px] font-mono whitespace-nowrap overflow-hidden text-ellipsis max-w-[120px] text-[#39ff14]">
                   {state.trackName || 'WAITING...'}
                 </p>
              </div>
              
              <div className="flex gap-1">
                <button 
                  onClick={() => onLoad('https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3')}
                  className="px-2 h-6 text-[9px] font-bold bg-[#ff5500]/10 border border-[#ff5500]/30 text-[#ff5500] rounded hover:bg-[#ff5500]/20 focus:ring-1 focus:ring-[#ff5500]/50"
                >
                  DEMO 1
                </button>
                <button 
                  onClick={() => onLoad('https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3')}
                  className="px-2 h-6 text-[9px] font-bold bg-[#ff5500]/10 border border-[#ff5500]/30 text-[#ff5500] rounded hover:bg-[#ff5500]/20 focus:ring-1 focus:ring-[#ff5500]/50"
                >
                  DEMO 2
                </button>
                <button 
                  onClick={() => fileInputRef.current?.click()}
                  className={`${LIQUID_GLASS_BTN} px-2 h-6 text-[10px] uppercase gap-1`}
                >
                  <Headphones className="w-3 h-3" />
                  FILE
                </button>
              </div>
           </div>

        <div className="flex justify-between items-center py-1 border-t border-white/5 mt-0.5">
              <div className="flex flex-col">
                <div className="flex items-center gap-1">
                   <span className="text-[7px] font-bold opacity-30 uppercase">Track Info</span>
                   {state.syncLocked && (
                     <span className="text-[6px] font-black bg-[#39ff14] text-black px-1 rounded-sm animate-pulse">SYNC ACTIVE</span>
                   )}
                </div>
                <div className="flex items-center gap-1.5 min-w-[60px]">
                  {state.isAnalyzing ? (
                    <span className="text-[9px] font-mono animate-pulse text-[#39ff14]">ANALYZING...</span>
                  ) : (
                    <div className="flex flex-col">
                      <div className="flex items-baseline gap-1">
                        <span className="text-lg font-black italic font-mono leading-none" style={{ color: id === 'A' ? '#00f2ff' : '#ffcc00' }}>
                          {state.bpm.toFixed(1)}
                        </span>
                        <span className="text-[10px] font-black italic opacity-60">
                          {state.key}
                        </span>
                      </div>
                      {state.bpm > 0 && (
                        <div className="flex items-center gap-1 -mt-0.5">
                           <span className="text-[7px] font-black opacity-30">PHRASE</span>
                           <span className={`text-[8px] font-black transition-all ${Math.floor(currentTime / (60/state.bpm)) % 32 >= 28 ? 'text-[#39ff14]' : 'text-white/40'}`}>
                              {(Math.floor(currentTime / (60/state.bpm)) % 32) + 1}/32
                           </span>
                        </div>
                      )}
                    {state.isMaster && (
                      <span className="text-[6px] font-black bg-red-500 text-white px-1 rounded-sm shadow-[0_0_8px_#ef4444]">MASTER</span>
                    )}
                    {/* AI Track Profile Badge */}
                    {state.aiProfile && !state.isAnalyzing && (
                      <div className="flex items-center gap-1 mt-0.5 flex-wrap">
                        <span
                          className="text-[6px] px-1 py-[1px] rounded font-black border"
                          style={{
                            background: ['','rgba(0,242,255,0.12)','rgba(255,204,0,0.12)','rgba(57,255,20,0.12)','rgba(255,0,68,0.12)'][state.aiProfile.phase],
                            color:      ['','#00f2ff','#ffcc00','#39ff14','#ff0044'][state.aiProfile.phase],
                            borderColor:['','#00f2ff40','#ffcc0040','#39ff1440','#ff004440'][state.aiProfile.phase],
                          }}
                        >
                          F{state.aiProfile.phase}
                        </span>
                        <span className="text-[7px] text-[#666] font-mono truncate max-w-[70px]" title={state.aiProfile.genre}>
                          {state.aiProfile.genre}
                        </span>
                        <span className="text-[6px] text-[#444]" title={`Mix out: ${state.aiProfile.mixOutTechnique}`}>
                          ↓{state.aiProfile.mixOutTechnique.slice(0,3).toUpperCase()}
                        </span>
                      </div>
                    )}
                    </div>
                  )}
                  <button 
                    onClick={(e) => {
                      e.stopPropagation();
                      const now = Date.now();
                      const taps = (window as any)[`taps${id}`] || [];
                      taps.push(now);
                      if (taps.length > 4) taps.shift();
                      (window as any)[`taps${id}`] = taps;
                      if (taps.length > 1) {
                         const intervals = [];
                         for(let i=1; i<taps.length; i++) intervals.push(taps[i] - taps[i-1]);
                         const avg = intervals.reduce((a,b)=>a+b, 0) / intervals.length;
                         const tappedBpm = Math.round(60000 / avg);
                         setState(p => ({ ...p, baseBpm: tappedBpm, bpm: tappedBpm * (p.playbackRate + p.bend) }));
                      }
                    }}
                    className="px-1 h-3 bg-white/5 hover:bg-white/10 text-[7px] font-bold text-white/40 border border-white/10 rounded"
                  >
                    TAP
                  </button>
                </div>
              </div>

              <div className="flex-1 flex justify-center">
                 <JogWheel 
                    id={id} 
                    isPlaying={state.isPlaying} 
                    currentTime={currentTime} 
                    bpm={state.bpm} 
                    playbackRate={state.playbackRate}
                    bend={state.bend}
                    color={id === 'A' ? '#00f2ff' : '#ffcc00'}
                    onNudge={(val) => setState(p => ({ ...p, bend: val }))}
                 />
              </div>

              <div className="text-right flex flex-col items-end">
                 <span className="text-[7px] font-bold opacity-30 uppercase">Pitch</span>
                 <div className="flex items-center gap-1">
                   <div className="flex gap-0.5 bg-black/40 p-0.5 rounded border border-white/5">
                      {(['linear', 'log', 'exp'] as const).map(c => (
                        <button 
                          key={c}
                          onClick={() => setState(p => ({ ...p, pitchCurve: c }))}
                          className={`text-[5px] px-1 rounded-sm uppercase font-bold transition-all ${state.pitchCurve === c ? 'bg-white/20 text-white' : 'text-white/20 hover:text-white/40'}`}
                        >
                          {c}
                        </button>
                      ))}
                   </div>
                   <span className="text-[10px] font-mono text-white/50 leading-none min-w-[30px]">
                     {((state.playbackRate + state.bend - 1) * 100).toFixed(1)}%
                   </span>
                 </div>
              </div>
            </div>

          <input 
            ref={fileInputRef}
            type="file" 
            accept="audio/*" 
            className="hidden" 
            onChange={(e) => e.target.files?.[0] && onLoad(e.target.files[0])}
          />
        </div>

        {/* Waveform Window */}
        <div className="flex-1 min-h-[50px] bg-black/60 border border-[#222] rounded-sm relative overflow-hidden group cursor-pointer shadow-[inset_0_0_20px_rgba(0,0,0,0.8)]">
          <canvas 
            ref={canvasRef} 
            className="w-full h-full" 
            width={600} 
            height={120} 
            onMouseDown={handleMouseDown}
          />
          
          {/* HOT CUES OVERLAY */}
          <div className="absolute top-0 left-0 w-full h-full pointer-events-none">
            {state.hotCues.map((t, i) => t !== null && (
               <div 
                 key={i}
                 className="absolute top-0 bottom-0 w-[1px] bg-[#39ff14]/70 z-20"
                 style={{ left: `${(t / (audioBuffer?.duration || 1)) * 100}%` }}
               >
                 <span className="absolute top-1 left-0.5 text-[8px] font-bold text-[#39ff14] bg-black/50 px-0.5 rounded">{i + 1}</span>
               </div>
            ))}
          </div>

          {/* Time Display Overlay */}
          <div className="absolute bottom-1 left-2 pointer-events-none flex gap-2">
            <span className="text-[10px] font-mono text-[#ff5500]">
              {Math.floor(currentTime / 60)}:{(currentTime % 60).toFixed(0).padStart(2, '0')}
            </span>
            <span className="text-[10px] font-mono text-white/20">
              / {audioBuffer ? `${Math.floor(audioBuffer.duration / 60)}:${(audioBuffer.duration % 60).toFixed(0).padStart(2, '0')}` : '0:00'}
            </span>
          </div>

          <div className="absolute top-0 right-0 p-1 flex flex-col gap-1 h-full pointer-events-none">
             <div className="flex-1 w-1 bg-[#121212] relative overflow-hidden">
                <div className="absolute bottom-0 w-full bg-[#39ff14] opacity-40 shadow-[0_0_5px_#39ff14]" style={{ height: state.isPlaying ? '60%' : '0%' }} />
             </div>
          </div>
        </div>

        <div className="flex flex-col gap-1 mt-auto">
          {/* Hot Cues */}
          <div className="grid grid-cols-4 gap-0.5">
            {[0, 1, 2, 3].map(i => (
              <button
                key={i}
                onClick={() => onHotCue(i)}
                className={`h-7 border border-white/5 rounded-sm text-[9px] font-bold transition-all ${state.hotCues[i] !== null ? 'bg-[#39ff14]/30 border-[#39ff14]/50 text-[#39ff14] shadow-[0_0_8px_rgba(57,255,20,0.2)]' : 'bg-white/5 text-white/20 hover:bg-white/10'}`}
              >
                {i + 1}
              </button>
            ))}
          </div>

          {/* Loop Controls */}
          <div className="grid grid-cols-4 gap-0.5 mb-1">
             {[4, 8, 16, 32].map(beats => (
               <button
                 key={beats}
                 onClick={() => {
                   if (state.isLooping && state.loopEnd === (state.loopStart || 0) + beats * (60 / state.bpm)) {
                     // Toggle off if same loop
                     onSetLoop(null, null, false);
                   } else {
                     const start = currentTime;
                     const end = start + beats * (60 / state.bpm);
                     onSetLoop(start, end, true);
                   }
                 }}
                 className={`h-6 border border-white/5 rounded-sm text-[8px] font-bold transition-all ${state.isLooping && Math.abs((state.loopEnd || 0) - (state.loopStart || 0) - beats * (60 / state.bpm)) < 0.1 ? 'bg-[#ffcc00] text-black border-[#ffcc00] shadow-[0_0_8px_#ffcc00]' : 'bg-white/5 text-white/40 hover:bg-white/10'}`}
               >
                 {beats}
               </button>
             ))}
          </div>
            <div className="grid grid-cols-[1fr_2fr_1fr_1fr] gap-1">
             <button 
               onMouseDown={onCueDown}
               onMouseUp={onCueUp}
               onMouseLeave={onCueUp}
               onClick={onCue}
               className={`${LIQUID_GLASS_BTN} h-10 text-[10px] font-black italic tracking-widest uppercase bg-[#1a1a1a] border-white/20 active:bg-white/10 active:border-white/40 active:shadow-[0_0_15px_rgba(255,255,255,0.2)]`}
               title={id === 'A' ? 'Key: A' : 'Key: K'}
             >
               CUE
             </button>
             <button 
               onClick={onPlay}
               className={`${LIQUID_GLASS_BTN} h-10 flex items-center justify-center gap-2 text-[10px] font-black italic tracking-widest uppercase ${state.isPlaying ? 'bg-[#00f2ff]/40 text-white border-[#00f2ff] shadow-[0_0_20px_rgba(0,242,255,0.4)]' : 'bg-[#00f2ff]/10 text-[#00f2ff] border-[#00f2ff]/30 hover:bg-[#00f2ff]/20'} active:scale-90`}
               title={id === 'A' ? 'Key: S' : 'Key: L'}
             >
               {state.isPlaying ? (
                 <div className="flex gap-1.5"><div className="w-1.5 h-3 bg-current" /><div className="w-1.5 h-3 bg-current" /></div>
               ) : (
                 <Play size={16} fill="currentColor" />
               )}
             </button>
             <button 
               onClick={onToggleMaster}
               className={`${LIQUID_GLASS_BTN} h-10 text-[10px] font-black italic tracking-widest uppercase transition-all ${state.isMaster ? 'bg-red-500 text-white border-red-500 shadow-[0_0_15px_#ef4444]' : 'text-red-400 bg-red-400/10 border-red-400/40 hover:bg-red-400/20'} active:shadow-[0_0_15px_rgba(239,68,68,0.3)]`}
             >
               MASTER
             </button>
             <button
               onClick={() => {
                 if (state.syncLocked) {
                   setState(p => ({ ...p, syncLocked: false }));
                 } else {
                   onSync();
                 }
               }}
               className={`${LIQUID_GLASS_BTN} h-10 text-[10px] font-black italic tracking-widest uppercase transition-all ${state.syncLocked ? 'bg-[#39ff14] text-black border-[#39ff14] shadow-[0_0_15px_#39ff14]' : 'text-[#39ff14] bg-[#39ff14]/10 border-[#39ff14]/40 hover:bg-[#39ff14]/20'} active:shadow-[0_0_15px_rgba(57,255,20,0.3)]`}
             >
               SYNC
             </button>

          </div>

          {/* FX Section */}
          <div className="flex gap-2 items-center justify-center mt-1 border-t border-white/5 pt-1">
            <span className="text-[7px] font-bold opacity-30 uppercase tracking-widest">FX</span>
            <div className="flex gap-3">
              <Knob
                label="REVERB"
                min={0} max={1}
                defaultValue={0}
                value={state.fxReverb}
                onChange={(v) => setState(p => ({ ...p, fxReverb: v }))}
                color={id === 'A' ? '#00f2ff' : '#ffcc00'}
              />
              <Knob
                label="DELAY"
                min={0} max={1}
                defaultValue={0}
                value={state.fxDelay}
                onChange={(v) => setState(p => ({ ...p, fxDelay: v }))}
                color={id === 'A' ? '#00f2ff' : '#ffcc00'}
              />
            </div>
          </div>

        </div>

        <div className="flex flex-col items-center gap-1 group/pitch ml-1 h-full py-1">
             <span className="text-[8px] font-bold opacity-30 tracking-widest text-white">SPEED</span>
             
             {/* BEND TOP */}
             <button 
               onMouseDown={() => setState(p => ({...p, bend: 0.05}))}
               onMouseUp={() => setState(p => ({...p, bend: 0}))}
               onMouseLeave={() => setState(p => ({...p, bend: 0}))}
               className="w-full bg-white/5 hover:bg-white/10 text-[8px] font-bold py-1 border border-white/5 rounded-t cursor-pointer"
             >
               +BEND
             </button>

             <div className="flex-1 w-full flex justify-center py-2">
                <VerticalFader 
                  label="" 
                  value={(state.playbackRate - 0.5) / 1.0} 
                  defaultValue={0.5}
                  onChange={(v) => {
                    let adjustedRate = 0.5 + v * 1.0;
                    
                    // Apply Curve Transformation
                    if (state.pitchCurve === 'log') {
                       const x = v * 2 - 1; 
                       const sign = x < 0 ? -1 : 1;
                       const absX = Math.abs(x);
                       const curvedX = (Math.log(1 + 9 * absX) / Math.log(10)) * sign;
                       adjustedRate = 1.0 + (curvedX * 0.5);
                    } else if (state.pitchCurve === 'exp') {
                       const x = v * 2 - 1; 
                       const sign = x < 0 ? -1 : 1;
                       const absX = Math.abs(x);
                       const curvedX = ((Math.pow(10, absX) - 1) / 9) * sign;
                       adjustedRate = 1.0 + (curvedX * 0.5);
                    }

                    setState(prev => ({ 
                      ...prev, 
                      playbackRate: adjustedRate,
                      bpm: adjustedRate * (prev.baseBpm || 128)
                    }));
                  }} 
                  accentColor={id === 'A' ? '#00f2ff' : '#ffcc00'} 
                />
             </div>

             {/* BEND BOTTOM */}
             <button 
               onMouseDown={() => setState(p => ({...p, bend: -0.05}))}
               onMouseUp={() => setState(p => ({...p, bend: 0}))}
               onMouseLeave={() => setState(p => ({...p, bend: 0}))}
               className="w-full bg-white/5 hover:bg-white/10 text-[8px] font-bold py-1 border border-white/5 rounded-b cursor-pointer"
             >
               -BEND
             </button>

             <span className="text-[9px] font-mono font-bold" style={{ color: id === 'A' ? '#00f2ff' : '#ffcc00' }}>
               {(state.playbackRate + state.bend).toFixed(2)}x
             </span>
          </div>
        </div>
      </div>
  );
}

function Knob({ label, value, onChange, min, max, defaultValue = 0, color = '#39ff14' }: { label: string, value: number, onChange: (v: number) => void, min: number, max: number, defaultValue?: number, color?: string }) {
  const [isDragging, setIsDragging] = useState(false);
  const startY = useRef(0);
  const startVal = useRef(0);

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
    startY.current = e.clientY;
    startVal.current = value;
    document.body.style.cursor = 'ns-resize';
  };

  const handleDoubleClick = () => {
    onChange(defaultValue);
  };

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      const delta = startY.current - e.clientY;
      const range = max - min;
      // High precision dragging
      const sensitivity = e.shiftKey ? 1500 : 400;
      const step = range / sensitivity; 
      const newVal = Math.min(max, Math.max(min, startVal.current + delta * step));
      onChange(newVal);
    };

    const handleMouseUp = () => {
      setIsDragging(false);
      document.body.style.cursor = 'default';
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, min, max, onChange]);

  const percentage = (value - min) / (max - min);
  const angle = percentage * 270 - 135;
  const isNeutral = Math.abs(value - defaultValue) < 0.1;

  return (
    <div className="flex flex-col items-center gap-1 group">
      <span className="text-[8px] font-black opacity-40 tracking-widest transition-opacity group-hover:opacity-100 uppercase">{label}</span>
      <div 
        onMouseDown={handleMouseDown}
        onDoubleClick={handleDoubleClick}
        className="w-12 h-12 rounded-full bg-[#1c1c1c] border border-white/5 shadow-[0_4px_10px_rgba(0,0,0,0.5),inset_0_1px_2px_rgba(255,255,255,0.05)] cursor-ns-resize relative flex items-center justify-center transition-all hover:border-white/20"
      >
        {/* Ring Background */}
        <svg className="absolute inset-0 w-full h-full -rotate-90 p-[2px] opacity-10">
          <circle cx="22" cy="22" r="18" fill="none" stroke="white" strokeWidth="3" strokeDasharray="85 100" strokeDashoffset="-15" />
        </svg>

        {/* Dynamic Value Ring */}
        <svg className="absolute inset-0 w-full h-full -rotate-90 p-[2px]">
          <circle 
            cx="22" 
            cy="22" 
            r="18" 
            fill="none" 
            stroke={color} 
            strokeWidth="3" 
            strokeDasharray={`${percentage * 85} 100`} 
            strokeDashoffset="-15"
            className="transition-all duration-75 shadow-[0_0_10px_rgba(0,0,0,1)]"
            style={{ filter: `drop-shadow(0 0 2px ${color})` }}
          />
        </svg>

        {/* Knob Body */}
        <div 
          className={`w-8 h-8 rounded-full bg-gradient-to-br from-[#3a3a3a] to-[#1a1a1a] shadow-lg flex items-center justify-center transition-transform duration-75 border border-white/5 ${isDragging ? 'scale-105' : ''}`}
          style={{ transform: `rotate(${angle}deg)` }}
        >
           {/* Top Indicator */}
           <div className={`w-[3px] h-3 rounded-full -translate-y-2 ${isNeutral ? 'bg-white/40' : ''}`} style={{ backgroundColor: isNeutral ? undefined : color }} />
           
           {/* Grooves for texture */}
           <div className="absolute inset-0 rounded-full border border-black/40 pointer-events-none" />
        </div>
      </div>
      <span className={`text-[9px] font-mono mt-1 transition-opacity ${isNeutral ? 'opacity-30' : 'opacity-80 font-bold'}`} style={{ color: isNeutral ? undefined : color }}>
        {value > 0 ? `+${value.toFixed(1)}` : value.toFixed(1)}
      </span>
    </div>
  );
}

function VerticalFader({ label, value, onChange, accentColor, defaultValue = 0.8 }: { label: string, value: number, onChange: (v: number) => void, accentColor?: string, defaultValue?: number }) {
  const [isDragging, setIsDragging] = useState(false);
  const startY = useRef(0);
  const startVal = useRef(0);

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    startY.current = e.clientY;
    startVal.current = value;
    document.body.style.cursor = 'grabbing';
  };

  const handleDoubleClick = () => {
    onChange(defaultValue);
  };

  useEffect(() => {
    if (!isDragging) return;
    const handleMouseMove = (e: MouseEvent) => {
      const sensitivity = e.shiftKey ? 1800 : 600;
      const delta = (startY.current - e.clientY) / sensitivity;
      onChange(Math.min(1, Math.max(0, startVal.current + delta)));
    };
    const handleMouseUp = () => {
      setIsDragging(false);
      document.body.style.cursor = 'default';
    };
    document.body.style.cursor = 'grabbing';
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, onChange]);

  return (
    <div className="flex flex-col items-center gap-2 h-full select-none group w-full">
      <div 
        onMouseDown={handleMouseDown}
        onDoubleClick={handleDoubleClick}
        className="flex-1 w-full max-w-[40px] relative bg-[#121212] rounded-sm border-t border-[#333] border-b border-[#000] border-x border-black/40 flex justify-center py-4 cursor-pointer shadow-inner transition-colors hover:border-white/10 overflow-hidden"
      >
        {/* dB Markings */}
        <div className="absolute inset-y-4 right-[4px] flex flex-col justify-between py-1 pointer-events-none opacity-20">
           {['+6', '0', '-6', '-12', '-24', '-48', '-un'].map((label, i) => (
             <div key={i} className="flex items-center gap-1">
                <span className="text-[6px] font-mono leading-none scale-75 origin-right">{label}</span>
                <div className="w-[5px] h-[1px] bg-white/40" />
             </div>
           ))}
        </div>
        
        {/* Track Channels (Aesthetics) */}
        <div className="absolute left-[4px] right-6 inset-y-4 flex flex-col gap-1 pointer-events-none">
           <div className="flex-1 w-full bg-black/40 rounded-sm overflow-hidden relative">
              <div 
                className="absolute bottom-0 w-full transition-all duration-75" 
                style={{ 
                  height: `${value * 100}%`, 
                  background: `linear-gradient(to top, #39ff14 0%, #39ff14 80%, #ffff00 90%, #ff0000 100%)`, 
                  opacity: value > 0 ? 0.3 : 0,
                  boxShadow: `0 0 10px ${accentColor || '#39ff14'}44`
                }} 
              />
           </div>
        </div>

        {/* Fader Track Line */}
        <div className="absolute left-[20px] top-4 bottom-4 w-[1px] bg-black shadow-[1px_0_0_rgba(255,255,255,0.05)] pointer-events-none" />

        {/* Fader Cap */}
        <div 
          className="absolute left-[6px] right-[-2px] h-7 bg-[#4c4c4c] border-t border-white/20 border-b border-black shadow-[0_6px_12px_rgba(0,0,0,0.8)] z-20 rounded-[1px] active:bg-[#555] transition-colors"
          style={{ bottom: `calc(${value * 100}% + 4px)`, transform: 'translateY(50%)' }}
        >
          {/* Cap Line Marks */}
          <div className="absolute top-[2px] inset-x-[2px] h-[1px] bg-white/10" />
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent pointer-events-none" />
          <div className="absolute top-1/2 left-0 right-0 h-[2px] bg-white shadow-[0_0_8px_rgba(255,255,255,0.6)] transform -translate-y-1/2" />
        </div>
      </div>
      <span className="text-[9px] font-bold opacity-30 uppercase tracking-tighter group-hover:opacity-70 transition-opacity whitespace-nowrap">
        {label}
      </span>
    </div>
  );
}

function MasterWaveform({ id, buffer, time, color, isPlaying, onSeek, bpm, name }: {
  id: 'A' | 'B',
  buffer: AudioBuffer | null,
  time: number,
  color: string,
  isPlaying: boolean,
  onSeek: (t: number) => void,
  bpm: number,
  name: string
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const peaksRef = useRef<number[]>([]);

  useEffect(() => {
    if (!buffer) {
      peaksRef.current = [];
      return;
    }
    const rawData = buffer.getChannelData(0);
    const step = Math.ceil(rawData.length / 10000); 
    const peaks = [];
    for (let i = 0; i < rawData.length; i += step) {
      let max = 0;
      for (let j = 0; j < step && i + j < rawData.length; j++) {
        const v = Math.abs(rawData[i + j]);
        if (v > max) max = v;
      }
      peaks.push(max);
    }
    peaksRef.current = peaks;
  }, [buffer]);

  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let raf: number;
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const peaks = peaksRef.current;
      if (peaks.length === 0) {
        ctx.fillStyle = '#333';
        ctx.font = '10px monospace';
        ctx.fillText(`DECK ${id} EMPTY`, 10, 20);
        return;
      }

      const duration = buffer?.duration || 1;
      const width = canvas.width;
      const height = canvas.height;
      const centerX = width / 2;
      const pixelsPerSecond = 200; 

      ctx.fillStyle = color;
      
      const barsToDraw = Math.ceil(width / 2);
      for (let i = -barsToDraw; i < barsToDraw; i += 2) {
        const timeOffset = i / pixelsPerSecond;
        const targetTime = time + timeOffset;
        
        if (targetTime < 0 || targetTime > duration) continue;

        const peakIdx = Math.floor((targetTime / duration) * peaks.length);
        const peak = peaks[peakIdx] || 0;
        const h = peak * height * 0.85;

        // Visual center guide intensity
        const distFromCenter = Math.abs(i);
        ctx.globalAlpha = distFromCenter < 2 ? 1 : Math.max(0.2, 0.7 - distFromCenter / width);
        
        ctx.fillRect(centerX + i, (height - h) / 2, 1.5, h);
      }

      // DRAW BEAT GRID ROLLING
      const beatDuration = bpm > 0 ? 60 / bpm : 1;
      
      // Calculate start time for grid to cover the canvas
      const startTimeToSearch = time - (width / 2) / pixelsPerSecond;
      const endTimeToSearch = time + (width / 2) / pixelsPerSecond;
      
      // Find the first beat before the canvas starts
      const firstBeat = Math.floor(startTimeToSearch / beatDuration) * beatDuration;
      
      ctx.lineWidth = 1;
      for (let t = firstBeat; t <= endTimeToSearch; t += beatDuration) {
        if (t < 0) continue;
        const x = centerX + (t - time) * pixelsPerSecond;
        
        const beatIndex = Math.round(t / beatDuration);
        const isBar = beatIndex % 4 === 0;
        
        ctx.strokeStyle = isBar ? 'rgba(255, 255, 255, 0.4)' : 'rgba(255, 255, 255, 0.15)';
        ctx.globalAlpha = 1;
        
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
        
        if (isBar) {
          ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
          ctx.font = '6px Arial';
          ctx.fillText(Math.floor(beatIndex / 4) + 1, x + 2, 10);
        }
      }
      
      raf = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(raf);
  }, [buffer, time, color, id]);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!buffer) return;
    const startX = e.clientX;
    const startTime = time;

    const onMove = (moveEvent: MouseEvent) => {
      const deltaX = moveEvent.clientX - startX;
      const pixelsPerSecond = 200; 
      const newTime = Math.max(0, Math.min(buffer.duration, startTime - (deltaX / pixelsPerSecond)));
      onSeek(newTime);
    };

    const onUp = () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  };

  return (
    <div className="flex-1 relative border-r border-white/5 cursor-grab active:cursor-grabbing h-1/2" onMouseDown={handleMouseDown}>
      <div className="absolute top-1 left-2 z-30 flex items-center gap-2 pointer-events-none">
        <span className="text-[9px] font-black italic opacity-50 px-1 rounded bg-black/40" style={{ color }}>{id}</span>
        <span className="text-[9px] font-mono text-white/50 truncate max-w-[150px] uppercase font-bold">{name || '---'}</span>
      </div>
      <div className="absolute top-1 right-2 z-30 pointer-events-none text-right flex items-baseline gap-1">
        <span className="text-[8px] font-bold opacity-30 uppercase">BPM</span>
        <span className="text-sm font-black italic font-mono" style={{ color }}>{bpm.toFixed(1)}</span>
      </div>
      <canvas ref={canvasRef} className="w-full h-full" width={1200} height={60} />
    </div>
  );
}
