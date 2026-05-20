declare module 'soundtouchjs' {
  export class PitchShifter {
    constructor(context: AudioContext, buffer: AudioBuffer, bufferSize: number, onEnd?: () => void);
    tempo: number;
    pitch: number;
    rate: number;
    percentagePlayed: number;
    sourcePosition: number;
    timePlayed: number;
    duration: number;
    sampleRate: number;
    readonly node: ScriptProcessorNode;
    connect(node: AudioNode): void;
    disconnect(): void;
    on(eventName: string, cb: (detail: any) => void): void;
    off(eventName?: string): void;
  }
  export class SoundTouch {}
  export class SimpleFilter {}
  export class WebAudioBufferSource {}
  export function getWebAudioNode(context: AudioContext, filter: any, cb?: (pos: number) => void, bufSize?: number): ScriptProcessorNode;
}
