# 🎛️ Liquid Glass DJ

> **Professional AI-powered two-channel DJ controller** built entirely in the browser using the Web Audio API, Google Gemini multimodal AI, and WebGPU-accelerated analysis.

---

## ✨ Features

### 🎚️ Dual-Deck Engine
- Two independent audio channels — **Deck A** (cyan) and **Deck B** (yellow)
- Real-time scrolling waveform with beat grid overlay
- 3-band EQ (**High / Mid / Low**) with per-band **Kill** switches
- Vertical gain fader with dB scale markings
- Pitch/speed fader with **Linear / Log / Exp** curve modes
- **+BEND / −BEND** momentary pitch nudge
- **TAP BPM** tempo tap

### 🎯 Transport & Performance
- **CUE** — Pioneer-style: anchors point when paused, stutter-plays while held
- **4 Hot Cues** per deck — set & jump to any position instantly
- **Loop controls** — 4 / 8 / 16 / 32 beat quantized loops with visual overlay
- **JogWheel** — rotates with the track; hover to reveal nudge (+/−) buttons
- **MASTER** — flag one deck as tempo master
- **SYNC** — magnetic phase-lock slave deck to master BPM

### 🤖 AI Director (Google Gemini)
| Feature | Description |
|---|---|
| **Multimodal Track Profiling** | Gemini *listens* to 12 s intro + outro snippets and returns genre, mood, instruments, vocal presence, energy arc, bass weight, and recommended mix techniques |
| **AI BPM Correction** | Detects and fixes halftime / doubletime errors from the audio analyser (e.g. 87 → 174 BPM for Liquid D&B) |
| **SMART MIX** | One-click AI transition — generates a JSON plan with technique, duration, bass-swap beat, energy target, and warning |
| **AI AUTOPILOT** | Fully autonomous DJ mode: loads a playlist, pre-analyses every track, fires transitions at the AI-recommended `mixPoint`, and advances the queue |

### 🎨 Transition Techniques
| Technique | Description |
|---|---|
| `cut` | Hard cut at the next quantized downbeat |
| `blend` | Long smooth crossfade (up to 32 s) |
| `filter_sweep` | High-pass sweep on outgoing + gain ramp on incoming |
| `echo_out` | Aggressive high-pass + fast fade — ideal for dense sub-bass tracks |

### ⚡ Performance & Analysis
- **WebGPU-accelerated** BPM refinement and musical key detection (falls back to CPU)
- **Beat Sync Overlay** — rolling scrolling waveform for both decks with phase-lock indicator
- **Master LED VU meters** — real-time stereo level display
- **RAM / GPU meters** in the footer status bar
- **Audio Distortion FX** — SVG turbulence displacement driven by live spectrum data

### 🎙️ Recording
- One-click session recording via `MediaRecorder` → downloads as `.webm` (320 kbps)

### ⌨️ Keyboard Shortcuts
| Key | Action |
|---|---|
| `S` | Play / Pause Deck A |
| `A` | CUE Deck A (hold = stutter) |
| `D` | Sync Deck A → Deck B BPM |
| `1–4` | Hot Cues 1–4 on Deck A |
| `L` | Play / Pause Deck B |
| `K` | CUE Deck B (hold = stutter) |
| `;` | Sync Deck B → Deck A BPM |
| `7–0` | Hot Cues 1–4 on Deck B |
| `← →` | Move crossfader left / right |
| `B` | Center crossfader |

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| UI Framework | React 19 + TypeScript |
| Bundler | Vite 6 |
| Styling | Tailwind CSS v4 |
| Animation | Framer Motion (motion/react) |
| Audio Engine | Web Audio API |
| BPM Detection | `web-audio-beat-detector` |
| GPU Analysis | WebGPU (`@webgpu/types`) |
| AI | Google Gemini 2.0 Flash (`@google/genai`) |
| Icons | Lucide React |
| Desktop | Electron 41 + electron-builder |

---

## 🚀 Getting Started

### Prerequisites
- **Node.js** 18+
- A **Google Gemini API key** → [Get one free](https://aistudio.google.com/app/apikey)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-user/liquid-glass-dj.git
cd liquid-glass-dj

# 2. Install dependencies
npm install

# 3. Set your Gemini API key
cp .env.example .env.local
# Edit .env.local and replace YOUR_API_KEY_HERE
```

### Running (Web)

```bash
npm run dev
# → http://localhost:3000
```

### Running (Desktop / Electron)

```bash
npm run electron:start
```

### Build for Distribution

```bash
# Web build
npm run build

# Desktop installer (Windows .exe / Mac .dmg / Linux .AppImage)
npm run electron:build
```

---

## ⚙️ Configuration

Create a `.env.local` file in the project root:

```env
GEMINI_API_KEY="your-gemini-api-key-here"
APP_URL="http://localhost:3000"
```

> **Note:** The app works without a Gemini API key — BPM detection, EQ, mixing, recording, and all manual controls still function. AI features (SMART MIX, AUTOPILOT, track profiling) fall back to sensible defaults.

---

## 🎛️ AI Autopilot — How It Works

1. **Upload a playlist** (any number of audio files) in the AUTOPILOT sidebar
2. Press **▶ START AUTOPILOT**
3. The system:
   - Loads Track 1 → Deck A, Track 2 → Deck B
   - Sends 12 s audio snippets (intro + outro) to Gemini → builds a `TrackProfile`
   - Corrects BPM if the detector returned a halftime/doubletime value
   - Waits until the AI-recommended `mixPoint` is reached (40–70% consumed)
   - Calls SMART MIX with full profile context from both decks
   - Executes the transition, then flips the active deck and loads the next track
   - Repeats until the playlist is exhausted

### mixPoint Reference

| Value | Trigger | Typical use |
|---|---|---|
| `early_cut` | 40% consumed | Short / repetitive tracks |
| `mid_break` | 50% consumed | Dense halftime tracks |
| `post_drop` | 65% consumed | Cut after second drop |
| `outro` | 70% consumed | Tracks with clear outros (default Liquid D&B) |

### Phase Dictionary (Director Prompt)

| Phase | Genre | BPM Range | Default Technique |
|---|---|---|---|
| 1 | Halftime / Neurohop | 85–110 | `cut` or `echo_out` |
| 2 | Breakbeat / Cinematic | 125–140 | `filter_sweep` |
| 3 | Liquid / Soulful D&B | 168–174 | `blend` |
| 4 | Rollers / Peak D&B | 172–178 | `filter_sweep` or `blend` |

---

## 📁 Project Structure

```
liquid-glass-dj/
├── src/
│   ├── App.tsx              # Main application (all logic + UI)
│   ├── webgpu-analyzer.ts   # WebGPU-accelerated BPM + key detection
│   ├── main.tsx             # React entry point
│   └── index.css            # Global styles + Tailwind directives
├── electron-main.cjs        # Electron main process
├── index.html               # HTML shell
├── vite.config.ts           # Vite configuration
├── tsconfig.json            # TypeScript configuration
├── .env.example             # Environment variable template
└── package.json
```

---

## 🔒 Security

- **API key** is read from `.env.local` at build time via `process.env.GEMINI_API_KEY`
- `.env.local` is excluded from git via `.gitignore` (`.env*` rule)
- No API key is ever hardcoded in source files
- Audio data sent to Gemini is **mono 16 kHz PCM WAV**, max ~256 KB per snippet

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

<div align="center">
  Built with ❤️ using React, Web Audio API, and Google Gemini
</div>
