# Liquid Glass DJ

Controlador de DJ profesional de dos canales impulsado por inteligencia artificial, construido íntegramente en el navegador con Web Audio API, Google Gemini multimodal y análisis acelerado por WebGPU.

---

## Características

### Motor de doble deck

- Dos canales de audio independientes: **Deck A** (cian) y **Deck B** (amarillo)
- Forma de onda con desplazamiento en tiempo real y cuadrícula de beats superpuesta
- Ecualizador de 3 bandas (**High / Mid / Low**) con botones **Kill** por banda
- Fader de ganancia vertical con escala en dB
- Fader de velocidad/tono con curvas **Linear / Log / Exp**
- Nudge momentáneo **+BEND / -BEND**
- Tap BPM manual

### Transporte y performance

- **CUE** — estilo Pioneer: ancla el punto en pausa, reproduce en stutter mientras se mantiene pulsado
- **4 Hot Cues** por deck — marca y salta a cualquier posición al instante
- **Loops cuantizados** de 4 / 8 / 16 / 32 beats con indicador visual en la forma de onda
- **JogWheel** — gira con la pista; al pasar el cursor muestra los botones de nudge (+/-)
- **MASTER** — designa un deck como maestro de tempo
- **SYNC** — bloqueo de fase magnético del deck esclavo al BPM maestro

### Director de IA (Google Gemini)

| Función | Descripción |
|---|---|
| Perfil de pista multimodal | Gemini escucha fragmentos de 12 s del intro y el outro, y devuelve género, mood, instrumentos, presencia vocal, arco de energía, peso del bajo y técnicas de mezcla recomendadas |
| Corrección de BPM por IA | Detecta y corrige errores de halftime/doubletime del analizador de audio (p. ej. 87 → 174 BPM en Liquid D&B) |
| SMART MIX | Transición con un clic: genera un plan JSON con técnica, duración, beat de intercambio de bajos, objetivo de energía y advertencias |
| AUTOPILOT | Modo DJ completamente autónomo: carga una playlist, pre-analiza cada pista, dispara las transiciones en el mixPoint recomendado por la IA y avanza la cola |

### Técnicas de transición

| Técnica | Descripción |
|---|---|
| `cut` | Corte duro en el siguiente tiempo cuantizado |
| `blend` | Crossfade largo y suave (hasta 32 s) |
| `filter_sweep` | Barrido high-pass en la pista saliente + rampa de ganancia en la entrante |
| `echo_out` | High-pass agresivo + fade rápido — ideal para pistas con sub-bajos densos |

### Rendimiento y análisis

- Detección de BPM y tonalidad musical acelerada por **WebGPU** (cae a CPU si no está disponible)
- Overlay de sincronización de beats con forma de onda rodante y bloqueo de fase
- Medidores VU estéreo LED en tiempo real
- Medidores de RAM y GPU en la barra de estado inferior
- Efecto de distorsión de audio: desplazamiento SVG turbulence reactivo al espectro en vivo

### Grabación

- Grabación de sesión con un clic mediante `MediaRecorder` — descarga como `.webm` a 320 kbps

### Atajos de teclado

| Tecla | Acción |
|---|---|
| `S` | Play / Pausa Deck A |
| `A` | CUE Deck A (mantener = stutter) |
| `D` | Sincronizar Deck A al BPM del Deck B |
| `1–4` | Hot Cues 1–4 del Deck A |
| `L` | Play / Pausa Deck B |
| `K` | CUE Deck B (mantener = stutter) |
| `;` | Sincronizar Deck B al BPM del Deck A |
| `7–0` | Hot Cues 1–4 del Deck B |
| `← →` | Mover el crossfader |
| `B` | Centrar el crossfader |

---

## Stack tecnológico

| Capa | Tecnología |
|---|---|
| Framework UI | React 19 + TypeScript |
| Bundler | Vite 6 |
| Estilos | Tailwind CSS v4 |
| Animaciones | Framer Motion (motion/react) |
| Motor de audio | Web Audio API |
| Detección de BPM | `web-audio-beat-detector` |
| Análisis GPU | WebGPU (`@webgpu/types`) |
| Inteligencia artificial | Google Gemini 2.0 Flash (`@google/genai`) |
| Iconos | Lucide React |
| Escritorio | Electron 41 + electron-builder |

---

## Instalación y uso

### Requisitos previos

- Node.js 18 o superior
- Clave de API de Google Gemini — [obtener gratis](https://aistudio.google.com/app/apikey)

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/amauri7399/-liquid-glass-dj.git
cd -liquid-glass-dj

# 2. Instalar dependencias
npm install

# 3. Configurar la clave de API
cp .env.example .env.local
# Editar .env.local y reemplazar YOUR_API_KEY_HERE con tu clave real
```

### Ejecutar en el navegador

```bash
npm run dev
# Abre http://localhost:3000
```

### Ejecutar como aplicación de escritorio (Electron)

```bash
npm run electron:start
```

### Compilar para distribución

```bash
# Build web
npm run build

# Instalador de escritorio (.exe en Windows, .dmg en Mac, .AppImage en Linux)
npm run electron:build
```

---

## Configuración

Crear el archivo `.env.local` en la raíz del proyecto:

```env
GEMINI_API_KEY="tu-clave-de-api-aqui"
APP_URL="http://localhost:3000"
```

> La aplicación funciona sin clave de API. La detección de BPM, el ecualizador, la mezcla manual, los loops y la grabación están disponibles sin conexión a Gemini. Las funciones de IA (SMART MIX, AUTOPILOT, perfil de pista) usarán valores predeterminados si no hay clave configurada.

---

## Autopilot — cómo funciona

1. Subir una playlist en el panel lateral de AUTOPILOT
2. Pulsar **Iniciar Autopilot**
3. El sistema:
   - Carga la pista 1 en el Deck A y la pista 2 en el Deck B
   - Envía fragmentos de audio a Gemini para construir un perfil por pista
   - Corrige el BPM si el detector devolvió un valor en halftime o doubletime
   - Espera hasta que se alcanza el mixPoint recomendado por la IA (entre 40 % y 70 % de la pista)
   - Ejecuta la transición con SMART MIX usando los perfiles completos de ambos decks
   - Cambia el deck activo y carga la siguiente pista
   - Repite hasta agotar la playlist

### Referencia de mixPoint

| Valor | Disparo | Uso típico |
|---|---|---|
| `early_cut` | 40 % consumido | Pistas cortas o repetitivas |
| `mid_break` | 50 % consumido | Halftime con bajos densos |
| `post_drop` | 65 % consumido | Cortar después del segundo drop |
| `outro` | 70 % consumido | Pistas con outro definido (Liquid D&B por defecto) |

### Diccionario de fases

| Fase | Género | Rango de BPM | Técnica por defecto |
|---|---|---|---|
| 1 | Halftime / Neurohop | 85–110 | `cut` o `echo_out` |
| 2 | Breakbeat / Cinematic | 125–140 | `filter_sweep` |
| 3 | Liquid / Soulful D&B | 168–174 | `blend` |
| 4 | Rollers / Peak D&B | 172–178 | `filter_sweep` o `blend` |

---

## Estructura del proyecto

```
liquid-glass-dj/
├── src/
│   ├── App.tsx              # Aplicacion principal (logica + UI)
│   ├── webgpu-analyzer.ts   # Analisis de BPM y tonalidad con WebGPU
│   ├── main.tsx             # Punto de entrada de React
│   └── index.css            # Estilos globales + directivas Tailwind
├── electron-main.cjs        # Proceso principal de Electron
├── index.html               # Shell HTML
├── vite.config.ts           # Configuracion de Vite
├── tsconfig.json            # Configuracion de TypeScript
├── .env.example             # Plantilla de variables de entorno
└── package.json
```

---

## Seguridad

- La clave de API se lee desde `.env.local` en tiempo de compilación mediante `process.env.GEMINI_API_KEY`
- `.env.local` está excluido del repositorio git por la regla `.env*` en `.gitignore`
- Ninguna clave de API está escrita directamente en el código fuente
- El audio enviado a Gemini es PCM WAV mono a 16 kHz, con un máximo de ~256 KB por fragmento

---

## Licencia

Apache 2.0
