---
name: add-concept
description: Research an electrical engineering concept, design a Three.js visualization for it, and add it as src/concepts/{id}/index.tsx with English/Korean content and sliders that visibly change the scene. Use when the user asks to add, create, or visualize a concept, formula, law, or phenomenon (circuits, electromagnetics, electronics, power systems, machines, control, installations).
argument-hint: "[concept keyword]"
---

# Add a Concept

Usage: `/add-concept [concept keyword]` — e.g. `/add-concept series RLC resonance`, `/add-concept 3상 회전자계`

Goal: from a single keyword, ship a concept that appears in the sidebar wheel, renders its scene when selected, and has sliders whose effect is visible. Work straight through all steps without stopping for approval; report the result at the end.

Read `AGENTS.md` and `DESIGN.md` first, and look at an existing concept (`src/concepts/rlc-series/index.tsx`) for the idiom.

## 1. Research

- Find the governing formula(s), units, and sign/phase conventions. Use WebSearch/WebFetch against reliable sources (textbooks, university course notes, standards bodies) whenever you are not certain; note the sources for the final report.
- Pick **one worked example with concrete numbers** (e.g. R = 100 Ω, L = 100 mH, C = 10 μF, f = 100 Hz → Z ≈ …). Compute it with a throwaway `node -e` / `npx tsx -e` one-liner and confirm it matches the formula. This number is your ground truth for step 4.
- Check that the concept doesn't already exist: `ls src/concepts`.

## 2. Design the scene

Write a short plan (keep it for the final report):

- **Core idea** in one sentence — what should the learner *see*?
- **Visual metaphor**: phasors rotating in the complex plane, time traces receding along −Z, field lines, a rotating machine, a block diagram with signal flow… Choose what makes the idea obvious, not what is easiest to draw.
- **Elements → color tokens** from DESIGN.md (voltage, current, resistance, …). Add a token only if no existing one fits.
- **Sliders**: 1–4 parameters; for each, the change it causes on screen. If a slider changes nothing visible, drop it or change the design.
- **3D vs 2D**: `Scene` (3D, with OrbitControls) is required; add `Scene2D` when a flat time-domain or front view helps.
- **Card spot**: the description card floats at `[2.6, 0.6, -1.5]` in 3D (DESIGN.md › Layout). Keep the visualization clear of it, or set `cardPosition` in the metadata.

## 3. Implement

Create `src/concepts/{id}/index.tsx`. Nothing else needs registering.

### Metadata

- **id**: kebab-case, must equal the folder name
- **subject**: `'electromagnetics' | 'circuits' | 'electronics' | 'power-systems' | 'machines' | 'control' | 'installations'` (new subject → the 4 places listed in AGENTS.md)
- **order**: position within the subject; check existing concepts in that subject
- **title, description, slider labels**: `{ en, ko }`, both required. English first, then natural Korean. Same formulas and symbols in both.
- **description**: follow DESIGN.md › Descriptions; color names must match the tokens used
- **sliderParams**: at least 1; SI `unit`; defaults that show the interesting behavior immediately

### Scene idiom

Put the physics in a module-scope pure function shared by `Scene` and `Scene2D`, so the formula lives in one place and matches your worked example.

```tsx
import { useRef, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import type { ConceptMeta } from '../../types/concept'

const POINTS = 200
const LINE_INIT: [number, number, number][] = Array.from({ length: POINTS }, () => [0, 0, 0])

const VM = 100 // source peak voltage [V]
const DISPLAY_SCALE = 1 // scene units per amp; pick so content stays within ±2
const SPEED = 1.5 // on-screen rad/s, slowed from the real ω so motion is readable

// Pure physics: same function feeds both scenes and matches the worked example
function solve({ R = 100 }: Record<string, number>) {
  const Im = VM / R // Ohm's law, peak current [A]
  return { Im }
}

function useSliderValues() {
  // Read sliders through a ref: no React re-render per frame
  const valuesRef = useRef<Record<string, number>>(useSliderStore.getState().values)
  useEffect(
    () => useSliderStore.subscribe((s) => s.values, (v) => { valuesRef.current = v }),
    []
  )
  return valuesRef
}

function MyScene() {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  // Allocate buffers once; mutate them in place every frame
  const positions = useMemo(() => new Float32Array(POINTS * 3), [])
  const lineRef = useRef<any>(null)
  const dotRef = useRef<THREE.Group>(null)

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime()
    const amp = solve(valuesRef.current).Im * DISPLAY_SCALE
    // i(t) trace: newest sample at z = 0, history recedes toward −Z
    for (let i = 0; i < POINTS; i++) {
      const frac = i / (POINTS - 1)
      const theta = SPEED * t - (1 - frac) * 4 * Math.PI
      positions[i * 3] = amp * Math.cos(theta)
      positions[i * 3 + 1] = amp * Math.sin(theta)
      positions[i * 3 + 2] = -(1 - frac) * 4
    }
    lineRef.current?.geometry.setPositions(positions)
    dotRef.current?.position.set(amp * Math.cos(SPEED * t), amp * Math.sin(SPEED * t), 0)
  })

  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />
      {/* Background, lights, grid, and bloom come from the shared SceneStage */}
      <TrailLine ref={lineRef} points={LINE_INIT} lineWidth={2.5} color={c.current} />
      <Marker ref={dotRef} color={c.marker} />
    </>
  )
}

const concept: ConceptMeta = {
  id: 'my-concept',
  title: { en: 'My Concept', ko: '내 개념' },
  subject: 'circuits',
  order: 3,
  description: { en: `## ...`, ko: `## ...` },
  sliderParams: [
    { key: 'R', label: { en: 'Resistance (R)', ko: '저항 (R)' }, unit: 'Ω', min: 50, max: 500, step: 10, defaultValue: 100 },
  ],
  Scene: MyScene,
}

export default concept
```

### Forbidden

- Hex color literals — use `usePhysicsColors()` (prebuild rejects hex)
- Own background, lights, grid, or post-processing — `SceneStage` provides them; use `TrailLine` / `Marker` for traces and dots (DESIGN.md › Scene conventions)
- `setState` or allocations (`new Float32Array`, arrays, objects) inside `useFrame`
- `sliderParams` built inside a component (unstable reference)
- URLs or `//` inside `description` (breaks prebuild comment stripping)
- CSS or other external file imports; `import.meta.glob` (core hooks only)
- One-language strings

## 4. Verify

1. `pnpm build` passes (validation + `tsc` + bundle). Fix and rerun until it does.
2. Re-check the physics: plug the worked example's slider values into `solve()` (throwaway `npx tsx -e`) and confirm it reproduces the step-1 numbers.
3. Look at it. With the Chrome extension (the `run` skill or Chrome tools): run `pnpm dev`, select the concept, move each slider, and check 3D/2D in light and dark. Without it, take headless screenshots of `pnpm build && pnpm preview --port 4199` (a fresh profile in the scratchpad; the page animates forever, so poll for the file and kill Chrome):
   ```bash
   "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new \
     --user-data-dir="$SCRATCH/cp-light" --no-first-run --use-angle=swiftshader --enable-unsafe-swiftshader \
     --window-size=1440,900 --virtual-time-budget=5000 --blink-settings=preferredColorScheme=1 \
     --enable-logging=stderr --screenshot="$SCRATCH/light.png" http://localhost:4199 2>"$SCRATCH/light.log"
   ```
   Use `--force-dark-mode` instead of `--blink-settings=…` for dark. Read the PNG, and grep the log for `Uncaught`, `THREE.`, and shader errors. Plain headless shows only the first concept in 3D. To select another concept, switch to 2D, or move sliders, drive Chrome with `puppeteer-core` installed in the scratchpad (never in the project), using the same `executablePath` and flags. Never claim a visual check you didn't do.

## 5. Report

Keep it short:
- File created and its subject/order
- The core idea and visual metaphor, and what each slider does
- The worked example and the sources used
- What was verified (build, physics check, visual check or not)
