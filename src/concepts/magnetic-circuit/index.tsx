import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import type { Group, Mesh } from 'three'
import type { LineMaterial } from 'three-stdlib'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { useLocaleStore } from '../../lib/i18n'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import {
  setLineOpacity, setOpacity, useStoryReveal,
} from '../../components/scene/storyReveal'
import type { ConceptMeta } from '../../types/concept'

const TAU = Math.PI * 2
const MU0 = 4 * Math.PI * 1e-7 // permeability of vacuum [H/m]
const L_PATH = 0.5 // mean magnetic path length of the core [m]
const AREA = 1e-3 // cross-section, 10 cm² [m²]
const B_SAT = 1.6 // saturation flux density of the modelled iron [T]
const HC_FRAC = 0.08 // coercivity as a fraction of the knee field, so the loop stays visible at any μr
const LOSS_N = 121 // trapezoid samples for the loop area (converged to 0.1% by 91)
const PHI_REF = 1.2566e-3 // the worked example's flux [Wb], used only to scale line widths

// Magnetic circuit = Ohm's law for flux. mmf drives it, reluctance opposes it:
//   F = N·I [AT],  R = l ÷ (μ0·μr·A) in iron,  R = l ÷ (μ0·A) in air,  Φ = F ÷ ΣR,  B = Φ ÷ A
// Worked example, verified with node -e (l = 0.5 m, A = 10 cm² = 1e-3 m², μr = 1000, N = 500, I = 1 A):
//   F     = 500 × 1                       =     500      AT
//   Riron = 0.5 ÷ (4πe-7 × 1000 × 1e-3)   = 397 887.4    AT/Wb
//   Φ     = 500 ÷ 397887.4                =       1.2566 mWb
//   B     = 1.2566e-3 ÷ 1e-3              =       1.2566 T
// Now cut a 1 mm air gap (iron path shortens to 0.499 m):
//   Rgap  = 1e-3 ÷ (4πe-7 × 1e-3)         = 795 774.7    AT/Wb   (2.004 × the iron's 397 091.6)
//   ΣR    = 397091.6 + 795774.7           = 1 192 866.3  AT/Wb
//   Φ     = 500 ÷ 1192866.3               =       0.4192 mWb     (a third of before, same current)
//   B     = 0.4192 T
// Hysteresis loss per cycle is the loop area ∮H dB; the trapezoid sum gives 350.1 J/m³ with the gap
// closed and 131.7 J/m³ with the 1 mm gap open. Steinmetz: Ph = k·f·B^1.6.
// One module-scope record holds the answers, so no frame allocates.
const S = {
  N: 500, I: 1, gap: 0, mur: 1000,
  F: 500, Riron: 397887.4, Rgap: 0, Rtot: 397887.4,
  phi: 1.2566e-3, B: 1.2566, gapShare: 0,
  a: 1273, Hc: 101.9, Hm: 1000, Bloop: 1.053, loss: 350.1,
}

/** The lagging branch of the B–H loop: H sweeps as cos, the coercive offset as sin. */
const loopB = (th: number, Hm: number, Hc: number, a: number) =>
  B_SAT * Math.tanh((Hm * Math.cos(th) + Hc * Math.sin(th)) / a)

function solve(v: Record<string, number>) {
  const N = v.N ?? 500
  const I = v.I ?? 1
  const gap = (v.gap ?? 0) * 1e-3 // slider is in mm
  const mur = v.mur ?? 1000
  const lIron = Math.max(L_PATH - gap, 1e-3)
  const Riron = lIron / (MU0 * mur * AREA)
  const Rgap = gap / (MU0 * AREA)
  const Rtot = Riron + Rgap
  const F = N * I
  const phi = F / Rtot
  S.N = N
  S.I = I
  S.gap = gap
  S.mur = mur
  S.F = F
  S.Riron = Riron
  S.Rgap = Rgap
  S.Rtot = Rtot
  S.phi = phi
  S.B = phi / AREA
  S.gapShare = Rgap / Rtot

  // The loop: knee field a is where the material gives up, so μ0·μr is its slope at the origin.
  // Driving H is the field the linear model predicts inside the iron, B ÷ (μ0·μr).
  const a = B_SAT / (MU0 * mur)
  const Hc = HC_FRAC * a
  const Hm = S.B / (MU0 * mur)
  let loss = 0
  let hPrev = Hm
  let bPrev = loopB(0, Hm, Hc, a)
  let bPeak = bPrev
  for (let i = 1; i <= LOSS_N; i++) {
    const th = (i / LOSS_N) * TAU
    const H = Hm * Math.cos(th)
    const B = loopB(th, Hm, Hc, a)
    loss += 0.5 * (hPrev + H) * (B - bPrev) // trapezoid ∮H dB
    if (B > bPeak) bPeak = B
    hPrev = H
    bPrev = B
  }
  S.a = a
  S.Hc = Hc
  S.Hm = Hm
  S.Bloop = bPeak
  S.loss = Math.abs(loss)
  return S
}

function useSliderValues() {
  const valuesRef = useRef<Record<string, number>>(useSliderStore.getState().values)
  useEffect(
    () => useSliderStore.subscribe((s) => s.values, (v) => { valuesRef.current = v }),
    []
  )
  return valuesRef
}

const fmtK = (x: number) => `${(x / 1000).toFixed(1)} kAT ÷ Wb`
const pct = (x: number) => `${Math.round(x * 100)}%`

// Labels read in the foreground color; only the two accents tint their own text.
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const accentClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums'
const axisClass = 'pointer-events-none whitespace-nowrap text-[11px] text-muted-foreground'

// ── Narration: the circuit is assembled one part at a time, the way it is reasoned about ──
// setLineOpacity works on a mesh too: a Mesh carries `material`, which is all it touches.
const REV_COIL = 0 // the winding, its N and I labels, and the mmf readout
const REV_FLUX = 1 // the flux loop, the Φ / B readout, and the 2D operating point
const REV_BAR = 2 // the series reluctance bar and its two segment labels
const REV_LOSS = 3 // the hysteresis-loss line, the saturation flag, and the 2D loop area
const REV_COUNT = 4

function revealTargets(step: number, out: Float32Array) {
  out[REV_COIL] = step >= 1 ? 1 : 0
  out[REV_FLUX] = step >= 2 ? 1 : 0
  out[REV_BAR] = step >= 3 ? 1 : 0
  out[REV_LOSS] = step >= 8 ? 1 : 0
}

// ── 3D: the iron ring, its winding, the flux, and the reluctance split as a series bar ──
const X_L = -1.25 // left leg: the winding sits here
const X_R = 0.75 // right leg: the air gap opens here
const XM = (X_L + X_R) / 2
const CORE_W = X_R - X_L
const CORE_HALF = 0.15 // half the iron's thickness
const CORE_Y = 0.75 // yoke center height
const LEG_TOP = CORE_Y + CORE_HALF
const GAP_PER_MM = 0.09 // the real gap is a thousandth of the path, so it is drawn far wider
const COIL_R = 0.3
const COIL_Y0 = -0.6
const COIL_Y1 = 0.6
const TURNS_MAX = 14 // drawn turns at the top of the N slider
const N_MAX = 1000

const LEG_ARGS: [number, number, number] = [2 * CORE_HALF, 2 * LEG_TOP, 2 * CORE_HALF]
const YOKE_ARGS: [number, number, number] = [CORE_W, 2 * CORE_HALF, 2 * CORE_HALF]
const UNIT: [number, number, number] = [1, 1, 1]

// The flux: one closed loop squared off to follow the iron
const FLUX_N = 161
const FLUX_PTS: [number, number, number][] = Array.from({ length: FLUX_N }, (_, i) => {
  const t = (i / (FLUX_N - 1)) * TAU
  const cs = Math.cos(t)
  const sn = Math.sin(t)
  return [
    XM + (CORE_W / 2) * Math.sign(cs) * Math.sqrt(Math.abs(cs)),
    CORE_Y * Math.sign(sn) * Math.sqrt(Math.abs(sn)),
    0,
  ]
})

const COIL_N = 193
const COIL_INIT: [number, number, number][] = Array.from(
  { length: COIL_N },
  (): [number, number, number] => [0, 0, 0]
)

// Series reluctance bar: iron segment then gap segment, standing on one foot.
// Total length grows as √ΣR so a 55-fold range stays on screen; the split between the two
// segments is the true ratio Riron : Rgap.
const BAR_X0 = -2.0
const BAR_Y = -1.5
const BAR_H = 0.17
const BAR_K = 1.72e-3 // scene units per √(AT/Wb)
const BAR_MAX = 3.6

function fillCoil(buf: Float32Array, turns: number) {
  const span = COIL_Y1 - COIL_Y0
  for (let i = 0; i < COIL_N; i++) {
    const u = i / (COIL_N - 1)
    const th = u * turns * TAU
    buf[i * 3] = X_L + COIL_R * Math.cos(th)
    buf[i * 3 + 1] = COIL_Y0 + u * span
    buf[i * 3 + 2] = COIL_R * Math.sin(th)
  }
}

function setSegment(m: Mesh | null | undefined, x0: number, len: number) {
  if (!m) return
  const w = Math.max(len, 1e-4)
  m.visible = len > 1e-3
  m.scale.set(w, BAR_H, BAR_H)
  m.position.set(x0 + w / 2, BAR_Y, 0)
}

function MagneticCircuitScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const coilBuf = useMemo(() => new Float32Array(COIL_N * 3), [])
  const coilRef = useRef<any>(null)
  const fluxRef = useRef<any>(null)
  const legRefs = useRef<(Mesh | null)[]>([]) // 0 upper half of the gapped leg, 1 lower half
  const segRefs = useRef<(Mesh | null)[]>([]) // 0 iron, 1 air gap
  const labelRefs = useRef<(Group | null)[]>([]) // 0 iron label, 1 gap label
  // 0 mmf, 1 flux and B, 2 current, 3 turns, 4 Riron, 5 Rgap, 6 loss, 7 saturation flag
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const flow = useRef([0, 0]) // winding current, flux
  const labelledFor = useRef<unknown>(null)
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(() => {
    const s = solve(valuesRef.current)

    // The winding drawn as the current itself: turns from N, line width from I
    fillCoil(coilBuf, (TURNS_MAX * s.N) / N_MAX)
    const coil = coilRef.current
    if (coil) {
      coil.geometry.setPositions(coilBuf)
      ;(coil.material as LineMaterial).linewidth = Math.min(1.2 + 1.8 * Math.sqrt(s.I), 7)
    }
    flow.current[0] = Math.min(0.6 + 1.2 * s.I, 8)

    // The flux: same closed path whatever the gap, but it thins out as the reluctance climbs
    const fluxMat = fluxRef.current?.material as LineMaterial | undefined
    const fluxFrac = s.phi / PHI_REF
    if (fluxMat) fluxMat.linewidth = Math.min(Math.max(0.4 + 5 * fluxFrac, 0.3), 9)
    flow.current[1] = Math.min(3 * fluxFrac, 9)

    // Opening the gap cuts the right leg in two
    const gapVis = s.gap * 1e3 * GAP_PER_MM
    const half = LEG_TOP - gapVis / 2
    for (let k = 0; k < 2; k++) {
      const m = legRefs.current[k]
      if (!m) continue
      m.scale.set(2 * CORE_HALF, Math.max(half, 1e-3), 2 * CORE_HALF)
      m.position.set(X_R, (k === 0 ? 1 : -1) * (LEG_TOP + gapVis / 2) / 2, 0)
    }

    const total = Math.min(BAR_K * Math.sqrt(s.Rtot), BAR_MAX)
    const ironLen = total * (1 - s.gapShare)
    setSegment(segRefs.current[0], BAR_X0, ironLen)
    setSegment(segRefs.current[1], BAR_X0 + ironLen, total - ironLen)
    // Both labels are far wider than the segments they name, so they are stacked on two lines
    // rather than placed side by side: at any split they would otherwise run into each other.
    labelRefs.current[0]?.position.set(BAR_X0 + ironLen / 2, BAR_Y - 0.24, 0)
    labelRefs.current[1]?.position.set(BAR_X0 + ironLen + (total - ironLen) / 2, BAR_Y - 0.52, 0)
    const hasGapSegment = total - ironLen > 0.12

    // ── the narration's fades, every frame, since the weights ease between steps ──
    const wCoil = reveal[REV_COIL]
    setLineOpacity(coilRef.current, wCoil)
    setOpacity(textRefs.current[0], wCoil)
    setOpacity(textRefs.current[2], wCoil)
    setOpacity(textRefs.current[3], wCoil)

    const wFlux = reveal[REV_FLUX]
    setLineOpacity(fluxRef.current, wFlux)
    setOpacity(textRefs.current[1], wFlux)

    const wBar = reveal[REV_BAR]
    setLineOpacity(segRefs.current[0], wBar)
    setLineOpacity(segRefs.current[1], wBar)
    setOpacity(textRefs.current[4], wBar)
    setOpacity(textRefs.current[5], hasGapSegment ? wBar : 0)

    setOpacity(textRefs.current[6], reveal[REV_LOSS])
    setOpacity(textRefs.current[7], reveal[REV_LOSS])

    if (labelledFor.current !== valuesRef.current && textRefs.current[7]) {
      labelledFor.current = valuesRef.current
      const ko = locale === 'ko'
      if (textRefs.current[0]) textRefs.current[0]!.textContent = `F = N·I = ${s.F.toFixed(0)} AT`
      if (textRefs.current[1]) {
        textRefs.current[1]!.textContent =
          `Φ = F ÷ ΣR = ${(s.phi * 1e3).toFixed(3)} mWb · B = Φ ÷ A = ${s.B.toFixed(3)} T`
      }
      if (textRefs.current[2]) textRefs.current[2]!.textContent = `I = ${s.I.toFixed(2)} A`
      if (textRefs.current[3]) textRefs.current[3]!.textContent = `N = ${s.N.toFixed(0)}`
      if (textRefs.current[4]) {
        textRefs.current[4]!.textContent =
          `${ko ? '철심' : 'iron'} ${fmtK(s.Riron)} · ${pct(1 - s.gapShare)}`
      }
      if (textRefs.current[5]) {
        textRefs.current[5]!.textContent =
          `${ko ? '공극' : 'air gap'} ${fmtK(s.Rgap)} · ${pct(s.gapShare)}`
      }
      if (textRefs.current[6]) {
        textRefs.current[6]!.textContent = ko
          ? `히스테리시스손 ${s.loss.toFixed(0)} J ÷ m³, 1사이클당`
          : `hysteresis loss ${s.loss.toFixed(0)} J ÷ m³ per cycle`
      }
      textRefs.current[7]!.textContent =
        s.B > B_SAT ? (ko ? `포화 · B > ${B_SAT} T` : `saturated · B > ${B_SAT} T`) : ''
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  return (
    <>
      {/* Pulled back and aimed right of the core: the scene runs from the bar at x −2 up to the
          readouts at y 1.55, and the narration panel takes the right of the frame */}
      <PerspectiveCamera makeDefault position={[3, 2, 5.8]} fov={50} />
      <OrbitControls enablePan={false} target={[0.5, -0.15, 0]} />

      {/* Background, lights, grid, and bloom come from the shared SceneStage */}
      {/* The iron core as glass, so the flux inside stays visible */}
      <mesh position={[X_L, 0, 0]}>
        <boxGeometry args={LEG_ARGS} />
        <meshStandardMaterial
          color={c.inductance} roughness={0.6} metalness={0.25}
          transparent opacity={0.4} depthWrite={false}
        />
      </mesh>
      {[CORE_Y, -CORE_Y].map((y, i) => (
        <mesh key={i} position={[XM, y, 0]}>
          <boxGeometry args={YOKE_ARGS} />
          <meshStandardMaterial
            color={c.inductance} roughness={0.6} metalness={0.25}
            transparent opacity={0.4} depthWrite={false}
          />
        </mesh>
      ))}
      {/* The gapped leg: two halves that pull apart as the gap opens */}
      {[0, 1].map((k) => (
        <mesh key={k} ref={(m) => { legRefs.current[k] = m }}>
          <boxGeometry args={UNIT} />
          <meshStandardMaterial
            color={c.inductance} roughness={0.6} metalness={0.25}
            transparent opacity={0.4} depthWrite={false}
          />
        </mesh>
      ))}

      {/* The flux: one closed loop round the iron, thinning as the reluctance grows */}
      <TrailLine
        ref={fluxRef}
        points={FLUX_PTS}
        lineWidth={5}
        color={c['b-field']}
        fade={false}
        flowSpeed={() => flow.current[1]}
      />

      {/* The winding, drawn as the current path: turns from N, thickness from I */}
      <TrailLine
        ref={coilRef}
        points={COIL_INIT}
        lineWidth={2.5}
        color={c.current}
        fade={false}
        flowSpeed={() => flow.current[0]}
      />

      {/* Series reluctance: iron then air gap, end to end like two resistors */}
      {[0, 1].map((k) => (
        <mesh key={k} ref={(m) => { segRefs.current[k] = m }}>
          <boxGeometry args={UNIT} />
          <meshStandardMaterial color={k === 0 ? c.resistance : c.capacitance} roughness={0.5} />
        </mesh>
      ))}

      <Html position={[XM, 1.55, 0]} center>
        <div className="pointer-events-none text-center leading-5">
          <div><span ref={(el) => { textRefs.current[0] = el }} className={labelClass} /></div>
          <div><span ref={(el) => { textRefs.current[1] = el }} className={labelClass} /></div>
        </div>
      </Html>
      <Html position={[X_L, -1.22, 0]} center>
        <div className="pointer-events-none text-center leading-5">
          <div>
            <span ref={(el) => { textRefs.current[2] = el }} className={accentClass} style={{ color: c.current }} />
          </div>
          <div><span ref={(el) => { textRefs.current[3] = el }} className={labelClass} /></div>
        </div>
      </Html>
      {[0, 1].map((k) => (
        <group key={k} ref={(g) => { labelRefs.current[k] = g }}>
          <Html center>
            <span ref={(el) => { textRefs.current[4 + k] = el }} className={labelClass} />
          </Html>
        </group>
      ))}
      <Html position={[XM, -2.05, 0]} center>
        <div className="pointer-events-none text-center leading-5">
          <div>
            <span ref={(el) => { textRefs.current[6] = el }} className={accentClass} style={{ color: c.power }} />
          </div>
          <div>
            <span ref={(el) => { textRefs.current[7] = el }} className={accentClass} style={{ color: c.power }} />
          </div>
        </div>
      </Html>
    </>
  )
}

// ── 2D: the B–H loop, where the linear model's μr is only the slope at the origin ──
// The H axis rescales with the material so the knees are always in frame; its current full-scale
// value is printed. B is on a fixed scale with the ±Bs ceilings marked.
const HALF_W = 3.5
const Y_PER_B = 1.6 / B_SAT
const LOOP_N = 361
const LOOP_INIT: [number, number, number][] = Array.from(
  { length: LOOP_N },
  (): [number, number, number] => [0, 0, 0]
)
const AXES: [number, number, number][] = [
  [-HALF_W, 0, 0], [HALF_W, 0, 0],
]
const V_AXIS: [number, number, number][] = [
  [0, -1.75, 0], [0, 1.75, 0],
]
const SAT_LINES: [number, number, number][][] = [
  [[-HALF_W, 1.6, 0], [HALF_W, 1.6, 0]],
  [[-HALF_W, -1.6, 0], [HALF_W, -1.6, 0]],
]

function HysteresisLoop2DScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const loopBuf = useMemo(() => new Float32Array(LOOP_N * 3), [])
  const loopRef = useRef<any>(null)
  const dotRef = useRef<Group>(null)
  // 0 the H full scale, 1 the operating point, 2 the loss
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<unknown>(null)
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(() => {
    const s = solve(valuesRef.current)
    // Keep both knees and the whole drive in frame
    const full = Math.max(2.5 * s.a, 1.15 * s.Hm, 1)
    const xPerH = HALF_W / full
    for (let i = 0; i < LOOP_N; i++) {
      const th = (i / (LOOP_N - 1)) * TAU
      loopBuf[i * 3] = s.Hm * Math.cos(th) * xPerH
      loopBuf[i * 3 + 1] = loopB(th, s.Hm, s.Hc, s.a) * Y_PER_B
      loopBuf[i * 3 + 2] = 0
    }
    loopRef.current?.geometry.setPositions(loopBuf)
    dotRef.current?.position.set(s.Hm * xPerH, loopB(0, s.Hm, s.Hc, s.a) * Y_PER_B, 0.01)

    // The loop is the material's own curve and is there from the first step; what the story
    // reveals is the operating point on it and the area under it.
    dotRef.current?.scale.setScalar(reveal[REV_FLUX])
    setOpacity(textRefs.current[1], reveal[REV_FLUX])
    setOpacity(textRefs.current[2], reveal[REV_LOSS])

    if (labelledFor.current !== valuesRef.current && textRefs.current[2]) {
      labelledFor.current = valuesRef.current
      const ko = locale === 'ko'
      if (textRefs.current[0]) {
        textRefs.current[0]!.textContent = `± ${full.toFixed(0)} A ÷ m`
      }
      if (textRefs.current[1]) {
        textRefs.current[1]!.textContent =
          `H = ${s.Hm.toFixed(0)} A ÷ m · B = ${s.Bloop.toFixed(3)} T` +
          ` · ${ko ? '선형 모델' : 'linear model'} ${s.B.toFixed(3)} T`
      }
      textRefs.current[2]!.textContent = ko
        ? `고리 면적 ${s.loss.toFixed(0)} J ÷ m³, 1사이클당 열`
        : `loop area ${s.loss.toFixed(0)} J ÷ m³ lost as heat each cycle`
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  return (
    <>
      <Line points={AXES} lineWidth={1} color={c.guide} />
      <Line points={V_AXIS} lineWidth={1} color={c.guide} />
      {SAT_LINES.map((pts, i) => (
        <Line key={i} points={pts} lineWidth={1} color={c.guide} dashed dashSize={0.1} gapSize={0.08} />
      ))}
      <TrailLine ref={loopRef} points={LOOP_INIT} lineWidth={3} color={c['b-field']} fade={false} flowSpeed={1.2} />
      <Marker ref={dotRef} color={c.marker} size={0.06} />

      <Html position={[HALF_W - 0.1, -0.22, 0]} center>
        <span className={axisClass}>H (A ÷ m)</span>
      </Html>
      <Html position={[0.42, 1.72, 0]} center>
        <span className={axisClass}>B (T)</span>
      </Html>
      <Html position={[-HALF_W + 0.42, 1.72, 0]} center>
        <span className={axisClass}>± Bs = {B_SAT} T</span>
      </Html>
      <Html position={[HALF_W - 0.1, 0.22, 0]} center>
        <span ref={(el) => { textRefs.current[0] = el }} className={axisClass} />
      </Html>
      <Html position={[-HALF_W + 0.1, 1.92, 0]}>
        <span ref={(el) => { textRefs.current[1] = el }} className={labelClass} />
      </Html>
      <Html position={[-HALF_W + 0.1, -1.9, 0]}>
        <span ref={(el) => { textRefs.current[2] = el }} className={accentClass} style={{ color: c.power }} />
      </Html>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'magnetic-circuit',
  title: { en: 'Magnetic Circuits & Hysteresis', ko: '자기회로와 히스테리시스' },
  subject: 'electromagnetics',
  order: 7,
  description: {
    en: `## The Magnetic Circuit and Hysteresis

**Flux in an iron core keeps the same books as current in a wire: mmf is the voltage, reluctance is the resistance, and a 1 mm air gap outweighs half a metre of iron.**

- The glassy ring: the iron core, the path the flux is confined to
- The deep cool loop inside it: the flux Φ. It thickens with mmf and thins as the reluctance climbs
- The cyan helix: the winding, drawn as the current itself. Its turn count follows N and its thickness follows I
- The two halves of the right leg: they pull apart as the air gap opens. The drawn gap is exaggerated, since 1 mm against a 0.5 m path would be invisible
- The horizontal bar: the reluctance in series, the iron segment first and the air gap second. Its total length grows as the square root of ΣR so a 55-fold range stays on screen, but the split between the segments is the true ratio
- The amber readout: the hysteresis loss, the part of the energy that never comes back

**Magnetomotive force**: F = N·I [AT]

**Reluctance**: R = l ÷ (μ0·μr·A) in iron, R = l ÷ (μ0·A) in air

**Ohm's law for magnetism**: Φ = F ÷ ΣR, and B = Φ ÷ A

**Steinmetz**: Ph = k·f·B^1.6, the hysteresis loss power per unit volume

Worked example: l = 0.5 m, A = 10 cm², μr = 1000, N = 500 turns, I = 1 A. Then F = 500 AT and Riron = 397 887 AT ÷ Wb, which gives Φ = 1.257 mWb and B = 1.257 T. Now cut a 1 mm air gap: it adds Rgap = 795 775 AT ÷ Wb, more than twice the iron's share, so ΣR nearly triples and Φ falls to 0.419 mWb. The current never moved.

Open the gap first and watch only that: the flux collapses to a third while I stays put, and the gap segment takes two thirds of the bar. Then drop μr and the iron segment grows until the gap stops mattering, which is why μr is a material's whole contribution here. Raise N or I to push the flux back up, and keep going until B passes Bs and the core saturates.

In 2D: the B–H loop. The linear model above uses μr as a constant, but μr is only the slope at the origin, so the real curve bends over at the knees and B falls short of the linear answer. The loop's area is the energy turned into heat each cycle, and a wider loop means a lossier core.`,
    ko: `## 자기회로와 히스테리시스

**철심 속 자속은 전선 속 전류와 같은 방식으로 계산됩니다. 기자력이 전압, 자기저항이 저항이고, 1 mm 공극 하나가 0.5 m 철심보다 무겁습니다.**

- 유리처럼 비치는 고리: 철심. 자속이 갇혀 지나가는 길입니다
- 안쪽을 도는 짙은 한색 고리: 자속 Φ. 기자력이 커지면 굵어지고 자기저항이 커지면 가늘어집니다
- 청록색 나선: 권선. 전류 그 자체로 그렸습니다. 감은 수는 N을 따르고 선의 굵기는 I를 따릅니다
- 오른쪽 기둥의 두 조각: 공극이 열리면 서로 벌어집니다. 0.5 m 경로에 1 mm는 보이지 않으므로 공극은 과장해 그렸습니다
- 가로 막대: 직렬로 놓인 자기저항. 앞쪽이 철심, 뒤쪽이 공극입니다. 전체 길이는 ΣR의 제곱근으로 늘려서 55배 범위를 화면에 담았지만, 두 구간의 비율은 실제 비율 그대로입니다
- 호박색 수치: 히스테리시스손. 돌아오지 않는 에너지입니다

**기자력**: F = N·I [AT]

**자기저항**: 철심은 R = l ÷ (μ0·μr·A), 공기는 R = l ÷ (μ0·A)

**자기회로의 옴의 법칙**: Φ = F ÷ ΣR, 그리고 B = Φ ÷ A

**스타인메츠 식**: Ph = k·f·B^1.6, 단위 체적당 히스테리시스손

계산 예: l = 0.5 m, A = 10 cm², μr = 1000, N = 500회, I = 1 A입니다. F = 500 AT, Riron = 397 887 AT ÷ Wb이므로 Φ = 1.257 mWb, B = 1.257 T입니다. 여기에 1 mm 공극을 내면 Rgap = 795 775 AT ÷ Wb가 더해집니다. 철심 몫의 두 배가 넘으므로 ΣR이 거의 세 배가 되고 Φ는 0.419 mWb로 떨어집니다. 전류는 전혀 건드리지 않았습니다.

먼저 공극만 열어 보십시오. I는 그대로인데 자속이 3분의 1로 주저앉고, 막대에서 공극 구간이 3분의 2를 차지합니다. 다음으로 μr을 낮추면 철심 구간이 자라서 공극이 더는 중요하지 않게 됩니다. 여기서 재료가 기여하는 것은 결국 μr뿐이라는 뜻입니다. N이나 I를 올리면 자속이 다시 커지고, 더 밀어붙이면 B가 Bs를 넘어 철심이 포화합니다.

2D: B–H 히스테리시스 고리입니다. 위의 선형 모델은 μr을 상수로 쓰지만 μr은 원점에서의 기울기일 뿐이어서, 실제 곡선은 무릎에서 꺾이고 B는 선형 모델보다 작게 나옵니다. 고리의 면적이 1사이클마다 열로 바뀌는 에너지이고, 고리가 넓을수록 손실이 큰 철심입니다.`,
  },
  story: [
    {
      set: { N: 500, I: 1, gap: 0, mur: 1000 },
      text: {
        en: 'This glassy ring is an iron core: one closed path made of material, and nothing else yet.',
        ko: '유리처럼 비치는 이 고리가 철심입니다. 재료로 닫아 놓은 길 하나, 아직 그것뿐입니다.',
      },
    },
    {
      set: { N: 500, I: 1, gap: 0, mur: 1000 },
      text: {
        en: 'Wind 500 turns around one leg and run 1 A through them. Multiply the two and you get 500 ampere-turns, the force that drives this whole circuit. `F = N·I`',
        ko: '한쪽 기둥에 코일을 500회 감고 1 A를 흘립니다. 둘을 곱한 500 AT이 이 회로 전체를 미는 힘, 기자력입니다. `F = N·I`',
      },
    },
    {
      set: { N: 500, I: 1, gap: 0, mur: 1000 },
      text: {
        en: 'That mmf drives a flux all the way round inside the iron: 1.257 mWb, which over a 10 cm² cross-section reads 1.257 T. `Φ = F ÷ ΣR`',
        ko: '그 기자력이 밀어낸 자속이 철심 안을 한 바퀴 돕니다. 1.257 mWb이고, 10 cm² 단면으로 나누면 1.257 T입니다. `Φ = F ÷ ΣR`',
      },
    },
    {
      set: { N: 500, I: 1, gap: 0, mur: 1000 },
      text: {
        en: 'What opposes the flux is reluctance, and it adds up in series exactly the way resistance does. Right now the iron is the only piece of it: 397.9 kAT ÷ Wb. `R = l ÷ (μ0 × μr × A)`',
        ko: '자속을 막는 쪽이 자기저항이고, 저항과 똑같이 직렬로 더해집니다. 지금은 철심 하나뿐, 397.9 kAT ÷ Wb입니다. `R = l ÷ (μ0 × μr × A)`',
      },
    },
    {
      set: { N: 500, I: 1, gap: 1, mur: 1000 },
      text: {
        en: 'Now cut the ring open by 1 mm, one five-hundredth of the path. That sliver of air on its own is 795.8 kAT ÷ Wb, more than twice what half a metre of iron costs.',
        ko: '이제 고리를 1 mm만 자릅니다. 전체 경로의 500분의 1입니다. 그런데 그 얇은 공기 틈 하나가 795.8 kAT ÷ Wb, 0.5 m 철심 값의 두 배가 넘습니다.',
      },
    },
    {
      set: { N: 500, I: 1, gap: 1, mur: 1000 },
      text: {
        en: 'Nobody touched the current. The flux still collapsed to a third of what it was, from 1.257 mWb down to 0.419 mWb, and the air gap now owns two thirds of the bar.',
        ko: '전류는 아무도 건드리지 않았습니다. 그런데 자속은 1.257 mWb에서 0.419 mWb로, 3분의 1까지 주저앉았습니다. 막대에서도 이제 공극이 3분의 2를 차지합니다.',
      },
    },
    {
      set: { N: 1000, I: 1.5, gap: 1, mur: 1000 },
      text: {
        en: 'Getting it back costs you. Take the winding to 1000 turns and the current to 1.5 A, so 1500 ampere-turns, and the flux returns to 1.257 mWb. Three times the mmf for the same flux: that is the price of an air gap.',
        ko: '되돌리려면 값을 치러야 합니다. 권수를 1000회, 전류를 1.5 A로 올려 1500 AT을 만들면 자속이 1.257 mWb로 돌아옵니다. 같은 자속에 기자력 세 배. 공극의 값은 이렇습니다.',
      },
    },
    {
      set: { N: 500, I: 1, gap: 0, mur: 200 },
      text: {
        en: 'Close the gap and change the material instead. Drop μr from 1000 to 200 and the iron itself becomes 1989.4 kAT ÷ Wb, leaving only 0.251 mWb. In a magnetic circuit a material contributes exactly one number, and μr is it.',
        ko: '공극을 닫고 대신 재료를 바꿔 봅니다. μr을 1000에서 200으로 내리면 철심 자체가 1989.4 kAT ÷ Wb가 되고, 자속은 0.251 mWb밖에 남지 않습니다. 자기회로에서 재료가 기여하는 숫자는 μr 하나뿐입니다.',
      },
    },
    {
      set: { N: 1000, I: 1, gap: 0, mur: 1000 },
      text: {
        en: 'And you cannot just keep pushing. At 1000 ampere-turns the linear formula promises 2.513 T, but real iron bends over at the knee and saturates past 1.6 T. The extra ampere-turns buy almost nothing.',
        ko: '그렇다고 무한정 밀 수도 없습니다. 1000 AT에서 선형 식은 2.513 T를 약속하지만, 실제 철심은 무릎에서 꺾이며 1.6 T를 넘어서면 포화합니다. 더 밀어넣은 AT은 거의 아무것도 사 오지 못합니다.',
      },
    },
    {
      set: { N: 500, I: 1, gap: 0, mur: 1000 },
      text: {
        en: 'Run it on AC and the flux walks the whole B–H loop once every cycle, throwing the loop\'s area away as heat. Here that is 350 J ÷ m³ per lap, and it is why a core gets warm with nothing attached to it. `Ph = k × f × B^1.6`',
        ko: '교류로 돌리면 자속이 매 사이클 B–H 고리를 한 바퀴씩 걷고, 고리의 면적만큼을 열로 버립니다. 여기서는 한 바퀴에 350 J ÷ m³. 아무것도 달지 않은 철심이 더워지는 이유입니다. `Ph = k × f × B^1.6`',
      },
    },
  ],
  sliderParams: [
    { key: 'N',   label: { en: 'Turns (N)', ko: '권수 (N)' },                                                     min: 50,  max: N_MAX, step: 50,   defaultValue: 500 },
    { key: 'I',   label: { en: 'Coil Current (I)', ko: '코일 전류 (I)' },                           unit: 'A',     min: 0,   max: 3,     step: 0.05, defaultValue: 1 },
    { key: 'gap', label: { en: 'Air-Gap Length (lg)', ko: '공극 길이 (lg)' },                       unit: 'mm',    min: 0,   max: 3,     step: 0.05, defaultValue: 0 },
    { key: 'mur', label: { en: 'Relative Permeability (μr)', ko: '비투자율 (μr)' },                               min: 200, max: 5000,  step: 100,  defaultValue: 1000 },
  ],
  Scene: MagneticCircuitScene,
  Scene2D: HysteresisLoop2DScene,
}

export default concept
