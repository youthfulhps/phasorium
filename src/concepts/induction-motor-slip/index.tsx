import { useRef, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import type { Group } from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import { useLocaleStore } from '../../lib/i18n'
import type { ConceptMeta } from '../../types/concept'

const TWO_PI = Math.PI * 2
const THIRD = TWO_PI / 3 // 120° between the winding axes

/*
 * Pure physics of a polyphase induction machine.
 * Ns = 120·f / P [rpm] (P = poles), s = (Ns − N) / Ns, slip speed = Ns − N = s·Ns, f2 = s·f.
 * s = 1 at standstill, s = 0 at synchronous speed (no relative motion, no induced emf, no torque).
 * Verified with node -e:
 *   f = 60 Hz, P = 4  -> Ns = 120 × 60 / 4 = 1800 rpm
 *   N = 1710 rpm      -> s = (1800 − 1710) / 1800 = 0.05 (5 %), slip speed 90 rpm, f2 = 0.05 × 60 = 3 Hz
 *   same f, P = 2 / 6 / 8 -> Ns = 3600 / 1200 / 900 rpm;  f = 120 Hz with P = 2 -> 7200 rpm
 *   slider default s = 5 % with f = 60, P = 4 -> N = 1800 × 0.95 = 1710 rpm, f2 = 3 Hz
 * The result is written into one module-scope record instead of a fresh object, so a frame never allocates.
 */
const S = { f: 60, P: 4, Ns: 1800, N: 1710, s: 0.05, slipSpeed: 90, f2: 3 }

function solve({ freq = 60, poles = 4, slip_pct = 5 }: Record<string, number>) {
  const P = Math.max(2, Math.round(poles / 2) * 2) // poles come in pairs
  const Ns = (120 * freq) / P
  const s = slip_pct / 100
  S.f = freq
  S.P = P
  S.Ns = Ns
  S.s = s
  S.N = Ns * (1 - s)
  S.slipSpeed = Ns - S.N
  S.f2 = s * freq
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

// ── display clocks ──
// A real 60 Hz field turns 30 times a second, so the whole scene is slowed down hard.
const K_RPM = 1 / 1800 // display rad/s per rpm: 1800 rpm -> 1.0 rad/s, so Ns is honest in proportion
const OMEGA_CAP = 4.0 // only the fastest corner (f = 120 Hz, P = 2 -> 7200 rpm) is clipped
// At a few percent slip the true relative creep is ~0.05 rad/s, far too slow to watch, so the
// rotor-frame circle and the 2D slip angle run on their own compressed clock: still zero at s = 0
// and monotone in f2, but square-root flattened so one slip cycle is a few seconds at any slip.
const F2_REF = 3 // Hz, the worked example's rotor frequency
const REL_REF = 1.3 // display rad/s at F2_REF
const relOmega = (f2: number) => REL_REF * Math.sqrt(f2 / F2_REF)

const fmtRpm = (r: number) => `${r.toFixed(0)} rpm`
// Scene labels read in the foreground color; only current (cyan) and power (amber) tint their text
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const smallClass = 'pointer-events-none whitespace-nowrap text-[10px] font-medium tabular-nums text-foreground/70 leading-4'

const L = {
  en: {
    field: 'stator field B · Ns',
    rotor: 'rotor · N',
    slipAngle: 'slip angle',
    rotorFrame: 'seen from the rotor',
    lap: '360° · the field gains a lap',
    zero: '0°',
    speed: 'speed',
    slipGap: 'slip speed',
  },
  ko: {
    field: '고정자 자기장 B · Ns',
    rotor: '회전자 · N',
    slipAngle: '슬립각',
    rotorFrame: '회전자에서 본 모습',
    lap: '360° · 자기장이 한 바퀴 앞섬',
    zero: '0°',
    speed: '속도',
    slipGap: '슬립 속도',
  },
} satisfies Record<string, Record<string, string>>

// Shared scratch: filled and handed straight to a geometry, so no frame ever allocates
const ARC_N = 40
const SEG = new Float32Array(6)
const ARC = new Float32Array(ARC_N * 3)

function setSeg(line: any, x0: number, y0: number, x1: number, y1: number) {
  SEG[0] = x0; SEG[1] = y0; SEG[2] = 0
  SEG[3] = x1; SEG[4] = y1; SEG[5] = 0
  line?.geometry.setPositions(SEG)
  line?.computeLineDistances()
}

function setArc(line: any, cx: number, cy: number, r: number, a0: number, a1: number) {
  for (let i = 0; i < ARC_N; i++) {
    const a = a0 + ((a1 - a0) * i) / (ARC_N - 1)
    ARC[i * 3] = cx + r * Math.cos(a)
    ARC[i * 3 + 1] = cy + r * Math.sin(a)
    ARC[i * 3 + 2] = 0
  }
  line?.geometry.setPositions(ARC)
  line?.computeLineDistances()
}

const SEG_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]
const ARC_INIT: [number, number, number][] = Array.from(
  { length: ARC_N },
  (): [number, number, number] => [0, 0, 0]
)
const put = (g: Group | null | undefined, x: number, y: number) => { g?.position.set(x, y, 0) }

const circlePts = (cx: number, cy: number, r: number, n = 96): [number, number, number][] =>
  Array.from({ length: n + 1 }, (_, i): [number, number, number] => {
    const a = (i / n) * TWO_PI
    return [cx + r * Math.cos(a), cy + r * Math.sin(a), 0]
  })

/** The four live numbers, rebuilt only when the sliders move. */
function setReadouts(refs: (HTMLSpanElement | null)[]) {
  if (refs[0]) refs[0].textContent = `Ns = 120 × ${S.f.toFixed(0)} ÷ ${S.P.toFixed(0)} = ${fmtRpm(S.Ns)}`
  if (refs[1]) refs[1].textContent = `N = (1 − s)·Ns = ${fmtRpm(S.N)}`
  if (refs[2]) {
    refs[2].textContent = `s = (Ns − N) ÷ Ns = ${S.s.toFixed(3)} = ${(S.s * 100).toFixed(0)} %`
  }
  if (refs[3]) refs[3].textContent = `f2 = s·f = ${S.f2.toFixed(2)} Hz · Ns − N = ${fmtRpm(S.slipSpeed)}`
}

function Readouts({ position, refs }: { position: [number, number, number]; refs: (HTMLSpanElement | null)[] }) {
  return (
    <group position={position}>
      <Html center>
        <div className={smallClass}>
          {[0, 1, 2, 3].map((i) => (
            <span key={i} ref={(el) => { refs[i] = el }} className="block" />
          ))}
        </div>
      </Html>
    </group>
  )
}

// ── 3D: the machine face-on, with both histories winding back into the depth ──
// The field vector sweeps at Ns while the rotor turns at N; the two trails recede toward −Z, and the
// field's trail has wound more turns than the rotor's over the very same stretch of history.
const OX = -0.9 // machine centre, kept clear of the floating description card on the right
const OY = 0
const R_STATOR = 1.3
const R_FIELD = 1.1
const R_ROTOR = 0.68
const R_WEDGE = 0.9
const HELIX_N = 130
const TSPAN = 2.6 // s of history shown
const DEPTH = 4.0
const HELIX_INIT: [number, number, number][] = Array.from(
  { length: HELIX_N },
  (): [number, number, number] => [0, 0, 0]
)
const STATOR_CIRCLE = circlePts(OX, OY, R_STATOR)
const ROTOR_CIRCLE = circlePts(OX, OY, R_ROTOR)
// Winding axes, 120° apart: A up, then B and C in the direction the field turns (counter-clockwise)
const WIND_ANG = [Math.PI / 2, Math.PI / 2 + THIRD, Math.PI / 2 - THIRD]
const WIND_SEGS = WIND_ANG.map((a): [number, number, number][] => [
  [OX + R_STATOR * Math.cos(a), OY + R_STATOR * Math.sin(a), 0],
  [OX + (R_STATOR + 0.24) * Math.cos(a), OY + (R_STATOR + 0.24) * Math.sin(a), 0],
])
const WIND_TAGS = WIND_ANG.map((a): [number, number, number] => [
  OX + (R_STATOR + 0.46) * Math.cos(a),
  OY + (R_STATOR + 0.46) * Math.sin(a),
  0,
])
const WIND_NAMES = ['A', 'B', 'C']
// The rotor frame: the rotor held still on the right, the field sweeping past it for ever
const IX = 1.35
const IY = -1.15
const R_INSET = 0.38
const INSET_CIRCLE = circlePts(IX, IY, R_INSET, 64)
const INSET_TICK: [number, number, number][] = [[IX, IY, 0], [IX + R_INSET, IY, 0]]

function RotatingFieldScene() {
  const c = usePhysicsColors()
  const t = L[useLocaleStore((s) => s.locale)]
  const valuesRef = useSliderValues()
  const fieldBuf = useMemo(() => new Float32Array(HELIX_N * 3), [])
  const rotorBuf = useMemo(() => new Float32Array(HELIX_N * 3), [])
  const fieldTrailRef = useRef<any>(null)
  const rotorTrailRef = useRef<any>(null)
  // 0 field vector, 1 rotor bar, 2 slip wedge, 3 rotor-frame field vector
  const lineRefs = useRef<any[]>([])
  // 0 field tip, 1 rotor rim mark, 2 rotor-frame tip
  const dotRefs = useRef<(Group | null)[]>([])
  // 0 field, 1 rotor, 2 slip angle
  const tagRefs = useRef<(Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<Record<string, number> | null>(null)

  useFrame(({ clock }) => {
    const v = solve(valuesRef.current)
    const time = clock.getElapsedTime()
    const ws = Math.min(v.Ns * K_RPM, OMEGA_CAP) // field, proportional to Ns
    const wr = ws * (1 - v.s) // rotor, slower by exactly the slip
    const ths = ws * time
    const thr = wr * time

    // Both histories, sample by sample: same depth of time, different number of turns
    for (let i = 0; i < HELIX_N; i++) {
      const back = 1 - i / (HELIX_N - 1) // 0 at the head (now), 1 at the oldest sample
      const dt = back * TSPAN
      const z = -back * DEPTH
      const af = ths - ws * dt
      const ar = thr - wr * dt
      fieldBuf[i * 3] = OX + R_FIELD * Math.cos(af)
      fieldBuf[i * 3 + 1] = OY + R_FIELD * Math.sin(af)
      fieldBuf[i * 3 + 2] = z
      rotorBuf[i * 3] = OX + R_ROTOR * Math.cos(ar)
      rotorBuf[i * 3 + 1] = OY + R_ROTOR * Math.sin(ar)
      rotorBuf[i * 3 + 2] = z
    }
    fieldTrailRef.current?.geometry.setPositions(fieldBuf)
    rotorTrailRef.current?.geometry.setPositions(rotorBuf)

    // The field vector and the rotor bar in the z = 0 face
    const fx = OX + R_FIELD * Math.cos(ths)
    const fy = OY + R_FIELD * Math.sin(ths)
    setSeg(lineRefs.current[0], OX, OY, fx, fy)
    dotRefs.current[0]?.position.set(fx, fy, 0)

    const rx = OX + R_ROTOR * Math.cos(thr)
    const ry = OY + R_ROTOR * Math.sin(thr)
    setSeg(lineRefs.current[1], OX - (rx - OX), OY - (ry - OY), rx, ry)
    dotRefs.current[1]?.position.set(rx, ry, 0)

    // The slip angle: the lead the field has built up, wrapping once per slip cycle
    const lead = (((ths - thr) % TWO_PI) + TWO_PI) % TWO_PI
    setArc(lineRefs.current[2], OX, OY, R_WEDGE, thr, thr + lead)

    // The same field in the rotor's own frame, on the compressed clock
    const thi = relOmega(v.f2) * time
    const ix = IX + R_INSET * Math.cos(thi)
    const iy = IY + R_INSET * Math.sin(thi)
    setSeg(lineRefs.current[3], IX, IY, ix, iy)
    dotRefs.current[2]?.position.set(ix, iy, 0)

    put(tagRefs.current[0], OX + (R_FIELD + 0.34) * Math.cos(ths), OY + (R_FIELD + 0.34) * Math.sin(ths))
    put(tagRefs.current[1], OX - (R_ROTOR + 0.3) * Math.cos(thr), OY - (R_ROTOR + 0.3) * Math.sin(thr))
    put(
      tagRefs.current[2],
      OX + (R_WEDGE + 0.22) * Math.cos(thr + lead / 2),
      OY + (R_WEDGE + 0.22) * Math.sin(thr + lead / 2)
    )

    // The numbers only move with the sliders, so the text is rebuilt only then
    if (labelledFor.current !== valuesRef.current) {
      labelledFor.current = valuesRef.current
      setReadouts(textRefs.current)
    }
  })

  const segs: { color: string; width: number; dashed?: boolean; arc?: boolean }[] = [
    { color: c['b-field'], width: 3 },
    { color: c.resistance, width: 3 },
    { color: c.guide, width: 1.5, dashed: true, arc: true },
    { color: c['b-field'], width: 2 },
  ]
  const tags = [t.field, t.rotor, t.slipAngle]

  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />

      {/* Background, lights, grid, and bloom come from the shared SceneStage */}
      <Line points={STATOR_CIRCLE} lineWidth={1} color={c.guide} dashed dashSize={0.09} gapSize={0.07} />
      <Line points={ROTOR_CIRCLE} lineWidth={1} color={c.guide} dashed dashSize={0.07} gapSize={0.05} />
      {WIND_SEGS.map((pts, i) => (
        <Line key={i} points={pts} lineWidth={2.5} color={c.current} />
      ))}

      {/* Two histories at two radii: the field's winds tighter than the rotor's */}
      <TrailLine ref={fieldTrailRef} points={HELIX_INIT} lineWidth={2.5} color={c['b-field']} />
      <TrailLine ref={rotorTrailRef} points={HELIX_INIT} lineWidth={2} color={c.resistance} />

      {segs.map((sg, i) => (
        <Line
          key={i}
          ref={(l) => { lineRefs.current[i] = l }}
          points={sg.arc ? ARC_INIT : SEG_INIT}
          lineWidth={sg.width}
          color={sg.color}
          dashed={sg.dashed}
          dashSize={0.07}
          gapSize={0.05}
        />
      ))}

      <Marker ref={(g) => { dotRefs.current[0] = g }} color={c['b-field']} size={0.07} />
      <Marker ref={(g) => { dotRefs.current[1] = g }} color={c.marker} size={0.055} />
      <Marker ref={(g) => { dotRefs.current[2] = g }} color={c['b-field']} size={0.045} />

      {tags.map((text, i) => (
        <group key={i} ref={(g) => { tagRefs.current[i] = g }}>
          <Html center>
            <span className={labelClass}>{text}</span>
          </Html>
        </group>
      ))}

      {WIND_TAGS.map((pos, i) => (
        <group key={i} position={pos}>
          <Html center>
            <span className={labelClass} style={{ color: c.current }}>{WIND_NAMES[i]}</span>
          </Html>
        </group>
      ))}

      {/* The rotor frame: the rotor held still, the field sweeping past it lap after lap */}
      <Line points={INSET_CIRCLE} lineWidth={1} color={c.guide} dashed dashSize={0.06} gapSize={0.05} />
      <Line points={INSET_TICK} lineWidth={2.5} color={c.resistance} />
      <group position={[IX, IY + R_INSET + 0.3, 0]}>
        <Html center>
          <span className={smallClass}>{t.rotorFrame}</span>
        </Html>
      </group>

      <Readouts position={[OX, -1.66, 0]} refs={textRefs.current} />
    </>
  )
}

// ── 2D: the slip angle climbing and resetting, over a speed bar of N against Ns ──
const BX0 = -3.4
const BX_SPAN = 6.1
const RPM_FULL = 7200 // the largest Ns the sliders can reach (f = 120 Hz, P = 2)
const X_PER_RPM = BX_SPAN / RPM_FULL
const BAR_Y = -1.0
const SY0 = 0.12 // slip-angle plot: 0° on this line, 360° at SY0 + SH
const SH = 1.45
const W_N = 220
const WIN = 6 // s of slip-angle history
const SAW_INIT: [number, number, number][] = Array.from(
  { length: W_N },
  (): [number, number, number] => [0, 0, 0]
)
const BASE_AXIS: [number, number, number][] = [[BX0 - 0.12, SY0, 0], [BX0 + BX_SPAN + 0.12, SY0, 0]]
const LAP_AXIS: [number, number, number][] = [
  [BX0 - 0.12, SY0 + SH, 0],
  [BX0 + BX_SPAN + 0.12, SY0 + SH, 0],
]
const SPEED_AXIS: [number, number, number][] = [[BX0, BAR_Y, 0], [BX0 + BX_SPAN, BAR_Y, 0]]
// A faint rpm scale, one tick every 1200 rpm
const RPM_TICKS = Array.from({ length: 7 }, (_, i): [number, number, number][] => {
  const x = BX0 + i * 1200 * X_PER_RPM
  return [[x, BAR_Y - 0.1, 0], [x, BAR_Y + 0.1, 0]]
})

function SlipBar2DScene() {
  const c = usePhysicsColors()
  const t = L[useLocaleStore((s) => s.locale)]
  const valuesRef = useSliderValues()
  const sawBuf = useMemo(() => new Float32Array(W_N * 3), [])
  const sawRef = useRef<any>(null)
  // 0 N bar, 1 Ns marker, 2 slip gap bracket
  const lineRefs = useRef<any[]>([])
  // 0 saw head, 1 N end, 2 Ns tip
  const dotRefs = useRef<(Group | null)[]>([])
  // 0 N, 1 Ns, 2 slip speed
  const tagRefs = useRef<(Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<Record<string, number> | null>(null)

  useFrame(({ clock }) => {
    const v = solve(valuesRef.current)
    const time = clock.getElapsedTime()
    const wrel = relOmega(v.f2)

    // The slip angle over the last WIN seconds, wrapped: one ramp per lap the field gains
    for (let k = 0; k < W_N; k++) {
      const frac = k / (W_N - 1)
      const tk = Math.max(time - (1 - frac) * WIN, 0)
      const ang = ((wrel * tk) % TWO_PI + TWO_PI) % TWO_PI
      sawBuf[k * 3] = BX0 + frac * BX_SPAN
      sawBuf[k * 3 + 1] = SY0 + (ang / TWO_PI) * SH
      sawBuf[k * 3 + 2] = 0
    }
    sawRef.current?.geometry.setPositions(sawBuf)
    const headAng = ((wrel * time) % TWO_PI + TWO_PI) % TWO_PI
    dotRefs.current[0]?.position.set(BX0 + BX_SPAN, SY0 + (headAng / TWO_PI) * SH, 0)

    // The speed bar: 0 rpm at the left, the rotor's bar ending short of the Ns mark for ever
    const xN = BX0 + v.N * X_PER_RPM
    const xNs = BX0 + v.Ns * X_PER_RPM
    setSeg(lineRefs.current[0], BX0, BAR_Y, xN, BAR_Y)
    setSeg(lineRefs.current[1], xNs, BAR_Y - 0.26, xNs, BAR_Y + 0.44)
    setSeg(lineRefs.current[2], xN, BAR_Y + 0.22, xNs, BAR_Y + 0.22)
    dotRefs.current[1]?.position.set(xN, BAR_Y, 0)
    dotRefs.current[2]?.position.set(xNs, BAR_Y, 0)

    put(tagRefs.current[0], xN * 0.5 + BX0 * 0.5, BAR_Y - 0.26)
    put(tagRefs.current[1], xNs + 0.3, BAR_Y + 0.5)
    put(tagRefs.current[2], (xN + xNs) / 2, BAR_Y + 0.46)

    if (labelledFor.current !== valuesRef.current) {
      labelledFor.current = valuesRef.current
      setReadouts(textRefs.current)
    }
  })

  const segs: { color: string; width: number; dashed?: boolean }[] = [
    { color: c.resistance, width: 3.5 },
    { color: c['b-field'], width: 2.5, dashed: true },
    { color: c.marker, width: 2 },
  ]
  const tags = ['N', 'Ns', t.slipGap]

  return (
    <>
      <Line points={BASE_AXIS} lineWidth={1} color={c.guide} />
      <Line points={LAP_AXIS} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      <Line points={SPEED_AXIS} lineWidth={1} color={c.guide} />
      {RPM_TICKS.map((pts, i) => (
        <Line key={i} points={pts} lineWidth={1} color={c.guide} />
      ))}

      {/* The slip angle: it climbs to 360° and starts again, once per slip cycle */}
      <TrailLine ref={sawRef} points={SAW_INIT} lineWidth={2.5} color={c['b-field']} fade={false} />

      {segs.map((sg, i) => (
        <Line
          key={i}
          ref={(l) => { lineRefs.current[i] = l }}
          points={SEG_INIT}
          lineWidth={sg.width}
          color={sg.color}
          dashed={sg.dashed}
          dashSize={0.07}
          gapSize={0.05}
        />
      ))}

      <Marker ref={(g) => { dotRefs.current[0] = g }} color={c['b-field']} size={0.055} />
      <Marker ref={(g) => { dotRefs.current[1] = g }} color={c.marker} size={0.06} />
      <Marker ref={(g) => { dotRefs.current[2] = g }} color={c['b-field']} size={0.05} />

      {tags.map((text, i) => (
        <group key={i} ref={(g) => { tagRefs.current[i] = g }}>
          <Html center>
            <span className={labelClass}>{text}</span>
          </Html>
        </group>
      ))}

      <group position={[BX0 + 1.1, SY0 + SH + 0.2, 0]}>
        <Html center>
          <span className={smallClass}>{t.lap}</span>
        </Html>
      </group>
      <group position={[BX0 - 0.3, SY0, 0]}>
        <Html center>
          <span className={smallClass}>{t.zero}</span>
        </Html>
      </group>
      <group position={[BX0 + 0.4, SY0 + SH * 0.5, 0]}>
        <Html center>
          <span className={smallClass}>{t.slipAngle}</span>
        </Html>
      </group>
      <group position={[BX0 - 0.26, BAR_Y - 0.34, 0]}>
        <Html center>
          <span className={smallClass}>{t.speed}</span>
        </Html>
      </group>

      <Readouts position={[BX0 + 1.9, -1.78, 0]} refs={textRefs.current} />
    </>
  )
}

const concept: ConceptMeta = {
  id: 'induction-motor-slip',
  title: { en: 'Rotating Field & Slip', ko: '회전자계와 슬립' },
  subject: 'machines',
  order: 2,
  description: {
    en: `## Rotating Field and Slip

**Three windings 120° apart make a magnetic field that rotates all by itself, and the rotor has to stay behind it: if it ever caught up, the flux through the rotor would stop changing and no torque could be made.**

- Deeper cyan vector and its receding trail: the stator field B, sweeping round at the synchronous speed Ns
- Mid neutral bar with the dot on its rim, and its own receding trail: the rotor, turning at N
- Faint dashed wedge between them: the slip angle, the lead the field has built up over the rotor
- Cyan ticks A, B, C: the three winding axes, 120° apart, carrying the three phase currents
- Small circle to the lower right: the same field seen from the rotor, sweeping past it once per slip cycle and standing still only when s = 0

Compare the two trails as they wind back into the depth: over the very same stretch of history the field has turned through more revolutions than the rotor. That difference is the slip, and it never stops growing.

**Synchronous speed**: Ns = 120·f ÷ P [rpm], with P the number of poles
**Slip**: s = (Ns − N) ÷ Ns
**Slip speed**: Ns − N = s·Ns [rpm]
**Rotor frequency**: f2 = s·f

At standstill N = 0, so s = 1 and the rotor bars see the full supply frequency. At synchronous speed s = 0: nothing moves relative to the field, the flux through the rotor is frozen, no emf is induced and the torque collapses to zero. That is why an induction motor can approach Ns but never reach it. A real machine at full load sits at only a few percent slip, close behind the field and never level with it.

Worked example: f = 60 Hz with P = 4 gives Ns = 120 × 60 ÷ 4 = 1800 rpm. A rotor at N = 1710 rpm is 90 rpm behind, so s = 0.05 (5 %) and f2 = 0.05 × 60 = 3 Hz.

Raise f and both speeds grow together while s stays put. Step P up through 4, 6, 8 and Ns is halved, then thirded, then quartered, the field slowing visibly for the same supply. Open the slip and the rotor falls back: at 0 % the rotor rides with the field and the rotor frame freezes, at 100 % the rotor is standing still and the field rushes past it at the full supply frequency.

In 2D: the slip angle climbing to 360° and starting over, once for every lap the field gains, above a speed bar that puts N against Ns with the slip gap bracketed. The rotation is slowed far below a real 60 Hz so it can be followed, and the rotor-frame circle and the slip angle run on their own faster, compressed clock so a slip cycle lasts a few seconds at any slip setting.`,
    ko: `## 회전자계와 슬립

**120°씩 벌어진 세 권선은 저절로 회전하는 자기장을 만듭니다. 회전자는 그 자기장보다 반드시 뒤처져 있어야 합니다. 만약 따라잡는다면 회전자를 지나는 자속이 변하지 않게 되고, 토크를 만들 수 없습니다.**

- 진한 시안 벡터와 뒤로 물러나는 자취: 고정자 자기장 B. 동기속도 Ns로 원을 쓸고 돕니다
- 중간 밝기 중성색 막대와 테두리의 점, 그리고 그 자취: 회전자. 속도는 N입니다
- 둘 사이의 흐린 점선 부채꼴: 슬립각. 자기장이 회전자보다 벌어 놓은 앞선 각입니다
- 시안 눈금 A, B, C: 120°씩 벌어진 세 권선 축. 세 상전류가 흐릅니다
- 오른쪽 아래 작은 원: 같은 자기장을 회전자에서 본 모습. 슬립 1주기마다 한 바퀴 지나가며, s = 0일 때만 멈춰 섭니다

깊이 쪽으로 감겨 들어가는 두 자취를 비교해 보세요. 똑같은 길이의 과거 동안 자기장이 회전자보다 더 많이 돌았습니다. 그 차이가 슬립이며, 차이는 결코 멈추지 않고 계속 쌓입니다.

**동기속도**: Ns = 120·f ÷ P [rpm], P는 극수
**슬립**: s = (Ns − N) ÷ Ns
**슬립 속도**: Ns − N = s·Ns [rpm]
**회전자 주파수**: f2 = s·f

정지 상태에서는 N = 0이므로 s = 1이고, 회전자 도체는 전원 주파수를 그대로 느낍니다. 동기속도에서는 s = 0입니다. 자기장에 대해 아무것도 움직이지 않으므로 회전자를 지나는 자속이 멈추고, 기전력이 유도되지 않아 토크가 0으로 사라집니다. 유도전동기가 Ns에 가까워질 수는 있어도 결코 도달할 수 없는 이유가 바로 이것입니다. 실제 전동기는 전부하에서 몇 퍼센트의 슬립으로 운전하며, 자기장 바로 뒤를 따라가되 결코 나란히 서지는 않습니다.

계산 예: f = 60 Hz, P = 4이면 Ns = 120 × 60 ÷ 4 = 1800 rpm입니다. 회전자가 N = 1710 rpm이면 90 rpm 뒤처진 것이므로 s = 0.05 (5 %)이고 f2 = 0.05 × 60 = 3 Hz입니다.

f를 올리면 두 속도가 함께 커지고 s는 그대로입니다. P를 4, 6, 8로 올리면 Ns가 1/2, 1/3, 1/4로 줄어들어 같은 전원에서도 자기장이 눈에 띄게 느려집니다. 슬립을 키우면 회전자가 뒤로 처집니다. 0 %에서는 회전자가 자기장과 함께 달려 회전자 기준 화면이 멈추고, 100 %에서는 회전자가 멈춰 선 채 자기장이 전원 주파수 그대로 스쳐 지나갑니다.

2D: 슬립각이 360°까지 올라가고 다시 시작하는 모습입니다. 자기장이 한 바퀴 앞설 때마다 한 번씩 반복됩니다. 그 아래 속도 막대는 N을 Ns와 나란히 놓고 그 사이 간격을 표시합니다. 회전은 실제 60 Hz보다 훨씬 느리게 재생하며, 회전자 기준 원과 슬립각은 어떤 슬립에서도 1주기가 몇 초가 되도록 자체의 더 빠른 압축 시계로 움직입니다.`,
  },
  sliderParams: [
    { key: 'freq',     label: { en: 'Supply Frequency (f)', ko: '전원 주파수 (f)' }, unit: 'Hz', min: 30, max: 120, step: 5, defaultValue: 60 },
    { key: 'poles',    label: { en: 'Poles (P)',            ko: '극수 (P)' },        unit: '',   min: 2,  max: 8,   step: 2, defaultValue: 4  },
    { key: 'slip_pct', label: { en: 'Slip (s)',             ko: '슬립 (s)' },        unit: '%',  min: 0,  max: 100, step: 1, defaultValue: 5  },
  ],
  Scene: RotatingFieldScene,
  Scene2D: SlipBar2DScene,
}

export default concept
