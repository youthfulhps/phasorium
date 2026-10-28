import { useRef, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import type { Group } from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { Marker } from '../../components/scene/Marker'
import { useLocaleStore } from '../../lib/i18n'
import type { ConceptMeta } from '../../types/concept'

const TWO_PI = Math.PI * 2

// ── machine constants (a 4-pole, 60 Hz machine, one phase of the approximate equivalent circuit) ──
const PHASES = 3
const POLES = 4
const FREQ = 60
const NS_RPM = (120 * FREQ) / POLES // 1800 rpm
const WS = (NS_RPM * TWO_PI) / 60 // 188.4956 rad/s, synchronous speed in mechanical rad/s
const R1 = 0.3 // stator resistance [Ω], held fixed so the torque axis can keep an honest scale
const S_RATED = 0.03 // the rated-ish slip the worked example is taken at

/*
 * Pure physics of the torque-slip characteristic, approximate equivalent circuit, per phase.
 *   T       = (3 ÷ ωs) · (V²·R2 ÷ s) ÷ ((R1 + R2 ÷ s)² + X²)
 *   s_maxT  = R2 ÷ sqrt(R1² + X²)          -- set by R2 alone, not by V
 *   Tmax    = 3V² ÷ (2·ωs·(R1 + sqrt(R1² + X²)))   -- no R2 in it at all
 *   Pag     = T·ωs,  P2cu = s·Pag,  Pmech = (1 − s)·Pag
 * The operating slip is where T(s) meets the constant shaft load TL, which rearranges to a
 * quadratic in s:  TL·ωs·(R1² + X²)·s² + (2·R1·R2·TL·ωs − 3V²·R2)·s + TL·ωs·R2² = 0.
 * The smaller root is the stable one (right of the peak); no real root means the load exceeds
 * Tmax and the machine cannot start.
 *
 * Verified with npx tsx against this file's own solve():
 *   V = 220 V per phase, f = 60 Hz, 4 poles -> ns = 1800 rpm, ωs = 188.496 rad/s, R1 = 0.3 Ω,
 *   R2 = 0.2 Ω, X = 1.0 Ω
 *     s_maxT = 0.1916,  Tmax = 286.57 N·m,  and T(s_maxT) = 286.57 N·m by the torque formula too
 *     s = 0.03  -> T = 103.67 N·m, N = 1746.0 rpm
 *     s = 1     -> T = 123.25 N·m (starting torque)
 *     at s = 0.03: Pag = T·ωs = 19.542 kW, Pmech = 18.956 kW, P2cu ÷ Pag = s = 0.0300
 * Results go into one module-scope record, so a frame never allocates.
 */
function torque(s: number, V: number, R2: number, X: number) {
  if (s <= 0) return 0
  const rs = R2 / s
  return ((PHASES / WS) * (V * V * rs)) / ((R1 + rs) * (R1 + rs) + X * X)
}

// The constant shaft load: exactly the torque the default machine makes at the rated slip,
// so the default operating point lands on s = 0.03 to machine precision.
const T_LOAD = torque(S_RATED, 220, 0.2, 1.0) // 103.67 N·m
const P_REF = T_LOAD * WS // 19.542 kW, the air-gap power the load draws when the motor runs

const OUT = {
  R2: 0.2,
  V: 220,
  X: 1,
  sMaxT: 0,
  Tmax: 0,
  Tstart: 0,
  sOp: 0,
  nOp: 0,
  Top: 0,
  stalled: false,
  Pag: 0,
  P2cu: 0,
  Pmech: 0,
}

function solve({ R2 = 0.2, V = 220, X = 1.0 }: Record<string, number>) {
  const root = Math.sqrt(R1 * R1 + X * X)
  const K = T_LOAD * WS
  const a = K * (R1 * R1 + X * X)
  const b = 2 * K * R1 * R2 - PHASES * V * V * R2
  const c = K * R2 * R2
  const disc = b * b - 4 * a * c
  let sOp = disc < 0 ? 1 : (-b - Math.sqrt(disc)) / (2 * a)
  const stalled = disc < 0 || !(sOp > 0) || sOp > 1
  if (stalled) sOp = 1
  const Top = torque(sOp, V, R2, X)
  const Pag = Top * WS
  OUT.R2 = R2
  OUT.V = V
  OUT.X = X
  OUT.sMaxT = R2 / root
  OUT.Tmax = (PHASES * V * V) / (2 * WS * (R1 + root))
  OUT.Tstart = torque(1, V, R2, X)
  OUT.sOp = sOp
  OUT.nOp = NS_RPM * (1 - sOp)
  OUT.Top = Top
  OUT.stalled = stalled
  OUT.Pag = Pag
  OUT.P2cu = sOp * Pag
  OUT.Pmech = (1 - sOp) * Pag
  return OUT
}

// ── plot geometry, pure so the layout can be swept over the slider ranges ──
const R2_GHOST = 0.2 // the ghosted curve is always the default rotor resistance
const T_FULL = 600 // N·m at the top of the torque axis; the sliders cannot exceed it
const S_VIEW = 1 // the plot spans s = 0 (synchronous, right) to s = 1 (standstill, left)

// 3D: speed grows to the right, so slip grows to the left
const PX1 = 1.15
const PX0 = -1.75
const PW = PX1 - PX0
const PY0 = -0.55
const PH = 2.05
const x3 = (s: number) => PX1 - (s / S_VIEW) * PW
const y3 = (T: number) => PY0 + (Math.min(T, T_FULL) / T_FULL) * PH

// 2D: the same curve, same orientation, with room to annotate
const QX1 = 3.25
const QX0 = -3.25
const QW = QX1 - QX0
const QY0 = -1.5
const QH = 2.95
const x2 = (s: number) => QX1 - (s / S_VIEW) * QW
const y2 = (T: number) => QY0 + (Math.min(T, T_FULL) / T_FULL) * QH

/* physics-end */

function useSliderValues() {
  const valuesRef = useRef<Record<string, number>>(useSliderStore.getState().values)
  useEffect(
    () => useSliderStore.subscribe((s) => s.values, (v) => { valuesRef.current = v }),
    []
  )
  return valuesRef
}

const fmtT = (v: number) => `${v.toFixed(1)} N·m`
const fmtKW = (w: number) => `${(w / 1000).toFixed(2)} kW`
const fmtRpm = (r: number) => `${r.toFixed(0)} rpm`

// Scene labels read in the foreground color; only current (cyan) and power (amber) tint their text
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const smallClass = 'pointer-events-none whitespace-nowrap text-[10px] font-medium tabular-nums text-foreground/70 leading-4'

const L = {
  en: {
    stable: 'stable',
    unstable: 'unstable',
    rotor: 'rotor',
    standstill: 'standstill · s = 1',
    sync: 's = 0 · 1800 rpm',
    torqueAxis: 'T [N·m]',
    stalled: 'load above Tmax · cannot start',
    ghost: 'R2 = 0.2 Ω',
  },
  ko: {
    stable: '안정 영역',
    unstable: '불안정 영역',
    rotor: '회전자',
    standstill: '정지 · s = 1',
    sync: 's = 0 · 1800 rpm',
    torqueAxis: 'T [N·m]',
    stalled: '부하가 Tmax보다 큼 · 기동 불가',
    ghost: 'R2 = 0.2 Ω',
  },
} satisfies Record<string, Record<string, string>>

type Labels = (typeof L)['en']

// Shared scratch: filled and handed straight to a geometry, so no frame ever allocates
const SEG = new Float32Array(6)
const N_CURVE = 180
const SEG_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]
const CURVE_INIT: [number, number, number][] = Array.from(
  { length: N_CURVE },
  (): [number, number, number] => [0, 0, 0]
)

function setSeg(line: any, x0: number, y0: number, x1: number, y1: number) {
  SEG[0] = x0; SEG[1] = y0; SEG[2] = 0
  SEG[3] = x1; SEG[4] = y1; SEG[5] = 0
  line?.geometry.setPositions(SEG)
  line?.computeLineDistances()
}

/** Samples T(s) over the whole window into a pre-allocated buffer. */
function fillCurve(
  buf: Float32Array,
  V: number,
  R2: number,
  X: number,
  mapX: (s: number) => number,
  mapY: (T: number) => number
) {
  for (let i = 0; i < N_CURVE; i++) {
    const s = (i / (N_CURVE - 1)) * S_VIEW
    buf[i * 3] = mapX(s)
    buf[i * 3 + 1] = mapY(torque(s, V, R2, X))
    buf[i * 3 + 2] = 0
  }
}

const put = (g: Group | null | undefined, x: number, y: number) => { g?.position.set(x, y, 0) }

const circlePts = (cx: number, cy: number, r: number, n = 72): [number, number, number][] =>
  Array.from({ length: n + 1 }, (_, i): [number, number, number] => {
    const a = (i / n) * TWO_PI
    return [cx + r * Math.cos(a), cy + r * Math.sin(a), 0]
  })

/** The five live numbers, rebuilt only when the sliders or the locale change. */
function setReadouts(refs: (HTMLSpanElement | null)[], t: Labels) {
  if (refs[0]) refs[0].textContent = `s_maxT = R2 ÷ √(R1² + X²) = ${OUT.sMaxT.toFixed(4)}`
  if (refs[1]) {
    refs[1].textContent = `Tmax = 3V² ÷ (2·ωs·(R1 + √(R1² + X²))) = ${fmtT(OUT.Tmax)}`
  }
  if (refs[2]) refs[2].textContent = `s = 1 → T = ${fmtT(OUT.Tstart)} · TL = ${fmtT(T_LOAD)}`
  if (refs[3]) {
    refs[3].textContent = OUT.stalled
      ? `${t.stalled} · s = 1 · T = ${fmtT(OUT.Tstart)}`
      : `s = ${OUT.sOp.toFixed(4)} · N = ${fmtRpm(OUT.nOp)} · T = ${fmtT(OUT.Top)}`
  }
  if (refs[4]) {
    refs[4].textContent =
      `Pag = ${fmtKW(OUT.Pag)} = P2cu ${fmtKW(OUT.P2cu)} + Pmech ${fmtKW(OUT.Pmech)} · P2cu ÷ Pag = s`
  }
}

function Readouts({ position, refs }: { position: [number, number, number]; refs: (HTMLSpanElement | null)[] }) {
  return (
    <group position={position}>
      <Html center>
        <div className={smallClass}>
          {[0, 1, 2, 3, 4].map((i) => (
            <span key={i} ref={(el) => { refs[i] = el }} className="block" />
          ))}
        </div>
      </Html>
    </group>
  )
}

// ── 3D: the torque-speed curve with the rotor turning at the operating point ──
const SPEED_AXIS: [number, number, number][] = [[PX0 - 0.12, PY0, 0], [PX1 + 0.14, PY0, 0]]
const TORQUE_AXIS: [number, number, number][] = [[PX0, PY0 - 0.1, 0], [PX0, PY0 + PH + 0.1, 0]]
const LOAD_LINE: [number, number, number][] = [
  [PX0, y3(T_LOAD), 0],
  [PX1 + 0.1, y3(T_LOAD), 0],
]
// A faint speed scale, one tick every 450 rpm (every 0.25 of slip)
const SPEED_TICKS = Array.from({ length: 5 }, (_, i): [number, number, number][] => {
  const x = x3(i / 4)
  return [[x, PY0 - 0.08, 0], [x, PY0 + 0.08, 0]]
})

const BRACKET_Y = PY0 - 0.31
const BRACKET_LABEL_Y = PY0 - 0.47
const BAR_Y = -1.53
const BX0 = -0.25
const BAR_W = 1.5
const ROT_X = -1.3
const ROT_Y = -1.53
const ROT_R = 0.36
const ROTOR_RIM = circlePts(ROT_X, ROT_Y, ROT_R)
const SPOKE: [number, number, number][] = [[-ROT_R, 0, 0], [ROT_R, 0, 0]]
const OMEGA_AT_NS = 2.2 // display rad/s at 1800 rpm: a real 1746 rpm shaft is far too fast to watch

function TorqueCurveScene() {
  const c = usePhysicsColors()
  const t = L[useLocaleStore((s) => s.locale)]
  const valuesRef = useSliderValues()
  const liveBuf = useMemo(() => new Float32Array(N_CURVE * 3), [])
  const ghostBuf = useMemo(() => new Float32Array(N_CURVE * 3), [])
  const liveRef = useRef<any>(null)
  const ghostRef = useRef<any>(null)
  // 0 Tmax level, 1 peak drop, 2 operating drop, 3 stable bracket, 4 unstable bracket,
  // 5 copper-loss bar, 6 mechanical bar, 7 split tick
  const lineRefs = useRef<any[]>([])
  const dotRefs = useRef<(Group | null)[]>([]) // 0 peak, 1 operating point
  // 0 Tmax tag, 1 operating tag, 2 stable, 3 unstable, 4 P2cu tag, 5 Pmech tag
  const tagRefs = useRef<(Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const rotorRef = useRef<Group>(null)
  const angle = useRef(0)
  const labelledFor = useRef<Record<string, number> | null>(null)
  const labelledIn = useRef<Labels | null>(null)

  useFrame((_, delta) => {
    const v = solve(valuesRef.current)

    fillCurve(liveBuf, v.V, v.R2, v.X, x3, y3)
    liveRef.current?.geometry.setPositions(liveBuf)
    fillCurve(ghostBuf, v.V, R2_GHOST, v.X, x3, y3)
    ghostRef.current?.geometry.setPositions(ghostBuf)
    ghostRef.current?.computeLineDistances()

    // The level of the maximum: it answers to V and X, never to R2
    const yMax = y3(v.Tmax)
    setSeg(lineRefs.current[0], PX0, yMax, PX1 + 0.1, yMax)
    put(tagRefs.current[0], PX1 + 0.32, yMax)

    // The peak itself, clamped to the window: once s_maxT passes 1 the best torque is at standstill
    const sPeak = Math.min(v.sMaxT, S_VIEW)
    const xPeak = x3(sPeak)
    const yPeak = y3(torque(sPeak, v.V, v.R2, v.X))
    dotRefs.current[0]?.position.set(xPeak, yPeak, 0)
    setSeg(lineRefs.current[1], xPeak, yPeak, xPeak, PY0)

    // The operating point: where the curve meets the constant load TL
    const xOp = x3(v.sOp)
    const yOp = y3(v.Top)
    dotRefs.current[1]?.position.set(xOp, yOp, 0)
    setSeg(lineRefs.current[2], xOp, yOp, xOp, PY0)
    put(tagRefs.current[1], xOp + 0.34, yOp - 0.19)

    // Stable to the right of the peak, unstable to its left
    setSeg(lineRefs.current[3], xPeak, BRACKET_Y, PX1, BRACKET_Y)
    setSeg(lineRefs.current[4], PX0, BRACKET_Y, xPeak, BRACKET_Y)
    put(tagRefs.current[2], (xPeak + PX1) / 2, BRACKET_LABEL_Y)
    put(tagRefs.current[3], (PX0 + xPeak) / 2, BRACKET_LABEL_Y)
    tagRefs.current[3]?.scale.setScalar(sPeak > 0.995 * S_VIEW ? 0 : 1)

    // The air-gap power divided s to (1 − s): copper loss in amber, mechanical output neutral
    const barLen = BAR_W * Math.min(v.Pag / P_REF, 1)
    const xSplit = BX0 + barLen * v.sOp
    setSeg(lineRefs.current[5], BX0, BAR_Y, xSplit, BAR_Y)
    setSeg(lineRefs.current[6], xSplit, BAR_Y, BX0 + barLen, BAR_Y)
    setSeg(lineRefs.current[7], xSplit, BAR_Y - 0.11, xSplit, BAR_Y + 0.11)
    put(tagRefs.current[4], (BX0 + xSplit) / 2, BAR_Y - 0.19)
    put(tagRefs.current[5], (xSplit + BX0 + barLen) / 2, BAR_Y - 0.19)
    tagRefs.current[5]?.scale.setScalar(v.sOp > 0.9 ? 0 : 1)

    // The shaft, turning at the operating speed and nowhere near synchronous when the slip is big
    angle.current += OMEGA_AT_NS * (1 - v.sOp) * delta
    if (rotorRef.current) rotorRef.current.rotation.z = angle.current

    // The numbers only move with the sliders or the locale, so the text is rebuilt only then
    if (labelledFor.current !== valuesRef.current || labelledIn.current !== t) {
      labelledFor.current = valuesRef.current
      labelledIn.current = t
      setReadouts(textRefs.current, t)
      if (textRefs.current[5]) {
        textRefs.current[5].textContent = `N = ${fmtRpm(OUT.nOp)}`
      }
    }
  })

  const segs: { color: string; width: number; dashed?: boolean }[] = [
    { color: c.marker, width: 1.5, dashed: true },
    { color: c.guide, width: 1, dashed: true },
    { color: c.guide, width: 1, dashed: true },
    { color: c.marker, width: 2.5 },
    { color: c.guide, width: 1.5, dashed: true },
    { color: c.power, width: 6 },
    { color: c.resistance, width: 6 },
    { color: c.marker, width: 1.5 },
  ]
  const tags = [
    { text: 'Tmax', small: false, accent: false },
    { text: '', small: false, accent: false },
    { text: t.stable, small: true, accent: false },
    { text: t.unstable, small: true, accent: false },
    { text: 'P2cu', small: true, accent: true },
    { text: 'Pmech', small: true, accent: false },
  ]

  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />

      {/* Background, lights, grid, and bloom come from the shared SceneStage */}
      <Line points={SPEED_AXIS} lineWidth={1} color={c.guide} />
      <Line points={TORQUE_AXIS} lineWidth={1} color={c.guide} />
      {SPEED_TICKS.map((pts, i) => (
        <Line key={i} points={pts} lineWidth={1} color={c.guide} />
      ))}
      <Line points={LOAD_LINE} lineWidth={2} color={c.resistance} />

      {/* The ghost sits at the default R2, so the crest is seen sliding along a level that holds */}
      <Line
        ref={ghostRef}
        points={CURVE_INIT}
        lineWidth={1.5}
        color={c.guide}
        dashed
        dashSize={0.08}
        gapSize={0.06}
      />
      <Line ref={liveRef} points={CURVE_INIT} lineWidth={3} color={c.voltage} />

      {segs.map((sg, i) => (
        <Line
          key={i}
          ref={(l) => { lineRefs.current[i] = l }}
          points={SEG_INIT}
          lineWidth={sg.width}
          color={sg.color}
          dashed={sg.dashed}
          dashSize={0.08}
          gapSize={0.06}
        />
      ))}

      <Marker ref={(g) => { dotRefs.current[0] = g }} color={c.marker} size={0.075} />
      <Marker ref={(g) => { dotRefs.current[1] = g }} color={c.marker} size={0.055} />

      {tags.map((tag, i) => (
        <group key={i} ref={(g) => { tagRefs.current[i] = g }}>
          <Html center>
            {i === 1 ? (
              <span ref={(el) => { textRefs.current[5] = el }} className={labelClass} />
            ) : (
              <span
                className={tag.small ? smallClass : labelClass}
                style={tag.accent ? { color: c.power } : undefined}
              >
                {tag.text}
              </span>
            )}
          </Html>
        </group>
      ))}

      <group position={[PX1 + 0.32, y3(T_LOAD) - 0.19, 0]}>
        <Html center>
          <span className={labelClass}>TL</span>
        </Html>
      </group>
      <group position={[PX0 + 0.26, PY0 - 0.17, 0]}>
        <Html center>
          <span className={smallClass}>{t.standstill}</span>
        </Html>
      </group>
      <group position={[PX1 - 0.22, PY0 - 0.17, 0]}>
        <Html center>
          <span className={smallClass}>{t.sync}</span>
        </Html>
      </group>
      <group position={[PX0 + 0.2, PY0 + PH + 0.15, 0]}>
        <Html center>
          <span className={smallClass}>{t.torqueAxis}</span>
        </Html>
      </group>

      {/* The divided bar: the air-gap power split s to (1 − s) */}
      <group position={[BX0 + BAR_W * 0.5, BAR_Y + 0.23, 0]}>
        <Html center>
          <span className={smallClass}>Pag</span>
        </Html>
      </group>

      {/* The shaft, turning at the speed the operating point picks out */}
      <Line points={ROTOR_RIM} lineWidth={1} color={c.guide} dashed dashSize={0.07} gapSize={0.05} />
      <group ref={rotorRef} position={[ROT_X, ROT_Y, 0]}>
        <Line points={SPOKE} lineWidth={3} color={c.resistance} />
        <group position={[ROT_R, 0, 0]}>
          <Marker color={c.marker} size={0.05} />
        </group>
      </group>
      <group position={[ROT_X, ROT_Y - ROT_R - 0.14, 0]}>
        <Html center>
          <span className={smallClass}>{t.rotor}</span>
        </Html>
      </group>

      <Readouts position={[-0.3, 1.78, 0]} refs={textRefs.current} />
    </>
  )
}

// ── 2D: the whole characteristic, annotated at the three points worth knowing ──
const Q_SPEED_AXIS: [number, number, number][] = [[QX0 - 0.15, QY0, 0], [QX1 + 0.18, QY0, 0]]
const Q_TORQUE_AXIS: [number, number, number][] = [[QX0, QY0 - 0.12, 0], [QX0, QY0 + QH + 0.12, 0]]
const Q_LOAD_LINE: [number, number, number][] = [[QX0, y2(T_LOAD), 0], [QX1 + 0.12, y2(T_LOAD), 0]]
const Q_TICKS = Array.from({ length: 5 }, (_, i): [number, number, number][] => {
  const x = x2(i / 4)
  return [[x, QY0 - 0.1, 0], [x, QY0 + 0.1, 0]]
})

function TorqueCurve2DScene() {
  const c = usePhysicsColors()
  const t = L[useLocaleStore((s) => s.locale)]
  const valuesRef = useSliderValues()
  const liveBuf = useMemo(() => new Float32Array(N_CURVE * 3), [])
  const ghostBuf = useMemo(() => new Float32Array(N_CURVE * 3), [])
  const liveRef = useRef<any>(null)
  const ghostRef = useRef<any>(null)
  const lineRefs = useRef<any[]>([]) // 0 Tmax level, 1 peak drop, 2 rated drop, 3 start drop
  const dotRefs = useRef<(Group | null)[]>([]) // 0 rated, 1 peak, 2 standstill
  const tagRefs = useRef<(Group | null)[]>([]) // 0 rated, 1 peak, 2 standstill
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<Record<string, number> | null>(null)
  const labelledIn = useRef<Labels | null>(null)

  useFrame(() => {
    const v = solve(valuesRef.current)

    fillCurve(liveBuf, v.V, v.R2, v.X, x2, y2)
    liveRef.current?.geometry.setPositions(liveBuf)
    fillCurve(ghostBuf, v.V, R2_GHOST, v.X, x2, y2)
    ghostRef.current?.geometry.setPositions(ghostBuf)
    ghostRef.current?.computeLineDistances()

    const yMax = y2(v.Tmax)
    setSeg(lineRefs.current[0], QX0, yMax, QX1 + 0.12, yMax)

    const sPeak = Math.min(v.sMaxT, S_VIEW)
    const pts: [number, number][] = [
      [x2(S_RATED), y2(torque(S_RATED, v.V, v.R2, v.X))],
      [x2(sPeak), y2(torque(sPeak, v.V, v.R2, v.X))],
      [QX0, y2(v.Tstart)],
    ]
    for (let i = 0; i < 3; i++) {
      dotRefs.current[i]?.position.set(pts[i][0], pts[i][1], 0)
    }
    setSeg(lineRefs.current[1], pts[1][0], pts[1][1], pts[1][0], QY0)
    setSeg(lineRefs.current[2], pts[0][0], pts[0][1], pts[0][0], QY0)
    put(tagRefs.current[0], pts[0][0] - 1.0, pts[0][1] + 0.26)
    put(tagRefs.current[1], pts[1][0], pts[1][1] + 0.26)
    put(tagRefs.current[2], QX0 + 0.95, pts[2][1] + 0.26)

    if (labelledFor.current !== valuesRef.current || labelledIn.current !== t) {
      labelledFor.current = valuesRef.current
      labelledIn.current = t
      setReadouts(textRefs.current, t)
      const marks = [
        `s = 0.03 → T = ${fmtT(torque(S_RATED, v.V, v.R2, v.X))}`,
        `Tmax = ${fmtT(v.Tmax)}`,
        `s = 1 → T = ${fmtT(v.Tstart)}`,
      ]
      for (let i = 0; i < 3; i++) {
        const el = textRefs.current[5 + i]
        if (el) el.textContent = marks[i]
      }
    }
  })

  const segs: { color: string; width: number; dashed?: boolean }[] = [
    { color: c.marker, width: 1.5, dashed: true },
    { color: c.guide, width: 1, dashed: true },
    { color: c.guide, width: 1, dashed: true },
  ]

  return (
    <>
      <Line points={Q_SPEED_AXIS} lineWidth={1} color={c.guide} />
      <Line points={Q_TORQUE_AXIS} lineWidth={1} color={c.guide} />
      {Q_TICKS.map((pts, i) => (
        <Line key={i} points={pts} lineWidth={1} color={c.guide} />
      ))}
      <Line points={Q_LOAD_LINE} lineWidth={2} color={c.resistance} />

      <Line
        ref={ghostRef}
        points={CURVE_INIT}
        lineWidth={1.5}
        color={c.guide}
        dashed
        dashSize={0.1}
        gapSize={0.08}
      />
      <Line ref={liveRef} points={CURVE_INIT} lineWidth={3} color={c.voltage} />

      {segs.map((sg, i) => (
        <Line
          key={i}
          ref={(l) => { lineRefs.current[i] = l }}
          points={SEG_INIT}
          lineWidth={sg.width}
          color={sg.color}
          dashed={sg.dashed}
          dashSize={0.1}
          gapSize={0.08}
        />
      ))}

      <Marker ref={(g) => { dotRefs.current[0] = g }} color={c.marker} size={0.055} />
      <Marker ref={(g) => { dotRefs.current[1] = g }} color={c.marker} size={0.075} />
      <Marker ref={(g) => { dotRefs.current[2] = g }} color={c.marker} size={0.055} />

      {[0, 1, 2].map((i) => (
        <group key={i} ref={(g) => { tagRefs.current[i] = g }}>
          <Html center>
            <span ref={(el) => { textRefs.current[5 + i] = el }} className={labelClass} />
          </Html>
        </group>
      ))}

      <group position={[QX1 + 0.42, y2(T_LOAD) - 0.2, 0]}>
        <Html center>
          <span className={labelClass}>TL</span>
        </Html>
      </group>
      <group position={[QX0 + 0.5, QY0 - 0.22, 0]}>
        <Html center>
          <span className={smallClass}>{t.standstill}</span>
        </Html>
      </group>
      <group position={[QX1 - 0.45, QY0 - 0.22, 0]}>
        <Html center>
          <span className={smallClass}>{t.sync}</span>
        </Html>
      </group>
      <group position={[QX0 + 0.34, QY0 + QH + 0.14, 0]}>
        <Html center>
          <span className={smallClass}>{t.torqueAxis}</span>
        </Html>
      </group>
      <group position={[QX1 - 1.5, y2(T_FULL) - 0.25, 0]}>
        <Html center>
          <span className={smallClass}>{t.ghost}</span>
        </Html>
      </group>

      <Readouts position={[-1.55, 1.74, 0]} refs={textRefs.current} />
    </>
  )
}

const concept: ConceptMeta = {
  id: 'induction-motor-torque',
  title: { en: 'Induction Motor Torque-Speed Curve', ko: '유도전동기의 토크-속도 특성' },
  subject: 'machines',
  order: 5,
  description: {
    en: `## Torque against Slip

**The torque curve has one maximum, and the slip at which it sits is set by the rotor resistance R2 alone, while the height of that maximum is set by the voltage and the leakage reactance and never by R2.**

- Strongest neutral curve: the torque T the machine makes at every slip, standstill at the left, synchronous speed at the right
- Faint dashed curve: the same machine with R2 left at its default 0.2 Ω, kept as a ghost so the crest is seen sliding sideways along a level that does not move
- Dashed level line Tmax: the height of the maximum. Move R2 and the peak travels along it; the line itself answers only to V and X
- Mid neutral line TL: the constant shaft load the motor has to drive, 103.7 N·m. Where the rising curve crosses it is the operating point
- Big dot with a dashed drop: the maximum-torque point. Small dot with its own drop: the operating point, and the speed on the axis below it. The rotor at the lower left turns at exactly that speed
- Solid bracket under the axis: the stable region, to the right of the peak. Faint dashed bracket: the unstable region, to its left
- Amber part of the bar: rotor copper loss. Neutral part: mechanical output. They divide the air-gap power Pag in the ratio s to 1 − s

**Torque**: T = (3 ÷ ωs) · (V²·R2 ÷ s) ÷ ((R1 + R2 ÷ s)² + X²)
**Maximum-torque slip**: s_maxT = R2 ÷ √(R1² + X²)
**Maximum torque**: Tmax = 3V² ÷ (2·ωs·(R1 + √(R1² + X²)))
**Power split**: Pag = T·ωs, P2cu = s·Pag, Pmech = (1 − s)·Pag

Read the two formulas side by side. R2 appears in s_maxT but not in Tmax, so rotor resistance decides **where** the peak sits and nothing else: raise it and the whole crest slides toward standstill at exactly the same height. That is how a wound-rotor machine buys starting torque, by inserting external rotor resistance for the start and shorting it out once the machine is up. V appears only in Tmax, squared, so it decides **how high** everything is: a 10 % sag leaves 0.9² = 0.81 of the torque, a loss of 19 % everywhere at once. Two parameters, two completely different sensitivities.

Why the right-hand side is the stable one: there, a dip in speed means more slip and therefore more torque, which pushes the speed back up. To the left of the peak more slip means **less** torque, so a dip runs away and the machine stalls.

Worked example, V = 220 V per phase, f = 60 Hz, 4 poles, R1 = 0.3 Ω, R2 = 0.2 Ω, X = 1.0 Ω, so ns = 1800 rpm and ωs = 188.496 rad/s: s_maxT = 0.2 ÷ √(0.09 + 1) = 0.1916 and Tmax = 286.57 N·m, and putting s = 0.1916 back into the torque formula returns 286.57 N·m as well. At s = 0.03 the torque is 103.67 N·m at 1746 rpm; at standstill, s = 1, it is 123.25 N·m. The rated point splits its air-gap power T·ωs = 19.542 kW into 0.586 kW of rotor copper loss and 18.956 kW of mechanical output, which is 0.03 to 0.97 exactly, because P2cu ÷ Pag is always s.

Drag R2 and watch only the horizontal position of the crest change, with the ghost curve showing the old peak at the identical height; at about 1.04 Ω, where R2 equals √(R1² + X²), the peak reaches standstill and the starting torque becomes the maximum torque. Drag V and the whole curve breathes as V², the load line holding still, until around 130 V the peak drops below the load and the motor can no longer start. Drag X and the peak both sinks and slides toward synchronous speed, since X sits under the square root in both formulas.

The shaft is slowed far below a real 1746 rpm so it can be followed. In 2D: the same characteristic across the full range of slip, marked at the three points worth knowing, the rated slip, the maximum, and the starting torque at s = 1.`,
    ko: `## 슬립에 따른 토크

**토크 곡선에는 최대점이 하나 있습니다. 그 최대점이 어느 슬립에 놓이는지는 회전자 저항 R2가 혼자서 결정하고, 그 최대점의 높이는 전압과 누설 리액턴스가 결정하며 R2는 전혀 관여하지 않습니다.**

- 가장 진한 중성색 곡선: 각 슬립에서 전동기가 내는 토크 T. 왼쪽이 정지, 오른쪽이 동기속도입니다
- 흐린 점선 곡선: R2를 기본값 0.2 Ω에 그대로 둔 같은 전동기. 잔상으로 남겨 두어, 봉우리가 움직이지 않는 높이를 따라 옆으로 미끄러지는 것을 볼 수 있습니다
- 점선 수평선 Tmax: 최대점의 높이. R2를 움직이면 봉우리가 이 선을 따라 이동하며, 선 자체는 V와 X에만 반응합니다
- 중간 밝기 중성색 선 TL: 전동기가 끌어야 하는 일정 부하 토크 103.7 N·m. 상승 구간의 곡선이 이 선과 만나는 곳이 운전점입니다
- 큰 점과 점선 수선: 최대 토크점. 작은 점과 그 수선: 운전점이며, 아래 축에서 그 속도를 읽습니다. 왼쪽 아래 회전자가 바로 그 속도로 돕니다
- 축 아래 실선 괄호: 봉우리 오른쪽의 안정 영역. 흐린 점선 괄호: 그 왼쪽의 불안정 영역
- 막대의 황색 부분: 회전자 동손. 중성색 부분: 기계적 출력. 이 둘이 공극 전력 Pag를 s 대 1 − s로 나눕니다

**토크**: T = (3 ÷ ωs) · (V²·R2 ÷ s) ÷ ((R1 + R2 ÷ s)² + X²)
**최대 토크 슬립**: s_maxT = R2 ÷ √(R1² + X²)
**최대 토크**: Tmax = 3V² ÷ (2·ωs·(R1 + √(R1² + X²)))
**전력 분배**: Pag = T·ωs, P2cu = s·Pag, Pmech = (1 − s)·Pag

두 식을 나란히 놓고 보세요. R2는 s_maxT에만 있고 Tmax에는 없습니다. 즉 회전자 저항은 봉우리의 **위치**만 정합니다. R2를 키우면 봉우리 전체가 같은 높이를 유지한 채 정지점 쪽으로 미끄러져 갑니다. 권선형 유도전동기가 기동 토크를 얻는 방법이 바로 이것으로, 기동할 때만 외부 저항을 회전자에 넣고 속도가 올라오면 단락시킵니다. V는 Tmax에만, 그것도 제곱으로 들어가므로 전체의 **높이**를 정합니다. 전압이 10 % 떨어지면 토크는 0.9² = 0.81배가 되어 모든 슬립에서 한꺼번에 19 %를 잃습니다. 두 파라미터, 전혀 다른 두 민감도입니다.

오른쪽이 안정한 이유는 이렇습니다. 그곳에서는 속도가 떨어지면 슬립이 커지고 따라서 토크가 커져 속도를 되밀어 올립니다. 봉우리 왼쪽에서는 슬립이 커질수록 토크가 **줄어들기** 때문에, 한 번 떨어진 속도가 계속 떨어져 전동기가 멈춰 섭니다.

계산 예로 V = 220 V(상전압), f = 60 Hz, 4극, R1 = 0.3 Ω, R2 = 0.2 Ω, X = 1.0 Ω이면 ns = 1800 rpm, ωs = 188.496 rad/s입니다. s_maxT = 0.2 ÷ √(0.09 + 1) = 0.1916이고 Tmax = 286.57 N·m이며, s = 0.1916을 토크 식에 다시 넣어도 286.57 N·m가 나옵니다. s = 0.03에서는 1746 rpm으로 토크가 103.67 N·m이고, 정지 상태 s = 1에서는 123.25 N·m입니다. 이 정격점에서 공극 전력 T·ωs = 19.542 kW는 회전자 동손 0.586 kW와 기계적 출력 18.956 kW로 나뉘며, 이는 정확히 0.03 대 0.97입니다. P2cu ÷ Pag는 언제나 s이기 때문입니다.

R2를 끌어 보면 봉우리의 가로 위치만 바뀌고, 잔상 곡선이 이전 봉우리가 똑같은 높이에 있었음을 보여 줍니다. R2가 √(R1² + X²)와 같아지는 약 1.04 Ω에서 봉우리가 정지점에 도달하여 기동 토크가 최대 토크가 됩니다. V를 끌면 부하선은 가만히 있는 채 곡선 전체가 V²로 숨을 쉬고, 130 V 근처에서 봉우리가 부하선 아래로 내려가면 전동기는 더 이상 기동하지 못합니다. X를 끌면 두 식 모두에서 X가 제곱근 안에 들어 있으므로 봉우리가 내려앉으면서 동기속도 쪽으로 미끄러집니다.

회전축은 실제 1746 rpm보다 훨씬 느리게 재생하여 눈으로 따라갈 수 있게 했습니다. 2D: 같은 특성을 슬립 전 구간에 걸쳐 보여 주며, 알아둘 만한 세 점, 즉 정격 슬립, 최대점, 그리고 s = 1의 기동 토크를 표시합니다.`,
  },
  sliderParams: [
    { key: 'R2', label: { en: 'Rotor Resistance (R2)', ko: '회전자 저항 (R2)' }, unit: 'Ω', min: 0.1, max: 1.2, step: 0.05, defaultValue: 0.2 },
    { key: 'V',  label: { en: 'Applied Voltage (V)',   ko: '인가 전압 (V)' },     unit: 'V', min: 100, max: 250, step: 5,    defaultValue: 220 },
    { key: 'X',  label: { en: 'Leakage Reactance (X)', ko: '누설 리액턴스 (X)' }, unit: 'Ω', min: 0.5, max: 2.0, step: 0.1,  defaultValue: 1.0 },
  ],
  Scene: TorqueCurveScene,
  Scene2D: TorqueCurve2DScene,
}

export default concept
