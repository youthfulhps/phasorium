import { useRef, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import type { Group, Mesh } from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import { useLocaleStore } from '../../lib/i18n'
import type { ConceptMeta } from '../../types/concept'

const TWO_PI = Math.PI * 2
const HALF_PI = Math.PI / 2

// Held fixed so the four sliders carry all of the physics
const POLES = 4        // P, field poles
const CONDUCTORS = 400 // Z, armature conductors
const V_TERM = 220     // V, terminal voltage in motor mode
const RA = 0.5         // Ω, armature resistance

/*
 * Pure physics of a DC machine. `a` is the number of parallel armature paths:
 * lap winding -> a = P, wave winding -> a = 2.
 *   E = P·Z·Φ·N ÷ (60·a) [V]    T = P·Z·Φ·Ia ÷ (2π·a) [N·m]    ω = 2π·N ÷ 60 [rad/s]
 * Both come from the same constant P·Z ÷ a, so
 *   E·Ia = P·Z·Φ·N·Ia ÷ (60·a) = T·ω
 * identically: the electrical power crossing into the armature IS the mechanical power
 * leaving it. Motor mode: Eb = V − Ia·Ra, and because E is proportional to Φ·N that same
 * back emf at the same flux fixes the speed, N = Eb ÷ (kE·Φ).
 *
 * Verified with npx tsx -e, for P = 4, Z = 400, Φ = 0.02 Wb, N = 1500 rpm, Ia = 10 A:
 *   lap  (a = 4) -> E = 200.00 V, T = 12.732 N·m
 *   wave (a = 2) -> E = 400.00 V, T = 25.465 N·m        (halving a doubles both)
 *   ω = 2π × 1500 ÷ 60 = 157.08 rad/s
 *   E·Ia = 2000.00 W, T·ω = 2000.00 W                   (relative difference 1.14e-16)
 *   motor: Eb = 220 − 10 × 0.5 = 215.00 V -> N = 215 ÷ (6.6667 × 0.02) = 1612.5 rpm
 * Sign convention. Driven shaft: the machine generates, E is raised inside the armature,
 * E > V at the terminals and power flows out. Fed terminals: the machine motors, the same
 * E now opposes the supply and is called Eb, Eb < V, and power flows in.
 *
 * The result goes into one module-scope record, so no frame ever allocates.
 */
const S = {
  Phi: 0.02, N: 1500, Ia: 10, a: 4, lap: true,
  kE: 0, kT: 0, E: 0, T: 0, omega: 0, Pelec: 0, Pmech: 0, Eb: 0, Nb: 0,
}

function solve({ flux = 0.02, speed = 1500, ia = 10, winding = 1 }: Record<string, number>) {
  const a = winding < 1.5 ? POLES : 2
  const kE = (POLES * CONDUCTORS) / (60 * a)     // V per Wb per rpm
  const kT = (POLES * CONDUCTORS) / (TWO_PI * a) // N·m per Wb per A
  S.Phi = flux
  S.N = speed
  S.Ia = ia
  S.a = a
  S.lap = a === POLES
  S.kE = kE
  S.kT = kT
  S.E = kE * flux * speed
  S.T = kT * flux * ia
  S.omega = (TWO_PI * speed) / 60
  S.Pelec = S.E * ia
  S.Pmech = S.T * S.omega
  S.Eb = V_TERM - ia * RA
  S.Nb = S.Eb / (kE * flux)
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

// Slider extremes, used only to pick display scales
const FLUX_MIN = 0.005
const FLUX_MAX = 0.04
const N_MAX = 3000
const IA_MAX = 30
const E_FULL = (POLES * CONDUCTORS * FLUX_MAX * N_MAX) / (60 * 2) // 1600 V, the wave-winding corner
const W_FULL = (TWO_PI * N_MAX) / 60                              // 314.16 rad/s

// Scene labels read in the foreground color; only current (cyan) and power (amber) tint their text
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const smallClass = 'pointer-events-none whitespace-nowrap text-[10px] font-medium tabular-nums text-foreground/70 leading-4'

const L = {
  en: {
    poles: 'field poles · flux Φ each',
    armature: 'armature · turning at N',
    paths: 'parallel paths · a',
    equal: 'equal area',
    lap: 'lap',
    wave: 'wave',
    motor: 'motor',
    speedPlot: 'speed N against Ia',
    torquePlot: 'torque T against Ia',
    atTen: 'at Ia = 10 A',
    iaAxis: 'Ia (A)',
  },
  ko: {
    poles: '계자 자극 · 극당 자속 Φ',
    armature: '전기자 · N으로 회전',
    paths: '병렬 회로 수 · a',
    equal: '같은 면적',
    lap: '중권',
    wave: '파권',
    motor: '전동기',
    speedPlot: '속도 N 대 Ia',
    torquePlot: '토크 T 대 Ia',
    atTen: 'Ia = 10 A에서',
    iaAxis: 'Ia (A)',
  },
} satisfies Record<string, Record<string, string>>

type Strings = (typeof L)['en']

// ── shared scratch: filled then handed straight to a geometry, so no frame allocates ──
const ARC_N = 40
const ARC = new Float32Array(ARC_N * 3)
const HYP_N = 48
const HYP = new Float32Array(HYP_N * 3)
const CURVE_N = 61
const SEG = new Float32Array(6)
const RECT = new Float32Array(15)

const zeros = (n: number): [number, number, number][] =>
  Array.from({ length: n }, (): [number, number, number] => [0, 0, 0])

const SEG_INIT = zeros(2)
const ARC_INIT = zeros(ARC_N)
const HYP_INIT = zeros(HYP_N)
const RECT_INIT = zeros(5)
const CURVE_INIT = zeros(CURVE_N)

function upload(line: any, buf: Float32Array) {
  line?.geometry.setPositions(buf)
  line?.computeLineDistances()
}

function setSeg(line: any, x0: number, y0: number, x1: number, y1: number) {
  SEG[0] = x0; SEG[1] = y0; SEG[2] = 0
  SEG[3] = x1; SEG[4] = y1; SEG[5] = 0
  upload(line, SEG)
}

function setArc(line: any, cx: number, cy: number, r: number, a0: number, a1: number) {
  for (let i = 0; i < ARC_N; i++) {
    const a = a0 + ((a1 - a0) * i) / (ARC_N - 1)
    ARC[i * 3] = cx + r * Math.cos(a)
    ARC[i * 3 + 1] = cy + r * Math.sin(a)
    ARC[i * 3 + 2] = 0
  }
  upload(line, ARC)
}

function setRect(line: any, x: number, y: number, w: number, h: number) {
  RECT[0] = x; RECT[1] = y; RECT[2] = 0
  RECT[3] = x + w; RECT[4] = y; RECT[5] = 0
  RECT[6] = x + w; RECT[7] = y + h; RECT[8] = 0
  RECT[9] = x; RECT[10] = y + h; RECT[11] = 0
  RECT[12] = x; RECT[13] = y; RECT[14] = 0
  upload(line, RECT)
}

const put = (g: Group | null | undefined, x: number, y: number) => { g?.position.set(x, y, 0) }

const arcPts = (cx: number, cy: number, r: number, a0: number, a1: number, n: number) =>
  Array.from({ length: n }, (_, i): [number, number, number] => {
    const a = a0 + ((a1 - a0) * i) / (n - 1)
    return [cx + r * Math.cos(a), cy + r * Math.sin(a), 0]
  })

const circlePts = (cx: number, cy: number, r: number, n = 96) => arcPts(cx, cy, r, 0, TWO_PI, n + 1)

// ── the six live numbers, rebuilt only when the sliders or the locale change ──
function setReadouts(refs: (HTMLSpanElement | null)[], t: Strings) {
  if (refs[0]) refs[0].textContent = `P = ${POLES} · Z = ${CONDUCTORS} · V = ${V_TERM} V · Ra = ${RA} Ω`
  if (refs[1]) refs[1].textContent = `a = ${S.a} (${S.lap ? t.lap : t.wave}) · Φ = ${S.Phi.toFixed(3)} Wb`
  if (refs[2]) refs[2].textContent = `E = P·Z·Φ·N ÷ (60·a) = ${S.E.toFixed(2)} V`
  if (refs[3]) refs[3].textContent = `T = P·Z·Φ·Ia ÷ (2π·a) = ${S.T.toFixed(3)} N·m`
  if (refs[4]) {
    refs[4].textContent = `E·Ia = ${S.Pelec.toFixed(2)} W = T·ω = ${S.Pmech.toFixed(2)} W`
  }
  if (refs[5]) {
    refs[5].textContent = `${t.motor}: Eb = V − Ia·Ra = ${S.Eb.toFixed(2)} V → ${S.Nb.toFixed(1)} rpm`
  }
}

function Readouts(
  { position, refs, accent }: { position: [number, number, number]; refs: (HTMLSpanElement | null)[]; accent: string }
) {
  return (
    <group position={position}>
      <Html center>
        <div className={smallClass}>
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <span
              key={i}
              ref={(el) => { refs[i] = el }}
              className="block"
              style={i === 4 ? { color: accent } : undefined}
            />
          ))}
        </div>
      </Html>
    </group>
  )
}

// ── 3D: the armature turning between the poles, beside the power rectangle ──
// Left: the machine face-on. The conductors swell while they pass under a pole, because only
// there do they cut flux and raise emf. Inside the armature the current runs in `a` cyan paths.
// Right: E, Ia, T and ω as the sides of two rectangles sharing a corner. The side scales obey
// K_E·K_IA = K_T·K_W, so both rectangles have exactly the area E·Ia = T·ω and both outer
// corners land on the same amber curve x·y = constant. Same area, same power, counted twice.
const MX = -1.0 // machine centre, kept well clear of the floating description card
const MY = 0.45
const R_ARM = 0.62
const R_PATH = 0.47
const R_POLE = 0.88
const R_LETTER = 1.04
const POLE_HALF = 0.42 // half the angular width of a pole shoe [rad]
const N_COND = 16
const K_RPM = 1 / 1500 // display rad/s per rpm: the 1500 rpm default turns at 1.0 rad/s

const PX = 0.15 // power frame origin
const PY = -1.25
const SPAN = 1.35 // scene units for a full-scale side
const K_E = SPAN / E_FULL
const K_IA = SPAN / IA_MAX
const K_W = SPAN / W_FULL
const K_T = (K_E * K_IA) / K_W // the one choice that makes the two areas comparable

const POLE_ANG = [0, HALF_PI, Math.PI, 3 * HALF_PI]
const POLE_ARCS = POLE_ANG.map((a) => arcPts(MX, MY, R_POLE, a - POLE_HALF, a + POLE_HALF, 20))
const POLE_LETTERS = POLE_ANG.map((a, i) => ({
  pos: [MX + R_LETTER * Math.cos(a), MY + R_LETTER * Math.sin(a), 0] as [number, number, number],
  text: i % 2 === 0 ? 'N' : 'S',
}))
// Four flux ticks per pole, filled from the middle outwards as Φ rises
const TICK_OFF = [-0.27, -0.09, 0.09, 0.27]
const TICK_RANK = [2, 0, 1, 3]
const FLUX_TICKS = POLE_ANG.flatMap((pa) =>
  TICK_OFF.map((off, j) => ({
    rank: TICK_RANK[j],
    pts: [
      [MX + (R_POLE + 0.04) * Math.cos(pa + off), MY + (R_POLE + 0.04) * Math.sin(pa + off), 0],
      [MX + (R_ARM + 0.06) * Math.cos(pa + off), MY + (R_ARM + 0.06) * Math.sin(pa + off), 0],
    ] as [number, number, number][],
  }))
)
const ARM_CIRCLE = circlePts(MX, MY, R_ARM)
const FRAME_X: [number, number, number][] = [[PX, PY, 0], [PX + SPAN + 0.1, PY, 0]]
const FRAME_Y: [number, number, number][] = [[PX, PY, 0], [PX, PY + SPAN + 0.1, 0]]

function setHyperbola(line: any, area: number) {
  const x0 = Math.max(area / SPAN, 0.02)
  for (let i = 0; i < HYP_N; i++) {
    const x = x0 + ((SPAN - x0) * i) / (HYP_N - 1)
    HYP[i * 3] = PX + x
    HYP[i * 3 + 1] = PY + Math.min(area / x, SPAN)
    HYP[i * 3 + 2] = 0
  }
  upload(line, HYP)
}

function DCMachineScene() {
  const c = usePhysicsColors()
  const t = L[useLocaleStore((s) => s.locale)]
  const valuesRef = useSliderValues()
  const pathRefs = useRef<any[]>([])
  const brushRefs = useRef<any[]>([])
  const fluxRefs = useRef<any[]>([])
  const rectRefs = useRef<any[]>([]) // 0 E×Ia outline, 1 T×ω outline
  const hypRef = useRef<any>(null)
  const fillRef = useRef<Mesh>(null)
  const condRefs = useRef<(Group | null)[]>([])
  const cornerRefs = useRef<(Group | null)[]>([])
  const tagRefs = useRef<(Group | null)[]>([]) // 0 E, 1 T, 2 Ia, 3 ω, 4 equal area
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<Record<string, number> | null>(null)
  const labelledIn = useRef<Strings | null>(null)
  const flow = useRef(2.8)
  const flowSpeed = useMemo(() => () => flow.current, [])

  useFrame(({ clock }) => {
    const v = solve(valuesRef.current)
    const time = clock.getElapsedTime()
    const th = v.N * K_RPM * time
    const fluxFrac = Math.min(Math.max((v.Phi - FLUX_MIN) / (FLUX_MAX - FLUX_MIN), 0), 1)

    // Flux: more ticks across each pole face as Φ rises
    const nTicks = 1 + Math.round(3 * fluxFrac)
    for (let i = 0; i < FLUX_TICKS.length; i++) {
      const l = fluxRefs.current[i]
      if (l) l.visible = FLUX_TICKS[i].rank < nTicks
    }

    // The conductors ride round at N and swell while they sweep under a pole
    for (let i = 0; i < N_COND; i++) {
      const a = th + (i * TWO_PI) / N_COND
      const g = condRefs.current[i]
      if (!g) continue
      g.position.set(MX + R_ARM * Math.cos(a), MY + R_ARM * Math.sin(a), 0)
      const m = ((a % HALF_PI) + HALF_PI) % HALF_PI
      const under = Math.min(m, HALF_PI - m) < POLE_HALF
      g.scale.setScalar(under ? 0.8 + fluxFrac : 0.45)
    }

    // The winding: `a` parallel paths round the armature, each with its own brush
    for (let k = 0; k < 4; k++) {
      const on = k < v.a
      const l = pathRefs.current[k]
      const b = brushRefs.current[k]
      if (l) l.visible = on
      if (b) b.visible = on
      if (!on) continue
      const a0 = (k * TWO_PI) / v.a
      setArc(l, MX, MY, R_PATH, a0, a0 + TWO_PI / v.a)
      setSeg(
        b,
        MX + R_PATH * Math.cos(a0), MY + R_PATH * Math.sin(a0),
        MX + (R_ARM + 0.16) * Math.cos(a0), MY + (R_ARM + 0.16) * Math.sin(a0)
      )
    }
    flow.current = 0.6 + 0.22 * v.Ia // the sheen runs faster as Ia rises

    // The power rectangle, read two ways
    const hE = Math.max(K_E * v.E, 1e-3)
    const wI = Math.max(K_IA * v.Ia, 1e-3)
    const hT = Math.max(K_T * v.T, 1e-3)
    const wW = Math.max(K_W * v.omega, 1e-3)
    const area = hE * wI
    setRect(rectRefs.current[0], PX, PY, wI, hE)
    setRect(rectRefs.current[1], PX, PY, wW, hT)
    setHyperbola(hypRef.current, area)
    const m = fillRef.current
    if (m) {
      m.position.set(PX + wI / 2, PY + hE / 2, -0.01)
      m.scale.set(wI, hE, 1)
    }
    cornerRefs.current[0]?.position.set(PX + wI, PY + hE, 0)
    cornerRefs.current[1]?.position.set(PX + wW, PY + hT, 0)

    const s = Math.sqrt(area)
    put(tagRefs.current[0], PX - 0.13, PY + hE / 2)
    put(tagRefs.current[1], PX + wW + 0.17, PY + hT)
    put(tagRefs.current[2], PX + wI / 2, PY - 0.16)
    put(tagRefs.current[3], PX + wW / 2, PY - 0.34)
    put(tagRefs.current[4], PX + s + 0.19, PY + s + 0.15)

    if (labelledFor.current !== valuesRef.current || labelledIn.current !== t) {
      labelledFor.current = valuesRef.current
      labelledIn.current = t
      setReadouts(textRefs.current, t)
    }
  })

  const tags = [
    { text: 'E', color: undefined },
    { text: 'T', color: undefined },
    { text: 'Ia', color: c.current },
    { text: 'ω', color: undefined },
    { text: t.equal, color: c.power },
  ]

  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />

      {/* Background, lights, grid, and bloom come from the shared SceneStage */}
      {POLE_ARCS.map((pts, i) => (
        <Line key={i} points={pts} lineWidth={3} color={c.inductance} />
      ))}
      {FLUX_TICKS.map((tick, i) => (
        <Line
          key={i}
          ref={(l) => { fluxRefs.current[i] = l }}
          points={tick.pts}
          lineWidth={1.5}
          color={c.inductance}
        />
      ))}
      <Line points={ARM_CIRCLE} lineWidth={1} color={c.guide} dashed dashSize={0.07} gapSize={0.05} />

      {/* The a parallel paths, and the brush that each one ends on */}
      {[0, 1, 2, 3].map((k) => (
        <TrailLine
          key={k}
          ref={(l) => { pathRefs.current[k] = l }}
          points={ARC_INIT}
          lineWidth={2.5}
          color={c.current}
          fade={false}
          flowSpeed={flowSpeed}
        />
      ))}
      {[0, 1, 2, 3].map((k) => (
        <Line
          key={k}
          ref={(l) => { brushRefs.current[k] = l }}
          points={SEG_INIT}
          lineWidth={2}
          color={c.current}
        />
      ))}
      {Array.from({ length: N_COND }, (_, i) => (
        <Marker key={i} ref={(g) => { condRefs.current[i] = g }} color={c.marker} size={0.05} />
      ))}

      {/* The power: one amber area, two pairs of sides */}
      <Line points={FRAME_X} lineWidth={1} color={c.guide} />
      <Line points={FRAME_Y} lineWidth={1} color={c.guide} />
      <mesh ref={fillRef}>
        <planeGeometry args={[1, 1]} />
        <meshBasicMaterial color={c.power} transparent opacity={0.24} depthWrite={false} />
      </mesh>
      <Line
        ref={(l) => { rectRefs.current[0] = l }}
        points={RECT_INIT}
        lineWidth={2.5}
        color={c.voltage}
      />
      <Line
        ref={(l) => { rectRefs.current[1] = l }}
        points={RECT_INIT}
        lineWidth={2.5}
        color={c.resistance}
      />
      <Line
        ref={hypRef}
        points={HYP_INIT}
        lineWidth={1.5}
        color={c.power}
        dashed
        dashSize={0.07}
        gapSize={0.05}
      />
      <Marker ref={(g) => { cornerRefs.current[0] = g }} color={c.power} size={0.055} />
      <Marker ref={(g) => { cornerRefs.current[1] = g }} color={c.power} size={0.055} />

      {tags.map((tag, i) => (
        <group key={i} ref={(g) => { tagRefs.current[i] = g }}>
          <Html center>
            <span className={labelClass} style={tag.color ? { color: tag.color } : undefined}>
              {tag.text}
            </span>
          </Html>
        </group>
      ))}

      {POLE_LETTERS.map((p, i) => (
        <group key={i} position={p.pos}>
          <Html center>
            <span className={labelClass}>{p.text}</span>
          </Html>
        </group>
      ))}
      <group position={[MX - 0.1, MY + 1.22, 0]}>
        <Html center>
          <span className={smallClass}>{t.poles}</span>
        </Html>
      </group>
      <group position={[MX, MY - 1.2, 0]}>
        <Html center>
          <span className={smallClass}>{t.armature}</span>
        </Html>
      </group>
      <group position={[MX + 0.62, MY + 0.98, 0]}>
        <Html center>
          <span className={smallClass} style={{ color: c.current }}>{t.paths}</span>
        </Html>
      </group>

      <Readouts position={[MX - 0.15, -1.62, 0]} refs={textRefs.current} accent={c.power} />
    </>
  )
}

// ── 2D: the motor characteristic, speed over torque, both against armature current ──
// Upper panel: N = (V − Ia·Ra) ÷ (kE·Φ), drooping only by the Ia·Ra the armature eats.
// Lower panel: T = kT·Φ·Ia, a straight line out of the origin. Weaken Φ and the speed line
// climbs while the torque line flattens, which is how a DC motor runs above its base speed.
const X0 = -3.2
const XSPAN = 5.6
const XK = XSPAN / IA_MAX
const NY0 = 0.25
const NH = 1.5
const N_FULL = 7000 // rpm, above the highest no-load speed the sliders can reach (6600)
const TY0 = -1.7
const TH = 1.4
const T_FULL = 160 // N·m, above the largest torque the sliders can reach (152.8)
const MARK_IA = 10

const nAxisX: [number, number, number][] = [[X0, NY0, 0], [X0 + XSPAN + 0.12, NY0, 0]]
const nAxisY: [number, number, number][] = [[X0, NY0, 0], [X0, NY0 + NH, 0]]
const tAxisX: [number, number, number][] = [[X0, TY0, 0], [X0 + XSPAN + 0.12, TY0, 0]]
const tAxisY: [number, number, number][] = [[X0, TY0, 0], [X0, TY0 + TH, 0]]
const MARK_LINE: [number, number, number][] = [
  [X0 + MARK_IA * XK, TY0, 0],
  [X0 + MARK_IA * XK, NY0 + NH, 0],
]
const tick = (x: number, y: number, dx: number, dy: number): [number, number, number][] =>
  [[x - dx, y - dy, 0], [x + dx, y + dy, 0]]
const N_TICKS = [2000, 4000, 6000].map((n) => tick(X0, NY0 + (n / N_FULL) * NH, 0.07, 0))
const T_TICKS = [40, 80, 120].map((q) => tick(X0, TY0 + (q / T_FULL) * TH, 0.07, 0))
const IA_TICKS = [10, 20, 30].map((i) => tick(X0 + i * XK, TY0, 0, 0.07))
const N_TICK_TAGS = [2000, 4000, 6000].map((n) => ({
  pos: [X0 - 0.34, NY0 + (n / N_FULL) * NH, 0] as [number, number, number],
  text: `${n}`,
}))
const T_TICK_TAGS = [40, 80, 120].map((q) => ({
  pos: [X0 - 0.3, TY0 + (q / T_FULL) * TH, 0] as [number, number, number],
  text: `${q}`,
}))
const IA_TICK_TAGS = [10, 20, 30].map((i) => ({
  pos: [X0 + i * XK, TY0 - 0.2, 0] as [number, number, number],
  text: `${i}`,
}))

const N_BUF = new Float32Array(CURVE_N * 3)
const T_BUF = new Float32Array(CURVE_N * 3)

function MotorCurves2DScene() {
  const c = usePhysicsColors()
  const t = L[useLocaleStore((s) => s.locale)]
  const valuesRef = useSliderValues()
  const curveRefs = useRef<any[]>([]) // 0 speed, 1 torque
  const dotRefs = useRef<(Group | null)[]>([]) // 0 speed at 10 A, 1 torque at 10 A
  const tagRefs = useRef<(Group | null)[]>([]) // 0 speed value, 1 torque value
  const valRefs = useRef<(HTMLSpanElement | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<Record<string, number> | null>(null)
  const labelledIn = useRef<Strings | null>(null)

  useFrame(() => {
    const v = solve(valuesRef.current)
    const kN = 1 / (v.kE * v.Phi) // rpm per volt of back emf

    for (let i = 0; i < CURVE_N; i++) {
      const ia = (i / (CURVE_N - 1)) * IA_MAX
      const x = X0 + ia * XK
      const n = Math.min(Math.max((V_TERM - ia * RA) * kN, 0), N_FULL)
      const q = Math.min(v.kT * v.Phi * ia, T_FULL)
      N_BUF[i * 3] = x
      N_BUF[i * 3 + 1] = NY0 + (n / N_FULL) * NH
      N_BUF[i * 3 + 2] = 0
      T_BUF[i * 3] = x
      T_BUF[i * 3 + 1] = TY0 + (q / T_FULL) * TH
      T_BUF[i * 3 + 2] = 0
    }
    upload(curveRefs.current[0], N_BUF)
    upload(curveRefs.current[1], T_BUF)

    const xm = X0 + MARK_IA * XK
    const nm = Math.min(Math.max((V_TERM - MARK_IA * RA) * kN, 0), N_FULL)
    const qm = Math.min(v.kT * v.Phi * MARK_IA, T_FULL)
    const yn = NY0 + (nm / N_FULL) * NH
    const yq = TY0 + (qm / T_FULL) * TH
    dotRefs.current[0]?.position.set(xm, yn, 0)
    dotRefs.current[1]?.position.set(xm, yq, 0)
    put(tagRefs.current[0], xm + 0.72, yn + 0.17)
    put(tagRefs.current[1], xm + 0.76, yq + 0.17)

    if (labelledFor.current !== valuesRef.current || labelledIn.current !== t) {
      labelledFor.current = valuesRef.current
      labelledIn.current = t
      setReadouts(textRefs.current, t)
      if (valRefs.current[0]) valRefs.current[0].textContent = `N = ${nm.toFixed(1)} rpm`
      if (valRefs.current[1]) valRefs.current[1].textContent = `T = ${qm.toFixed(3)} N·m`
    }
  })

  return (
    <>
      <Line points={nAxisX} lineWidth={1} color={c.guide} />
      <Line points={nAxisY} lineWidth={1} color={c.guide} />
      <Line points={tAxisX} lineWidth={1} color={c.guide} />
      <Line points={tAxisY} lineWidth={1} color={c.guide} />
      {[...N_TICKS, ...T_TICKS, ...IA_TICKS].map((pts, i) => (
        <Line key={i} points={pts} lineWidth={1} color={c.guide} />
      ))}
      <Line points={MARK_LINE} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />

      <Line
        ref={(l) => { curveRefs.current[0] = l }}
        points={CURVE_INIT}
        lineWidth={2.5}
        color={c.inductance}
      />
      <Line
        ref={(l) => { curveRefs.current[1] = l }}
        points={CURVE_INIT}
        lineWidth={2.5}
        color={c.resistance}
      />
      <Marker ref={(g) => { dotRefs.current[0] = g }} color={c.marker} size={0.06} />
      <Marker ref={(g) => { dotRefs.current[1] = g }} color={c.marker} size={0.06} />

      {[0, 1].map((i) => (
        <group key={i} ref={(g) => { tagRefs.current[i] = g }}>
          <Html center>
            <span ref={(el) => { valRefs.current[i] = el }} className={labelClass} />
          </Html>
        </group>
      ))}

      {[...N_TICK_TAGS, ...T_TICK_TAGS, ...IA_TICK_TAGS].map((tag, i) => (
        <group key={i} position={tag.pos}>
          <Html center>
            <span className={smallClass}>{tag.text}</span>
          </Html>
        </group>
      ))}
      <group position={[X0 + 1.5, NY0 + NH - 0.14, 0]}>
        <Html center>
          <span className={smallClass}>{t.speedPlot}</span>
        </Html>
      </group>
      <group position={[X0 + 1.5, TY0 + TH - 0.14, 0]}>
        <Html center>
          <span className={smallClass}>{t.torquePlot}</span>
        </Html>
      </group>
      <group position={[X0 + MARK_IA * XK, NY0 + NH + 0.17, 0]}>
        <Html center>
          <span className={smallClass}>{t.atTen}</span>
        </Html>
      </group>
      <group position={[X0 + XSPAN + 0.42, TY0 - 0.2, 0]}>
        <Html center>
          <span className={smallClass}>{t.iaAxis}</span>
        </Html>
      </group>

      <Readouts position={[X0 + 1.42, -1.95, 0]} refs={textRefs.current} accent={c.power} />
    </>
  )
}

const concept: ConceptMeta = {
  id: 'dc-machine',
  title: { en: 'DC Machine: EMF, Torque & Speed', ko: '직류기의 유기기전력·토크·속도' },
  subject: 'machines',
  order: 3,
  description: {
    en: `## DC Machine: EMF, Torque and Speed

**One machine, one pair of equations: the armature that generates E when you drive it develops T when you feed it current, and the two are locked together by E·Ia = T·ω, so the electrical power crossing into the armature is exactly the mechanical power leaving it.**

- Light neutral arcs with the radial ticks across them: the field poles and the flux Φ each one sends through the armature. More ticks means more flux
- Faint dashed circle with the dots riding it: the armature, turning at N. A dot swells while it passes under a pole, because only there does it cut flux and raise emf
- Cyan arcs inside the armature, each ending on its own brush tick: the a parallel paths the conductors are wired into. The sheen runs faster as Ia rises
- Strongest neutral rectangle, tall and narrow: E as its height, Ia as its width
- Mid neutral rectangle, short and wide: T as its height, ω as its width
- Amber fill and the amber curve through both outer corners: the power. Both corners sit on the same curve x·y = constant, so the two rectangles enclose exactly the same area

**Generated emf**: E = P·Z·Φ·N ÷ (60·a) [V]
**Torque**: T = P·Z·Φ·Ia ÷ (2π·a) [N·m]
**Parallel paths**: lap winding gives a = P, wave winding gives a = 2
**Power identity**: E·Ia = T·ω, with ω = 2π·N ÷ 60 [rad/s]
**Motor back emf**: Eb = V − Ia·Ra, and N is proportional to Eb ÷ Φ

Held fixed here: P = 4 poles, Z = 400 armature conductors, V = 220 V, Ra = 0.5 Ω.

Sign convention. Drive the shaft and the machine generates: E is raised inside the armature, E is larger than V at the terminals, and power flows out. Feed the terminals instead and the same machine motors: that same E now opposes the supply and is called the back emf Eb, Eb is smaller than V, the difference Ia·Ra is what pushes the current in, and power flows in. Neither equation changes. Only the direction of Ia and of the shaft power does.

Worked example: Φ = 0.02 Wb, N = 1500 rpm, Ia = 10 A, lap winding so a = P = 4. Then E = 4 × 400 × 0.02 × 1500 ÷ (60 × 4) = 200.00 V and T = 4 × 400 × 0.02 × 10 ÷ (2π × 4) = 12.732 N·m. With ω = 2π × 1500 ÷ 60 = 157.08 rad/s, E·Ia = 2000.00 W and T·ω = 2000.00 W. The same 2000 W, counted twice.

Switch to wave winding and a falls from 4 to 2: E doubles to 400.00 V and T doubles to 25.465 N·m, and the cyan paths re-route from four arcs into two. One integer decides whether the same iron and copper make a higher-voltage machine or a higher-torque one.

As a motor on 220 V drawing Ia = 10 A, Eb = 220 − 10 × 0.5 = 215.00 V, which at this flux corresponds to 1612.5 rpm.

Try the sliders. Raise Φ and E grows, T grows with it, and the amber area grows faster than either. Raise N and E grows while T stays put, so the rectangle turns tall and narrow and the other turns short and wide, always sweeping the same area. Raise Ia and the sheen in the paths speeds up. Flux is the interesting one: it lifts both E and T, yet in motor mode a weaker Φ means a higher speed, because the same Eb has to be made with less flux.

In 2D: the motor characteristic, with armature current across the bottom. Light neutral above is the speed N = (V − Ia·Ra) ÷ (kE·Φ), drooping gently because only Ia·Ra is lost. Mid neutral below is the torque T = kT·Φ·Ia, a straight line out of the origin. Markers at Ia = 10 A read 1612.5 rpm and 12.732 N·m at the default settings. Weaken Φ and watch the speed line climb while the torque line flattens.`,
    ko: `## 직류기의 유기기전력·토크·속도

**하나의 기계, 하나의 식 쌍입니다. 축을 돌려 주면 E를 유기하는 그 전기자가, 전류를 넣어 주면 T를 냅니다. 두 값은 E·Ia = T·ω로 묶여 있어서, 전기자로 들어가는 전기적 전력이 곧 축으로 나가는 기계적 전력입니다.**

- 밝은 중성색 호와 그 위를 가로지르는 방사형 눈금: 계자 자극과 각 극이 전기자로 보내는 자속 Φ. 눈금이 많을수록 자속이 큽니다
- 흐린 점선 원과 그 위를 달리는 점들: 전기자. 속도는 N입니다. 점은 자극 아래를 지날 때 커집니다. 그곳에서만 자속을 끊어 기전력을 만들기 때문입니다
- 전기자 안쪽의 시안 호와 각각의 브러시 눈금: 도체가 엮인 a개의 병렬 회로. Ia가 커지면 광택이 더 빠르게 흐릅니다
- 가장 진한 중성색 직사각형, 높고 좁은 쪽: 높이가 E, 너비가 Ia입니다
- 중간 밝기 중성색 직사각형, 낮고 넓은 쪽: 높이가 T, 너비가 ω입니다
- 황색 채움과 두 바깥 꼭짓점을 지나는 황색 곡선: 전력입니다. 두 꼭짓점이 같은 곡선 x·y = 일정 위에 있으므로, 두 직사각형의 면적은 정확히 같습니다

**유기기전력**: E = P·Z·Φ·N ÷ (60·a) [V]
**토크**: T = P·Z·Φ·Ia ÷ (2π·a) [N·m]
**병렬 회로 수**: 중권은 a = P, 파권은 a = 2
**전력 항등식**: E·Ia = T·ω, 여기서 ω = 2π·N ÷ 60 [rad/s]
**전동기의 역기전력**: Eb = V − Ia·Ra, 그리고 N은 Eb ÷ Φ에 비례합니다

여기서 고정한 값: P = 4극, Z = 400개 전기자 도체, V = 220 V, Ra = 0.5 Ω.

부호 규약. 축을 돌려 주면 발전기입니다. E가 전기자 안에서 세워지고, 단자에서 E가 V보다 크며, 전력은 밖으로 나갑니다. 반대로 단자에 전압을 걸면 같은 기계가 전동기가 됩니다. 똑같은 E가 이제 전원을 거스르며, 이를 역기전력 Eb라 부릅니다. Eb는 V보다 작고, 그 차이 Ia·Ra가 전류를 밀어 넣으며, 전력은 안으로 들어옵니다. 두 식은 조금도 바뀌지 않습니다. 바뀌는 것은 Ia와 축 전력의 방향뿐입니다.

계산 예: Φ = 0.02 Wb, N = 1500 rpm, Ia = 10 A, 중권이므로 a = P = 4입니다. 그러면 E = 4 × 400 × 0.02 × 1500 ÷ (60 × 4) = 200.00 V, T = 4 × 400 × 0.02 × 10 ÷ (2π × 4) = 12.732 N·m입니다. ω = 2π × 1500 ÷ 60 = 157.08 rad/s이므로 E·Ia = 2000.00 W이고 T·ω = 2000.00 W입니다. 똑같은 2000 W를 두 번 센 것입니다.

파권으로 바꾸면 a가 4에서 2로 줄어듭니다. E는 400.00 V로, T는 25.465 N·m로 둘 다 두 배가 되고, 시안 경로는 네 개의 호에서 두 개로 다시 엮입니다. 정수 하나가 같은 철과 구리로 고전압 기계를 만들지 고토크 기계를 만들지를 정합니다.

220 V 전동기로 Ia = 10 A를 흘리면 Eb = 220 − 10 × 0.5 = 215.00 V이고, 이 자속에서는 1612.5 rpm에 해당합니다.

슬라이더를 움직여 보세요. Φ를 올리면 E가 커지고 T도 함께 커지며, 황색 면적은 둘보다 더 빠르게 커집니다. N을 올리면 E만 커지고 T는 그대로여서, 한 직사각형은 높고 좁아지고 다른 쪽은 낮고 넓어지지만 쓸고 있는 면적은 언제나 같습니다. Ia를 올리면 경로의 광택이 빨라집니다. 재미있는 것은 자속입니다. Φ는 E와 T를 모두 끌어올리지만, 전동기에서는 Φ가 약할수록 속도가 올라갑니다. 같은 Eb를 더 적은 자속으로 만들어야 하기 때문입니다.

2D: 전동기 특성입니다. 가로축은 전기자 전류입니다. 위쪽 밝은 중성색은 속도 N = (V − Ia·Ra) ÷ (kE·Φ)이며, 손실이 Ia·Ra뿐이므로 완만하게 처집니다. 아래쪽 중간 밝기 중성색은 토크 T = kT·Φ·Ia로, 원점에서 나가는 직선입니다. Ia = 10 A의 표시점은 기본 설정에서 1612.5 rpm과 12.732 N·m을 가리킵니다. Φ를 줄이면 속도 선이 올라가고 토크 선이 눕는 것을 볼 수 있습니다.`,
  },
  sliderParams: [
    { key: 'flux',    label: { en: 'Flux per Pole (Φ)',       ko: '극당 자속 (Φ)' },        unit: 'Wb',  min: 0.005, max: 0.04, step: 0.001, defaultValue: 0.02 },
    { key: 'speed',   label: { en: 'Speed (N)',               ko: '회전 속도 (N)' },        unit: 'rpm', min: 500,   max: 3000, step: 50,    defaultValue: 1500 },
    { key: 'ia',      label: { en: 'Armature Current (Ia)',   ko: '전기자 전류 (Ia)' },     unit: 'A',   min: 2,     max: 30,   step: 1,     defaultValue: 10 },
    { key: 'winding', label: { en: 'Winding (1 lap, 2 wave)', ko: '권선 (1 중권, 2 파권)' }, unit: '',    min: 1,     max: 2,    step: 1,     defaultValue: 1 },
  ],
  Scene: DCMachineScene,
  Scene2D: MotorCurves2DScene,
}

export default concept
