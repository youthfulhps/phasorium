import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import type { Group } from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import { setLineOpacity, setOpacity, useSettledLabels, useStoryReveal } from '../../components/scene/storyReveal'
import { useLocaleStore, type Locale } from '../../lib/i18n'
import type { ConceptMeta } from '../../types/concept'

// Pure physics. A current wraps a magnetic field around itself; Biot–Savart integrates the
// contribution of every current element, and Ampère's law ∮B·dl = μ0·I_enclosed is the shortcut
// that solves the symmetric cases outright.
//   straight wire, distance r:   B = μ0·I ÷ (2π·r)       falls as 1 ÷ r
//   circular loop, at centre:    B = μ0·I ÷ (2·R)
//   long solenoid, inside:       B = μ0·n·I              uniform, n = turns per metre
// μ0 = 4π × 10⁻⁷ H ÷ m
//
// Worked example, verified with node -e:
//   I = 10 A, r = 5 cm from a straight wire
//     B = 4πe−7 × 10 ÷ (2π × 0.05) = 4.000e−5 T = 40.0 μT
//     Ampère check: B × 2π·r ÷ μ0 = 10.000 A, the enclosed current
//   loop of R = 5 cm carrying 10 A, at its centre
//     B = 4πe−7 × 10 ÷ (2 × 0.05)  = 1.2566e−4 T = 125.7 μT
//   solenoid of n = 1000 turns ÷ m carrying 2 A, inside
//     B = 4πe−7 × 1000 × 2         = 2.5133e−3 T = 2.513 mT
//
// One module-scope record holds the result so no frame allocates.
const MU0 = 4 * Math.PI * 1e-7
const N_TURNS_PER_M = 1000 // fixed winding density of the drawn solenoid

const S = { shape: 1, I: 10, r: 0.05, B: 4e-5, bWire: 4e-5, bLoop: 1.2566e-4, bSol: 1.2566e-2 }

function solve(v: Record<string, number>) {
  const sel = v.shape ?? 1
  const shape = sel >= 2.5 ? 3 : sel >= 1.5 ? 2 : 1
  const I = v.I_A ?? 10
  const r = (v.r_cm ?? 5) / 100
  S.shape = shape
  S.I = I
  S.r = r
  S.bWire = (MU0 * I) / (2 * Math.PI * r)
  S.bLoop = (MU0 * I) / (2 * r)
  S.bSol = MU0 * N_TURNS_PER_M * I
  S.B = shape === 1 ? S.bWire : shape === 2 ? S.bLoop : S.bSol
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

const fmtB = (t: number) =>
  t >= 1e-3 ? `${(t * 1e3).toFixed(3)} mT`
    : t >= 1e-6 ? `${(t * 1e6).toFixed(1)} μT`
      : `${(t * 1e9).toFixed(0)} nT`
const fmtLen = (m: number) => (m >= 1 ? `${m.toFixed(2)} m` : `${(m * 100).toFixed(1)} cm`)

// Labels read in the foreground color; only the two accents tint their text, and the only accent
// here is the current, so the current readout is cyan and everything else is plain foreground.
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const currentClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-current'
const axisClass = 'pointer-events-none whitespace-nowrap text-[11px] tabular-nums text-muted-foreground'

const SHAPE_NAMES: Record<Locale, [string, string, string]> = {
  en: ['Straight wire', 'Single loop', 'Solenoid'],
  ko: ['직선 도체', '원형 코일 1회', '솔레노이드'],
}
const SHAPE_FORMULA = ['B = μ0·I ÷ (2π·r)', 'B = μ0·I ÷ (2·R)', 'B = μ0·n·I']

// Narration slots. Nothing here scales the physics: the shape slider already swaps the whole
// conductor, so the story drives that and reveals the measuring apparatus around it.
const REV_FIELD = 0 // the field lines wrapped around the conductor
const REV_PROBE = 1 // where B is read: the probe, its arrow, the ruler and the two readouts
const REV_SQ = 2 // 2D only: the 1 ÷ r² curve a point charge would give instead
const REV_AMPERE = 3 // the Ampèrian loop drawn through the probe, for the straight wire
const REV_COUNT = 4

function revealTargets(step: number, out: Float32Array) {
  out[REV_FIELD] = step >= 1 ? 1 : 0
  out[REV_PROBE] = step >= 2 ? 1 : 0
  out[REV_SQ] = step >= 3 ? 1 : 0
  out[REV_AMPERE] = step >= 9 ? 1 : 0
}

// ── Shared geometry tables, built once ──
const N_PTS = 72
const UNIT_C = new Float32Array(N_PTS)
const UNIT_S = new Float32Array(N_PTS)
// A field line of an axisymmetric coil is a closed loop in the meridional half plane: out through
// the bore at a small radius, back along the outside at a large one. The shape is fixed, so it is
// tabulated once as a rounded rectangle (axial from UX, radial from UY) and only stretched.
const UX = new Float32Array(N_PTS)
const UY = new Float32Array(N_PTS)
for (let i = 0; i < N_PTS; i++) {
  const t = (i / (N_PTS - 1)) * 2 * Math.PI
  UNIT_C[i] = Math.cos(t)
  UNIT_S[i] = Math.sin(t)
  UX[i] = Math.sign(UNIT_C[i]) * Math.sqrt(Math.abs(UNIT_C[i]))
  UY[i] = Math.sign(UNIT_S[i]) * Math.sqrt(Math.abs(UNIT_S[i]))
}

const FIELD_LINES = 7
const FIELD_INIT: [number, number, number][] = Array.from(
  { length: N_PTS },
  (): [number, number, number] => [0, 0, 0]
)
// Inner lines are drawn thicker: that is the field being stronger close in, the 1 ÷ r falloff
// seen as line weight rather than as a number.
const FIELD_W = [3, 2, 1.3, 3, 2, 1.3, 2.5]
// Straight wire: three radii at two heights, plus the probe circle as line 6
const WIRE_R = [0.4, 0.75, 1.25, 0.4, 0.75, 1.25]
const WIRE_Y = [0.8, 0.8, 0.8, -0.8, -0.8, -0.8]
// Coil cases: seven meridional loops fanned around the axis, three bore sizes cycling
const BORE_IN = [0.3, 0.6, 0.85]
const BORE_OUT = [1.3, 1.65, 2.0]

const AXIS_Y = 1.7 // half length of the straight wire
const SOL_HALF = 0.9 // half length of the solenoid
const SOL_TURNS = 9
const CON_N = 241 // samples along the drawn conductor
const CON_INIT: [number, number, number][] = Array.from(
  { length: CON_N },
  (): [number, number, number] => [0, 0, 0]
)

/**
 * Scene radius for a radius given in cm, kept strictly proportional so the shrink is honest.
 * The constant is set by the widest thing drawn: at the slider's 20 cm the outermost field loop
 * reaches rScene(20) × BORE_OUT[2] = 1.0 × 2.0 = 2.0, which is the ±2 the default camera frames.
 */
const rScene = (cm: number) => 0.05 * cm

const ARROW_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
const RULER_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]
const HEAD = 0.13
const HEAD_W = 0.07

/** Writes a 5-point arrow (shaft, then the two head strokes) into 15 floats. Allocates nothing. */
function setArrow(
  buf: Float32Array,
  tx: number, ty: number, tz: number,
  dx: number, dy: number, dz: number,
  len: number
) {
  const m = Math.hypot(dx, dy, dz) || 1
  const ux = dx / m
  const uy = dy / m
  const uz = dz / m
  // A unit vector perpendicular to u, taken as u × ŷ and falling back to x̂ when u is along ŷ
  const pm = Math.hypot(uz, ux)
  const px = pm > 1e-6 ? -uz / pm : 1
  const pz = pm > 1e-6 ? ux / pm : 0
  const hx = tx + ux * len
  const hy = ty + uy * len
  const hz = tz + uz * len
  const bx = hx - ux * HEAD
  const by = hy - uy * HEAD
  const bz = hz - uz * HEAD
  buf[0] = tx; buf[1] = ty; buf[2] = tz
  buf[3] = hx; buf[4] = hy; buf[5] = hz
  buf[6] = bx + px * HEAD_W; buf[7] = by; buf[8] = bz + pz * HEAD_W
  buf[9] = hx; buf[10] = hy; buf[11] = hz
  buf[12] = bx - px * HEAD_W; buf[13] = by; buf[14] = bz - pz * HEAD_W
}

// The arrow stands for B measured against each case's own reference, so one shape's arrow never
// has to be ten times another's; the readout carries the honest number.
const B_REF = [4e-5, 1.2566e-4, 1.2566e-2]
const armOf = (B: number, shape: number) => {
  const a = (0.95 * B) / B_REF[shape - 1]
  return a < 0.12 ? 0.12 : a > 1.7 ? 1.7 : a
}

// ── 3D: the conductor and the field it wraps around itself ──
function AmpereFieldScene() {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  const locale = useLocaleStore((s) => s.locale)

  const conBuf = useMemo(() => new Float32Array(CON_N * 3), [])
  const fieldBufs = useMemo(
    () => Array.from({ length: FIELD_LINES }, () => new Float32Array(N_PTS * 3)),
    []
  )
  const bArrow = useMemo(() => new Float32Array(15), [])
  const iArrow = useMemo(() => new Float32Array(15), [])
  const ruler = useMemo(() => new Float32Array(6), [])

  const conRef = useRef<any>(null)
  const fieldRefs = useRef<any[]>([])
  const bArrowRef = useRef<any>(null)
  const iArrowRef = useRef<any>(null)
  const rulerRef = useRef<any>(null)
  const probeRef = useRef<Group>(null)
  const bTagRef = useRef<Group>(null)
  const iLabelRef = useRef<Group>(null)
  const bLabelRef = useRef<Group>(null)
  // 0 title, 1 B readout, 2 current readout, 3 radius readout, 4 the B letter on the arrow head
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const flow = useRef(1.5)
  const labels = useSettledLabels(revealTargets, REV_COUNT, [locale])
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(() => {
    const s = solve(valuesRef.current)
    const rs = rScene(s.r * 100)
    flow.current = 0.15 * s.I // sheen bands per second, so a larger current visibly runs faster

    // The conductor. Index 0 is the tail and the last index the head, so the sheen runs with the
    // current, and every field line below is drawn circulating the way that current demands.
    if (s.shape === 1) {
      for (let i = 0; i < CON_N; i++) {
        conBuf[i * 3] = 0
        conBuf[i * 3 + 1] = -AXIS_Y + (2 * AXIS_Y * i) / (CON_N - 1)
        conBuf[i * 3 + 2] = 0
      }
    } else if (s.shape === 2) {
      for (let i = 0; i < CON_N; i++) {
        const a = (i / (CON_N - 1)) * 2 * Math.PI
        conBuf[i * 3] = rs * Math.cos(a)
        conBuf[i * 3 + 1] = 0
        conBuf[i * 3 + 2] = -rs * Math.sin(a)
      }
    } else {
      for (let i = 0; i < CON_N; i++) {
        const u = i / (CON_N - 1)
        const a = u * SOL_TURNS * 2 * Math.PI
        conBuf[i * 3] = rs * Math.cos(a)
        conBuf[i * 3 + 1] = -SOL_HALF + 2 * SOL_HALF * u
        conBuf[i * 3 + 2] = -rs * Math.sin(a)
      }
    }
    conRef.current?.geometry.setPositions(conBuf)

    // The field lines
    if (s.shape === 1) {
      // Circles threaded on the wire. With I along +Y the circulation at +X is along −Z, which is
      // the right-hand rule: thumb along the current, fingers curling the way these lines run.
      for (let j = 0; j < 6; j++) {
        const buf = fieldBufs[j]
        const rad = WIRE_R[j]
        const y = WIRE_Y[j]
        for (let i = 0; i < N_PTS; i++) {
          buf[i * 3] = rad * UNIT_C[i]
          buf[i * 3 + 1] = y
          buf[i * 3 + 2] = -rad * UNIT_S[i]
        }
        fieldRefs.current[j]?.geometry.setPositions(buf)
      }
      const buf = fieldBufs[6]
      for (let i = 0; i < N_PTS; i++) {
        buf[i * 3] = rs * UNIT_C[i]
        buf[i * 3 + 1] = 0
        buf[i * 3 + 2] = -rs * UNIT_S[i]
      }
      fieldRefs.current[6]?.geometry.setPositions(buf)
    } else {
      // Coiled: closed meridional loops. For the single turn they are short and fat; stretched
      // along a solenoid their inner legs run parallel to the axis, which is the uniform bore.
      const half = s.shape === 2 ? 0.5 * rs + 0.35 : SOL_HALF + 0.35
      for (let j = 0; j < FIELD_LINES; j++) {
        const size = j % 3
        const phi = (j / FIELD_LINES) * 2 * Math.PI
        const cos = Math.cos(phi)
        const sin = Math.sin(phi)
        const rin = rs * BORE_IN[size]
        const rout = rs * BORE_OUT[size]
        const rm = (rin + rout) / 2
        const rh = (rout - rin) / 2
        const buf = fieldBufs[j]
        for (let i = 0; i < N_PTS; i++) {
          const rho = rm + rh * UY[i]
          buf[i * 3] = rho * cos
          buf[i * 3 + 1] = half * UX[i]
          buf[i * 3 + 2] = rho * sin
        }
        fieldRefs.current[j]?.geometry.setPositions(buf)
      }
    }

    // Where the field is read off, and how strong it is there
    const arm = armOf(s.B, s.shape)
    if (s.shape === 1) {
      probeRef.current?.position.set(rs, 0, 0)
      setArrow(bArrow, rs, 0, 0, 0, 0, -1, arm)
      bTagRef.current?.position.set(rs + 0.1, 0.34, -arm * 0.5)
      ruler[0] = 0; ruler[1] = 0; ruler[2] = 0
      ruler[3] = rs; ruler[4] = 0; ruler[5] = 0
    } else {
      probeRef.current?.position.set(0, 0, 0)
      setArrow(bArrow, 0, 0, 0, 0, 1, 0, arm)
      bTagRef.current?.position.set(0.34, arm + 0.14, 0)
      ruler[0] = 0; ruler[1] = 0; ruler[2] = 0
      ruler[3] = rs; ruler[4] = 0; ruler[5] = 0
    }
    bArrowRef.current?.geometry.setPositions(bArrow)
    rulerRef.current?.geometry.setPositions(ruler)
    rulerRef.current?.computeLineDistances()

    // The current arrow sits on the conductor itself, taken from two nearby samples of its buffer
    const k = Math.round(CON_N * 0.6)
    const k2 = k + 4
    setArrow(
      iArrow,
      conBuf[k * 3], conBuf[k * 3 + 1], conBuf[k * 3 + 2],
      conBuf[k2 * 3] - conBuf[k * 3],
      conBuf[k2 * 3 + 1] - conBuf[k * 3 + 1],
      conBuf[k2 * 3 + 2] - conBuf[k * 3 + 2],
      0.42
    )
    iArrowRef.current?.geometry.setPositions(iArrow)

    // The I and B letters ride on their own arrow heads (indices 3, 4, 5 of each buffer) with a
    // small fixed offset, so neither can drift away from the arrow it names.
    iLabelRef.current?.position.set(iArrow[3] + 0.19, iArrow[4] + 0.11, iArrow[5])
    bLabelRef.current?.position.set(bArrow[3] + 0.16, bArrow[4] + 0.14, bArrow[5])

    // The field arrives after the bare conductor, and the measuring apparatus after the field.
    // Line 6 is the Ampèrian loop through the probe for a straight wire, but one of the seven
    // ordinary meridional loops once the wire is coiled, so it follows whichever is on screen.
    const wField = reveal[REV_FIELD]
    const wProbe = reveal[REV_PROBE]
    for (let j = 0; j < 6; j++) setLineOpacity(fieldRefs.current[j], wField)
    setLineOpacity(fieldRefs.current[6], s.shape === 1 ? reveal[REV_AMPERE] : wField)
    setLineOpacity(bArrowRef.current, wProbe)
    setLineOpacity(rulerRef.current, wProbe)
    probeRef.current?.scale.setScalar(wProbe)
    setOpacity(textRefs.current[4], wProbe)
    setOpacity(textRefs.current[0], wProbe)
    setOpacity(textRefs.current[1], wProbe)

    const w = textRefs.current[3] ? labels.settled(valuesRef.current) : null
    if (w) {
      const idx = s.shape - 1
      if (textRefs.current[0]) {
        textRefs.current[0]!.textContent = `${SHAPE_NAMES[locale][idx]} · ${SHAPE_FORMULA[idx]}`
      }
      if (textRefs.current[1]) textRefs.current[1]!.textContent = `B = ${fmtB(s.B)}`
      if (textRefs.current[2]) textRefs.current[2]!.textContent = `I = ${s.I.toFixed(1)} A`
      textRefs.current[3]!.textContent =
        s.shape === 1 ? `r = ${fmtLen(s.r)}` : `R = ${fmtLen(s.r)}`
    }
  })

  return (
    <>
      {/* Aimed right of centre so the field and its readouts clear the narration panel */}
      <PerspectiveCamera makeDefault position={[3, 2, 5.4]} fov={50} />
      <OrbitControls enablePan={false} target={[0.75, 0, 0]} />

      {/* Background, lights, grid, and bloom come from the shared SceneStage */}
      {/* The field wrapped around the current; inner lines are thicker where the field is stronger */}
      {FIELD_W.map((w, j) => (
        <TrailLine
          key={j}
          ref={(l) => { fieldRefs.current[j] = l }}
          points={FIELD_INIT}
          lineWidth={w}
          color={c['b-field']}
          fade={false}
          flowSpeed={() => flow.current}
        />
      ))}

      {/* The conductor, carrying the current that makes all of it */}
      <TrailLine
        ref={conRef}
        points={CON_INIT}
        lineWidth={3}
        color={c.current}
        fade={false}
        flowSpeed={() => flow.current}
      />

      {/* The right-hand rule, drawn: thumb along I, fingers curling along B */}
      <Line ref={iArrowRef} points={ARROW_INIT} lineWidth={2.5} color={c.current} />
      <Line ref={bArrowRef} points={ARROW_INIT} lineWidth={2.5} color={c['b-field']} />

      {/* The distance being measured */}
      <Line
        ref={rulerRef}
        points={RULER_INIT}
        lineWidth={1}
        color={c.guide}
        dashed
        dashSize={0.07}
        gapSize={0.05}
      />

      <Marker ref={probeRef} color={c.marker} size={0.055} />

      <group ref={bTagRef}>
        <Html center>
          <span ref={(el) => { textRefs.current[1] = el }} className={labelClass} />
        </Html>
      </group>
      <Html position={[0, 1.95, 0]} center>
        <span ref={(el) => { textRefs.current[0] = el }} className={labelClass} />
      </Html>
      <Html position={[0, -1.95, 0]} center>
        <div className="pointer-events-none flex items-center gap-2">
          <span ref={(el) => { textRefs.current[2] = el }} className={currentClass} />
          <span className={axisClass}>·</span>
          <span ref={(el) => { textRefs.current[3] = el }} className={labelClass} />
        </div>
      </Html>
      <group ref={iLabelRef}>
        <Html center>
          <span className={currentClass}>I</span>
        </Html>
      </group>
      <group ref={bLabelRef}>
        <Html center>
          <span ref={(el) => { textRefs.current[4] = el }} className={labelClass}>B</span>
        </Html>
      </group>
    </>
  )
}

// ── 2D: how fast the field dies away, against the 1 ÷ r² the electric field would give ──
// Log axes on both sides, so 1 ÷ r is a straight line of slope −1 and 1 ÷ r² one of slope −2:
// the whole contrast becomes "one line is twice as steep as the other".
const X0 = -3.05
const PW = 6.1
const Y0 = -1.45
const PH = 2.6
const R_LOG_MIN = -2 // 1 cm
const R_LOG_SPAN = 2 // up to 1 m
const B_LOG_MIN = -7 // 0.1 μT
const B_LOG_SPAN = 4 // up to 1 mT

const clamp01 = (u: number) => (u < 0 ? 0 : u > 1 ? 1 : u)
const xOf = (r: number) => X0 + clamp01((Math.log10(r) - R_LOG_MIN) / R_LOG_SPAN) * PW
const yOf = (b: number) => Y0 + clamp01((Math.log10(b) - B_LOG_MIN) / B_LOG_SPAN) * PH

const PTS2 = 121
const CURVE_INIT: [number, number, number][] = Array.from(
  { length: PTS2 },
  (): [number, number, number] => [0, 0, 0]
)
const SEG_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]

const R_TICKS = [0.01, 0.03, 0.1, 0.3, 1]
const R_TICK_TEXT = ['1 cm', '3 cm', '10 cm', '30 cm', '1 m']
const B_TICKS = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
const B_TICK_TEXT = ['0.1 μT', '1 μT', '10 μT', '100 μT', '1 mT']

const CURVE_WORDS: Record<Locale, [string, string]> = {
  en: ['wire, B ∝ 1 ÷ r', 'point charge, E ∝ 1 ÷ r²'],
  ko: ['도선, B ∝ 1 ÷ r', '점전하, E ∝ 1 ÷ r²'],
}

function AmpereFalloff2DScene() {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  const locale = useLocaleStore((s) => s.locale)

  const bBuf = useMemo(() => new Float32Array(PTS2 * 3), [])
  const sqBuf = useMemo(() => new Float32Array(PTS2 * 3), [])
  const seg = useMemo(() => new Float32Array(6), [])

  const axes = useMemo<[number, number, number][]>(
    () => [[X0, Y0 + PH, 0], [X0, Y0, 0], [X0 + PW, Y0, 0]],
    []
  )
  const grid = useMemo<[number, number, number][][]>(
    () => B_TICKS.map((b) => [[X0, yOf(b), 0], [X0 + PW, yOf(b), 0]]),
    []
  )

  const bRef = useRef<any>(null)
  const sqRef = useRef<any>(null)
  const probeLineRef = useRef<any>(null)
  const dotRef = useRef<Group>(null)
  // 0 the 1 ÷ r curve tag, 1 the 1 ÷ r² curve tag, 2 the readout
  const tagRefs = useRef<(Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const tagTextRefs = useRef<(HTMLSpanElement | null)[]>([]) // the two curve names, faded with their curves
  const flow = useRef(1.5)
  const labels = useSettledLabels(revealTargets, REV_COUNT, [locale])
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(() => {
    const s = solve(valuesRef.current)
    flow.current = 0.15 * s.I
    const rp = s.r < 0.01 ? 0.01 : s.r > 1 ? 1 : s.r
    const bp = (MU0 * s.I) / (2 * Math.PI * rp)

    for (let i = 0; i < PTS2; i++) {
      const frac = i / (PTS2 - 1)
      const r = Math.pow(10, R_LOG_MIN + frac * R_LOG_SPAN)
      const x = X0 + frac * PW
      bBuf[i * 3] = x
      bBuf[i * 3 + 1] = yOf((MU0 * s.I) / (2 * Math.PI * r))
      bBuf[i * 3 + 2] = 0.02
      // The same value at the probe, but falling as 1 ÷ r²: the two curves cross there and the
      // steeper one is what a point charge's electric field would do instead.
      sqBuf[i * 3] = x
      sqBuf[i * 3 + 1] = yOf(bp * (rp / r) * (rp / r))
      sqBuf[i * 3 + 2] = 0.01
    }
    bRef.current?.geometry.setPositions(bBuf)
    sqRef.current?.geometry.setPositions(sqBuf)
    sqRef.current?.computeLineDistances()

    const xp = xOf(rp)
    const yp = yOf(bp)
    seg[0] = xp; seg[1] = Y0; seg[2] = 0
    seg[3] = xp; seg[4] = yp; seg[5] = 0
    probeLineRef.current?.geometry.setPositions(seg)
    probeLineRef.current?.computeLineDistances()
    dotRef.current?.position.set(xp, yp, 0.03)

    tagRefs.current[0]?.position.set(X0 + PW - 0.55, bBuf[(PTS2 - 1) * 3 + 1] + 0.26, 0)
    tagRefs.current[1]?.position.set(X0 + PW - 0.75, sqBuf[(PTS2 - 1) * 3 + 1] - 0.26, 0)
    tagRefs.current[2]?.position.set(xp, yp + 0.3, 0)

    const wField = reveal[REV_FIELD]
    const wProbe = reveal[REV_PROBE]
    const wSq = reveal[REV_SQ]
    setLineOpacity(bRef.current, wField)
    setOpacity(tagTextRefs.current[0], wField)
    setLineOpacity(sqRef.current, wSq)
    setOpacity(tagTextRefs.current[1], wSq)
    setLineOpacity(probeLineRef.current, wProbe)
    dotRef.current?.scale.setScalar(wProbe)
    setOpacity(textRefs.current[0], wProbe)

    const w = textRefs.current[0] ? labels.settled(valuesRef.current) : null
    if (w) {
      textRefs.current[0]!.textContent = `B = ${fmtB(bp)} · r = ${fmtLen(rp)}`
    }
  })

  return (
    <>
      <Line points={axes} lineWidth={1.5} color={c.guide} />
      {grid.map((g, i) => (
        <Line key={i} points={g} lineWidth={1} color={c.guide} dashed dashSize={0.06} gapSize={0.07} />
      ))}

      {/* The electric field of a point charge, scaled to agree at the probe: twice as steep */}
      <Line
        ref={sqRef}
        points={CURVE_INIT}
        lineWidth={1.8}
        color={c['e-field']}
        dashed
        dashSize={0.09}
        gapSize={0.07}
      />
      {/* The magnetic field of the straight wire */}
      <TrailLine
        ref={bRef}
        points={CURVE_INIT}
        lineWidth={2.8}
        color={c['b-field']}
        fade={false}
        flowSpeed={() => flow.current}
      />

      <Line
        ref={probeLineRef}
        points={SEG_INIT}
        lineWidth={1}
        color={c.guide}
        dashed
        dashSize={0.07}
        gapSize={0.05}
      />
      <Marker ref={dotRef} color={c.marker} size={0.06} />

      {R_TICKS.map((r, i) => (
        <Html key={r} position={[xOf(r), Y0 - 0.18, 0]} center>
          <span className={axisClass}>{R_TICK_TEXT[i]}</span>
        </Html>
      ))}
      {B_TICKS.map((b, i) => (
        <Html key={b} position={[X0 - 0.42, yOf(b), 0]} center>
          <span className={axisClass}>{B_TICK_TEXT[i]}</span>
        </Html>
      ))}

      <group ref={(g) => { tagRefs.current[0] = g }}>
        <Html center>
          <span ref={(el) => { tagTextRefs.current[0] = el }} className={labelClass}>{CURVE_WORDS[locale][0]}</span>
        </Html>
      </group>
      <group ref={(g) => { tagRefs.current[1] = g }}>
        <Html center>
          <span ref={(el) => { tagTextRefs.current[1] = el }} className={axisClass}>{CURVE_WORDS[locale][1]}</span>
        </Html>
      </group>
      <group ref={(g) => { tagRefs.current[2] = g }}>
        <Html center>
          <span ref={(el) => { textRefs.current[0] = el }} className={labelClass} />
        </Html>
      </group>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'ampere-biot-savart',
  title: { en: "Ampère's & Biot–Savart Law", ko: '앙페르 법칙과 비오-사바르 법칙' },
  subject: 'electromagnetics',
  order: 5,
  description: {
    en: `## The Magnetic Field a Current Makes

**A current wraps a magnetic field around itself, and coiling the wire gathers that field where you want it.**

- Cyan line: the conductor and the current in it, the sheen running the way the current flows
- Deep cool lines: the magnetic field B. They are thicker close in, because the field is stronger there
- The two arrows: the right-hand rule itself. Point your thumb along the cyan arrow, the current, and your fingers curl the way the field arrow turns
- Dashed segment: the distance r being measured, or the coil radius R
- The dot: where B is read off, at r from the wire or at the centre of the coil

**Ampère's law**: ∮B·dl = μ0·I_enclosed
**Straight wire**: B = μ0·I ÷ (2π·r)
**Loop, at its centre**: B = μ0·I ÷ (2·R)
**Solenoid, inside**: B = μ0·n·I, with n turns per metre
**μ0** = 4π × 10⁻⁷ H ÷ m

Note the exponent. The field of a wire falls as 1 ÷ r, not as 1 ÷ r². A point charge's electric field spreads over a sphere, so it dies as 1 ÷ r², but a current line only has a circle to spread around, so it dies as 1 ÷ r. Ampère's law says exactly that: the circle of circumference 2π·r carries all of the enclosed current, so B × 2π·r = μ0·I.

Worked example: I = 10 A at r = 5 cm from a straight wire gives B = 40.0 μT, and B × 2π·r ÷ μ0 returns the 10 A back. Bend that same wire into a loop of R = 5 cm and its centre sees 125.7 μT, three times more from the same current. Wind it as a solenoid of 1000 turns per metre carrying 2 A and the bore holds 2.513 mT, uniform all the way along.

The solenoid here is wound at a fixed 1000 turns per metre, so its field depends on the current alone. The field arrow is measured against each shape's own reference value, since the three cases differ by orders of magnitude; the readout is the true number.

Step the shape slider through the three cases and watch the field rearrange: circles threaded on a wire, then short fat loops gathered at a centre, then long loops whose inner legs run parallel, which is the uniform bore. Raise the current and every line's sheen runs faster while B climbs. Move r and the read-off point slides out, or the coil grows and its centre field weakens.

In 2D: B against r for the straight wire on log axes, where 1 ÷ r is a straight line. Behind it, dashed, is the 1 ÷ r² a point charge would give, scaled to agree at the probe so the only difference left is the slope, twice as steep.`,
    ko: `## 전류가 만드는 자기장

**전류는 자기장을 자기 몸 주위로 감아 돌립니다. 도선을 감으면 그 자기장을 원하는 곳에 모을 수 있습니다.**

- 하늘색 선: 도체와 그 안을 흐르는 전류. 광택이 전류가 흐르는 방향으로 흐릅니다
- 짙은 한색 선: 자기장 B. 가까운 쪽이 더 굵은데, 그곳의 자기장이 더 세기 때문입니다
- 두 개의 화살표: 오른나사 법칙 그 자체입니다. 엄지를 하늘색 화살표, 즉 전류 방향으로 두면 네 손가락이 자기장 화살표가 도는 방향으로 감깁니다
- 점선 선분: 지금 재고 있는 거리 r, 또는 코일 반지름 R
- 점: B를 읽는 지점. 직선 도체에서는 r만큼 떨어진 곳, 코일에서는 중심입니다

**앙페르 법칙**: ∮B·dl = μ0·I_enclosed
**직선 도체**: B = μ0·I ÷ (2π·r)
**원형 코일 중심**: B = μ0·I ÷ (2·R)
**솔레노이드 내부**: B = μ0·n·I, n은 1 m당 권수
**μ0** = 4π × 10⁻⁷ H ÷ m

지수를 잘 보세요. 도선의 자기장은 1 ÷ r로 줄어들며, 1 ÷ r²이 아닙니다. 점전하의 전기장은 구면으로 퍼지니 1 ÷ r²로 줄지만, 전류가 흐르는 선은 원을 따라 퍼질 수밖에 없으니 1 ÷ r로 줄어듭니다. 앙페르 법칙이 바로 그 말입니다. 둘레가 2π·r인 원이 그 안에 둘러싼 전류 전부를 담당하므로 B × 2π·r = μ0·I입니다.

계산 예: 직선 도체에 I = 10 A가 흐를 때 r = 5 cm 지점의 B는 40.0 μT이고, B × 2π·r ÷ μ0를 계산하면 10 A가 그대로 돌아옵니다. 같은 도선을 R = 5 cm인 원형 코일로 구부리면 중심에서 125.7 μT, 같은 전류로 세 배가 됩니다. 1 m당 1000회로 감은 솔레노이드에 2 A를 흘리면 내부는 2.513 mT로, 길이 방향 어디서나 균일합니다.

여기 그린 솔레노이드는 1 m당 1000회로 고정되어 있어서 자기장이 전류만으로 정해집니다. 자기장 화살표의 길이는 각 모양의 기준값에 대한 비율입니다. 세 경우의 크기가 자릿수 단위로 다르기 때문이며, 실제 값은 수치 표시가 알려 줍니다.

모양 슬라이더로 세 경우를 차례로 넘기며 자기장이 재배열되는 모습을 보세요. 도선에 꿰어진 원들, 중심에 모인 짧고 통통한 고리들, 그리고 안쪽 변이 나란히 달리는 긴 고리들 - 그 나란한 부분이 균일한 내부입니다. 전류를 올리면 모든 선의 광택이 빨라지고 B가 커집니다. r을 움직이면 읽는 지점이 밖으로 밀려나거나, 코일이 커지면서 중심 자기장이 약해집니다.

2D: 직선 도체의 B를 r에 대해 로그 축으로 그렸습니다. 로그 축에서 1 ÷ r은 직선입니다. 그 뒤의 점선은 점전하가 만들 1 ÷ r²로, 읽는 지점에서 값이 같아지도록 맞췄으니 남은 차이는 기울기뿐입니다. 두 배 더 급합니다.`,
  },
  story: [
    {
      set: { I_A: 10, r_cm: 5, shape: 1 },
      text: {
        en: 'A straight wire with 10 A running up it. That is all there is so far — charge moving in one direction.',
        ko: '10 A가 위로 흐르는 직선 도선 하나입니다. 지금 있는 건 이게 전부입니다. 전하가 한 방향으로 움직이고 있을 뿐입니다.',
      },
    },
    {
      set: { I_A: 10, r_cm: 5, shape: 1 },
      text: {
        en: 'And the moment it moves, a magnetic field wraps itself around the wire. Not lines going somewhere — rings, closed on themselves, with no start and no end.',
        ko: '그런데 전하가 움직이는 순간, 자기장이 도선을 감싸고 돕니다. 어디론가 뻗어 나가는 선이 아니라, 스스로 닫혀 있는 고리입니다. 시작도 끝도 없습니다.',
      },
    },
    {
      set: { I_A: 10, r_cm: 5, shape: 1 },
      text: {
        en: 'Measure it 5 cm out and you get 40.0 μT. The strength is set by how much current there is and how far out you stand: `B = μ0·I ÷ (2π·r)`.',
        ko: '5 cm 떨어진 곳에서 재면 40.0 μT가 나옵니다. 세기를 정하는 건 전류가 얼마나 흐르는지와, 얼마나 떨어져 서 있는지 둘뿐입니다. `B = μ0·I ÷ (2π·r)`',
      },
    },
    {
      set: { I_A: 10, r_cm: 20, shape: 1 },
      text: {
        en: 'Step back to 20 cm — four times as far — and it falls to 10.0 μT, exactly a quarter. Note that it is r, not r squared: a point charge\'s electric field would have fallen sixteen times over the same move.',
        ko: '20 cm로 물러나 봅니다. 네 배 멀어졌더니 10.0 μT, 정확히 4분의 1입니다. r의 제곱이 아니라 r이라는 점을 보세요. 점전하의 전기장이었다면 같은 거리에서 16분의 1로 떨어졌을 겁니다.',
      },
    },
    {
      set: { I_A: 30, r_cm: 20, shape: 1 },
      text: {
        en: 'Current is the other half. Triple it to 30 A and B triples with it, to 30.0 μT. The field is simply proportional to the current that makes it.',
        ko: '나머지 절반은 전류입니다. 30 A로 세 배 올리면 B도 그대로 세 배, 30.0 μT가 됩니다. 자기장은 그것을 만드는 전류에 그냥 비례합니다.',
      },
    },
    {
      set: { I_A: 10, r_cm: 5, shape: 2 },
      text: {
        en: 'So if the field always circles the wire, what happens when you bend the wire into a circle? Every part of the loop now pushes its field through the same hole. At the centre: 125.7 μT, π times what the straight wire gave at the same 5 cm. `B = μ0·I ÷ (2·R)`',
        ko: '자기장이 늘 도선을 감는다면, 도선 자체를 동그랗게 구부리면 어떻게 될까요? 이제 고리의 모든 부분이 같은 구멍으로 자기장을 밀어 넣습니다. 중심에서 125.7 μT, 같은 5 cm에서 직선 도선이 주던 값의 π배입니다. `B = μ0·I ÷ (2·R)`',
      },
    },
    {
      set: { I_A: 10, r_cm: 20, shape: 2 },
      text: {
        en: 'Make the loop bigger and you lose it again: at R = 20 cm the centre reads only 31.4 μT. The wire has moved further from the spot you are measuring, and that is the whole reason.',
        ko: '고리를 키우면 다시 잃습니다. R = 20 cm에서는 중심이 31.4 μT밖에 안 됩니다. 도선이 재는 지점에서 멀어졌기 때문이고, 이유는 그것 하나입니다.',
      },
    },
    {
      set: { I_A: 10, r_cm: 5, shape: 3 },
      text: {
        en: 'Now stack a thousand of those loops per metre. Each one adds its field to the next, and the bore fills with a field that runs straight down the axis: 12.566 mT, a hundred times the single turn. `B = μ0·n·I`',
        ko: '이제 그런 고리를 1 m에 천 개씩 쌓습니다. 하나하나가 옆의 것에 자기장을 보태고, 속은 축을 따라 곧게 달리는 자기장으로 가득 찹니다. 12.566 mT, 한 바퀴짜리의 백 배입니다. `B = μ0·n·I`',
      },
    },
    {
      set: { I_A: 10, r_cm: 20, shape: 3 },
      text: {
        en: 'And now the odd part. Widen the solenoid four times over and the number does not move: still 12.566 mT. Inside a long coil the radius is not in the formula at all — only turns per metre and current are.',
        ko: '그리고 여기가 이상한 부분입니다. 솔레노이드를 네 배로 굵게 키워도 숫자가 꿈쩍하지 않습니다. 여전히 12.566 mT입니다. 긴 코일 안쪽에서는 반지름이 식에 아예 없습니다. 1 m당 감은 횟수와 전류, 그 둘뿐입니다.',
      },
    },
    {
      set: { I_A: 10, r_cm: 5, shape: 1 },
      text: {
        en: 'Three shapes, three formulas — but one law underneath. Walk once around any closed loop, adding up B as you go, and the total only ever depends on the current threading that loop: `∮B·dl = μ0·I`. Here 40.0 μT around a 5 cm circle gives back exactly the 10 A inside it.',
        ko: '모양 셋, 공식 셋 — 그런데 바닥에 깔린 법칙은 하나입니다. 아무 닫힌 고리나 한 바퀴 돌면서 B를 더해 나가면, 그 합은 오직 그 고리를 꿰고 지나가는 전류에만 달려 있습니다. `∮B·dl = μ0·I`. 여기서도 5 cm 원을 따라 돈 40.0 μT가 그 안의 10 A를 정확히 되돌려 줍니다.',
      },
    },
  ],
  sliderParams: [
    { key: 'I_A',   label: { en: 'Current (I)',                                               ko: '전류 (I)' },                                        unit: 'A',  min: 1, max: 30, step: 0.5, defaultValue: 10 },
    { key: 'r_cm',  label: { en: 'Distance (r), or coil radius (R)',                          ko: '거리 (r), 또는 코일 반지름 (R)' },                   unit: 'cm', min: 1, max: 20, step: 0.5, defaultValue: 5 },
    { key: 'shape', label: { en: 'Conductor shape — 1 straight wire, 2 loop, 3 solenoid',     ko: '도체 모양 — 1 직선 도체, 2 원형 코일, 3 솔레노이드' },             min: 1, max: 3,  step: 1,   defaultValue: 1 },
  ],
  Scene: AmpereFieldScene,
  Scene2D: AmpereFalloff2DScene,
}

export default concept
