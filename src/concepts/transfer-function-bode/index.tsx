import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import { setLineOpacity, setOpacity, useSettledLabels, useStoryReveal } from '../../components/scene/storyReveal'
import { useLocaleStore, type Locale } from '../../lib/i18n'
import type { ConceptMeta } from '../../types/concept'

// Five decades of frequency, drawn on a log axis so a first-order lag becomes two straight lines.
const F_MIN = 0.01
const F_MAX = 1000
const LOG_MIN = Math.log10(F_MIN)
const LOG_SPAN = Math.log10(F_MAX) - LOG_MIN

const DB_TOP = 20
const DB_BOT = -100

// Plot box, in BodePlot's local coordinates: magnitude above, phase below, one shared x axis.
const MAG_BOT = 0.35
const MAG_H = 1.35
const MAG_PER_DB = MAG_H / (DB_TOP - DB_BOT)
const PH_TOP = -0.35
const PH_H = 1.1
const PH_PER_DEG = PH_H / 180

const PTS = 241
const CURVE_INIT: [number, number, number][] = Array.from({ length: PTS }, () => [0, 0, 0])
const SEG_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]
const P3_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
const P4_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]

const TICKS = [0.01, 0.1, 1, 10, 100, 1000]
const DB_TICKS = [0, -20, -40, -80]
const DEG_TICKS = [0, -45, -90, -180]

// Narration slots. The sliders already sweep everything this concept can show, so the story drives
// them and reveals the plot one layer at a time; BodePlot is shared by 2D and 3D, so one set covers
// both views.
const REV_PROBE = 0 // the probe dot, the readouts, and the in/out wave pair in 3D
const REV_CORNER = 1 // the corner vertical and its fc readout
const REV_ASY = 2 // the straight-line sketch
const REV_PHASE = 3 // the phase curve, its dot and the link down to it
const REV_COUNT = 4

function revealTargets(step: number, out: Float32Array) {
  out[REV_PROBE] = step >= 1 ? 1 : 0
  out[REV_CORNER] = step >= 2 ? 1 : 0
  out[REV_ASY] = step >= 4 ? 1 : 0
  out[REV_PHASE] = step >= 5 ? 1 : 0
}

const clamp = (x: number, lo: number, hi: number) => (x < lo ? lo : x > hi ? hi : x)
const clamp01 = (u: number) => (u < 0 ? 0 : u > 1 ? 1 : u)
/** 0 at F_MIN, 1 at F_MAX, on a logarithmic frequency axis. */
const fracOf = (f: number) => (Math.log10(f) - LOG_MIN) / LOG_SPAN

/** |G(jω)| for n identical first-order lags, without building an object — the sweeps call it per sample. */
function magAt(f: number, K: number, tau: number, n: number) {
  const wt = 2 * Math.PI * f * tau
  const m = Math.sqrt(1 + wt * wt)
  return K / (n === 2 ? m * m : m)
}
const dbAt = (f: number, K: number, tau: number, n: number) => 20 * Math.log10(magAt(f, K, tau, n))
/** ∠G(jω) in degrees, also allocation-free. */
const degAt = (f: number, tau: number, n: number) =>
  -n * Math.atan(2 * Math.PI * f * tau) * (180 / Math.PI)

const yDb = (db: number) => MAG_BOT + (clamp(db, DB_BOT, DB_TOP) - DB_BOT) * MAG_PER_DB
const yDeg = (d: number) => PH_TOP + clamp(d, -180, 0) * PH_PER_DEG

// Reused result object so solve() allocates nothing when called inside useFrame.
const OUT = { K: 0, tau: 0, n: 1, fc: 0, f: 0, mag: 0, db: 0, deg: 0, lag: 0, dbK: 0, slope: 0 }

// Pure physics of the first-order lag G(s) = K ÷ (1 + sτ), raised to n for n identical poles:
//   |G(jω)| = K ÷ sqrt(1 + (ωτ)²)^n, ∠G(jω) = −n·arctan(ωτ), corner ωc = 1 ÷ τ, fc = 1 ÷ (2πτ).
// Worked example, checked with node:
//   K = 1, τ = 10 ms -> fc = 15.9155 Hz; at fc the gain is −3.0103 dB and the phase −45.00°
//   one decade higher, f = 159.155 Hz -> gain −20.0432 dB, phase −84.29°
//   with n = 2 at fc -> gain −6.0206 dB, phase −90.00°; at 159.155 Hz -> −40.0864 dB, −168.58°
function solve({ K = 1, tau_ms = 10, poles = 1, fDec = 0 }: Record<string, number>) {
  const tau = tau_ms * 1e-3
  const n = poles >= 1.5 ? 2 : 1
  const fc = 1 / (2 * Math.PI * tau)
  const f = fc * Math.pow(10, fDec)
  OUT.K = K
  OUT.tau = tau
  OUT.n = n
  OUT.fc = fc
  OUT.f = f
  OUT.mag = magAt(f, K, tau, n)
  OUT.db = 20 * Math.log10(OUT.mag)
  OUT.deg = degAt(f, tau, n)
  OUT.lag = -OUT.deg / 360 / f // seconds the output trails the input at this frequency
  OUT.dbK = 20 * Math.log10(K)
  OUT.slope = -20 * n // dB per decade, far above the corner
  return OUT
}

function useSliderValues() {
  const valuesRef = useRef<Record<string, number>>(useSliderStore.getState().values)
  useEffect(
    () => useSliderStore.subscribe((s) => s.values, (v) => { valuesRef.current = v }),
    []
  )
  return valuesRef
}

const WAVE_WORDS: Record<Locale, [string, string]> = {
  en: ['in', 'out'],
  ko: ['입력', '출력'],
}

const fmtF = (f: number) =>
  f < 1 ? `${f.toFixed(3)} Hz` : f < 1000 ? `${f.toFixed(2)} Hz` : `${(f / 1e3).toFixed(2)} kHz`
const fmtT = (s: number) =>
  s < 1e-3 ? `${(s * 1e6).toFixed(0)} μs` : s < 1 ? `${(s * 1e3).toFixed(2)} ms` : `${s.toFixed(2)} s`
const tickLabel = (f: number) => (f >= 1000 ? `${f / 1000}k` : `${f}`)

// Labels read in the foreground color; nothing here is current or power, so nothing is tinted
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const axisClass = 'pointer-events-none whitespace-nowrap text-[11px] tabular-nums text-muted-foreground'

// ── The Bode plot itself, shared by both scenes ──
// Local coordinates: x runs 0…width along log f, magnitude sits above y = 0, phase below.
type PlotProps = {
  position: [number, number, number]
  width: number
  scale?: number
}

function BodePlot({ position, width, scale = 1 }: PlotProps) {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  const locale = useLocaleStore((s) => s.locale)

  const magBuf = useMemo(() => new Float32Array(PTS * 3), [])
  const phBuf = useMemo(() => new Float32Array(PTS * 3), [])
  const asyM = useMemo(() => new Float32Array(9), [])
  const asyP = useMemo(() => new Float32Array(12), [])
  const seg = useMemo(() => new Float32Array(6), [])

  const magAxis = useMemo<[number, number, number][]>(
    () => [[0, yDb(DB_TOP), 0], [0, yDb(DB_BOT), 0], [width, yDb(DB_BOT), 0]],
    [width]
  )
  const phAxis = useMemo<[number, number, number][]>(
    () => [[0, yDeg(0), 0], [0, yDeg(-180), 0], [width, yDeg(-180), 0]],
    [width]
  )
  const gridLines = useMemo<[number, number, number][][]>(
    () => [yDb(0), yDeg(-45), yDeg(-90)].map((y) => [[0, y, 0], [width, y, 0]]),
    [width]
  )

  const magRef = useRef<any>(null)
  const phRef = useRef<any>(null)
  const asyMRef = useRef<any>(null)
  const asyPRef = useRef<any>(null)
  const cornerRef = useRef<any>(null)
  const linkRef = useRef<any>(null)
  const magDot = useRef<THREE.Group>(null)
  const phDot = useRef<THREE.Group>(null)
  const tagRefs = useRef<(THREE.Group | null)[]>([]) // corner, gain readout, phase readout
  const textRefs = useRef<(HTMLSpanElement | null)[]>([]) // the above, then the bottom summary
  const labels = useSettledLabels(revealTargets, REV_COUNT, [locale])
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(() => {
    const s = solve(valuesRef.current)

    // The true curves: gain in dB and phase in degrees, sampled evenly in log f
    for (let i = 0; i < PTS; i++) {
      const frac = i / (PTS - 1)
      const f = Math.pow(10, LOG_MIN + frac * LOG_SPAN)
      const x = frac * width
      magBuf[i * 3] = x
      magBuf[i * 3 + 1] = yDb(dbAt(f, s.K, s.tau, s.n))
      magBuf[i * 3 + 2] = 0.01
      phBuf[i * 3] = x
      phBuf[i * 3 + 1] = yDeg(degAt(f, s.tau, s.n))
      phBuf[i * 3 + 2] = 0.01
    }
    magRef.current?.geometry.setPositions(magBuf)
    phRef.current?.geometry.setPositions(phBuf)

    const xc = clamp01(fracOf(s.fc)) * width

    // Magnitude asymptote: flat at 20 log K to the corner, then −20n dB per decade.
    // Stop it where it meets the axis floor so the straight segment keeps its true slope.
    const decToFloor = (s.dbK - DB_BOT) / (20 * s.n)
    const dec = Math.min(decToFloor, Math.log10(F_MAX / s.fc))
    const xEnd = clamp01(fracOf(s.fc * Math.pow(10, dec))) * width
    asyM[0] = 0; asyM[1] = yDb(s.dbK); asyM[2] = 0
    asyM[3] = xc; asyM[4] = yDb(s.dbK); asyM[5] = 0
    asyM[6] = xEnd; asyM[7] = yDb(s.dbK + s.slope * dec); asyM[8] = 0
    asyMRef.current?.geometry.setPositions(asyM)
    asyMRef.current?.computeLineDistances()

    // Phase asymptote: 0° below the corner, −90n° above it. The true curve crosses −45n° right at fc.
    const degFar = -90 * s.n
    asyP[0] = 0; asyP[1] = yDeg(0); asyP[2] = 0
    asyP[3] = xc; asyP[4] = yDeg(0); asyP[5] = 0
    asyP[6] = xc; asyP[7] = yDeg(degFar); asyP[8] = 0
    asyP[9] = width; asyP[10] = yDeg(degFar); asyP[11] = 0
    asyPRef.current?.geometry.setPositions(asyP)
    asyPRef.current?.computeLineDistances()

    // The corner, through both plots
    seg[0] = xc; seg[1] = yDeg(-180); seg[2] = 0
    seg[3] = xc; seg[4] = yDb(DB_TOP); seg[5] = 0
    cornerRef.current?.geometry.setPositions(seg)
    cornerRef.current?.computeLineDistances()

    // The probe: two dots and the link that ties this gain to this phase
    const xp = clamp01(fracOf(s.f)) * width
    const yg = yDb(s.db)
    const yp = yDeg(s.deg)
    seg[0] = xp; seg[1] = yp; seg[2] = 0
    seg[3] = xp; seg[4] = yg; seg[5] = 0
    linkRef.current?.geometry.setPositions(seg)
    linkRef.current?.computeLineDistances()
    magDot.current?.position.set(xp, yg, 0.03)
    phDot.current?.position.set(xp, yp, 0.03)

    tagRefs.current[0]?.position.set(xc, yDb(DB_TOP) + 0.2, 0)
    tagRefs.current[1]?.position.set(xp, yg + 0.22, 0)
    tagRefs.current[2]?.position.set(xp, yp - 0.22, 0)

    // One layer of the plot per step. The magnitude curve and both axes stay, so the lower plot
    // reads as the phase panel waiting to be filled rather than as empty space.
    const wProbe = reveal[REV_PROBE]
    const wPhase = reveal[REV_PHASE]
    setLineOpacity(cornerRef.current, reveal[REV_CORNER])
    setLineOpacity(asyMRef.current, reveal[REV_ASY])
    setLineOpacity(asyPRef.current, reveal[REV_ASY] * wPhase)
    setLineOpacity(phRef.current, wPhase)
    setLineOpacity(linkRef.current, wProbe * wPhase)
    magDot.current?.scale.setScalar(wProbe)
    phDot.current?.scale.setScalar(wProbe * wPhase)
    const fades = [reveal[REV_CORNER], wProbe, wProbe * wPhase, wProbe]
    fades.forEach((v, t) => setOpacity(textRefs.current[t], v))

    const w = labels.settled(valuesRef.current)
    if (w) {
      const texts = [
        `fc = ${fmtF(s.fc)} · ${s.slope} dB ÷ decade`,
        `f = ${fmtF(s.f)} · 20 log |G| = ${s.db.toFixed(2)} dB`,
        `∠G = ${s.deg.toFixed(1)}°`,
        `K = ${s.K.toFixed(2)} (${s.dbK.toFixed(2)} dB) · τ = ${fmtT(s.tau)} · n = ${s.n} · |G| = ${s.mag.toFixed(4)} · lag ${fmtT(s.lag)}`,
      ]
      for (let i = 0; i < texts.length; i++) {
        const el = textRefs.current[i]
        if (el) el.textContent = texts[i]
      }
    }
  })

  return (
    <group position={position} scale={scale}>
      <Line points={magAxis} lineWidth={1} color={c.guide} />
      <Line points={phAxis} lineWidth={1} color={c.guide} />
      {gridLines.map((pts, g) => (
        <Line key={g} points={pts} lineWidth={1} color={c.guide} />
      ))}

      {/* The straight-line sketch, against the true curve */}
      <Line ref={asyMRef} points={P3_INIT} lineWidth={1.5} color={c.guide} dashed dashSize={0.1} gapSize={0.07} />
      <Line ref={asyPRef} points={P4_INIT} lineWidth={1.5} color={c.guide} dashed dashSize={0.1} gapSize={0.07} />
      <Line ref={cornerRef} points={SEG_INIT} lineWidth={1.5} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      <Line ref={linkRef} points={SEG_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.05} gapSize={0.05} />

      <TrailLine ref={magRef} points={CURVE_INIT} lineWidth={3} color={c.voltage} fade={false} />
      <Line ref={phRef} points={CURVE_INIT} lineWidth={2.5} color={c.inductance} />

      <Marker ref={magDot} color={c.marker} size={0.07} />
      <Marker ref={phDot} color={c.marker} size={0.07} />

      {[0, 1, 2].map((t) => (
        <group key={t} ref={(g) => { tagRefs.current[t] = g }}>
          <Html center>
            <span ref={(el) => { textRefs.current[t] = el }} className={labelClass} />
          </Html>
        </group>
      ))}
      <Html position={[0, yDeg(-180) - 0.62, 0]}>
        <span ref={(el) => { textRefs.current[3] = el }} className={labelClass} />
      </Html>

      {TICKS.map((f) => (
        <Html key={f} position={[fracOf(f) * width, yDeg(-180) - 0.2, 0]} center>
          <span className={axisClass}>{tickLabel(f)}</span>
        </Html>
      ))}
      <Html position={[width + 0.3, yDeg(-180), 0]} center>
        <span className={axisClass}>f (Hz)</span>
      </Html>
      {DB_TICKS.map((db) => (
        <Html key={db} position={[-0.55, yDb(db), 0]}>
          <span className={`${axisClass} -translate-x-full`}>{`${db} dB`}</span>
        </Html>
      ))}
      {DEG_TICKS.map((d) => (
        <Html key={d} position={[-0.55, yDeg(d), 0]}>
          <span className={`${axisClass} -translate-x-full`}>{`${d}°`}</span>
        </Html>
      ))}
    </group>
  )
}

// ── The concrete layer: one input sine, one output sine, at the probe frequency ──
// Same axis for both, so the dB number is the shrink and the phase number is the shift.
const WAVE_PTS = 161
const WAVE_INIT: [number, number, number][] = Array.from({ length: WAVE_PTS }, () => [0, 0, 0])
const WAVE_CYCLES = 2 // cycles drawn across the width, whatever the real frequency is
const WAVE_SPEED = 1.6 // on-screen rad ÷ s, slowed from the real ω so the motion is readable
const A_IN = 0.3 // scene units for the input peak; the output is A_IN × |G|, kept honest

function WavePair({ position, width }: { position: [number, number, number]; width: number }) {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  const locale = useLocaleStore((s) => s.locale)

  const inBuf = useMemo(() => new Float32Array(WAVE_PTS * 3), [])
  const outBuf = useMemo(() => new Float32Array(WAVE_PTS * 3), [])
  const seg = useMemo(() => new Float32Array(6), [])
  const zeroAxis = useMemo<[number, number, number][]>(() => [[0, 0, 0], [width, 0, 0]], [width])
  const inLevel = useMemo<[number, number, number][]>(
    () => [[0, A_IN, 0], [width, A_IN, 0]],
    [width]
  )

  const inRef = useRef<any>(null)
  const outRef = useRef<any>(null)
  const levelRef = useRef<any>(null)
  const axisRef = useRef<any>(null)
  const inLevelRef = useRef<any>(null)
  const outTag = useRef<THREE.Group>(null)
  const textRefs = useRef<(HTMLSpanElement | null)[]>([]) // out level, summary, the "in" label
  const labels = useSettledLabels(revealTargets, REV_COUNT, [locale])
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(({ clock }) => {
    const s = solve(valuesRef.current)
    const phi = (s.deg * Math.PI) / 180
    const t = clock.getElapsedTime()
    const aOut = A_IN * s.mag

    for (let i = 0; i < WAVE_PTS; i++) {
      const u = i / (WAVE_PTS - 1)
      const x = u * width
      const ph = u * WAVE_CYCLES * 2 * Math.PI - WAVE_SPEED * t
      inBuf[i * 3] = x
      inBuf[i * 3 + 1] = A_IN * Math.sin(ph)
      inBuf[i * 3 + 2] = 0
      outBuf[i * 3] = x
      outBuf[i * 3 + 1] = aOut * Math.sin(ph + phi)
      outBuf[i * 3 + 2] = 0.01
    }
    inRef.current?.geometry.setPositions(inBuf)
    outRef.current?.geometry.setPositions(outBuf)

    seg[0] = 0; seg[1] = aOut; seg[2] = 0
    seg[3] = width; seg[4] = aOut; seg[5] = 0
    levelRef.current?.geometry.setPositions(seg)
    levelRef.current?.computeLineDistances()
    outTag.current?.position.set(width + 0.24, aOut, 0)

    // The concrete pair arrives with the probe it belongs to
    const wProbe = reveal[REV_PROBE]
    for (const l of [inRef, outRef, levelRef, axisRef, inLevelRef]) setLineOpacity(l.current, wProbe)
    for (const t of [0, 1, 2]) setOpacity(textRefs.current[t], wProbe)

    const w = labels.settled(valuesRef.current)
    if (w) {
      const [wIn, wOut] = WAVE_WORDS[locale]
      const texts = [
        wOut,
        locale === 'ko'
          ? `${wIn} sin(2πft), f = ${fmtF(s.f)} → ${wOut} ×${s.mag.toFixed(3)} (${s.db.toFixed(2)} dB), ${(-s.deg).toFixed(1)}° 늦음 (${fmtT(s.lag)})`
          : `${wIn} sin(2πft) at f = ${fmtF(s.f)} → ${wOut} ×${s.mag.toFixed(3)} (${s.db.toFixed(2)} dB), lagging ${(-s.deg).toFixed(1)}° (${fmtT(s.lag)})`,
      ]
      for (let i = 0; i < texts.length; i++) {
        const el = textRefs.current[i]
        if (el) el.textContent = texts[i]
      }
    }
  })

  return (
    <group position={position}>
      <Line ref={axisRef} points={zeroAxis} lineWidth={1} color={c.guide} />
      <Line ref={inLevelRef} points={inLevel} lineWidth={1} color={c.guide} dashed dashSize={0.06} gapSize={0.05} />
      <Line ref={levelRef} points={SEG_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.06} gapSize={0.05} />

      <Line ref={inRef} points={WAVE_INIT} lineWidth={2} color={c.voltage} />
      <TrailLine ref={outRef} points={WAVE_INIT} lineWidth={2.5} color={c.resistance} fade={false} />

      <Html position={[width + 0.24, A_IN, 0]}>
        <span ref={(el) => { textRefs.current[2] = el }} className={axisClass}>{WAVE_WORDS[locale][0]}</span>
      </Html>
      <group ref={outTag}>
        <Html>
          <span ref={(el) => { textRefs.current[0] = el }} className={axisClass} />
        </Html>
      </group>
      <Html position={[0, -0.78, 0]}>
        <span ref={(el) => { textRefs.current[1] = el }} className={labelClass} />
      </Html>
    </group>
  )
}

// ── 3D: the plots behind, the input and output sine waves on a plane nearer the camera ──
function BodeScene() {
  return (
    <>
      {/* Aimed right of centre so the plots and the wave pair clear the narration panel */}
      <PerspectiveCamera makeDefault position={[3, 2, 5.6]} fov={50} />
      <OrbitControls enablePan={false} target={[0.8, 0, 0]} />
      <BodePlot position={[-2.4, 0.75, -0.7]} width={7.2} scale={0.6} />
      <WavePair position={[-2.1, -1.0, 0.85]} width={3.9} />
    </>
  )
}

// ── 2D: the plots alone, full size — this concept is the plot ──
function Bode2DScene() {
  return <BodePlot position={[-3.1, 0.02, 0]} width={6.4} />
}

const concept: ConceptMeta = {
  id: 'transfer-function-bode',
  title: { en: 'Transfer Function & Bode Plot', ko: '전달함수와 보드선도' },
  subject: 'control',
  order: 1,
  cardPosition: [3.3, 1.2, -0.6],
  description: {
    en: `## Transfer Function and the Bode Plot

**A transfer function says what a system does to a sine wave at every frequency: how much it shrinks it, and how far it delays it.**

- Bright curve, upper plot: the true gain in decibels, 20 log |G(jω)|
- Lighter curve, lower plot: the true phase ∠G(jω) in degrees
- Dashed straight lines: the asymptotes. Flat at 20 log K, then falling at 20n dB per decade from the corner. That pair of lines is the whole hand sketch
- Dashed vertical: the corner frequency fc. The two dots sit at the probe frequency, and the faint link between them is the gain and the phase you are reading off
- In 3D, on the plane nearer the camera: the input sine and the output sine at that probe frequency. The dB number **is** that shrink, and the phase number **is** that shift

**First-order lag**: G(s) = K ÷ (1 + sτ), or K ÷ (1 + sτ)² for two identical poles
**Gain**: |G(jω)| = K ÷ sqrt(1 + (ωτ)²), reported as 20 log |G| in decibels
**Phase**: ∠G(jω) = −arctan(ωτ)
**Corner**: ωc = 1 ÷ τ, so fc = 1 ÷ (2πτ)

Three numbers are worth memorizing. At the corner the gain is exactly −3.01 dB, a factor of 1 ÷ sqrt(2), and the phase is exactly −45°. One decade above the corner the gain is −20.04 dB and the phase −84.3°. Far above it the straight line simply is the answer: every decade costs another 20 dB and the phase settles at −90°.

With K = 1 and τ = 10 ms: fc = 15.92 Hz, where the gain is −3.01 dB and the phase −45.0°. At 159.2 Hz, one decade higher, the gain is −20.04 dB and the phase −84.3°.

K lifts or drops the flat part by 20 log K dB and never touches the phase at all. τ slides the corner along the log axis: a slower system, with a larger τ, bends down at a lower frequency. Add the second pole and the slope doubles to −40 dB per decade while the phase heads for −180°, passing through exactly −90° at the corner. Sweep the probe frequency and watch the output wave shrink and slide behind the input.`,
    ko: `## 전달함수와 보드선도

**전달함수는 어떤 시스템이 주파수마다 사인파를 얼마나 줄이고 얼마나 늦추는지를 알려 줍니다.**

- 위쪽 그래프의 밝은 곡선: 실제 이득을 데시벨로 나타낸 20 log |G(jω)|
- 아래쪽 그래프의 연한 곡선: 실제 위상 ∠G(jω), 단위는 도
- 점선 직선: 점근선. 20 log K에서 평탄했다가 꺾은점부터 데케이드마다 20n dB씩 내려갑니다. 손으로 그리는 보드선도는 이 두 직선이 전부입니다
- 세로 점선: 꺾은점 주파수 fc. 두 점은 측정 주파수에 놓이고, 그 사이를 잇는 흐린 선이 지금 읽고 있는 이득과 위상입니다
- 3D에서 카메라에 가까운 평면: 그 측정 주파수에서의 입력 사인파와 출력 사인파. dB 값이 **곧** 저 줄어든 크기이고, 위상 값이 **곧** 저 밀린 양입니다

**1차 지연**: G(s) = K ÷ (1 + sτ), 동일한 극이 두 개면 K ÷ (1 + sτ)²
**이득**: |G(jω)| = K ÷ sqrt(1 + (ωτ)²), 데시벨로는 20 log |G|
**위상**: ∠G(jω) = −arctan(ωτ)
**꺾은점**: ωc = 1 ÷ τ, 즉 fc = 1 ÷ (2πτ)

외워 둘 숫자는 세 개입니다. 꺾은점에서 이득은 정확히 −3.01 dB, 즉 1 ÷ sqrt(2)배이고 위상은 정확히 −45°입니다. 꺾은점보다 한 데케이드 높은 곳에서는 이득이 −20.04 dB, 위상이 −84.3°입니다. 꺾은점에서 충분히 멀어지면 직선이 그대로 정답입니다. 데케이드마다 20 dB씩 더 떨어지고 위상은 −90°에 수렴합니다.

K = 1, τ = 10 ms이면 fc = 15.92 Hz이고 그곳의 이득은 −3.01 dB, 위상은 −45.0°입니다. 한 데케이드 높은 159.2 Hz에서는 이득 −20.04 dB, 위상 −84.3°입니다.

K는 평탄한 부분을 20 log K dB만큼 올리거나 내릴 뿐 위상에는 전혀 손대지 않습니다. τ는 꺾은점을 로그 축 위에서 밀고 다닙니다. τ가 클수록 느린 시스템이라 더 낮은 주파수에서 꺾입니다. 극을 하나 더 넣으면 기울기가 데케이드당 −40 dB로 두 배가 되고, 위상은 −180°를 향하면서 꺾은점에서 정확히 −90°를 지납니다. 측정 주파수를 훑으며 출력 파형이 작아지고 입력 뒤로 밀려나는 모습을 보세요.`,
  },
  story: [
    {
      set: { K: 1, tau_ms: 10, poles: 1, fDec: -1 },
      text: {
        en: 'Feed a system a sine wave and a sine wave comes back out — same frequency, but smaller. This curve is the whole answer to “how much smaller,” at every frequency at once. Flat and generous on the left, falling off a cliff on the right.',
        ko: '시스템에 사인파를 넣으면 사인파가 나옵니다. 주파수는 그대로인데 크기가 줄어 있습니다. 이 곡선은 “얼마나 줄어드는가”에 대한 답을, 모든 주파수에 대해 한꺼번에 그려 놓은 것입니다. 왼쪽은 평탄하고 후하지만, 오른쪽은 절벽처럼 떨어집니다.',
      },
    },
    {
      set: { K: 1, tau_ms: 10, poles: 1, fDec: -1 },
      text: {
        en: 'Pick a frequency low on the flat part, 1.59 Hz, and look at what comes out. −0.04 dB — the output sine is sitting right on top of the input. Nothing was taken away.',
        ko: '평탄한 쪽 낮은 주파수 하나, 1.59 Hz를 골라서 나오는 걸 봅니다. −0.04 dB — 출력 사인파가 입력 위에 거의 포개져 있습니다. 깎인 게 없습니다.',
      },
    },
    {
      set: { K: 1, tau_ms: 10, poles: 1, fDec: 0 },
      text: {
        en: 'Climb to 15.92 Hz and the curve starts bending. This is the corner, and it is not an arbitrary spot: `fc = 1 ÷ (2πτ)`. The gain here is −3.01 dB, which is exactly 1 ÷ sqrt(2) of what went in.',
        ko: '15.92 Hz까지 올라가면 곡선이 꺾이기 시작합니다. 여기가 꺾은점이고, 아무 데나 찍은 자리가 아닙니다. `fc = 1 ÷ (2πτ)`. 여기서 이득은 −3.01 dB, 들어간 것의 정확히 1 ÷ sqrt(2)입니다.',
      },
    },
    {
      set: { K: 1, tau_ms: 10, poles: 1, fDec: 1 },
      text: {
        en: 'One decade further out, at 159.15 Hz, it is −20.04 dB. The output wave has shrunk to a tenth of the input — the system simply cannot keep up any more.',
        ko: '거기서 한 데케이드 더 가면, 159.15 Hz에서 −20.04 dB입니다. 출력 파형이 입력의 10분의 1로 쪼그라들었습니다. 시스템이 더는 따라가지 못하는 겁니다.',
      },
    },
    {
      set: { K: 1, tau_ms: 10, poles: 1, fDec: 1 },
      text: {
        en: 'Here is why engineers sketch these by hand in seconds. Forget the curve: draw one flat line, draw one line falling 20 dB per decade, and let them meet at the corner. The true curve misses that sketch by 3 dB at one single point and hugs it everywhere else.',
        ko: '엔지니어들이 이걸 몇 초 만에 손으로 그리는 이유가 여기 있습니다. 곡선은 잊어버리세요. 평탄한 직선 하나, 데케이드마다 20 dB 내려가는 직선 하나를 긋고 꺾은점에서 만나게 하면 끝입니다. 실제 곡선은 딱 한 점에서 3 dB 어긋날 뿐, 나머지는 이 직선에 붙어 갑니다.',
      },
    },
    {
      set: { K: 1, tau_ms: 10, poles: 1, fDec: 0 },
      text: {
        en: 'But size is only half of it. The output also comes out late, and the lower plot is that delay. At the corner the lag is exactly −45.0° — the same place the gain was exactly −3.01 dB. One frequency, two signatures.',
        ko: '그런데 크기는 절반일 뿐입니다. 출력은 늦게도 나오고, 아래쪽 그래프가 그 늦음입니다. 꺾은점에서 지연은 정확히 −45.0°입니다. 이득이 정확히 −3.01 dB였던 바로 그 자리입니다. 주파수 하나에 지문이 두 개인 셈입니다.',
      },
    },
    {
      set: { K: 1, tau_ms: 10, poles: 1, fDec: 1 },
      text: {
        en: 'Push the frequency up and the lag deepens toward a limit: −84.3° here, creeping toward −90° and never past it. Watch the output wave slide right as it shrinks — that slide is this number.',
        ko: '주파수를 더 올리면 지연이 한계를 향해 깊어집니다. 여기서는 −84.3°, −90° 쪽으로 기어가지만 결코 넘지 않습니다. 출력 파형이 작아지면서 오른쪽으로 밀리는 게 보이죠. 그 밀린 양이 바로 이 숫자입니다.',
      },
    },
    {
      set: { K: 2, tau_ms: 10, poles: 1, fDec: 1 },
      text: {
        en: 'Now the two knobs. K is pure volume: double it and the entire magnitude plot lifts 6.02 dB, from −20.04 to −14.02 dB — and the phase does not move by a hair. Gain and timing are separate controls.',
        ko: '이제 손잡이 두 개입니다. K는 순수한 음량입니다. 두 배로 올리면 크기 그래프 전체가 6.02 dB 들리면서 −20.04 dB에서 −14.02 dB가 됩니다. 그리고 위상은 머리카락 한 올만큼도 움직이지 않습니다. 크기와 타이밍은 따로 노는 손잡이입니다.',
      },
    },
    {
      set: { K: 1, tau_ms: 100, poles: 1, fDec: 1 },
      text: {
        en: 'τ is the other one, and it moves the corner instead. Ten times slower a system, and fc drops ten times to 1.59 Hz: the whole shape slides left without changing shape at all. A slow system gives up on high frequencies sooner.',
        ko: 'τ는 다른 쪽이고, 이건 꺾은점을 옮깁니다. 시스템이 열 배 느려지면 fc도 열 배 낮은 1.59 Hz로 내려갑니다. 모양은 전혀 바뀌지 않고 전체가 왼쪽으로 미끄러집니다. 느린 시스템은 높은 주파수를 더 일찍 포기합니다.',
      },
    },
    {
      set: { K: 1, tau_ms: 10, poles: 2, fDec: 0 },
      text: {
        en: 'Stack a second identical pole and everything doubles: the slope goes to −40 dB per decade, the corner reads −6.02 dB instead of −3.01, and the phase passes through exactly −90.0° on its way to −180°. That last part is why control engineers care: what decides whether a feedback loop holds together or sings is `∠G = −n·arctan(ωτ)`',
        ko: '같은 극을 하나 더 쌓으면 전부 두 배가 됩니다. 기울기는 데케이드당 −40 dB가 되고, 꺾은점은 −3.01이 아니라 −6.02 dB를 가리키고, 위상은 −180°로 가는 길에 정확히 −90.0°를 지납니다. 마지막 부분이 제어공학자들이 이 그림을 보는 이유입니다. 피드백 루프가 버틸지 발진할지를 결정하는 건 바로 이것이거든요. `∠G = −n·arctan(ωτ)`',
      },
    },
  ],
  sliderParams: [
    { key: 'K',      label: { en: 'DC gain (K)',                                    ko: '직류 이득 (K)' },                             min: 0.2,  max: 2,   step: 0.05, defaultValue: 1 },
    { key: 'tau_ms', label: { en: 'Time constant (τ) — slides the corner',          ko: '시정수 (τ) — 꺾은점을 옮깁니다' },            unit: 'ms',  min: 5,    max: 200, step: 1,    defaultValue: 10 },
    { key: 'fDec',   label: { en: 'Probe frequency (f ÷ fc) — decades past fc',     ko: '측정 주파수 (f ÷ fc) — 꺾은점에서의 데케이드' }, unit: 'dec', min: -1.5, max: 1.5, step: 0.05, defaultValue: 0 },
    { key: 'poles',  label: { en: 'Identical poles (n)',                            ko: '동일한 극의 개수 (n)' },                      min: 1,    max: 2,   step: 1,    defaultValue: 1 },
  ],
  Scene: BodeScene,
  Scene2D: Bode2DScene,
}

export default concept
