import { useRef, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import { setLineOpacity, setOpacity, useSettledLabels, useStoryReveal } from '../../components/scene/storyReveal'
import type { ConceptMeta } from '../../types/concept'

const VM = 10 // source peak voltage [V]
const F_MIN = 10 // frequency axis start [Hz]
const F_MAX = 2000 // frequency axis end [Hz]
const LOG_MIN = Math.log10(F_MIN)
const LOG_SPAN = Math.log10(F_MAX) - LOG_MIN
const R_MIN = 10
const R_MAX = 500

/** Peak current at one frequency, without building an object — the sweeps call this once per sample. */
function currentAt(f: number, R: number, L: number, C: number) {
  const w = 2 * Math.PI * f
  return VM / Math.hypot(R, w * L - 1 / (w * C))
}

/** 0 at F_MIN, 1 at F_MAX, on a logarithmic frequency axis. */
const fracOf = (f: number) => (Math.log10(f) - LOG_MIN) / LOG_SPAN
const clamp01 = (u: number) => (u < 0 ? 0 : u > 1 ? 1 : u)

// Pure physics of the series RLC at the resonant point: XL = ωL and XC = 1 ÷ (ωC) cancel at
// ω0 = 1 ÷ sqrt(LC), leaving Z = R, so I peaks at Vm ÷ R and VL = VC = Q·Vm.
// Worked example: R = 50 Ω, L = 100 mH, C = 10 μF, Vm = 10 V ->
// ω0 = 1000 rad/s, f0 = 159.15 Hz, Q = 2.00, BW = 79.58 Hz, half-power at 124.26 and 203.84 Hz,
// at resonance Z = R = 50 Ω, I = 0.200 A, VL = VC = 20.0 V (= Q x Vm).
function solve({ R = 50, L_mH = 100, C_uF = 10, freq = 100 }: Record<string, number>) {
  const L = L_mH * 1e-3
  const C = C_uF * 1e-6
  const w0 = 1 / Math.sqrt(L * C)
  const f0 = w0 / (2 * Math.PI)
  const Q = (w0 * L) / R
  const BW = f0 / Q
  const root = Math.sqrt(1 + 1 / (4 * Q * Q)) // half-power frequencies are f0·(root ∓ 1 ÷ 2Q)
  const w = 2 * Math.PI * freq
  const I = currentAt(freq, R, L, C)
  const Ipeak = VM / R
  return {
    R, L, C, freq, w0, f0, Q, BW, I, Ipeak,
    f1: f0 * (root - 1 / (2 * Q)),
    f2: f0 * (root + 1 / (2 * Q)),
    norm: I / Ipeak, // = R ÷ Z, 1 at resonance and 0.707 at the band edges
    XL: w * L,
    XC: 1 / (w * C),
    Z0: Math.sqrt(L / C), // XL = XC = Z0 at resonance
    Vmag: Q * VM, // VL = VC at resonance
  }
}

function useSliderValues() {
  const valuesRef = useRef<Record<string, number>>(useSliderStore.getState().values)
  useEffect(
    () => useSliderStore.subscribe((s) => s.values, (v) => { valuesRef.current = v }),
    []
  )
  return valuesRef
}

const fmtF = (f: number) => (f < 1000 ? `${f.toFixed(1)} Hz` : `${(f / 1e3).toFixed(2)} kHz`)
const fmtI = (a: number) => (a < 1 ? `${(a * 1e3).toFixed(1)} mA` : `${a.toFixed(3)} A`)
const fmtR = (x: number) => `${x.toFixed(x < 100 ? 1 : 0)} Ω`
// Labels read in the foreground color; only accent quantities (current) tint their text
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const axisClass = 'pointer-events-none whitespace-nowrap text-[11px] tabular-nums text-muted-foreground'

// Narration slots. Nothing here scales the physics: a frequency sweep has no part that can be
// taken out of the circuit, so the story reveals the sweep one layer at a time instead.
const REV_CROSS = 0 // the XL = XC crossing and the f0 it marks
const REV_RIDGE = 1 // the learner's own response curve and the dot riding it
const REV_FAMILY = 2 // the seven reference curves behind it, and the crest tying their peaks together
const REV_BAND = 3 // the half-power level, the bandwidth bar and the Q readout
const REV_VMAG = 4 // the voltage magnification line
const REV_COUNT = 5

function revealTargets(step: number, out: Float32Array) {
  out[REV_CROSS] = step >= 1 ? 1 : 0
  out[REV_RIDGE] = step >= 2 ? 1 : 0
  out[REV_FAMILY] = step >= 4 ? 1 : 0
  out[REV_BAND] = step >= 6 ? 1 : 0
  out[REV_VMAG] = step >= 9 ? 1 : 0
}

/** What resonance actually buys you, in one line: Z collapses to R and L and C swing Q·Vm. */
const magnifyLine = (s: ReturnType<typeof solve>) =>
  `f0: Z = R = ${fmtR(s.R)} · I = ${fmtI(s.Ipeak)} · VL = VC = ${s.Vmag.toFixed(1)} V`

const SEG_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]

// ── 3D: the resonance ridge — one response curve per resistance, stacked along Z ──
// Front of the ridge is small R (sharp), back is large R (broad); the crest stays at f0 because
// only L and C set the resonant frequency. In front of it, XL and XC cross at that same f0.
const X3 = -2.6
const X3_SPAN = 5.2
const Y3 = -1.0 // the frequency axis floor, well above the scene grid at y = −2.2
const RIDGE_H = 1.5
const Z_FRONT = 0
const Z_BACK = -2.4
const REACT_Z = 1.4 // the XL / XC plane sits nearest the camera
// XL and XC are drawn in decades, against a frequency axis that is already logarithmic, so the pair
// is two straight lines crossing in an X. A linear scale needs a clamp at both ends, and a clamped
// tail is a flat line that perspective tilts — which reads as "XL falls again", the one thing this
// view must not say. REACT_CLIP is wide enough that nothing in the story reaches it.
const REACT_MID = 0.62 // height of the XL = XC crossing above the frequency axis
const REACT_DECADE = 0.42 // scene units per decade of reactance
const REACT_CLIP = 1.45
const GHOST_R = [10, 20, 40, 80, 160, 320, 500] // the resistances drawn as faint reference curves
const RIDGE_PTS = 121
const REACT_PTS = 81
const RIDGE_INIT: [number, number, number][] = Array.from({ length: RIDGE_PTS }, () => [0, 0, 0])
const REACT_INIT: [number, number, number][] = Array.from({ length: REACT_PTS }, () => [0, 0, 0])
const FLOOR_AXIS: [number, number, number][] = [[X3, Y3, Z_FRONT], [X3 + X3_SPAN, Y3, Z_FRONT]]
const REACT_AXIS: [number, number, number][] = [[X3, Y3, REACT_Z], [X3 + X3_SPAN, Y3, REACT_Z]]

const x3Of = (f: number) => X3 + clamp01(fracOf(f)) * X3_SPAN
/** How far XL ÷ Z0 is from 1, in decades, clipped so a far-off f0 still leaves the pair in frame. */
const decadesOf = (f: number, f0: number) => {
  const d = Math.log10(f / f0)
  return d > REACT_CLIP ? REACT_CLIP : d < -REACT_CLIP ? -REACT_CLIP : d
}
/** Where a given resistance sits on the depth axis: small R at the front, large R at the back. */
const z3Of = (R: number) =>
  Z_FRONT + clamp01(Math.log10(R / R_MIN) / Math.log10(R_MAX / R_MIN)) * (Z_BACK - Z_FRONT)

function ResonanceRidgeScene() {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  const ghostBufs = useMemo(() => GHOST_R.map(() => new Float32Array(RIDGE_PTS * 3)), [])
  const ridgeBuf = useMemo(() => new Float32Array(RIDGE_PTS * 3), [])
  const xlBuf = useMemo(() => new Float32Array(REACT_PTS * 3), [])
  const xcBuf = useMemo(() => new Float32Array(REACT_PTS * 3), [])
  const segBuf = useMemo(() => new Float32Array(6), [])
  const ghostRefs = useRef<any[]>([])
  const ridgeRef = useRef<any>(null)
  const xlRef = useRef<any>(null)
  const xcRef = useRef<any>(null)
  const crestRef = useRef<any>(null)
  const meridianRef = useRef<any>(null)
  const opDot = useRef<THREE.Group>(null)
  const crossDot = useRef<THREE.Group>(null)
  const tagRefs = useRef<(THREE.Group | null)[]>([]) // crossing, operating point, Q summary, XL, XC
  const textRefs = useRef<(HTMLSpanElement | null)[]>([]) // those five, then the magnification line
  const flow = useRef(0)
  const labels = useSettledLabels(revealTargets, REV_COUNT)
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(() => {
    const s = solve(valuesRef.current)

    // Reference curves: the same L and C at seven fixed resistances, so the family is visible at once
    for (let g = 0; g < GHOST_R.length; g++) {
      const buf = ghostBufs[g]
      const zg = z3Of(GHOST_R[g])
      for (let i = 0; i < RIDGE_PTS; i++) {
        const frac = i / (RIDGE_PTS - 1)
        const f = Math.pow(10, LOG_MIN + frac * LOG_SPAN)
        buf[i * 3] = X3 + frac * X3_SPAN
        buf[i * 3 + 1] = Y3 + (currentAt(f, GHOST_R[g], s.L, s.C) / (VM / GHOST_R[g])) * RIDGE_H
        buf[i * 3 + 2] = zg
      }
      ghostRefs.current[g]?.geometry.setPositions(buf)
    }

    // The live curve: the learner's own R, riding forward along Z as R drops and sharpening with Q
    const zLive = z3Of(s.R)
    for (let i = 0; i < RIDGE_PTS; i++) {
      const frac = i / (RIDGE_PTS - 1)
      const f = Math.pow(10, LOG_MIN + frac * LOG_SPAN)
      ridgeBuf[i * 3] = X3 + frac * X3_SPAN
      ridgeBuf[i * 3 + 1] = Y3 + (currentAt(f, s.R, s.L, s.C) / s.Ipeak) * RIDGE_H
      ridgeBuf[i * 3 + 2] = zLive
    }
    ridgeRef.current?.geometry.setPositions(ridgeBuf)
    flow.current = 4 * s.norm // the sheen runs fastest where the current is largest

    // XL and XC, both divided by Z0, so they cross at exactly 1 — which is 0 decades, the middle
    for (let i = 0; i < REACT_PTS; i++) {
      const frac = i / (REACT_PTS - 1)
      const f = Math.pow(10, LOG_MIN + frac * LOG_SPAN)
      const x = X3 + frac * X3_SPAN
      const d = decadesOf(f, s.f0) // XL ÷ Z0 in decades; XC ÷ Z0 is its mirror
      xlBuf[i * 3] = x
      xlBuf[i * 3 + 1] = Y3 + REACT_MID + d * REACT_DECADE
      xlBuf[i * 3 + 2] = REACT_Z
      xcBuf[i * 3] = x
      xcBuf[i * 3 + 1] = Y3 + REACT_MID - d * REACT_DECADE
      xcBuf[i * 3 + 2] = REACT_Z
    }
    xlRef.current?.geometry.setPositions(xlBuf)
    xcRef.current?.geometry.setPositions(xcBuf)

    const xf0 = x3Of(s.f0)
    // Crest line: every curve in the family peaks at the same f0, whatever R is
    segBuf[0] = xf0; segBuf[1] = Y3 + RIDGE_H; segBuf[2] = Z_FRONT
    segBuf[3] = xf0; segBuf[4] = Y3 + RIDGE_H; segBuf[5] = Z_BACK
    crestRef.current?.geometry.setPositions(segBuf)
    crestRef.current?.computeLineDistances()
    // Floor meridian: ties the XL = XC crossing to the crest behind it
    segBuf[0] = xf0; segBuf[1] = Y3; segBuf[2] = REACT_Z
    segBuf[3] = xf0; segBuf[4] = Y3; segBuf[5] = Z_BACK
    meridianRef.current?.geometry.setPositions(segBuf)
    meridianRef.current?.computeLineDistances()

    const xOp = x3Of(s.freq)
    opDot.current?.position.set(xOp, Y3 + s.norm * RIDGE_H, zLive)
    crossDot.current?.position.set(xf0, Y3 + REACT_MID, REACT_Z)
    tagRefs.current[0]?.position.set(xf0, Y3 + REACT_MID + 0.3, REACT_Z)
    tagRefs.current[1]?.position.set(xOp, Y3 + s.norm * RIDGE_H + 0.28, zLive)
    tagRefs.current[2]?.position.set(X3 + X3_SPAN * 0.5, Y3 - 0.3, Z_BACK)
    const dHigh = decadesOf(F_MAX, s.f0) * REACT_DECADE
    const dLow = decadesOf(F_MIN, s.f0) * REACT_DECADE
    tagRefs.current[3]?.position.set(X3 + X3_SPAN - 0.15, Y3 + REACT_MID + dHigh + 0.2, REACT_Z)
    tagRefs.current[4]?.position.set(X3 + 0.15, Y3 + REACT_MID - dLow + 0.2, REACT_Z)

    // Each layer of the sweep fades in on the step that first talks about it
    const wCross = reveal[REV_CROSS]
    const wRidge = reveal[REV_RIDGE]
    const wFam = reveal[REV_FAMILY]
    setLineOpacity(meridianRef.current, wCross)
    crossDot.current?.scale.setScalar(wCross)
    setLineOpacity(ridgeRef.current, wRidge)
    opDot.current?.scale.setScalar(wRidge)
    setLineOpacity(crestRef.current, wFam)
    for (const g of ghostRefs.current) setLineOpacity(g, wFam)
    const fades = [wCross, wRidge, reveal[REV_BAND], 1, 1, reveal[REV_VMAG]]
    fades.forEach((v, t) => setOpacity(textRefs.current[t], v))

    const w = textRefs.current[5] ? labels.settled(valuesRef.current) : null
    if (w) {
      const texts = [
        `f0 ${fmtF(s.f0)} · XL = XC = ${fmtR(s.Z0)}`,
        `f ${fmtF(s.freq)} · I ${fmtI(s.I)}`,
        `R ${fmtR(s.R)} · Q ${s.Q.toFixed(2)} · BW ${fmtF(s.BW)}`,
        'XL',
        'XC',
        magnifyLine(s),
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  return (
    <>
      {/* Backed off and aimed right of centre: the ridge is 5.2 units wide and its high end runs
          under the narration panel at the default framing */}
      <PerspectiveCamera makeDefault position={[3.2, 2.2, 6.2]} fov={50} />
      <OrbitControls enablePan={false} target={[0.9, 0, -0.4]} />

      <Line points={FLOOR_AXIS} lineWidth={1} color={c.guide} />
      <Line points={REACT_AXIS} lineWidth={1} color={c.guide} />
      <Line ref={meridianRef} points={SEG_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      <Line ref={crestRef} points={SEG_INIT} lineWidth={1.5} color={c.guide} dashed dashSize={0.1} gapSize={0.07} />

      {/* The family of responses: same L and C, seven resistances, receding into depth */}
      {GHOST_R.map((Rg, g) => (
        <Line
          key={Rg}
          ref={(l) => { ghostRefs.current[g] = l }}
          points={RIDGE_INIT}
          lineWidth={1}
          color={c.guide}
        />
      ))}
      {/* The learner's own curve */}
      <TrailLine
        ref={ridgeRef}
        points={RIDGE_INIT}
        lineWidth={3.5}
        color={c.current}
        fade={false}
        flowSpeed={() => flow.current}
      />

      {/* XL rising and XC falling; where they cross, the circuit is purely resistive */}
      <Line ref={xlRef} points={REACT_INIT} lineWidth={2} color={c.inductance} />
      <Line ref={xcRef} points={REACT_INIT} lineWidth={2} color={c.capacitance} />

      <Marker ref={crossDot} color={c.marker} size={0.06} />
      <Marker ref={opDot} color={c.marker} size={0.07} />

      <Html position={[X3 + X3_SPAN * 0.2, Y3 + RIDGE_H + 0.5, Z_BACK]} center>
        <span ref={(el) => { textRefs.current[5] = el }} className={labelClass} />
      </Html>

      {[0, 1, 2, 3, 4].map((t) => (
        <group key={t} ref={(g) => { tagRefs.current[t] = g }}>
          <Html center>
            <span
              ref={(el) => { textRefs.current[t] = el }}
              className={labelClass}
              style={t === 1 ? { color: c.current } : undefined}
            />
          </Html>
        </group>
      ))}
    </>
  )
}

// ── 2D: the frequency response curve itself, normalized to its own peak ──
// I ÷ Imax = R ÷ Z, so the peak is always 1 and the only thing that changes shape is Q.
const X0 = -3.3
const X_SPAN = 6.6
const Y0 = -1.45
const PLOT_H = 2.75
const HALF_POWER = Math.SQRT1_2 // 0.707 of the peak current, half the power
const CURVE_PTS = 201
const CURVE_INIT: [number, number, number][] = Array.from({ length: CURVE_PTS }, () => [0, 0, 0])
const BASE_AXIS: [number, number, number][] = [[X0, Y0, 0], [X0 + X_SPAN, Y0, 0]]
const TICKS = [10, 100, 1000, 2000]

const xOf = (f: number) => X0 + clamp01(fracOf(f)) * X_SPAN
const Y_HALF = Y0 + HALF_POWER * PLOT_H

function ResponseCurve2DScene() {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  const curveBuf = useMemo(() => new Float32Array(CURVE_PTS * 3), [])
  const segBuf = useMemo(() => new Float32Array(6), [])
  const curveRef = useRef<any>(null)
  const f0Ref = useRef<any>(null)
  const bandRef = useRef<any>(null)
  const halfRef = useRef<any>(null)
  const opDot = useRef<THREE.Group>(null)
  const hp1 = useRef<THREE.Group>(null)
  const hp2 = useRef<THREE.Group>(null)
  const tagRefs = useRef<(THREE.Group | null)[]>([]) // f0, bandwidth, operating point
  const textRefs = useRef<(HTMLSpanElement | null)[]>([]) // those three, the magnification line, the 0.707 tick
  const flow = useRef(0)
  const labels = useSettledLabels(revealTargets, REV_COUNT)
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(() => {
    const s = solve(valuesRef.current)

    for (let i = 0; i < CURVE_PTS; i++) {
      const frac = i / (CURVE_PTS - 1)
      const f = Math.pow(10, LOG_MIN + frac * LOG_SPAN)
      curveBuf[i * 3] = X0 + frac * X_SPAN
      curveBuf[i * 3 + 1] = Y0 + (currentAt(f, s.R, s.L, s.C) / s.Ipeak) * PLOT_H
      curveBuf[i * 3 + 2] = 0
    }
    curveRef.current?.geometry.setPositions(curveBuf)
    flow.current = 4 * s.norm

    // Vertical line at f0, where the peak always lands
    const xf0 = xOf(s.f0)
    segBuf[0] = xf0; segBuf[1] = Y0; segBuf[2] = 0
    segBuf[3] = xf0; segBuf[4] = Y0 + PLOT_H + 0.15; segBuf[5] = 0
    f0Ref.current?.geometry.setPositions(segBuf)
    f0Ref.current?.computeLineDistances()

    // Half-power level across the whole plot, and the band it cuts out of the curve
    segBuf[0] = X0; segBuf[1] = Y_HALF; segBuf[2] = 0
    segBuf[3] = X0 + X_SPAN; segBuf[4] = Y_HALF; segBuf[5] = 0
    halfRef.current?.geometry.setPositions(segBuf)
    halfRef.current?.computeLineDistances()

    const x1 = xOf(s.f1)
    const x2 = xOf(s.f2)
    segBuf[0] = x1; segBuf[1] = Y_HALF; segBuf[2] = 0.01
    segBuf[3] = x2; segBuf[4] = Y_HALF; segBuf[5] = 0.01
    bandRef.current?.geometry.setPositions(segBuf)
    hp1.current?.position.set(x1, Y_HALF, 0.02)
    hp2.current?.position.set(x2, Y_HALF, 0.02)

    const xOp = xOf(s.freq)
    const yOp = Y0 + s.norm * PLOT_H
    opDot.current?.position.set(xOp, yOp, 0.02)
    // f0 reads at the foot of its own line, not the top: the peak is where the operating dot parks
    // whenever the narration tunes to resonance, and the two labels land on each other up there
    tagRefs.current[0]?.position.set(xf0 + 0.55, Y0 + 0.22, 0)
    tagRefs.current[1]?.position.set((x1 + x2) / 2, Y_HALF - 0.28, 0)
    tagRefs.current[2]?.position.set(xOp, yOp + 0.3, 0)

    // The half-power level and the band arrive together with the Q readout, since the band is what
    // Q measures; the 0.707 tick belongs to that line and goes with it
    const wRidge = reveal[REV_RIDGE]
    const wCross = reveal[REV_CROSS]
    const wBand = reveal[REV_BAND]
    setLineOpacity(curveRef.current, wRidge)
    opDot.current?.scale.setScalar(wRidge)
    setLineOpacity(f0Ref.current, wCross)
    setLineOpacity(halfRef.current, wBand)
    setLineOpacity(bandRef.current, wBand)
    hp1.current?.scale.setScalar(wBand)
    hp2.current?.scale.setScalar(wBand)
    const fades = [wCross, wBand, wRidge, reveal[REV_VMAG], wBand]
    fades.forEach((v, t) => setOpacity(textRefs.current[t], v))

    const w = textRefs.current[3] ? labels.settled(valuesRef.current) : null
    if (w) {
      const texts = [
        `f0 ${fmtF(s.f0)}`,
        `BW ${fmtF(s.BW)} · Q ${s.Q.toFixed(2)}`,
        `f ${fmtF(s.freq)} · I ${fmtI(s.I)}`,
        magnifyLine(s),
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  return (
    <>
      <Line points={BASE_AXIS} lineWidth={1} color={c.guide} />
      <Line ref={halfRef} points={SEG_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.07} gapSize={0.06} />
      <Line ref={f0Ref} points={SEG_INIT} lineWidth={1.5} color={c.guide} dashed dashSize={0.09} gapSize={0.06} />
      <Line ref={bandRef} points={SEG_INIT} lineWidth={2.5} color={c.resistance} />
      <TrailLine
        ref={curveRef}
        points={CURVE_INIT}
        lineWidth={3}
        color={c.current}
        fade={false}
        flowSpeed={() => flow.current}
      />
      <Marker ref={hp1} color={c.marker} size={0.045} />
      <Marker ref={hp2} color={c.marker} size={0.045} />
      <Marker ref={opDot} color={c.marker} size={0.07} />

      {[0, 1, 2].map((t) => (
        <group key={t} ref={(g) => { tagRefs.current[t] = g }}>
          <Html center>
            <span
              ref={(el) => { textRefs.current[t] = el }}
              className={labelClass}
              style={t === 2 ? { color: c.current } : undefined}
            />
          </Html>
        </group>
      ))}
      <Html position={[X0, Y0 - 0.5, 0]}>
        <span ref={(el) => { textRefs.current[3] = el }} className={labelClass} />
      </Html>

      {TICKS.map((f) => (
        <Html key={f} position={[xOf(f), Y0 - 0.2, 0]} center>
          <span className={axisClass}>{f < 1000 ? f : `${f / 1000}k`}</span>
        </Html>
      ))}
      <Html position={[X0 + X_SPAN + 0.3, Y0, 0]} center>
        <span className={axisClass}>f (Hz)</span>
      </Html>
      <Html position={[X0 - 0.45, Y0 + PLOT_H, 0]} center>
        <span className={axisClass}>I ÷ Imax</span>
      </Html>
      <Html position={[X0 - 0.45, Y_HALF, 0]} center>
        <span ref={(el) => { textRefs.current[4] = el }} className={axisClass}>0.707</span>
      </Html>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'resonance',
  title: { en: 'Resonance', ko: '공진' },
  subject: 'circuits',
  order: 15,
  cardPosition: [3.3, 1.2, -0.6],
  description: {
    en: `## Resonance

**A series RLC circuit has exactly one frequency where XL and XC cancel, and there the current peaks.**

The frequency response is the whole story: sweep the drive frequency and the current traces a hill with its top at f0.
- Cyan liquid line: the current response, drawn against frequency and scaled to its own peak
- Dashed vertical: f0, the resonant frequency. The peak never leaves it, no matter what R does
- Dashed horizontal at 0.707: the half-power level. The solid bar between the two dots is the bandwidth
- XL (rising) and XC (falling) in 3D: their crossing **is** resonance

**Resonant frequency**: f0 = 1 ÷ (2π·sqrt(LC)), or ω0 = 1 ÷ sqrt(LC)
**At resonance**: XL = XC, so Z = R, its smallest value, and I = Vm ÷ R, its largest
**Quality factor**: Q = ω0·L ÷ R
**Bandwidth**: BW = f0 ÷ Q = f2 − f1, measured between the half-power points

With R = 50 Ω, L = 100 mH, C = 10 μF and a 10 V peak source: f0 = 159.2 Hz, Q = 2.00, BW = 79.6 Hz, and the current reaches 0.200 A. At that point VL = VC = Q × Vm = 20.0 V — twice the source. A high-Q circuit magnifies voltage across L and C.

Lower R and the hill grows sharper: Q rises, the band narrows, and the circuit gets picky about frequency. Move L or C and the whole hill slides sideways, because only L and C set f0.

In 3D: the same curve for seven resistances at once, stacked into depth. The front edge is the sharpest, the back the broadest, and the dashed crest shows every peak sitting at the same f0.`,
    ko: `## 공진

**RLC 직렬 회로에는 XL과 XC가 정확히 상쇄되는 주파수가 꼭 하나 있고, 그곳에서 전류가 최대가 됩니다.**

주파수 응답이 이 개념의 전부입니다. 구동 주파수를 훑으면 전류가 f0에서 정점을 찍는 언덕을 그립니다.
- 시안 액체 선: 주파수에 대한 전류 응답. 자기 최댓값으로 정규화했습니다
- 세로 점선: 공진 주파수 f0. R을 아무리 바꿔도 정점은 이 선을 떠나지 않습니다
- 0.707 높이의 가로 점선: 반전력 레벨. 두 점 사이의 굵은 선이 대역폭입니다
- 3D의 XL(상승)과 XC(하강): 두 선이 만나는 지점이 **곧** 공진입니다

**공진 주파수**: f0 = 1 ÷ (2π·sqrt(LC)), 즉 ω0 = 1 ÷ sqrt(LC)
**공진점에서**: XL = XC이므로 Z = R로 최소가 되고, I = Vm ÷ R로 최대가 됩니다
**선택도**: Q = ω0·L ÷ R
**대역폭**: BW = f0 ÷ Q = f2 − f1, 반전력점 사이의 폭입니다

R = 50 Ω, L = 100 mH, C = 10 μF, 최댓값 10 V 전원이면 f0 = 159.2 Hz, Q = 2.00, BW = 79.6 Hz이고 전류는 0.200 A까지 올라갑니다. 이때 VL = VC = Q × Vm = 20.0 V로 전원 전압의 두 배입니다. Q가 높은 회로는 L과 C 양단의 전압을 이렇게 키웁니다.

R을 낮추면 언덕이 뾰족해집니다. Q가 커지고 대역이 좁아져서 회로가 주파수를 까다롭게 가립니다. L이나 C를 움직이면 언덕 전체가 옆으로 미끄러집니다. f0을 정하는 것은 L과 C뿐이기 때문입니다.

3D: 같은 곡선을 저항 일곱 개에 대해 한꺼번에 깊이 방향으로 쌓았습니다. 앞쪽이 가장 뾰족하고 뒤쪽이 가장 완만하며, 점선 능선은 모든 정점이 같은 f0에 있음을 보여 줍니다.`,
  },
  story: [
    {
      set: { R: 50, L_mH: 100, C_uF: 10, freq: 100 },
      text: {
        en: 'Two curves pulling opposite ways. A coil\'s reactance grows as the frequency rises; a capacitor\'s shrinks. One climbs, the other falls.',
        ko: '서로 반대로 당기는 두 선입니다. 코일의 리액턴스는 주파수가 올라갈수록 커지고, 커패시터의 리액턴스는 작아집니다. 하나는 올라가고 하나는 내려갑니다.',
      },
    },
    {
      set: { R: 50, L_mH: 100, C_uF: 10, freq: 100 },
      text: {
        en: 'A rising line and a falling line can only ever meet once. These two meet at 159.2 Hz, both worth 100 Ω there, and that one frequency has a name: `f0 = 1 ÷ (2π·sqrt(LC))`.',
        ko: '올라가는 선과 내려가는 선은 딱 한 번밖에 만날 수 없습니다. 이 둘은 159.2 Hz에서 만나고, 그 지점에서 둘 다 100 Ω입니다. 이 하나뿐인 주파수에 이름이 붙어 있습니다. `f0 = 1 ÷ (2π·sqrt(LC))`',
      },
    },
    {
      set: { R: 50, L_mH: 100, C_uF: 10, freq: 100 },
      text: {
        en: 'Now sweep the drive frequency and watch the current. It traces a hill. At 100 Hz we are still on the slope, pulling only 92.1 mA.',
        ko: '이제 구동 주파수를 훑으면서 전류를 봅니다. 전류가 언덕을 그립니다. 100 Hz는 아직 비탈이라 92.1 mA밖에 못 끌어옵니다.',
      },
    },
    {
      set: { R: 50, L_mH: 100, C_uF: 10, freq: 160 },
      text: {
        en: 'Tune to f0 and the dot climbs to the top: 200.0 mA. The two reactances have cancelled, so nothing is left but R — impedance at its smallest, current at its largest.',
        ko: 'f0에 맞추면 점이 꼭대기까지 올라갑니다. 200.0 mA. 두 리액턴스가 서로를 지웠으니 남은 건 R뿐입니다. 임피던스가 가장 작고, 전류가 가장 커지는 지점입니다.',
      },
    },
    {
      set: { R: 50, L_mH: 100, C_uF: 10, freq: 160 },
      text: {
        en: 'So what does R do? Here is the same L and C at seven resistances, stacked into the depth. The hills differ in height and in width — but every single peak sits on that one dashed line. R has no say in f0.',
        ko: '그럼 R은 무슨 일을 할까요? 같은 L과 C로 저항 일곱 개의 곡선을 깊이 방향으로 쌓았습니다. 언덕의 높이도 너비도 전부 다른데, 정점은 하나도 빠짐없이 저 점선 하나 위에 있습니다. f0에 R은 발언권이 없습니다.',
      },
    },
    {
      set: { R: 10, L_mH: 100, C_uF: 10, freq: 160 },
      text: {
        en: 'Drop R to 10 Ω and the hill turns into a spike. The current shoots up to 994.4 mA — five times what 50 Ω could manage.',
        ko: 'R을 10 Ω으로 낮추면 언덕이 바늘처럼 뾰족해집니다. 전류는 994.4 mA까지 치솟습니다. 50 Ω일 때의 다섯 배입니다.',
      },
    },
    {
      set: { R: 10, L_mH: 100, C_uF: 10, freq: 160 },
      text: {
        en: 'That sharpness has a name. `Q = ω0·L ÷ R` is 10.00 here, and the band the circuit lets through is `BW = f0 ÷ Q` — a mere 15.9 Hz wide.',
        ko: '이 뾰족한 정도에 이름이 있습니다. `Q = ω0·L ÷ R`이고 여기선 10.00입니다. 이 회로가 통과시키는 대역의 폭은 `BW = f0 ÷ Q`, 겨우 15.9 Hz입니다.',
      },
    },
    {
      set: { R: 500, L_mH: 100, C_uF: 10, freq: 160 },
      text: {
        en: 'Push R to 500 Ω and it all goes the other way. Q collapses to 0.20, the band blows out to 795.8 Hz, and the circuit stops caring which frequency you feed it — along with the current, down to 20.0 mA.',
        ko: 'R을 500 Ω으로 올리면 전부 반대가 됩니다. Q는 0.20으로 주저앉고 대역은 795.8 Hz로 퍼집니다. 어떤 주파수를 넣든 상관하지 않는 회로가 되고, 전류도 20.0 mA까지 떨어집니다.',
      },
    },
    {
      set: { R: 50, L_mH: 400, C_uF: 10, freq: 80 },
      text: {
        en: 'Want to move the peak itself? Only L and C can do that. Four times the inductance drops f0 to half, 79.6 Hz, and the whole hill slides left with it.',
        ko: '정점 자체를 옮기고 싶다면 L과 C만이 할 수 있습니다. 인덕턴스를 네 배로 키우면 f0이 절반인 79.6 Hz로 내려가고, 언덕 전체가 그대로 왼쪽으로 미끄러집니다.',
      },
    },
    {
      set: { R: 10, L_mH: 100, C_uF: 10, freq: 160 },
      text: {
        en: 'And here is the sting. At f0 the two reactances cancel in the sum, but each one is still 100 Ω carrying the full current — so `VL = VC = Q × Vm` puts 100.0 V across the coil from a 10 V source. Nothing is created: L and C are handing the same energy back and forth, and Q counts how many times.',
        ko: '그리고 여기 함정이 있습니다. f0에서 두 리액턴스는 합으로는 서로를 지우지만, 각자는 여전히 100 Ω짜리 리액턴스에 전체 전류가 흐르고 있습니다. 그래서 `VL = VC = Q × Vm`, 10 V 전원인데 코일 양단에는 100.0 V가 걸립니다. 없던 게 생긴 게 아닙니다. L과 C가 같은 에너지를 주고받고 있을 뿐이고, 그 횟수를 세는 값이 Q입니다.',
      },
    },
  ],
  sliderParams: [
    { key: 'R',    label: { en: 'Resistance (R)',    ko: '저항 (R)' },       unit: 'Ω',  min: R_MIN, max: R_MAX, step: 10, defaultValue: 50 },
    { key: 'L_mH', label: { en: 'Inductance (L)',    ko: '인덕턴스 (L)' },   unit: 'mH', min: 10,    max: 500,   step: 10, defaultValue: 100 },
    { key: 'C_uF', label: { en: 'Capacitance (C)',   ko: '커패시턴스 (C)' }, unit: 'μF', min: 1,     max: 100,   step: 1,  defaultValue: 10 },
    { key: 'freq', label: { en: 'Drive frequency (f)', ko: '구동 주파수 (f)' }, unit: 'Hz', min: F_MIN, max: F_MAX, step: 10, defaultValue: 100 },
  ],
  Scene: ResonanceRidgeScene,
  Scene2D: ResponseCurve2DScene,
}

export default concept
