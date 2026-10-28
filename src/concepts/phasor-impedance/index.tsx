import { useRef, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import type { Group } from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import { setLineOpacity, setOpacity, useSettledLabels, useStoryReveal } from '../../components/scene/storyReveal'
import type { ConceptMeta } from '../../types/concept'

const VM = 10 // source peak voltage [V]
const DISPLAY_OMEGA = 1.4 // on-screen rad/s: a real 60 Hz phasor is far too fast to follow

// Pure physics, voltage as the phase reference: v = Vm·sin(ωt), i = (Vm / |Z|)·sin(ωt − θ).
// `wL` and `wC` scale the two reactances, so a narration step that has not introduced the coil or
// the capacitor yet solves the circuit that is actually on screen: wL = wC = 0 is a plain R on AC,
// where θ = 0 and the current arrow sits right on top of the voltage arrow.
// Verified with node -e:
// R = 50 Ω, L = 100 mH, C = 50 μF, f = 60 Hz →
// XL = 37.70 Ω, XC = 53.05 Ω, X = −15.35 Ω, |Z| = 52.30 Ω, θ = −17.1° (capacitive, current leads).
function solve({ R = 50, L_mH = 100, C_uF = 50, freq = 60 }: Record<string, number>, wL = 1, wC = 1) {
  const omega = 2 * Math.PI * freq
  const XL = omega * L_mH * 1e-3 * wL
  const XC = (1 / (omega * C_uF * 1e-6)) * wC
  const X = XL - XC
  const Zmag = Math.hypot(R, X)
  return { R, XL, XC, X, Zmag, theta: Math.atan2(X, R), Im: VM / Zmag }
}

function useSliderValues() {
  const valuesRef = useRef<Record<string, number>>(useSliderStore.getState().values)
  useEffect(
    () => useSliderStore.subscribe((s) => s.values, (v) => { valuesRef.current = v }),
    []
  )
  return valuesRef
}

// Narration slots. The wedge between the two arrows rides on REV_L because it only has an angle to
// show once the coil is in, and the triangle arrives after both reactances do.
const REV_I = 0 // the current phasor, its trail and its readout
const REV_L = 1 // the coil, and with it XL and the wedge
const REV_C = 2 // the capacitor, and with it XC
const REV_TRI = 3 // the impedance triangle's three sides
const REV_TARC = 4 // the triangle's corner angle, the θ that matches the wedge
const REV_COUNT = 5

function revealTargets(step: number, out: Float32Array) {
  out[REV_I] = step >= 1 ? 1 : 0
  out[REV_L] = step >= 2 ? 1 : 0
  out[REV_C] = step >= 3 ? 1 : 0
  out[REV_TRI] = step >= 4 ? 1 : 0
  out[REV_TARC] = step >= 5 ? 1 : 0
}

const fmtI = (a: number) => (a < 1 ? `${(a * 1e3).toFixed(0)} mA` : `${a.toFixed(2)} A`)
const fmtDeg = (rad: number) => `θ ${((rad * 180) / Math.PI).toFixed(1)}°`
// Labels read in the foreground color; only accent quantities (current) tint their text
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'

// Shared scratch buffers: every writer fills them and hands them straight to a geometry,
// so no frame ever allocates.
const ARC_N = 24
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

// ── 3D: the complex plane — two phasors turning, their history receding, the Z triangle beside them ──
const PH_X = 0.7 // the rotating pair sits right of centre, the impedance triangle to its left
const V_LEN = 1.25 // scene units for the voltage phasor (Vm is fixed)
const I_SCALE = 2.5 // scene units per amp; R ≥ 20 Ω caps Im at Vm / 20 = 0.5 A, so i never exceeds V_LEN
const ZT_X = -3.3 // impedance triangle corner
const ZT_Y = -0.7
const TRAIL_N = 150
const TRAIL_TURNS = 2
const TRAIL_DEPTH = 4.2
const TRAIL_INIT: [number, number, number][] = Array.from(
  { length: TRAIL_N },
  (): [number, number, number] => [0, 0, 0]
)
const AXIS_R = 1.55
const RE_AXIS: [number, number, number][] = [[PH_X - AXIS_R, 0, 0], [PH_X + AXIS_R, 0, 0]]
const IM_AXIS: [number, number, number][] = [[PH_X, -AXIS_R, 0], [PH_X, AXIS_R, 0]]
const CIRCLE: [number, number, number][] = Array.from(
  { length: 97 },
  (_, i): [number, number, number] => {
    const a = (i / 96) * Math.PI * 2
    return [PH_X + V_LEN * Math.cos(a), V_LEN * Math.sin(a), 0]
  }
)
const FLOW_AT_MAX = 3.5 // sheen bands per second when the current is at its largest possible peak

function PhasorPlaneScene() {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  // 0 v phasor, 1 i phasor, 2 phase arc, 3 R, 4 X, 5 Z, 6 triangle arc
  const lineRefs = useRef<any[]>([])
  const tagRefs = useRef<(Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const vTipRef = useRef<Group>(null)
  const iTipRef = useRef<Group>(null)
  const vTrailRef = useRef<any>(null)
  const iTrailRef = useRef<any>(null)
  const vBuf = useMemo(() => new Float32Array(TRAIL_N * 3), [])
  const iBuf = useMemo(() => new Float32Array(TRAIL_N * 3), [])
  const flow = useRef(0)
  const labels = useSettledLabels(revealTargets, REV_COUNT)
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(({ clock }) => {
    const s = solve(valuesRef.current, reveal[REV_L], reveal[REV_C])
    const wt = clock.getElapsedTime() * DISPLAY_OMEGA
    const iLen = s.Im * I_SCALE
    const ia = wt - s.theta // the current phasor is the voltage phasor turned back by θ

    // The two rotating arrows and the wedge between them
    setSeg(lineRefs.current[0], PH_X, 0, PH_X + V_LEN * Math.cos(wt), V_LEN * Math.sin(wt))
    setSeg(lineRefs.current[1], PH_X, 0, PH_X + iLen * Math.cos(ia), iLen * Math.sin(ia))
    setArc(lineRefs.current[2], PH_X, 0, 0.62, ia, wt)
    vTipRef.current?.position.set(PH_X + V_LEN * Math.cos(wt), V_LEN * Math.sin(wt), 0)
    iTipRef.current?.position.set(PH_X + iLen * Math.cos(ia), iLen * Math.sin(ia), 0)

    // Time traces: the same two arrows a moment ago, receding toward −Z
    for (let k = 0; k < TRAIL_N; k++) {
      const back = 1 - k / (TRAIL_N - 1) // 0 at the head (now), 1 at the tail (oldest)
      const av = wt - back * TRAIL_TURNS * 2 * Math.PI
      const z = -back * TRAIL_DEPTH
      vBuf[k * 3] = PH_X + V_LEN * Math.cos(av)
      vBuf[k * 3 + 1] = V_LEN * Math.sin(av)
      vBuf[k * 3 + 2] = z
      iBuf[k * 3] = PH_X + iLen * Math.cos(av - s.theta)
      iBuf[k * 3 + 1] = iLen * Math.sin(av - s.theta)
      iBuf[k * 3 + 2] = z
    }
    vTrailRef.current?.geometry.setPositions(vBuf)
    iTrailRef.current?.geometry.setPositions(iBuf)
    flow.current = FLOW_AT_MAX * (iLen / V_LEN)

    // Impedance triangle: R across, X up (down when capacitive), Z closing the corner
    const kz = Math.min(1.9 / s.R, 1.25 / Math.max(Math.abs(s.X), 1e-6))
    const ax = ZT_X + s.R * kz
    const by = ZT_Y + s.X * kz
    setSeg(lineRefs.current[3], ZT_X, ZT_Y, ax, ZT_Y)
    setSeg(lineRefs.current[4], ax, ZT_Y, ax, by)
    setSeg(lineRefs.current[5], ZT_X, ZT_Y, ax, by)
    setArc(lineRefs.current[6], ZT_X, ZT_Y, 0.42, 0, s.theta)

    // θ sits inside the wedge and the current reads outside it: only a radial gap keeps the two
    // apart, since a small θ brings them to within a degree of the same bearing.
    const lr = Math.max(iLen, 0.4) + 0.55
    tagRefs.current[0]?.position.set(PH_X + (V_LEN + 0.3) * Math.cos(wt), (V_LEN + 0.3) * Math.sin(wt), 0)
    tagRefs.current[1]?.position.set(PH_X + lr * Math.cos(ia), lr * Math.sin(ia), 0)
    tagRefs.current[2]?.position.set(PH_X + 0.4 * Math.cos(wt - s.theta / 2), 0.4 * Math.sin(wt - s.theta / 2), 0)
    // R reads near the far end of the base and |Z| past the middle of the hypotenuse, because the
    // corner they share is already taken by the triangle's own θ
    tagRefs.current[3]?.position.set(ZT_X + (ax - ZT_X) * 0.78, ZT_Y - 0.24, 0)
    tagRefs.current[4]?.position.set(ax + 0.36, (ZT_Y + by) / 2, 0)
    tagRefs.current[5]?.position.set(
      ZT_X + (ax - ZT_X) * 0.55 - 0.1,
      ZT_Y + (by - ZT_Y) * 0.55 + (s.X >= 0 ? 0.26 : -0.26),
      0
    )
    tagRefs.current[6]?.position.set(ZT_X + 0.62 * Math.cos(s.theta / 2), ZT_Y + 0.62 * Math.sin(s.theta / 2), 0)

    // Each element fades in on the step that first talks about it
    const wI = reveal[REV_I]
    const wTri = reveal[REV_TRI]
    const fades = [1, wI, reveal[REV_L], wTri, wTri, wTri, reveal[REV_TARC]]
    fades.forEach((v, i) => {
      setLineOpacity(lineRefs.current[i], v)
      setOpacity(textRefs.current[i], v)
    })
    setLineOpacity(iTrailRef.current, wI)
    iTipRef.current?.scale.setScalar(wI)

    // The readouts follow the step as well as the sliders, since the step decides which reactances
    // are in the circuit at all
    const w = textRefs.current[6] ? labels.settled(valuesRef.current) : null
    if (w) {
      const q = solve(valuesRef.current, w[REV_L], w[REV_C])
      const texts = [
        `v ${VM.toFixed(0)} V`,
        `i ${fmtI(q.Im)}`,
        fmtDeg(q.theta),
        `R ${q.R.toFixed(0)} Ω`,
        `X ${q.X.toFixed(1)} Ω`,
        `|Z| ${q.Zmag.toFixed(1)} Ω`,
        fmtDeg(q.theta),
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  const segs: { color: string; width: number; dashed?: boolean; arc?: boolean }[] = [
    { color: c.voltage, width: 3 },
    { color: c.current, width: 2.5 },
    { color: c.guide, width: 1.5, dashed: true, arc: true },
    { color: c.resistance, width: 2 },
    { color: c.guide, width: 2, dashed: true },
    { color: c.voltage, width: 2.5 },
    { color: c.guide, width: 1.5, dashed: true, arc: true },
  ]

  return (
    <>
      {/* Backed off and aimed right of centre: the plane and the triangle together are nearly six
          units wide, and at the default framing the phasors sit under the narration panel */}
      <PerspectiveCamera makeDefault position={[3.2, 2.2, 6.6]} fov={50} />
      <OrbitControls enablePan={false} target={[1.0, 0, 0]} />

      {/* Complex plane: real axis across, imaginary axis up, reference circle at |v| */}
      <Line points={RE_AXIS} lineWidth={1} color={c.guide} />
      <Line points={IM_AXIS} lineWidth={1} color={c.guide} />
      <Line points={CIRCLE} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />

      <TrailLine ref={vTrailRef} points={TRAIL_INIT} lineWidth={2.5} color={c.voltage} />
      <TrailLine
        ref={iTrailRef}
        points={TRAIL_INIT}
        lineWidth={2.5}
        color={c.current}
        flowSpeed={() => flow.current}
      />

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

      <Marker ref={vTipRef} color={c.marker} size={0.06} />
      <Marker ref={iTipRef} color={c.current} size={0.06} />

      {segs.map((_, i) => (
        <group key={i} ref={(g) => { tagRefs.current[i] = g }}>
          <Html center>
            <span
              ref={(el) => { textRefs.current[i] = el }}
              className={labelClass}
              style={i === 1 ? { color: c.current } : undefined}
            />
          </Html>
        </group>
      ))}
    </>
  )
}

// ── 2D: one cycle of v and i, with the gap between their peaks bracketed ──
const W_N = 220
const W_HALF = 3.6
const V_AMP = 1.4
const I_AMP_SCALE = 2.8 // same cap as 3D: Im ≤ 0.5 A maps to V_AMP
const BRACKET_Y = -1.82
const xOfPhase = (p: number) => -W_HALF + (p / (2 * Math.PI)) * 2 * W_HALF
const ZERO_AXIS: [number, number, number][] = [[-W_HALF - 0.15, 0, 0], [W_HALF + 0.15, 0, 0]]
const WAVE_INIT: [number, number, number][] = Array.from(
  { length: W_N },
  (): [number, number, number] => [0, 0, 0]
)

function PhasorWave2DScene() {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  // 0 now line, 1 peak-gap bracket, 2 dropline from the v peak, 3 dropline from the i peak
  const lineRefs = useRef<any[]>([])
  const tagRefs = useRef<(Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const vWaveRef = useRef<any>(null)
  const iWaveRef = useRef<any>(null)
  const vTipRef = useRef<Group>(null)
  const iTipRef = useRef<Group>(null)
  const vBuf = useMemo(() => new Float32Array(W_N * 3), [])
  const iBuf = useMemo(() => new Float32Array(W_N * 3), [])
  const flow = useRef(0)
  const labels = useSettledLabels(revealTargets, REV_COUNT)
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(({ clock }) => {
    const s = solve(valuesRef.current, reveal[REV_L], reveal[REV_C])
    const iAmp = s.Im * I_AMP_SCALE

    for (let k = 0; k < W_N; k++) {
      const p = (k / (W_N - 1)) * 2 * Math.PI
      const x = xOfPhase(p)
      vBuf[k * 3] = x
      vBuf[k * 3 + 1] = V_AMP * Math.sin(p)
      vBuf[k * 3 + 2] = 0
      iBuf[k * 3] = x
      iBuf[k * 3 + 1] = iAmp * Math.sin(p - s.theta)
      iBuf[k * 3 + 2] = 0
    }
    vWaveRef.current?.geometry.setPositions(vBuf)
    iWaveRef.current?.geometry.setPositions(iBuf)
    flow.current = FLOW_AT_MAX * (iAmp / V_AMP)

    // "Now" sweeps across the single cycle and wraps
    const span = 2 * Math.PI
    const pn = ((clock.getElapsedTime() * DISPLAY_OMEGA) % span + span) % span
    const xn = xOfPhase(pn)
    setSeg(lineRefs.current[0], xn, -1.55, xn, 1.55)
    vTipRef.current?.position.set(xn, V_AMP * Math.sin(pn), 0)
    iTipRef.current?.position.set(xn, iAmp * Math.sin(pn - s.theta), 0)

    // The two peaks: v at ωt = 90°, i at ωt = 90° + θ. θ < 0 puts the current peak earlier (to the left)
    const xv = xOfPhase(Math.PI / 2)
    const xi = xOfPhase(Math.PI / 2 + s.theta)
    setSeg(lineRefs.current[1], xi, BRACKET_Y, xv, BRACKET_Y)
    setSeg(lineRefs.current[2], xv, V_AMP, xv, BRACKET_Y)
    setSeg(lineRefs.current[3], xi, iAmp, xi, BRACKET_Y)

    const q = Math.PI / 4
    tagRefs.current[0]?.position.set(xOfPhase(q), V_AMP * Math.sin(q) + 0.26, 0)
    tagRefs.current[1]?.position.set(xOfPhase(q), iAmp * Math.sin(q - s.theta) - 0.26, 0)
    tagRefs.current[2]?.position.set((xv + xi) / 2, BRACKET_Y + 0.24, 0)

    // The current wave arrives with the resistor; the peak-gap bracket with the coil, which is the
    // first step that has a gap to bracket at all
    const wI = reveal[REV_I]
    const wGap = reveal[REV_L]
    setLineOpacity(iWaveRef.current, wI)
    iTipRef.current?.scale.setScalar(wI)
    for (const i of [1, 2, 3]) setLineOpacity(lineRefs.current[i], wGap)
    const fades = [1, wI, wGap]
    fades.forEach((v, t) => setOpacity(textRefs.current[t], v))

    const w = textRefs.current[2] ? labels.settled(valuesRef.current) : null
    if (w) {
      const q = solve(valuesRef.current, w[REV_L], w[REV_C])
      const texts = [`v ${VM.toFixed(0)} V`, `i ${fmtI(q.Im)}`, fmtDeg(q.theta)]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  const segs: { color: string; width: number; dashed?: boolean }[] = [
    { color: c.guide, width: 1, dashed: true },
    { color: c.marker, width: 2 },
    { color: c.guide, width: 1, dashed: true },
    { color: c.guide, width: 1, dashed: true },
  ]

  return (
    <>
      <Line points={ZERO_AXIS} lineWidth={1} color={c.guide} />
      <TrailLine ref={vWaveRef} points={WAVE_INIT} lineWidth={2.5} color={c.voltage} />
      <TrailLine
        ref={iWaveRef}
        points={WAVE_INIT}
        lineWidth={2.5}
        color={c.current}
        flowSpeed={() => flow.current}
      />

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

      <Marker ref={vTipRef} color={c.marker} size={0.055} />
      <Marker ref={iTipRef} color={c.current} size={0.055} />

      {[0, 1, 2].map((t) => (
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

const concept: ConceptMeta = {
  id: 'phasor-impedance',
  title: { en: 'Phasors & Complex Impedance', ko: '페이저·복소 임피던스' },
  subject: 'circuits',
  order: 12,
  description: {
    en: `## Phasors and Complex Impedance

**A sinusoid is a rotating vector, and impedance is a complex number that stretches it and turns it at the same time.**

In the complex plane v = Vm·sin(ωt) is a vector of length Vm turning counter-clockwise; its height above the real axis is the value at this instant. Divide that vector by Z and you get the current vector.
- Strongest neutral line: the voltage phasor v, the reference, at angle ωt
- Cyan liquid line: the current phasor i, the same rotation turned back by θ
- The two trails receding into the back: the same two phasors a moment ago, so depth reads as time
- Triangle on the left: R across, X up, Z along the hypotenuse, θ at the corner

R scales a phasor without turning it, jXL turns it forward by 90°, and −jXC turns it back by 90°. Add them and you have one complex number that does both jobs at once.

**Impedance**: Z = R + jX, with XL = ωL, XC = 1 ÷ (ωC), and X = XL − XC
**Magnitude and angle**: |Z| = sqrt(R² + X²), θ = arctan(X ÷ R)
**Current**: i = (Vm ÷ |Z|)·sin(ωt − θ)

The θ in the triangle is exactly the gap between the two arrows. Positive θ (inductive) and the current trails the voltage; negative θ (capacitive) and it runs ahead.

Raise f or L to tip the triangle up and make the current lag; raise C to tip it down and make the current lead; raise R to flatten the triangle toward θ = 0, and shrink the current with it.

In 2D: one cycle of v and i, with the gap between their peaks bracketed and labelled with the same θ. The rotation is slowed far below a real 60 Hz so it can be followed; labels show peak values.`,
    ko: `## 페이저와 복소 임피던스

**정현파는 돌아가는 벡터이고, 임피던스는 그 벡터를 늘이는 동시에 돌리는 복소수입니다.**

복소평면에서 v = Vm·sin(ωt)는 길이가 Vm이고 반시계 방향으로 도는 벡터이며, 실수축 위로 올라간 높이가 그 순간의 값입니다. 이 벡터를 Z로 나누면 전류 벡터가 됩니다.
- 가장 진한 중성색 선: 기준이 되는 전압 페이저 v, 각도는 ωt
- 시안 액체 선: 전류 페이저 i, 같은 회전을 θ만큼 되돌린 것
- 뒤로 물러나는 두 자취: 조금 전의 같은 두 페이저. 깊이가 곧 시간입니다
- 왼쪽 삼각형: 가로가 R, 세로가 X, 빗변이 Z, 모서리 각이 θ

R은 페이저를 돌리지 않고 크기만 바꾸고, jXL은 90° 앞으로, −jXC는 90° 뒤로 돌립니다. 이 셋을 더하면 크기 변환과 회전을 한꺼번에 하는 하나의 복소수가 됩니다.

**임피던스**: Z = R + jX, XL = ωL, XC = 1 ÷ (ωC), X = XL − XC
**크기와 각**: |Z| = sqrt(R² + X²), θ = arctan(X ÷ R)
**전류**: i = (Vm ÷ |Z|)·sin(ωt − θ)

삼각형의 θ는 두 화살표 사이의 벌어진 각과 정확히 같습니다. θ가 양수(유도성)면 전류가 전압보다 뒤지고, 음수(용량성)면 앞섭니다.

f나 L을 올리면 삼각형이 위로 기울어 전류가 뒤지고, C를 올리면 아래로 기울어 전류가 앞섭니다. R을 올리면 삼각형이 납작해져 θ가 0에 가까워지고 전류도 함께 작아집니다.

2D: v와 i의 한 주기입니다. 두 파형의 최댓점 사이 간격을 표시하고 같은 θ를 붙였습니다. 회전은 실제 60 Hz보다 훨씬 느리게 재생하며, 라벨은 최댓값입니다.`,
  },
  story: [
    {
      set: { R: 50, L_mH: 100, C_uF: 50, freq: 60 },
      text: {
        en: 'In the complex plane a voltage is not a squiggly curve. It is an arrow spinning counter-clockwise, and how high it reaches above the flat axis is the voltage right now.',
        ko: '복소평면에서 전압은 구불구불한 곡선이 아닙니다. 반시계로 도는 화살표 하나이고, 이 화살표가 수평축 위로 올라간 높이가 바로 지금 이 순간의 전압입니다.',
      },
    },
    {
      set: { R: 50, L_mH: 100, C_uF: 50, freq: 60 },
      text: {
        en: 'Hang a plain 50 Ω resistor on it and a second arrow shows up glued to the first: `10 V ÷ 50 Ω = 200 mA`. A resistor changes the length and nothing else, so the two turn as one body.',
        ko: '여기에 그냥 50 Ω 저항만 달면 두 번째 화살표가 첫 번째에 딱 붙어서 나타납니다. `10 V ÷ 50 Ω = 200 mA`. 저항은 길이만 바꾸기 때문에 둘은 한 몸처럼 돕니다.',
      },
    },
    {
      set: { R: 50, L_mH: 100, C_uF: 50, freq: 60 },
      text: {
        en: 'Now add the coil. Its reactance `XL = ωL` works out to 37.7 Ω, and it drags the current arrow 37.0° behind the voltage. The current shrinks to 160 mA, and a wedge opens between the two.',
        ko: '이제 코일을 넣습니다. 리액턴스 `XL = ωL`가 37.7 Ω으로 나오고, 전류 화살표를 전압보다 37.0° 뒤로 끌어당깁니다. 전류는 160 mA로 줄고, 두 화살표 사이에 부채꼴이 벌어집니다.',
      },
    },
    {
      set: { R: 50, L_mH: 100, C_uF: 50, freq: 60 },
      text: {
        en: 'The capacitor shoves the other way. `XC = 1 ÷ (ωC)` is 53.1 Ω, bigger than XL, so the two cancel down to −15.4 Ω and the sign flips: the current now runs 17.1° ahead, and climbs to 191 mA.',
        ko: '커패시터는 반대쪽으로 밀어붙입니다. `XC = 1 ÷ (ωC)`가 53.1 Ω으로 XL보다 커서, 둘이 상쇄되면 −15.4 Ω이 남고 부호가 뒤집힙니다. 전류는 이제 17.1° 앞서 달리고 191 mA까지 올라갑니다.',
      },
    },
    {
      set: { R: 50, L_mH: 100, C_uF: 50, freq: 60 },
      text: {
        en: 'Lay those three out as a right triangle on the left: R along the base, X up the side — downward here, because the capacitor is winning. The hypotenuse is what the circuit actually pushes back with, `|Z| = sqrt(R² + X²)`, which is 52.3 Ω.',
        ko: '이 셋을 왼쪽에 직각삼각형으로 펼쳐 놓습니다. 밑변이 R, 세로가 X인데 지금은 커패시터가 이기고 있어서 아래로 내려갑니다. 빗변은 회로가 실제로 버티는 양, `|Z| = sqrt(R² + X²)`이고 52.3 Ω입니다.',
      },
    },
    {
      set: { R: 50, L_mH: 100, C_uF: 50, freq: 60 },
      text: {
        en: 'And the corner angle of that triangle? It is exactly the wedge between the two spinning arrows. 17.1° in both places, the same number twice, because `θ = arctan(X ÷ R)` is what tilts them apart.',
        ko: '그럼 삼각형 모서리의 각은요? 돌아가는 두 화살표 사이의 부채꼴과 정확히 같습니다. 두 곳 모두 17.1°, 같은 숫자가 두 번 나옵니다. 두 화살표를 벌려 놓는 값이 `θ = arctan(X ÷ R)` 바로 그것이기 때문입니다.',
      },
    },
    {
      set: { R: 50, L_mH: 140, C_uF: 50, freq: 60 },
      text: {
        en: 'Nudge L up to 140 mH and XL catches XC. The triangle lies flat on the floor, θ drops to −0.3°, and the current is back to the full 200 mA — the coil and the capacitor just erased each other.',
        ko: 'L을 140 mH까지 올리면 XL이 XC를 따라잡습니다. 삼각형이 바닥에 납작하게 눕고, θ는 −0.3°까지 떨어지고, 전류는 다시 200 mA를 꽉 채웁니다. 코일과 커패시터가 서로를 지워버린 겁니다.',
      },
    },
    {
      set: { R: 50, L_mH: 140, C_uF: 50, freq: 120 },
      text: {
        en: 'Double the frequency and that balance breaks hard: XL doubles to 105.6 Ω while XC halves to 26.5 Ω. The triangle shoots up, the current falls 57.7° behind, and only 107 mA gets through.',
        ko: '주파수를 두 배로 올리면 그 균형이 크게 깨집니다. XL은 105.6 Ω으로 두 배가 되고 XC는 26.5 Ω으로 절반이 됩니다. 삼각형은 위로 솟고, 전류는 57.7°나 뒤처지고, 107 mA밖에 흐르지 못합니다.',
      },
    },
    {
      set: { R: 200, L_mH: 140, C_uF: 50, freq: 120 },
      text: {
        en: 'Raise R to 200 Ω and the two jobs split apart in front of you. The base got long, so the angle shrinks to 21.6° — but |Z| swells to 215 Ω, so the current falls again to 47 mA. R never turns the arrow. It only shortens it.',
        ko: 'R을 200 Ω으로 키우면 두 가지 역할이 눈앞에서 갈라집니다. 밑변이 길어져서 각은 21.6°로 작아지는데, |Z|는 215 Ω으로 불어나 전류는 47 mA로 다시 떨어집니다. R은 화살표를 돌리는 일은 절대 하지 않고, 길이만 줄입니다.',
      },
    },
    {
      set: { R: 50, L_mH: 100, C_uF: 50, freq: 60 },
      text: {
        en: 'So all of AC analysis collapses back into Ohm\'s law, with one complex number doing both jobs at once: `Z = R + jX`. Divide the voltage phasor by it and the current phasor falls out, `i = (Vm ÷ |Z|)·sin(ωt − θ)` — size and timing, in a single divide.',
        ko: '결국 교류 해석은 옴의 법칙으로 되돌아옵니다. 복소수 하나가 두 가지 일을 동시에 해주니까요. `Z = R + jX`. 전압 페이저를 이걸로 나누면 전류 페이저가 그대로 떨어집니다. `i = (Vm ÷ |Z|)·sin(ωt − θ)`. 크기와 타이밍을, 나눗셈 한 번으로요.',
      },
    },
  ],
  sliderParams: [
    { key: 'R',    label: { en: 'Resistance (R)',  ko: '저항 (R)' },       unit: 'Ω',  min: 20, max: 200, step: 5,  defaultValue: 50  },
    { key: 'L_mH', label: { en: 'Inductance (L)',  ko: '인덕턴스 (L)' },   unit: 'mH', min: 20, max: 300, step: 10, defaultValue: 100 },
    { key: 'C_uF', label: { en: 'Capacitance (C)', ko: '커패시턴스 (C)' }, unit: 'μF', min: 20, max: 300, step: 10, defaultValue: 50  },
    { key: 'freq', label: { en: 'Frequency (f)',   ko: '주파수 (f)' },     unit: 'Hz', min: 30, max: 120, step: 5,  defaultValue: 60  },
  ],
  Scene: PhasorPlaneScene,
  Scene2D: PhasorWave2DScene,
}

export default concept
