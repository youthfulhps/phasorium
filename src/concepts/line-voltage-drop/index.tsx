import { useRef, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { Marker } from '../../components/scene/Marker'
import { TrailLine } from '../../components/scene/TrailLine'
import { FloorCaption, PotentialRuler, useRiseIn } from '../../components/scene/PotentialLandscape'
import { Resistor, setResistorHeat } from '../../components/scene/Resistor'
import type { ConceptMeta } from '../../types/concept'

const SQRT3 = Math.sqrt(3)

// Fixed receiving-end line voltage [V] — a standard Korean distribution level
const VR = 22900
// Slider edges, kept here so the visuals can be normalized against the worst case
const I_MAX = 200
const R_MIN = 1
const R_MAX = 20
const X_MAX = 20
const COS_MIN = 0.6
const E_MAX = SQRT3 * I_MAX * (R_MAX * COS_MIN + X_MAX * Math.sqrt(1 - COS_MIN ** 2))
const VS_MAX = VR + E_MAX // 32599 V
const LOSS_MAX = 3 * I_MAX * I_MAX * R_MAX // 2.4 MW

// Pure physics: the standard approximate three-phase line-drop formula, Joule loss, and the
// transmitted power that ties current to voltage.
//   e    = sqrt(3)·I·(R·cosθ + X·sinθ)        [V]   (approximation: the receiving voltage is taken
//                                                    as the phase reference, so the quadrature part
//                                                    sqrt(3)·I·(X·cosθ − R·sinθ) is dropped)
//   Vs   = Vr + e                              [V]
//   drop = e ÷ Vr × 100                        [%]
//   loss = 3·I²·R                              [W]
//   P    = sqrt(3)·Vr·I·cosθ                   [W]  so I = P ÷ (sqrt(3)·Vr·cosθ)
//   and therefore loss = P²·R ÷ (V²·cos²θ) ∝ 1 ÷ V² at fixed P
//
// Worked example, verified with node -e:
//   R = 5 Ω, X = 8 Ω, I = 100 A, cosθ = 0.8 (sinθ = 0.6), Vr = 22900 V
//   e = sqrt(3) × 100 × (5 × 0.8 + 8 × 0.6) = sqrt(3) × 100 × 8.8 = 1524.2047 V
//   Vs = 22900 + 1524.2047 = 24424.2047 V
//   drop = 1524.2047 ÷ 22900 × 100 = 6.6559 %
//   loss at I = 100 A: 3 × 100² × 5 = 150000 W = 150.0 kW
//   delivering P = 3 MW at 22.9 kV: I = 3e6 ÷ (sqrt(3) × 22900 × 0.8) = 94.5443 A
//     loss = 3 × 94.5443² × 5 = 134079.25 W = 134.08 kW
//   the same 3 MW at 45.8 kV: I = 47.2721 A, loss = 3 × 47.2721² × 5 = 33519.81 W = 33.52 kW
//     134079.25 ÷ 33519.81 = 4.0000 — exactly one quarter
function solve({ I = 100, R = 5, X = 8, cosphi = 0.8 }: Record<string, number>) {
  const sinphi = Math.sqrt(Math.max(0, 1 - cosphi * cosphi))
  const e = SQRT3 * I * (R * cosphi + X * sinphi)
  const loss = 3 * I * I * R
  return {
    I,
    R,
    e,
    Vs: VR + e,
    drop: (e / VR) * 100,
    loss,
    lossAtDoubleV: loss / 4, // same P at 2V needs I ÷ 2, so I²R falls to a quarter
    P: SQRT3 * VR * I * cosphi,
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

const fmtKV = (v: number) => `${(v / 1000).toFixed(2)} kV`
const fmtKW = (w: number) => `${(w / 1000).toFixed(1)} kW`
const fmtMW = (w: number) => `${(w / 1e6).toFixed(2)} MW`
// Scene text reads in the foreground color; only current (cyan) and power (amber) tint their labels
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const accentClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums'
const axisClass = 'pointer-events-none whitespace-nowrap text-[10px] tabular-nums text-muted-foreground'

// Shared scratch, written then handed straight to a geometry, so no frame ever allocates
const SEG = new Float32Array(6)
function setSeg(
  line: any,
  x0: number, y0: number, z0: number,
  x1: number, y1: number, z1: number
) {
  SEG[0] = x0; SEG[1] = y0; SEG[2] = z0
  SEG[3] = x1; SEG[4] = y1; SEG[5] = z1
  line?.geometry.setPositions(SEG)
  line?.computeLineDistances()
}

// ── 3D: the route as a potential landscape (height = line voltage) ──
// Walked in the direction of conventional current:
// A→B the source lifts the bus to Vs, B→C the line sheds e on the way, C→D the load drops Vr, D→A at 0 V.
// Raised and shallowed so the front of the loop and its labels clear the bottom slider HUD
const BASE_Y = -0.95
const HEIGHT_UNITS = 2.0 // scene units for the tallest sending voltage
const HEIGHT_PER_VOLT = HEIGHT_UNITS / VS_MAX
const RULER_VOLTS = [0, VR]
const A = new THREE.Vector3(-1.9, 0, 0.95)
const B = new THREE.Vector3(-1.9, 0, -0.95)
const C = new THREE.Vector3(0.9, 0, -0.95)
const D = new THREE.Vector3(0.9, 0, 0.95)
const LEN_AB = A.distanceTo(B)
const LEN_BC = B.distanceTo(C)
const LOOP_LEN = 2 * (LEN_AB + LEN_BC)

const PATH_POINTS = 181
const FLOW_SPEED = 0.018 // scene units per second per amp (on-screen sheen, not drift velocity)
const SHEEN_BANDS = 14 // bands along the loop in TrailLine's shader

/** Point at arc-length fraction s ∈ [0, 1) along the loop, with height set by the line voltage. */
function pathPoint(s: number, hs: number, hr: number, out: THREE.Vector3) {
  let d = s * LOOP_LEN
  if (d < LEN_AB) {
    const f = d / LEN_AB
    return out.lerpVectors(A, B, f).setY(BASE_Y + hs * f)
  }
  d -= LEN_AB
  if (d < LEN_BC) {
    const f = d / LEN_BC
    return out.lerpVectors(B, C, f).setY(BASE_Y + hs + (hr - hs) * f)
  }
  d -= LEN_BC
  if (d < LEN_AB) {
    const f = d / LEN_AB
    return out.lerpVectors(C, D, f).setY(BASE_Y + hr * (1 - f))
  }
  d -= LEN_AB
  return out.lerpVectors(D, A, d / LEN_BC).setY(BASE_Y)
}

const PATH_INIT: [number, number, number][] = Array.from(
  { length: PATH_POINTS },
  (): [number, number, number] => [0, 0, 0]
)
const SEG_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]
const FOOTPRINT: [number, number, number][] = [A, B, C, D, A].map(
  (p): [number, number, number] => [p.x, BASE_Y, p.z]
)
const X_AXIS = new THREE.Vector3(1, 0, 0)
// Scratch vectors reused every frame
const tmp = new THREE.Vector3()
const top = new THREE.Vector3()
const bot = new THREE.Vector3()
const dir = new THREE.Vector3()

function LineDropScene() {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  const rise = useRiseIn()
  const pathBuf = useMemo(() => new Float32Array(PATH_POINTS * 3), [])
  const pathRef = useRef<any>(null)
  // 0 riser at B, 1 riser at C, 2 sending level carried across, 3 the e dimension
  const lineRefs = useRef<any[]>([])
  // 0 Vs, 1 Vr, 2 e, 3 I, 4 Ploss
  const tagRefs = useRef<(THREE.Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const cmpRefs = useRef<(HTMLSpanElement | null)[]>([])
  const resistorRef = useRef<THREE.Group>(null)
  const sendDotRef = useRef<THREE.Group>(null)
  const recvDotRef = useRef<THREE.Group>(null)
  const labelledFor = useRef<Record<string, number> | null>(null)

  useFrame(() => {
    const s = solve(valuesRef.current)
    const k = HEIGHT_PER_VOLT * rise.current
    const hs = s.Vs * k
    const hr = VR * k

    // The voltage profile all the way round the loop
    for (let i = 0; i < PATH_POINTS; i++) {
      pathPoint(i / (PATH_POINTS - 1), hs, hr, tmp)
      pathBuf[i * 3] = tmp.x
      pathBuf[i * 3 + 1] = tmp.y
      pathBuf[i * 3 + 2] = tmp.z
    }
    pathRef.current?.geometry.setPositions(pathBuf)

    // Risers down to the footprint, so height reads as height
    setSeg(lineRefs.current[0], B.x, BASE_Y, B.z, B.x, BASE_Y + hs, B.z)
    setSeg(lineRefs.current[1], C.x, BASE_Y, C.z, C.x, BASE_Y + hr, C.z)
    // The sending level carried across to the receiving end, and the gap it leaves open: that gap is e
    setSeg(lineRefs.current[2], B.x, BASE_Y + hs, B.z, C.x + 0.3, BASE_Y + hs, C.z)
    setSeg(lineRefs.current[3], C.x, BASE_Y + hr, C.z, C.x, BASE_Y + hs, C.z)

    // The line's resistance: a glass tube along B→C, thinner for larger R, filament thickening with loss
    top.set(B.x, BASE_Y + hs, B.z)
    bot.set(C.x, BASE_Y + hr, C.z)
    dir.subVectors(bot, top)
    const length = dir.length()
    const g = resistorRef.current
    if (g) {
      g.position.addVectors(top, bot).multiplyScalar(0.5)
      g.quaternion.setFromUnitVectors(X_AXIS, dir.normalize())
      g.scale.set(length * 0.78, 0.12 * (R_MIN / s.R) ** 0.25, 0.12 * (R_MIN / s.R) ** 0.25)
    }
    setResistorHeat(g, Math.sqrt(s.loss / LOSS_MAX) * 1.4) // sqrt: filament area tracks the loss

    sendDotRef.current?.position.set(B.x, BASE_Y + hs, B.z)
    recvDotRef.current?.position.set(C.x, BASE_Y + hr, C.z)

    // Vs sits to the right of the bus so it never crowds the ruler's 22900 V tick on the left
    tagRefs.current[0]?.position.set(B.x + 0.45, BASE_Y + hs + 0.26, B.z)
    tagRefs.current[1]?.position.set(C.x + 0.52, BASE_Y + hr - 0.26, C.z)
    tagRefs.current[2]?.position.set(C.x + 0.52, BASE_Y + hs + 0.2, C.z)
    tagRefs.current[3]?.position.set((A.x + D.x) / 2, BASE_Y - 0.22, A.z)
    tagRefs.current[4]?.position.set((B.x + C.x) / 2, BASE_Y + (hs + hr) / 2 + 0.34, B.z)

    // Readouts move only with the sliders, so the text is rebuilt only then
    if (labelledFor.current !== valuesRef.current && cmpRefs.current[1]) {
      labelledFor.current = valuesRef.current
      if (textRefs.current[0]) textRefs.current[0]!.textContent = `Vs = ${fmtKV(s.Vs)}`
      if (textRefs.current[1]) textRefs.current[1]!.textContent = `Vr = ${fmtKV(VR)}`
      if (textRefs.current[2]) {
        textRefs.current[2]!.textContent = `e = ${s.e.toFixed(0)} V (${s.drop.toFixed(2)} %)`
      }
      if (textRefs.current[3]) textRefs.current[3]!.textContent = `I = ${s.I.toFixed(0)} A`
      if (textRefs.current[4]) {
        textRefs.current[4]!.textContent = `Ploss = 3·I²·R = ${fmtKW(s.loss)}`
      }
      if (cmpRefs.current[0]) {
        cmpRefs.current[0]!.textContent = `P = ${fmtMW(s.P)} @ ${fmtKV(VR)} → Ploss = ${fmtKW(s.loss)}`
      }
      cmpRefs.current[1]!.textContent =
        `P = ${fmtMW(s.P)} @ ${fmtKV(2 * VR)} → Ploss = ${fmtKW(s.lossAtDoubleV)} (÷ 4)`
    }
  })

  const segs: { color: string; width: number; dashed?: boolean }[] = [
    { color: c.guide, width: 1, dashed: true },
    { color: c.guide, width: 1, dashed: true },
    { color: c.guide, width: 1, dashed: true },
    { color: c.voltage, width: 2.5 },
  ]
  const tags: { accent?: string }[] = [{}, {}, {}, { accent: c.current }, { accent: c.power }]

  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />

      {/* Background, lights, grid, and bloom come from the shared SceneStage */}
      <Line points={FOOTPRINT} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      <FloorCaption position={[A.x + 0.75, BASE_Y, A.z + 0.38]} />
      <PotentialRuler
        x={B.x - 0.38}
        z={B.z}
        baseY={BASE_Y}
        volts={RULER_VOLTS}
        unitsPerVolt={() => HEIGHT_PER_VOLT * rise.current}
      />

      {/* The load current itself: a liquid loop at the height of the voltage, sheen flowing ∝ I */}
      <TrailLine
        ref={pathRef}
        points={PATH_INIT}
        lineWidth={4}
        color={c.current}
        fade={false}
        flowSpeed={() => (solve(valuesRef.current).I * FLOW_SPEED * SHEEN_BANDS) / LOOP_LEN}
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
          gapSize={0.055}
        />
      ))}

      <Resistor ref={resistorRef} />
      <Marker ref={sendDotRef} color={c.marker} size={0.055} />
      <Marker ref={recvDotRef} color={c.marker} size={0.055} />

      {tags.map((tag, i) => (
        <group key={i} ref={(g) => { tagRefs.current[i] = g }}>
          <Html center>
            <span
              ref={(el) => { textRefs.current[i] = el }}
              className={tag.accent ? accentClass : labelClass}
              style={tag.accent ? { color: tag.accent } : undefined}
            />
          </Html>
        </group>
      ))}

      {/* Raise the voltage, the loss falls as the square: the same power priced at two voltages.
          Kept at the top of the scene, clear of the bottom slider HUD. */}
      <Html position={[0.1, 1.72, 0]} center>
        <div className="pointer-events-none leading-4" style={{ color: c.power }}>
          <div ref={(el) => { cmpRefs.current[0] = el }} className={accentClass} />
          <div ref={(el) => { cmpRefs.current[1] = el }} className={accentClass} />
        </div>
      </Html>
    </>
  )
}

// ── 2D: the loss against the transmission voltage, at the power the sliders are delivering ──
// Heights are the loss relative to its value at 22.9 kV, so the curve is exactly 1 ÷ V² and the
// marker at twice the voltage literally sits a quarter as high.
const X0 = -3.4
const X1 = 3.2
const Y0 = -1.45
const PLOT_H = 2.9
const xOfV = (v: number) => X0 + ((v - VR) / (2 * VR)) * (X1 - X0)
const yOfRatio = (r: number) => Y0 + r * PLOT_H

const CURVE_N = 181
const CURVE: [number, number, number][] = Array.from(
  { length: CURVE_N },
  (_, i): [number, number, number] => {
    const v = VR + (i / (CURVE_N - 1)) * 2 * VR
    return [xOfV(v), yOfRatio((VR / v) ** 2), 0]
  }
)
const AXES: [number, number, number][] = [
  [X0 - 0.18, Y0 + PLOT_H + 0.22, 0], [X0 - 0.18, Y0, 0], [X1 + 0.22, Y0, 0],
]
const QUARTER_GUIDE: [number, number, number][] = [
  [X0 - 0.18, yOfRatio(0.25), 0], [xOfV(2 * VR), yOfRatio(0.25), 0], [xOfV(2 * VR), Y0, 0],
]
const BASE_GUIDE: [number, number, number][] = [
  [xOfV(VR), yOfRatio(1), 0], [xOfV(VR), Y0, 0],
]
const RATIO_TICKS: { r: number; text: string }[] = [
  { r: 1, text: '1' },
  { r: 0.25, text: '1 ÷ 4' },
  { r: 1 / 9, text: '1 ÷ 9' },
]
const V_TICKS: { v: number; text: string }[] = [
  { v: VR, text: `${(VR / 1000).toFixed(1)} kV` },
  { v: 2 * VR, text: `${(2 * VR) / 1000} kV` },
  { v: 3 * VR, text: `${(3 * VR) / 1000} kV` },
]

function LineDrop2DScene() {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  // 0 loss at Vr, 1 loss at 2·Vr, 2 the drop readout
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<Record<string, number> | null>(null)

  useFrame(() => {
    if (labelledFor.current === valuesRef.current || !textRefs.current[2]) return
    labelledFor.current = valuesRef.current
    const s = solve(valuesRef.current)
    if (textRefs.current[0]) textRefs.current[0]!.textContent = `Ploss = ${fmtKW(s.loss)}`
    if (textRefs.current[1]) {
      textRefs.current[1]!.textContent = `${fmtKW(s.lossAtDoubleV)} = Ploss ÷ 4`
    }
    textRefs.current[2]!.textContent =
      `P = ${fmtMW(s.P)} · e = ${s.e.toFixed(0)} V (${s.drop.toFixed(2)} %)`
  })

  return (
    <>
      <Line points={AXES} lineWidth={1} color={c.guide} />
      <Line
        points={QUARTER_GUIDE}
        lineWidth={1}
        color={c.guide}
        dashed
        dashSize={0.08}
        gapSize={0.06}
      />
      <Line
        points={BASE_GUIDE}
        lineWidth={1}
        color={c.guide}
        dashed
        dashSize={0.08}
        gapSize={0.06}
      />
      {/* loss ∝ 1 ÷ V² at fixed power */}
      <Line points={CURVE} lineWidth={3} color={c.power} />

      <group position={[xOfV(VR), yOfRatio(1), 0.01]}>
        <Marker color={c.marker} size={0.065} />
      </group>
      <group position={[xOfV(2 * VR), yOfRatio(0.25), 0.01]}>
        <Marker color={c.marker} size={0.065} />
      </group>

      <Html position={[xOfV(VR) + 0.16, yOfRatio(1) + 0.17, 0]}>
        <span
          ref={(el) => { textRefs.current[0] = el }}
          className={accentClass}
          style={{ color: c.power }}
        />
      </Html>
      <Html position={[xOfV(2 * VR) + 0.16, yOfRatio(0.25) + 0.17, 0]}>
        <span
          ref={(el) => { textRefs.current[1] = el }}
          className={accentClass}
          style={{ color: c.power }}
        />
      </Html>
      <Html position={[(X0 + X1) / 2, Y0 - 0.45, 0]} center>
        <span ref={(el) => { textRefs.current[2] = el }} className={labelClass} />
      </Html>

      {RATIO_TICKS.map((t) => (
        <Html key={t.text} position={[X0 - 0.3, yOfRatio(t.r), 0]} center>
          <span className={axisClass} style={{ transform: 'translateX(-50%)' }}>{t.text}</span>
        </Html>
      ))}
      {V_TICKS.map((t) => (
        <Html key={t.text} position={[xOfV(t.v), Y0 - 0.2, 0]} center>
          <span className={axisClass}>{t.text}</span>
        </Html>
      ))}
      <Html position={[X0 - 0.1, Y0 + PLOT_H + 0.4, 0]} center>
        <span className={axisClass}>Ploss ÷ Ploss(22.9 kV)</span>
      </Html>
      <Html position={[X1 + 0.45, Y0, 0]} center>
        <span className={axisClass}>V</span>
      </Html>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'line-voltage-drop',
  title: { en: 'Line Voltage Drop & Loss', ko: '송전선로 전압강하와 손실' },
  subject: 'power-systems',
  order: 1,
  description: {
    en: `## Transmission Line Voltage Drop and Loss

**A line has resistance and reactance, so the receiving end always sits below the sending end — and the current that opens that gap also burns I²R in the conductors.**

The route is drawn as a potential landscape: height = line voltage, floor = 0 V.
- Height of the loop: the line voltage — the source lifts the bus to Vs on the left, the load lets it back down to 0 V on the right
- The sloping stretch between the two markers: the line itself, shedding e on the way
- e: the gap between the dashed sending level and the receiving end, labelled with the drop rate
- Glass tube along the line: the line resistance R; thinner means larger R
- Amber filament: the power lost as heat inside that resistance
- Cyan liquid line: the load current I; the faster its sheen flows, the larger I
- Amber text: the loss, and the same power priced at twice the voltage

**Voltage drop** (standard approximation): e = sqrt(3) × I × (R × cosθ + X × sinθ)

**Sending end**: Vs = Vr + e, and **drop rate** = e ÷ Vr × 100 %

**Line loss**: Ploss = 3 × I² × R

**Power and current**: P = sqrt(3) × Vr × I × cosθ, so I = P ÷ (sqrt(3) × Vr × cosθ)

The drop formula is the usual engineering approximation: the receiving voltage is taken as the phase reference and the quadrature part sqrt(3) × I × (X × cosθ − R × sinθ) is dropped, which is accurate to a fraction of a percent at the few-percent drops a real line is designed for.

Why transmission is high voltage: for a fixed delivered power P, raising V lowers I in exact proportion, and the loss follows I², so Ploss = P² × R ÷ (V² × cos²θ) ∝ 1 ÷ V². Double the voltage and the loss falls to a quarter. The 2D plot is that law, with the marker at 45.8 kV sitting literally a quarter as high as the one at 22.9 kV.

Note how little of the voltage the drop actually is: a few percent of 22.9 kV is a small step in the landscape, which is exactly how a line is designed. Raise I to open the gap and brighten the filament together. Raise R and the tube thins while both the drop and the loss grow. Raise X and only the drop grows — reactance costs voltage but no heat. Lower cosθ and the X × sinθ term takes over, so a poor power factor drops more voltage at the very same current.

Worked example: R = 5 Ω, X = 8 Ω, I = 100 A, cosθ = 0.8 into 22.9 kV gives e = 1524 V, Vs = 24.42 kV, a drop rate of 6.66 %. Delivering 3 MW at 22.9 kV costs 134.1 kW of loss; the same 3 MW at 45.8 kV costs 33.5 kW.`,
    ko: `## 송전선로의 전압강하와 손실

**선로에는 저항과 리액턴스가 있으므로 수전단은 언제나 송전단보다 낮습니다. 그 차이를 만드는 전류가 동시에 도체에서 I²R로 타 없어집니다.**

송전 경로를 전위 지형으로 그렸습니다. 높이 = 선간전압, 바닥 = 0 V입니다.
- 회로의 높이: 선간전압 — 왼쪽에서 전원이 모선을 Vs까지 끌어올리고, 오른쪽에서 부하가 0 V로 되돌립니다
- 두 마커 사이의 기울어진 구간: 선로 자체. 지나가면서 e만큼을 떨어뜨립니다
- e: 점선으로 그은 송전단 높이와 수전단 사이의 간격. 전압강하율을 함께 적었습니다
- 선로를 따라 놓인 유리관: 선로 저항 R. 가늘수록 R이 큽니다
- 앰버 필라멘트: 그 저항 안에서 열로 사라지는 전력
- 시안 액체 선: 부하 전류 I. 광택이 빨리 흐를수록 I가 큽니다
- 앰버 글자: 손실, 그리고 같은 전력을 두 배 전압으로 보냈을 때의 손실

**전압강하** (표준 근사식): e = sqrt(3) × I × (R × cosθ + X × sinθ)

**송전단 전압**: Vs = Vr + e, **전압강하율** = e ÷ Vr × 100 %

**선로 손실**: Ploss = 3 × I² × R

**전력과 전류**: P = sqrt(3) × Vr × I × cosθ, 따라서 I = P ÷ (sqrt(3) × Vr × cosθ)

이 전압강하 식은 실무에서 쓰는 근사식입니다. 수전단 전압을 위상 기준으로 잡고 직각 성분 sqrt(3) × I × (X × cosθ − R × sinθ)를 버린 것이며, 실제 선로가 설계되는 수 % 수준의 강하에서는 오차가 1 %에도 한참 못 미칩니다.

송전을 고압으로 하는 이유가 여기 있습니다. 보내는 전력 P가 정해져 있으면 V를 올리는 만큼 I가 정확히 반비례로 줄고, 손실은 I²를 따라가므로 Ploss = P² × R ÷ (V² × cos²θ) ∝ 1 ÷ V²가 됩니다. 전압을 두 배로 올리면 손실은 1/4로 떨어집니다. 2D 그래프가 바로 그 법칙이며, 45.8 kV 마커는 22.9 kV 마커의 정확히 1/4 높이에 놓여 있습니다.

전압강하가 전압에 비해 얼마나 작은지도 함께 보십시오. 22.9 kV의 수 %는 지형에서 아주 낮은 단차이고, 선로는 원래 그렇게 설계합니다. I를 올리면 간격이 벌어지고 필라멘트가 함께 굵어집니다. R을 올리면 관이 가늘어지면서 강하와 손실이 같이 커집니다. X를 올리면 강하만 커집니다. 리액턴스는 전압은 먹지만 열은 내지 않습니다. cosθ를 낮추면 X × sinθ 항이 주도하므로, 전류가 똑같아도 역률이 나쁘면 전압이 더 많이 떨어집니다.

계산 예: R = 5 Ω, X = 8 Ω, I = 100 A, cosθ = 0.8으로 22.9 kV에 보내면 e = 1524 V, Vs = 24.42 kV, 전압강하율 6.66 %입니다. 22.9 kV로 3 MW를 보내면 손실이 134.1 kW이고, 같은 3 MW를 45.8 kV로 보내면 33.5 kW입니다.`,
  },
  sliderParams: [
    { key: 'I',      label: { en: 'Load Current (I)',     ko: '부하 전류 (I)' },   unit: 'A', min: 20,      max: I_MAX, step: 5,    defaultValue: 100 },
    { key: 'R',      label: { en: 'Line Resistance (R)',  ko: '선로 저항 (R)' },   unit: 'Ω', min: R_MIN,   max: R_MAX, step: 0.5,  defaultValue: 5   },
    { key: 'X',      label: { en: 'Line Reactance (X)',   ko: '선로 리액턴스 (X)' }, unit: 'Ω', min: 0,     max: X_MAX, step: 0.5,  defaultValue: 8   },
    { key: 'cosphi', label: { en: 'Power Factor (cosθ)',  ko: '역률 (cosθ)' },     unit: '',  min: COS_MIN, max: 1,     step: 0.01, defaultValue: 0.8 },
  ],
  Scene: LineDropScene,
  Scene2D: LineDrop2DScene,
}

export default concept
