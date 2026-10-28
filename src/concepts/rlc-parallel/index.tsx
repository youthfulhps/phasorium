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

const VM = 10 // shared node voltage, peak [V]
const DISPLAY_OMEGA = 1.2 // on-screen rad/s: real 10–500 Hz is far too fast to follow
const HALF_PI = Math.PI / 2

// Pure physics, voltage as the phase reference (the dual of the series case, where current was).
// The three elements share v, so the currents add: I = V·Y with Y = G + j(BC − BL).
// Worked example: R = 100 Ω, L = 100 mH, C = 10 μF, f = 100 Hz →
// G = 0.01000 S, BL = 0.01592 S, BC = 0.00628 S, B = −0.00963 S, |Y| = 0.01388 S, |Z| = 72.0 Ω,
// θ = −43.9° (inductive, the source current lags), resonance at f0 = 159.2 Hz.
// `wR`, `wL` and `wC` scale the three branch admittances. Here a weight of 0 means exactly what it
// looks like — that branch is simply not connected yet — because elements in parallel drop out of
// the sum when they are removed, so a narration step always solves the circuit that is on screen.
function solve(
  { R = 100, L_mH = 100, C_uF = 10, freq = 100 }: Record<string, number>,
  wR = 1, wL = 1, wC = 1,
) {
  const omega = 2 * Math.PI * freq
  const G = wR / R
  const BL = wL / (omega * L_mH * 1e-3)
  const BC = wC * omega * C_uF * 1e-6
  const B = BC - BL
  const Y = Math.hypot(G, B)
  const theta = Math.atan2(B, G)
  return {
    G, BL, BC, B, Y, theta,
    Z: 1 / Y,
    f0: 1 / (2 * Math.PI * Math.sqrt(L_mH * 1e-3 * C_uF * 1e-6)),
    IR: VM * G,
    IL: VM * BL,
    IC: VM * BC,
    I: VM * Y,
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

const fmtI = (a: number) => (a < 1 ? `${(a * 1e3).toFixed(1)} mA` : `${a.toFixed(2)} A`)
const fmtS = (s: number) => (s < 0.1 ? `${(s * 1e3).toFixed(2)} mS` : `${s.toFixed(3)} S`)
const deg = (rad: number) => `${((rad * 180) / Math.PI).toFixed(1)}°`
// Labels read in the foreground color; only accent quantities (current) tint their text
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'

// Narration slots: one branch per step, then the arrow that chains them into the source current.
const REV_R = 0
const REV_L = 1
const REV_C = 2
const REV_SUM = 3 // the source-current arrow, its tip and the admittance hypotenuse
const REV_COUNT = 4

function revealTargets(step: number, out: Float32Array) {
  out[REV_R] = step >= 1 ? 1 : 0
  out[REV_L] = step >= 2 ? 1 : 0
  out[REV_C] = step >= 3 ? 1 : 0
  out[REV_SUM] = step >= 4 ? 1 : 0
}

/** Write one straight segment into a drei Line and park its label group at the midpoint. */
function setSeg(
  buf: Float32Array,
  lines: any[],
  tags: (THREE.Group | null)[],
  i: number,
  x0: number, y0: number, x1: number, y1: number,
) {
  buf[0] = x0; buf[1] = y0; buf[2] = 0
  buf[3] = x1; buf[4] = y1; buf[5] = 0
  lines[i]?.geometry.setPositions(buf)
  lines[i]?.computeLineDistances()
  tags[i]?.position.set((x0 + x1) / 2, (y0 + y1) / 2, 0)
}

// ── 3D: five helices around one axis — radius = amplitude, twist = phase ──
// Each quantity spirals back in time toward −Z; the heads sit in the z = 0 plane, where the
// branch-current arrows are chained head to tail onto the source current.
const HELIX_POINTS = 140
const TURNS = 2
const DEPTH = 4
const OX = -1.0 // helix axis, kept clear of the floating description card
const V_RADIUS = 0.55 // volts and amps cannot share a scale; V is drawn at a fixed reference radius
const I_RADIUS = 1.35 // the largest current fills this radius
const HELIX_INIT: [number, number, number][] = Array.from({ length: HELIX_POINTS }, () => [0, 0, 0])
const SEG_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]
// V, IR, IL, IC, I
const TRACE_WIDTH = [2, 2.5, 2.5, 2.5, 3.5]

function ParallelCurrentHelixScene() {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  const helixBufs = useMemo(
    () => Array.from({ length: 5 }, () => new Float32Array(HELIX_POINTS * 3)),
    []
  )
  const amps = useMemo(() => new Float64Array(5), [])
  const phases = useMemo(() => new Float64Array(5), [])
  const segBuf = useMemo(() => new Float32Array(6), [])
  const helixRefs = useRef<any[]>([])
  const armRefs = useRef<any[]>([]) // V, IR, IL, IC, I
  const tagRefs = useRef<(THREE.Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const tipRef = useRef<THREE.Group>(null)
  const labels = useSettledLabels(revealTargets, REV_COUNT)
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(({ clock }) => {
    const s = solve(valuesRef.current, reveal[REV_R], reveal[REV_L], reveal[REV_C])
    const theta = clock.getElapsedTime() * DISPLAY_OMEGA
    // One current scale for all four currents, so their lengths compare honestly. It is set by the
    // fully connected circuit, so an arrow already on screen keeps its length when the next branch
    // arrives — and it never divides by zero while every branch is still off.
    const full = solve(valuesRef.current)
    const k = I_RADIUS / Math.max(full.IR, full.IL, full.IC, full.I)
    amps[0] = V_RADIUS; phases[0] = 0
    amps[1] = s.IR * k; phases[1] = 0 // in phase with V
    amps[2] = s.IL * k; phases[2] = -HALF_PI // lags V
    amps[3] = s.IC * k; phases[3] = HALF_PI // leads V
    amps[4] = s.I * k; phases[4] = s.theta

    for (let tr = 0; tr < 5; tr++) {
      const buf = helixBufs[tr]
      const a = amps[tr]
      const ph = phases[tr]
      for (let i = 0; i < HELIX_POINTS; i++) {
        const back = 1 - i / (HELIX_POINTS - 1)
        const ang = theta - back * TURNS * 2 * Math.PI + ph
        buf[i * 3] = OX + a * Math.cos(ang)
        buf[i * 3 + 1] = a * Math.sin(ang)
        buf[i * 3 + 2] = -back * DEPTH
      }
      helixRefs.current[tr]?.geometry.setPositions(buf)
    }

    // Arrows in the z = 0 plane: V sets the reference direction, the branch currents chain onto it
    const ux = Math.cos(theta)
    const uy = Math.sin(theta)
    const px = -uy // +90°, the leading direction
    const py = ux
    const rx = OX + amps[1] * ux
    const ry = amps[1] * uy
    const net = amps[3] - amps[2] // (IC − IL), what survives of the two reactive branches
    setSeg(segBuf, armRefs.current, tagRefs.current, 0, OX, 0, OX + V_RADIUS * ux, V_RADIUS * uy)
    setSeg(segBuf, armRefs.current, tagRefs.current, 1, OX, 0, rx, ry)
    setSeg(segBuf, armRefs.current, tagRefs.current, 2, rx, ry, rx - amps[2] * px, ry - amps[2] * py)
    setSeg(segBuf, armRefs.current, tagRefs.current, 3, rx, ry, rx + amps[3] * px, ry + amps[3] * py)
    const ix = rx + net * px
    const iy = ry + net * py
    setSeg(segBuf, armRefs.current, tagRefs.current, 4, OX, 0, ix, iy)
    // Every arrow starts at the same origin, so midpoint labels pile up there whenever the arrows
    // are short. V is a constant, so it reads from a fixed spot under the whole star; the source
    // current reads just past its own tip, which keeps it clear of IR's midpoint at any length.
    tagRefs.current[0]?.position.set(OX, -I_RADIUS - 0.38, 0)
    const ilen = Math.hypot(ix - OX, iy) || 1
    tagRefs.current[4]?.position.set(ix + ((ix - OX) / ilen) * 0.34, iy + (iy / ilen) * 0.34, 0)
    tipRef.current?.position.set(ix, iy, 0)

    // One branch fades in per step, and the source-current arrow once all three are there
    const fades = [1, reveal[REV_R], reveal[REV_L], reveal[REV_C], reveal[REV_SUM]]
    fades.forEach((v, i) => {
      setLineOpacity(helixRefs.current[i], v)
      setLineOpacity(armRefs.current[i], v)
      setOpacity(textRefs.current[i], v)
    })
    tipRef.current?.scale.setScalar(reveal[REV_SUM])

    const w = textRefs.current[4] ? labels.settled(valuesRef.current) : null
    if (w) {
      const q = solve(valuesRef.current, w[REV_R], w[REV_L], w[REV_C])
      if (textRefs.current[0]) textRefs.current[0]!.textContent = `V ${VM.toFixed(1)} V`
      if (textRefs.current[1]) textRefs.current[1]!.textContent = `IR ${fmtI(q.IR)}`
      if (textRefs.current[2]) textRefs.current[2]!.textContent = `IL ${fmtI(q.IL)}`
      if (textRefs.current[3]) textRefs.current[3]!.textContent = `IC ${fmtI(q.IC)}`
      textRefs.current[4]!.textContent = `I ${fmtI(q.I)} · ${deg(q.theta)}`
    }
  })

  const arms: { color: string; width: number; offset: [number, number, number] }[] = [
    { color: c.voltage, width: 2.5, offset: [0, 0.22, 0] },
    { color: c.resistance, width: 2.5, offset: [0, -0.26, 0] },
    { color: c.inductance, width: 2.5, offset: [0.34, 0, 0] },
    { color: c.capacitance, width: 2.5, offset: [-0.34, 0, 0] },
    { color: c.current, width: 3, offset: [0, 0.3, 0] },
  ]
  const traceColors = [c.voltage, c.resistance, c.inductance, c.capacitance, c.current]

  return (
    <>
      {/* Aimed right of centre so the helices and their labels clear the narration panel */}
      <PerspectiveCamera makeDefault position={[3, 2, 5.6]} fov={50} />
      <OrbitControls enablePan={false} target={[0.7, 0, 0]} />

      {/* Background, lights, grid, and bloom come from the shared SceneStage */}
      {traceColors.map((color, tr) => (
        <TrailLine
          key={tr}
          ref={(l) => { helixRefs.current[tr] = l }}
          points={HELIX_INIT}
          lineWidth={TRACE_WIDTH[tr]}
          color={color}
        />
      ))}

      {arms.map((arm, i) => (
        <group key={i}>
          <Line
            ref={(l) => { armRefs.current[i] = l }}
            points={SEG_INIT}
            lineWidth={arm.width}
            color={arm.color}
          />
          <group ref={(g) => { tagRefs.current[i] = g }}>
            <Html position={arm.offset} center>
              <span
                ref={(el) => { textRefs.current[i] = el }}
                className={labelClass}
                style={arm.color === c.current ? { color: arm.color } : undefined}
              />
            </Html>
          </group>
        </group>
      ))}
      <Marker ref={tipRef} color={c.marker} size={0.07} />
    </>
  )
}

// ── 2D: the admittance triangle beside the branch-current sum ──
const PH_ORIGIN = new THREE.Vector2(-3.2, -0.1)
const ADM_ORIGIN = new THREE.Vector2(0.6, -0.4)
const PH_SPAN = 1.5
const ADM_X = 2.6
const ADM_Y = 1.4

function AdmittanceTriangle2DScene() {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  const buf = useMemo(() => new Float32Array(6), [])
  // Currents: V (reference), IR, IL, IC, I · Admittance: G, B, Y
  const lineRefs = useRef<any[]>([])
  const tagRefs = useRef<(THREE.Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const tipRef = useRef<THREE.Group>(null)
  const labels = useSettledLabels(revealTargets, REV_COUNT)
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(() => {
    const s = solve(valuesRef.current, reveal[REV_R], reveal[REV_L], reveal[REV_C])
    const full = solve(valuesRef.current) // both scales come from the fully connected circuit
    const lines = lineRefs.current
    const tags = tagRefs.current

    // Branch currents, drawn to one current scale so lengths compare honestly
    const k = PH_SPAN / Math.max(full.IR, full.IL, full.IC, full.I)
    const ox = PH_ORIGIN.x
    const oy = PH_ORIGIN.y
    const xR = ox + s.IR * k
    const yEnd = oy + (s.IC - s.IL) * k
    setSeg(buf, lines, tags, 0, ox, oy, ox - 0.7, oy) // V, direction only, pointing back from the origin
    setSeg(buf, lines, tags, 1, ox, oy, xR, oy) // IR along V
    // IL (lags) down and IC (leads) up from the IR tip; together they leave IC − IL
    setSeg(buf, lines, tags, 2, xR, oy, xR, oy - s.IL * k)
    setSeg(buf, lines, tags, 3, xR, oy, xR, oy + s.IC * k)
    setSeg(buf, lines, tags, 4, ox, oy, xR, yEnd) // I = IR + IL + IC
    // Past its own tip, for the same reason as in 3D: I's midpoint sweeps straight through IR's label
    const ilen = Math.hypot(xR - ox, yEnd - oy) || 1
    tags[4]?.position.set(xR + ((xR - ox) / ilen) * 0.32, yEnd + ((yEnd - oy) / ilen) * 0.32, 0)
    tipRef.current?.position.set(xR, yEnd, 0)

    // Admittance triangle, to its own siemens scale
    const kz = Math.min(ADM_X / full.G, ADM_Y / Math.max(Math.abs(full.B), 1e-9))
    const ax = ADM_ORIGIN.x
    const ay = ADM_ORIGIN.y
    const gx = ax + s.G * kz
    setSeg(buf, lines, tags, 5, ax, ay, gx, ay) // G
    setSeg(buf, lines, tags, 6, gx, ay, gx, ay + s.B * kz) // B = BC − BL
    setSeg(buf, lines, tags, 7, ax, ay, gx, ay + s.B * kz) // Y
    // G reads near the far end of the base and Y on the outer side of the hypotenuse, which is below
    // it whenever the coil wins — otherwise the two land on each other at the shared corner
    tags[5]?.position.set(ax + (gx - ax) * 0.78, ay, 0)
    tags[7]?.position.set((ax + gx) / 2, ay + s.B * kz * 0.5 + (s.B >= 0 ? 0.3 : -0.3), 0)

    // G arrives with the resistor branch, B with the first reactive one, Y with the sum
    const wSum = reveal[REV_SUM]
    const wB = Math.max(reveal[REV_L], reveal[REV_C])
    const fades = [1, reveal[REV_R], reveal[REV_L], reveal[REV_C], wSum, reveal[REV_R], wB, wSum]
    fades.forEach((v, i) => {
      setLineOpacity(lineRefs.current[i], v)
      setOpacity(textRefs.current[i], v)
    })
    tipRef.current?.scale.setScalar(wSum)

    const w = textRefs.current[7] ? labels.settled(valuesRef.current) : null
    if (w) {
      const q = solve(valuesRef.current, w[REV_R], w[REV_L], w[REV_C])
      if (textRefs.current[0]) textRefs.current[0]!.textContent = `V ${VM.toFixed(1)} V`
      if (textRefs.current[1]) textRefs.current[1]!.textContent = `IR ${fmtI(q.IR)}`
      if (textRefs.current[2]) textRefs.current[2]!.textContent = `IL ${fmtI(q.IL)}`
      if (textRefs.current[3]) textRefs.current[3]!.textContent = `IC ${fmtI(q.IC)}`
      if (textRefs.current[4]) textRefs.current[4]!.textContent = `I ${fmtI(q.I)}`
      if (textRefs.current[5]) textRefs.current[5]!.textContent = `G ${fmtS(q.G)}`
      if (textRefs.current[6]) textRefs.current[6]!.textContent = `B ${fmtS(Math.abs(q.B))} · f0 ${q.f0.toFixed(1)} Hz`
      // With no branch connected yet the circuit really is open, and 1 ÷ 0 should read as such
      const z = q.Y > 0 ? `${q.Z.toFixed(1)} Ω` : '∞'
      textRefs.current[7]!.textContent = `Y ${fmtS(q.Y)} · θ ${deg(q.theta)} · Z ${z}`
    }
  })

  const segs: { color: string; width: number; dashed?: boolean; offset: [number, number, number] }[] = [
    { color: c.voltage, width: 2.5, offset: [0, 0.22, 0] },
    { color: c.resistance, width: 2.5, offset: [0, -0.24, 0] },
    { color: c.inductance, width: 2.5, offset: [-0.5, 0, 0] },
    { color: c.capacitance, width: 2.5, offset: [0.5, 0, 0] },
    { color: c.current, width: 3, offset: [0, 0.24, 0] },
    { color: c.resistance, width: 2, offset: [0, -0.24, 0] },
    { color: c.guide, width: 2, dashed: true, offset: [0.42, 0, 0] },
    { color: c.voltage, width: 2.5, offset: [-0.2, 0, 0] },
  ]

  return (
    <>
      {segs.map((sg, i) => (
        <group key={i}>
          <Line
            ref={(l) => { lineRefs.current[i] = l }}
            points={SEG_INIT}
            lineWidth={sg.width}
            color={sg.color}
            dashed={sg.dashed}
            dashSize={0.08}
            gapSize={0.06}
          />
          <group ref={(g) => { tagRefs.current[i] = g }}>
            <Html position={sg.offset} center>
              <span
                ref={(el) => { textRefs.current[i] = el }}
                className={labelClass}
                style={sg.color === c.current ? { color: sg.color } : undefined}
              />
            </Html>
          </group>
        </group>
      ))}
      <Marker ref={tipRef} color={c.marker} size={0.06} />
    </>
  )
}

const concept: ConceptMeta = {
  id: 'rlc-parallel',
  title: { en: 'Parallel RLC Circuit', ko: 'RLC 병렬 회로' },
  subject: 'circuits',
  order: 14,
  description: {
    en: `## Parallel RLC Circuit

**One voltage, three currents that split between the branches.**

All three elements hang across the same pair of nodes, so they share one v(t) and the source current splits among them. That is the mirror image of the series circuit, where they shared one current and split the voltage — and it flips the bookkeeping from impedance to admittance.
- V: the shared node voltage, the phase reference for everything else
- IR: in phase with V
- IL: lags V by a quarter cycle
- IC: leads V by a quarter cycle
- Cyan liquid line: I, the source current, the sum of the three branch currents

**Admittance**: Y = G + j(BC − BL), with G = 1 ÷ R, BL = 1 ÷ (ωL), BC = ωC
**Magnitude and angle**: B = BC − BL, |Y| = sqrt(G² + B²), θ = arctan(B ÷ G), |Z| = 1 ÷ |Y|
**Branch currents**: IR = V·G, IL = V·BL, IC = V·BC, I = V·|Y|

In 3D each quantity is a helix around one axis: its radius is the amplitude, its twist is the phase, and the newest instant is the head at the front. There the arrows chain head to tail — IR along V, then IL down and IC up — and what is left over is I. Adding them that way is what makes Y = G + j(BC − BL) obvious.

Tune f toward parallel resonance (BC = BL): the two reactive branch currents cancel, the source sees only G, and I shrinks to IR. Yet IL and IC each keep circulating, often far larger than I itself — the dual of the series circuit's voltage magnification.

In 2D: the admittance triangle (G across, B up, Y the hypotenuse, angle θ) beside the branch-current sum. Voltage and current are drawn to separate scales; the time scale is slowed down for viewing.`,
    ko: `## RLC 병렬 회로

**전압은 하나, 전류는 셋 — 가지마다 나뉘어 흐릅니다.**

세 소자가 같은 두 마디에 걸려 있으므로 v(t)를 함께 쓰고, 전원 전류가 세 갈래로 나뉩니다. 전류 하나를 공유하고 전압이 나뉘던 직렬 회로의 거울상이며, 그래서 계산의 기준이 임피던스에서 어드미턴스로 바뀝니다.
- V: 공유되는 마디 전압. 나머지 모두의 위상 기준입니다
- IR: V와 같은 위상
- IL: V보다 1/4 주기 뒤짐
- IC: V보다 1/4 주기 앞섬
- 시안 액체 선: I, 전원 전류. 세 가지 전류의 합입니다

**어드미턴스**: Y = G + j(BC − BL), G = 1 ÷ R, BL = 1 ÷ (ωL), BC = ωC
**크기와 각**: B = BC − BL, |Y| = sqrt(G² + B²), θ = arctan(B ÷ G), |Z| = 1 ÷ |Y|
**가지 전류**: IR = V·G, IL = V·BL, IC = V·BC, I = V·|Y|

3D에서는 각 양이 하나의 축을 감는 나선입니다. 반지름이 진폭, 비틀림이 위상이고, 가장 최근 순간이 앞쪽 머리입니다. 그 앞면에서 화살표를 이어 붙입니다. V를 따라 IR, 거기서 IL은 아래로 IC는 위로, 남는 것이 I입니다. 이렇게 더하는 그림이 Y = G + j(BC − BL)를 한눈에 보여 줍니다.

f를 병렬 공진(BC = BL) 쪽으로 옮기면 두 무효 가지 전류가 상쇄되어 전원은 G만 보게 되고 I는 IR까지 줄어듭니다. 그래도 IL과 IC는 계속 순환하며, 종종 I보다 훨씬 큽니다. 직렬 회로의 전압 확대에 대응하는 현상입니다.

2D: 어드미턴스 삼각형(가로 G, 세로 B, 빗변 Y, 각 θ)과 가지 전류의 합을 나란히 보여 줍니다. 전압과 전류는 서로 다른 축척으로 그렸고, 시간은 보기 좋게 느리게 재생합니다.`,
  },
  story: [
    {
      set: { R: 100, L_mH: 100, C_uF: 10, freq: 100 },
      text: {
        en: 'Three elements hanging across the same two nodes have no choice: they all see the same voltage. That is the only thing they share, and this helix is it.',
        ko: '같은 두 마디에 걸린 소자 셋은 선택권이 없습니다. 전부 같은 전압을 봅니다. 셋이 공유하는 건 그것 하나뿐이고, 이 나선이 바로 그 전압입니다.',
      },
    },
    {
      set: { R: 100, L_mH: 100, C_uF: 10, freq: 100 },
      text: {
        en: 'Connect the resistor branch first. Its current keeps step with the voltage, exactly in phase, and pulls 100.0 mA.',
        ko: '먼저 저항 가지를 연결합니다. 이 전류는 전압과 발을 맞춰 같은 위상으로 돌고, 100.0 mA를 끌어옵니다.',
      },
    },
    {
      set: { R: 100, L_mH: 100, C_uF: 10, freq: 100 },
      text: {
        en: 'Now the coil branch. It lags a quarter cycle behind — and at 100 Hz it is pulling 159.2 mA, more than the resistor does.',
        ko: '이제 코일 가지입니다. 1/4 주기만큼 뒤처져서 돕니다. 그리고 100 Hz에서는 159.2 mA나 흘러서, 저항보다도 많습니다.',
      },
    },
    {
      set: { R: 100, L_mH: 100, C_uF: 10, freq: 100 },
      text: {
        en: 'The capacitor branch does the opposite: a quarter cycle ahead, 62.8 mA. Note that these two point in exactly opposite directions.',
        ko: '커패시터 가지는 정반대입니다. 1/4 주기 앞서고, 62.8 mA입니다. 이 둘이 정확히 반대 방향을 가리킨다는 점을 봐 두세요.',
      },
    },
    {
      set: { R: 100, L_mH: 100, C_uF: 10, freq: 100 },
      text: {
        en: 'So the current the source actually delivers is the three arrows chained head to tail. The two reactive ones cancel down to 96.3 mA, which meets the resistor\'s 100.0 mA at a right angle — 138.8 mA, lagging 43.9°.',
        ko: '그래서 전원이 실제로 내보내는 전류는 세 화살표를 머리-꼬리로 이어 붙인 결과입니다. 무효 가지 둘이 상쇄되고 96.3 mA가 남아서, 저항의 100.0 mA와 직각으로 만납니다. 138.8 mA, 43.9° 뒤처집니다.',
      },
    },
    {
      set: { R: 100, L_mH: 100, C_uF: 10, freq: 160 },
      text: {
        en: 'Tune to 160 Hz and the two opposing arrows become the same length. They wipe each other out, the source sees nothing but the resistor, and I drops back to 100.0 mA with almost no phase left: 0.6°.',
        ko: '160 Hz로 맞추면 서로 반대인 두 화살표의 길이가 같아집니다. 둘이 서로를 지워 버리고, 전원은 저항밖에 보지 못하게 되고, I는 100.0 mA로 돌아오며 위상도 0.6°밖에 남지 않습니다.',
      },
    },
    {
      set: { R: 500, L_mH: 100, C_uF: 10, freq: 160 },
      text: {
        en: 'Here is the part that surprises people. Raise R to 500 Ω and the source current collapses to 20.0 mA — but IL and IC are still 99.5 and 100.5 mA. Five times the source current is circulating between the coil and the capacitor, never passing through the source at all.',
        ko: '여기서부터가 놀라운 부분입니다. R을 500 Ω으로 올리면 전원 전류는 20.0 mA까지 주저앉는데, IL과 IC는 여전히 99.5 mA와 100.5 mA입니다. 전원 전류의 다섯 배가 코일과 커패시터 사이에서만 돌고 있고, 전원 쪽으로는 한 번도 나오지 않습니다.',
      },
    },
    {
      set: { R: 500, L_mH: 100, C_uF: 10, freq: 500 },
      text: {
        en: 'Leave resonance and the balance breaks. At 500 Hz the capacitor branch runs away with 314.2 mA, the coil drops to 31.8 mA, and the source current climbs to 283.0 mA, leading by 85.9°.',
        ko: '공진에서 벗어나면 균형이 깨집니다. 500 Hz에서는 커패시터 가지가 314.2 mA로 치고 나가고 코일은 31.8 mA로 줄며, 전원 전류는 283.0 mA까지 올라가 85.9° 앞섭니다.',
      },
    },
    {
      set: { R: 500, L_mH: 100, C_uF: 10, freq: 10 },
      text: {
        en: 'Go the other way, down to 10 Hz, and the coil takes over: 1.59 A through it, and the source has to supply all of it. So parallel resonance is where the current is smallest — the exact inverse of the series circuit, where it was largest.',
        ko: '반대쪽으로 10 Hz까지 내려가면 이번엔 코일이 다 가져갑니다. 1.59 A가 흐르고, 전원이 그걸 전부 대야 합니다. 그러니까 병렬 공진은 전류가 가장 작아지는 지점입니다. 전류가 가장 커지던 직렬 회로와 정확히 뒤집힌 셈입니다.',
      },
    },
    {
      set: { R: 500, L_mH: 100, C_uF: 10, freq: 160 },
      text: {
        en: 'Series circuits share a current and split the voltage, so you add impedances. Parallel circuits share a voltage and split the current, so you add their reciprocals instead: `Y = G + j(BC − BL)`, and `I = V × |Y|`. Same physics, read from the other end.',
        ko: '직렬 회로는 전류를 공유하고 전압이 나뉘니까 임피던스를 더합니다. 병렬 회로는 전압을 공유하고 전류가 나뉘니까 그 역수를 더합니다. `Y = G + j(BC − BL)`, 그리고 `I = V × |Y|`. 같은 물리를 반대쪽에서 읽은 것뿐입니다.',
      },
    },
  ],
  sliderParams: [
    { key: 'R',    label: { en: 'Resistance (R)',  ko: '저항 (R)' },       unit: 'Ω',  min: 10, max: 500, step: 10, defaultValue: 100 },
    { key: 'L_mH', label: { en: 'Inductance (L)',  ko: '인덕턴스 (L)' },   unit: 'mH', min: 10, max: 500, step: 10, defaultValue: 100 },
    { key: 'C_uF', label: { en: 'Capacitance (C)', ko: '커패시턴스 (C)' }, unit: 'μF', min: 1,  max: 100, step: 1,  defaultValue: 10  },
    { key: 'freq', label: { en: 'Frequency (f)',   ko: '주파수 (f)' },     unit: 'Hz', min: 10, max: 500, step: 10, defaultValue: 100 },
  ],
  Scene: ParallelCurrentHelixScene,
  Scene2D: AdmittanceTriangle2DScene,
}

export default concept
