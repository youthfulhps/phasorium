import { useRef, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import { FloorCaption, PotentialRuler, useRiseIn } from '../../components/scene/PotentialLandscape'
import { Resistor, setResistorHeat } from '../../components/scene/Resistor'
import { setOpacity, useSettledLabels, useStoryReveal } from '../../components/scene/storyReveal'
import type { ConceptMeta } from '../../types/concept'

const VM = 10 // source peak voltage [V]
const DISPLAY_OMEGA = 1.2 // on-screen rad/s: real 10–500 Hz is far too fast to follow

// Pure physics, current as the phase reference: i = Im·sin θ, vR in phase, vL leads 90°, vC lags 90°.
// `wL` and `wC` scale the two reactances, so a narration step that has not introduced the coil or
// the plates yet gets the circuit that is actually on screen: wL = wC = 0 is a plain R on AC.
// Worked example: R = 100 Ω, L = 100 mH, C = 10 μF, f = 100 Hz →
// XL = 62.8 Ω, XC = 159.2 Ω, Z = 138.8 Ω, φ = −43.9°, Im = 72.0 mA, VR = 7.20 V, VL = 4.53 V, VC = 11.46 V.
function solve({ R = 100, L_mH = 100, C_uF = 10, freq = 100 }: Record<string, number>, wL = 1, wC = 1) {
  const omega = 2 * Math.PI * freq
  const XL = omega * L_mH * 1e-3 * wL
  const XC = (1 / (omega * C_uF * 1e-6)) * wC
  const Z = Math.hypot(R, XL - XC)
  const phi = Math.atan2(XL - XC, R)
  const Im = VM / Z
  return { R, XL, XC, Z, phi, Im, VR: Im * R, VL: Im * XL, VC: Im * XC }
}

function useSliderValues() {
  const valuesRef = useRef<Record<string, number>>(useSliderStore.getState().values)
  useEffect(
    () => useSliderStore.subscribe((s) => s.values, (v) => { valuesRef.current = v }),
    []
  )
  return valuesRef
}

const fmtV = (v: number) => `${v.toFixed(v < 10 ? 2 : 1)} V`
const fmtI = (a: number) => (a < 1 ? `${(a * 1e3).toFixed(1)} mA` : `${a.toFixed(2)} A`)
// Labels read in the foreground color; only accent quantities (current) tint their text
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'

// ── 3D: the same potential landscape as Ohm's law, now alternating ──
// Walk the loop in the direction of positive current: source (rises by vs), R, L, C (each drops by
// its own voltage), then back to ground. At every instant vs = vR + vL + vC, so the loop closes (KVL).
const BASE_Y = -0.3
const A = new THREE.Vector3(-1.8, 0, 1.2)
const B = new THREE.Vector3(-1.8, 0, -1.2)
const C = new THREE.Vector3(1.0, 0, -1.2)
const D = new THREE.Vector3(1.0, 0, 1.2)
const CORNERS = [A, B, C, D, A]
const SIDES = [2.4, 2.8, 2.4, 2.8] // |AB|, |BC|, |CD|, |DA|
const LOOP_LEN = 10.4
// Distances along the loop where each element starts and ends
const SRC: [number, number] = [0, 2.4]
const EL_R: [number, number] = [2.7, 3.6]
const EL_L: [number, number] = [3.95, 4.85]
const EL_C: [number, number] = [6.0, 6.8]
const SPANS = [EL_R, EL_L, EL_C]
const RULER_VOLTS = [-VM, 0, VM]

const PATH_POINTS = 241
const FLOW_AT_RESONANCE = 3.5 // sheen bands per second at peak current when Im is at its resonant maximum

/** Horizontal position at distance d along the loop. */
function loopXZ(d: number, out: THREE.Vector3) {
  let rest = ((d % LOOP_LEN) + LOOP_LEN) % LOOP_LEN
  for (let k = 0; k < 4; k++) {
    if (rest <= SIDES[k] || k === 3) return out.lerpVectors(CORNERS[k], CORNERS[k + 1], Math.min(rest / SIDES[k], 1))
    rest -= SIDES[k]
  }
  return out
}

const ramp = (d: number, [a, b]: [number, number]) => Math.min(Math.max((d - a) / (b - a), 0), 1)

/** Potential at distance d, given the instantaneous source and element voltages. */
function potential(d: number, vs: number, vR: number, vL: number, vC: number) {
  return vs * ramp(d, SRC) - vR * ramp(d, EL_R) - vL * ramp(d, EL_L) - vC * ramp(d, EL_C)
}

// Coil drawn along local x ∈ [−0.5, 0.5]; its group is scaled to the element's length
const COIL: [number, number, number][] = Array.from({ length: 121 }, (_, i) => {
  const u = i / 120
  const a = u * 7 * 2 * Math.PI
  return [u - 0.5, 0.11 * Math.sin(a), 0.11 * Math.cos(a)]
})
const PATH_INIT: [number, number, number][] = Array.from({ length: PATH_POINTS }, () => [0, 0, 0])
const FOOTPRINT: [number, number, number][] = CORNERS.map((p) => [p.x, BASE_Y, p.z])
const X_AXIS = new THREE.Vector3(1, 0, 0)
const p0 = new THREE.Vector3()
const p1 = new THREE.Vector3()
const dir = new THREE.Vector3()
const unit = new THREE.Vector3()

// ── Narration steps drive what exists on screen. Each reveal eases 0 → 1 so nothing pops in. ──
const REV_LABEL = 0 // the voltage ruler and every readout
const REV_L = 1 // the coil, and with it XL
const REV_C = 2 // the plates, and with it XC
const REV_COUNT = 3

/** Which reveals the sentence at `step` asks for. Step 0 is a bare resistive loop, unlabelled. */
function revealTargets(step: number, out: Float32Array) {
  out[REV_LABEL] = step >= 1 ? 1 : 0
  out[REV_L] = step >= 2 ? 1 : 0
  out[REV_C] = step >= 3 ? 1 : 0
}

function RLCLandscapeScene() {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  const pathBuf = useMemo(() => new Float32Array(PATH_POINTS * 3), [])
  const pathRef = useRef<any>(null)
  const elRefs = useRef<(THREE.Group | null)[]>([]) // R, L, C
  const tagRefs = useRef<(THREE.Group | null)[]>([]) // Vs, R, L, C
  const textRefs = useRef<(HTMLSpanElement | null)[]>([]) // Vs, R, L, C, I
  const flow = useRef(0)
  const rise = useRiseIn()
  const scale = useRef(0) // scene units per volt, shared with the ruler
  const labels = useSettledLabels(revealTargets, REV_COUNT)
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(({ clock }) => {
    const revLabel = reveal[REV_LABEL]
    const wL = reveal[REV_L]
    const wC = reveal[REV_C]
    const s = solve(valuesRef.current, wL, wC)
    const theta = clock.getElapsedTime() * DISPLAY_OMEGA
    const sin = Math.sin(theta)
    const cos = Math.cos(theta)
    const vR = s.VR * sin
    const vL = s.VL * cos // leads i by 90°
    const vC = -s.VC * cos // lags i by 90°
    const vs = vR + vL + vC // = Im·Z·sin(θ + φ)
    // Every node potential is bounded by max(Vm, VC), so this keeps the landscape in frame
    const k = (1.2 / Math.max(VM, s.VC)) * rise.current
    scale.current = k

    for (let i = 0; i < PATH_POINTS; i++) {
      const d = (i / (PATH_POINTS - 1)) * LOOP_LEN
      loopXZ(d, p0)
      pathBuf[i * 3] = p0.x
      pathBuf[i * 3 + 1] = BASE_Y + k * potential(d, vs, vR, vL, vC)
      pathBuf[i * 3 + 2] = p0.z
    }
    pathRef.current?.geometry.setPositions(pathBuf)

    // Current reverses every half cycle; its peak is largest at resonance (Im = Vm / R)
    flow.current = FLOW_AT_RESONANCE * (s.R / s.Z) * sin

    // Seat each element on its stretch of the landscape
    for (let e = 0; e < 3; e++) {
      const [a, b] = SPANS[e]
      loopXZ(a, p0).setY(BASE_Y + k * potential(a, vs, vR, vL, vC))
      loopXZ(b, p1).setY(BASE_Y + k * potential(b, vs, vR, vL, vC))
      dir.subVectors(p1, p0)
      const g = elRefs.current[e]
      if (g) {
        g.position.addVectors(p0, p1).multiplyScalar(0.5)
        g.quaternion.setFromUnitVectors(X_AXIS, unit.copy(dir).normalize())
        // The coil and the plates grow out of nothing when their sentence introduces them
        if (e === 0) g.scale.set(dir.length(), 0.08, 0.08)
        else if (e === 1) g.scale.set(dir.length() * wL, wL, wL)
        else g.scale.setScalar(wC)
      }
      // C sits on the right side, below the coil in camera view, so its tag goes beside it instead
      const [dx, dy] = e === 2 ? [0.5, 0] : [0, 0.35]
      tagRefs.current[e + 1]?.position.set((p0.x + p1.x) / 2 + dx, Math.max(p0.y, p1.y) + dy, (p0.z + p1.z) / 2)
    }
    loopXZ((SRC[0] + SRC[1]) / 2, p0)
    tagRefs.current[0]?.position.set(p0.x - 0.35, BASE_Y + (k * vs) / 2, p0.z)

    // Filament follows instantaneous power p = i²R (pulses at twice the frequency); √p/p_max = |sin|·√(R/Z)
    setResistorHeat(elRefs.current[0], Math.abs(sin) * Math.sqrt(s.R / s.Z))

    // Peak-value labels change with the sliders, and with the step, which decides whether the coil
    // and the plates are in the circuit at all. Read off the step's settled weights rather than the
    // easing ones, so the number a sentence quotes is on screen as soon as the sentence is.
    const w = textRefs.current[4] ? labels.settled(valuesRef.current) : null
    if (w) {
      const q = solve(valuesRef.current, w[REV_L], w[REV_C])
      const texts = [`Vs ${fmtV(VM)}`, `R ${fmtV(q.VR)}`, `L ${fmtV(q.VL)}`, `C ${fmtV(q.VC)}`, `I ${fmtI(q.Im)}`]
      texts.forEach((text, i) => { if (textRefs.current[i]) textRefs.current[i]!.textContent = text })
    }

    setOpacity(textRefs.current[0], revLabel)
    setOpacity(textRefs.current[1], revLabel)
    setOpacity(textRefs.current[2], Math.min(revLabel, wL))
    setOpacity(textRefs.current[3], Math.min(revLabel, wC))
    setOpacity(textRefs.current[4], revLabel)
  })


  return (
    <>
      {/* Camera and orbit target share an x offset to the right, which slides the circuit
          left out from under the narration floating on the right. */}
      <PerspectiveCamera makeDefault position={[3.5, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} target={[0.5, 0, 0]} />

      <Line points={FOOTPRINT} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      <FloorCaption position={[D.x + 0.2, BASE_Y, D.z + 0.45]} />
      <PotentialRuler x={B.x - 0.4} z={B.z} baseY={BASE_Y} volts={RULER_VOLTS} unitsPerVolt={() => scale.current * reveal[REV_LABEL]} />
      <TrailLine
        ref={pathRef}
        points={PATH_INIT}
        lineWidth={4}
        color={c.current}
        fade={false}
        flowSpeed={() => flow.current}
      />

      {/* Resistor: glass tube whose filament follows instantaneous power */}
      <Resistor ref={(g) => { elRefs.current[0] = g }} />
      {/* Inductor: a coil wound around the path */}
      <group ref={(g) => { elRefs.current[1] = g }}>
        <Line points={COIL} lineWidth={1.5} color={c.inductance} />
      </group>
      {/* Capacitor: two plates across the path */}
      <group ref={(g) => { elRefs.current[2] = g }}>
        {[-0.07, 0.07].map((x) => (
          <mesh key={x} position={[x, 0, 0]}>
            <boxGeometry args={[0.02, 0.26, 0.26]} />
            <meshBasicMaterial color={c.capacitance} />
          </mesh>
        ))}
      </group>

      {[0, 1, 2, 3].map((t) => (
        <group key={t} ref={(g) => { tagRefs.current[t] = g }}>
          <Html center>
            <span ref={(el) => { textRefs.current[t] = el }} className={labelClass} />
          </Html>
        </group>
      ))}
      <Html position={[(A.x + D.x) / 2, BASE_Y - 0.3, A.z]} center>
        <span ref={(el) => { textRefs.current[4] = el }} className={labelClass} style={{ color: c.current }} />
      </Html>
    </>
  )
}

// ── 2D: the textbook view — phasor diagram (head to tail) and impedance triangle ──
const PH_ORIGIN = new THREE.Vector2(-3.6, 0)
const Z_ORIGIN = new THREE.Vector2(0.6, 0)
const SEG_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]

function RLCPhasor2DScene() {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  const buf = useMemo(() => new Float32Array(6), [])
  // Phasors: I (reference), VR, VL, VC, Vs · Impedance: R, X, Z
  const lineRefs = useRef<any[]>([])
  const tagRefs = useRef<(THREE.Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const tipRef = useRef<THREE.Group>(null)
  const labels = useSettledLabels(revealTargets, REV_COUNT)
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(() => {
    const revLabel = reveal[REV_LABEL]
    const wL = reveal[REV_L]
    const wC = reveal[REV_C]
    // The VL and VC phasors are Im·X long, so a zero weight shrinks them away on its own
    const s = solve(valuesRef.current, wL, wC)
    const X = s.XL - s.XC
    const set = (i: number, x0: number, y0: number, x1: number, y1: number) => {
      buf[0] = x0; buf[1] = y0; buf[2] = 0; buf[3] = x1; buf[4] = y1; buf[5] = 0
      lineRefs.current[i]?.geometry.setPositions(buf)
      lineRefs.current[i]?.computeLineDistances()
      tagRefs.current[i]?.position.set((x0 + x1) / 2, (y0 + y1) / 2, 0)
    }

    // Phasor diagram, drawn to one voltage scale so lengths compare honestly
    const k = 1.8 / Math.max(VM, s.VR, s.VL, s.VC)
    const { x: ox, y: oy } = PH_ORIGIN
    const xR = ox + s.VR * k
    const yL = oy + s.VL * k
    const yEnd = yL - s.VC * k
    set(0, ox, oy, ox + 0.6, oy) // I, direction only
    set(1, ox, oy, xR, oy) // VR along I
    // VL (leads) up and VC (lags) down from the VR tip; together they leave VL − VC
    set(2, xR, oy, xR, yL)
    set(3, xR, oy, xR, oy - s.VC * k)
    set(4, ox, oy, xR, yEnd) // Vs = VR + VL + VC
    tipRef.current?.position.set(xR, yEnd, 0)

    // Impedance triangle, to its own ohm scale
    const kz = Math.min(3.0 / s.R, 1.8 / Math.max(Math.abs(X), 1e-6))
    const { x: zx, y: zy } = Z_ORIGIN
    set(5, zx, zy, zx + s.R * kz, zy) // R
    set(6, zx + s.R * kz, zy, zx + s.R * kz, zy + X * kz) // X = XL − XC
    set(7, zx, zy, zx + s.R * kz, zy + X * kz) // Z

    const w = textRefs.current[7] ? labels.settled(valuesRef.current) : null
    if (w) {
      const q = solve(valuesRef.current, w[REV_L], w[REV_C])
      const qX = q.XL - q.XC
      const texts = [
        'I', `VR ${fmtV(q.VR)}`, `VL ${fmtV(q.VL)}`, `VC ${fmtV(q.VC)}`, `Vs ${fmtV(VM)}`,
        `R ${q.R.toFixed(0)} Ω`, `X ${qX.toFixed(1)} Ω`, `Z ${q.Z.toFixed(1)} Ω · φ ${((q.phi * 180) / Math.PI).toFixed(1)}°`,
      ]
      texts.forEach((text, i) => { if (textRefs.current[i]) textRefs.current[i]!.textContent = text })
    }

    const reactive = Math.max(wL, wC)
    const fades = [revLabel, revLabel, Math.min(revLabel, wL), Math.min(revLabel, wC), revLabel, revLabel, Math.min(revLabel, reactive), revLabel]
    fades.forEach((v, i) => setOpacity(textRefs.current[i], v))
  })

  const segs: { color: string; width: number; dashed?: boolean; offset: [number, number] }[] = [
    { color: c.current, width: 2.5, offset: [-0.1, 0.22] },
    { color: c.resistance, width: 2.5, offset: [0.25, -0.22] },
    { color: c.inductance, width: 2.5, offset: [0.55, 0] },
    { color: c.capacitance, width: 2.5, offset: [0.55, 0] },
    { color: c.voltage, width: 3, offset: [-0.45, 0.15] },
    { color: c.resistance, width: 2, offset: [0, -0.2] },
    { color: c.guide, width: 2, dashed: true, offset: [0.55, 0] },
    { color: c.voltage, width: 2.5, offset: [-0.95, 0] },
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
            <Html position={[sg.offset[0], sg.offset[1], 0]} center>
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
  id: 'rlc-series',
  title: { en: 'Series RLC Circuit', ko: 'RLC 직렬 회로' },
  subject: 'circuits',
  order: 13,
  description: {
    en: `## Series RLC Circuit

**One current, three voltages that peak at different moments.**

The same potential landscape as Ohm's law, now with AC: height = potential at this instant.
- Cyan liquid line: the current; it flows one way, then the other, every half cycle
- R (glass tube): its drop rises and falls with the current; its filament pulses with i² R
- L (coil): its drop peaks a quarter cycle before the current
- C (plates): its drop peaks a quarter cycle after the current

At every instant the source's rise equals the three drops together (KVL): vs = vR + vL + vC.

**Impedance**: Z = sqrt(R² + (XL − XC)²), with XL = ωL and XC = 1 ÷ (ωC)
**Phase angle**: φ = arctan((XL − XC) ÷ R)

Tune f toward resonance (XL = XC): the L and C drops cancel, Z = R, and the current flows hardest. Note that VL or VC alone can exceed the source voltage.

In 2D: the phasor diagram (VR, VL, VC added head to tail give Vs) and the impedance triangle. Labels show peak values; the time scale is slowed down for viewing.`,
    ko: `## RLC 직렬 회로

**전류는 하나, 전압은 셋 — 각자 다른 순간에 최대가 됩니다.**

옴의 법칙과 같은 전위 지형을 교류로 그렸습니다. 높이 = 이 순간의 전위입니다.
- 시안 액체 선: 전류. 반 주기마다 흐르는 방향이 바뀝니다
- R (유리관): 전압 강하가 전류와 함께 오르내리고, 필라멘트가 i² R에 따라 맥동합니다
- L (코일): 전압 강하가 전류보다 1/4 주기 먼저 최대가 됩니다
- C (두 판): 전압 강하가 전류보다 1/4 주기 늦게 최대가 됩니다

어느 순간이든 전원이 올린 높이는 세 강하의 합과 같습니다(KVL): vs = vR + vL + vC.

**임피던스**: Z = sqrt(R² + (XL − XC)²), XL = ωL, XC = 1 ÷ (ωC)
**위상각**: φ = arctan((XL − XC) ÷ R)

f를 공진(XL = XC) 쪽으로 옮기면 L과 C의 강하가 상쇄되어 Z = R이 되고 전류가 가장 크게 흐릅니다. VL이나 VC 하나가 전원 전압보다 커질 수도 있습니다.

2D: 페이저도(VR, VL, VC를 이어 더하면 Vs)와 임피던스 삼각형입니다. 라벨은 최댓값이며, 시간은 보기 좋게 느리게 재생합니다.`,
  },
  story: [
    {
      set: { R: 100, L_mH: 100, C_uF: 10, freq: 100 },
      text: {
        en: 'An AC source and one resistor. The source lifts one side of the loop, lets it down, lifts it again — on and on. So the current runs one way, then the other, every half cycle.',
        ko: '교류 전원 하나, 저항 하나. 전원이 고리의 한쪽을 들었다 내리고, 또 들었다 내립니다. 그래서 전류도 반 주기마다 방향을 바꿉니다.',
      },
    },
    {
      set: { R: 100, L_mH: 100, C_uF: 10, freq: 100 },
      text: {
        en: 'Height is potential here, so a voltage is a height difference. The source lifts 10 V, the resistor hands all 10 V back, 100 mA flows. Nothing yet that Ohm’s law does not already own. It is just wobbling.',
        ko: '여기서 높이는 전위입니다. 전압은 곧 높이 차입니다. 전원이 10 V 올리면 저항이 10 V를 그대로 돌려주고, 100 mA가 흐릅니다. 아직은 옴의 법칙 그대로입니다. 출렁이고 있을 뿐입니다.',
      },
    },
    {
      set: { R: 100, L_mH: 100, C_uF: 10, freq: 100 },
      text: {
        en: 'Wind a coil into the loop. A coil hates any change in current, so its drop peaks a quarter cycle ahead of the current. `XL = ωL = 62.8 Ω` The current slides to 84.7 mA.',
        ko: '고리에 코일을 감습니다. 코일은 전류가 변하는 걸 싫어합니다. 그래서 전압 강하가 전류보다 1/4 주기 앞섭니다. `XL = ωL = 62.8 Ω` 전류는 84.7 mA로 밀려납니다.',
      },
    },
    {
      set: { R: 100, L_mH: 100, C_uF: 10, freq: 100 },
      text: {
        en: 'Now slip in two plates. A capacitor hates any change in voltage, so its drop peaks a quarter cycle behind the current. `XC = 1 ÷ (ωC) = 159.2 Ω` Current down to 72.0 mA — and look at the plates. 11.5 V. More than the source ever puts out.',
        ko: '이번엔 두 판을 끼웁니다. 커패시터는 전압이 변하는 걸 싫어합니다. 그래서 전압 강하가 전류보다 1/4 주기 뒤집니다. `XC = 1 ÷ (ωC) = 159.2 Ω` 전류는 72.0 mA. 그리고 두 판을 보세요. 11.5 V. 전원이 내놓는 것보다 큽니다.',
      },
    },
    {
      set: { R: 100, L_mH: 100, C_uF: 10, freq: 100 },
      text: {
        en: 'It can do that because the coil and the plates push opposite ways. Only the difference survives. `X = XL − XC = 62.8 − 159.2 = −96.3 Ω` `Z = √(R² + X²) = 138.8 Ω` And the current runs 43.9° ahead of the source.',
        ko: '그게 가능한 건 코일과 두 판이 서로 반대로 밀기 때문입니다. 남는 건 차이뿐입니다. `X = XL − XC = 62.8 − 159.2 = −96.3 Ω` `Z = √(R² + X²) = 138.8 Ω` 전류는 전원보다 43.9° 앞섭니다.',
      },
    },
    {
      set: { R: 100, L_mH: 100, C_uF: 10, freq: 160 },
      text: {
        en: 'So push the frequency up. XL climbs, XC falls, and at 160 Hz they meet: 100.5 Ω against 99.5 Ω. They erase each other. Z collapses to R alone — 100 Ω — and the current hits the biggest it can ever be. 100 mA. This is resonance.',
        ko: '그럼 주파수를 올려 봅니다. XL은 커지고 XC는 작아집니다. 160 Hz에서 둘이 만납니다. 100.5 Ω 대 99.5 Ω. 서로를 지워 버립니다. Z는 R 하나, 100 Ω으로 주저앉고, 전류는 낼 수 있는 최대치에 닿습니다. 100 mA. 이게 공진입니다.',
      },
    },
    {
      set: { R: 20, L_mH: 100, C_uF: 10, freq: 160 },
      text: {
        en: 'Drop R to 20 Ω. Now that 20 Ω is the only thing holding the current back: 499.3 mA. The coil and the plates are carrying 50.2 V and 49.7 V — five times the source. And they still cancel. The source never has to give more than 10 V.',
        ko: 'R을 20 Ω으로 낮춥니다. 이제 전류를 붙잡는 건 그 20 Ω뿐입니다. 499.3 mA. 코일과 두 판에는 각각 50.2 V, 49.7 V가 걸립니다. 전원의 다섯 배입니다. 그런데도 서로 지워집니다. 전원은 끝까지 10 V만 내놓으면 됩니다.',
      },
    },
    {
      set: { R: 100, L_mH: 100, C_uF: 10, freq: 160 },
      text: {
        en: 'XL = XC happens here. `f₀ = 1 ÷ (2π√(LC))` 159.2 Hz for this L and C, and R gets no say in it. Off resonance, `Z = √(R² + (XL − XC)²)` `φ = arctan((XL − XC) ÷ R)` Picking one frequency out this sharply is how a radio finds one station in a sky full of them.',
        ko: 'XL = XC가 되는 지점은 여기입니다. `f₀ = 1 ÷ (2π√(LC))` 지금의 L과 C라면 159.2 Hz. R은 여기 끼어들지 못합니다. 공진에서 벗어나면 `Z = √(R² + (XL − XC)²)` `φ = arctan((XL − XC) ÷ R)` 이렇게 한 주파수만 날카롭게 집어내는 것이, 라디오가 전파로 가득한 하늘에서 방송국 하나를 찾아내는 방법입니다.',
      },
    },
  ],
  sliderParams: [
    { key: 'R',    label: { en: 'Resistance (R)',  ko: '저항 (R)' },       unit: 'Ω',  min: 10, max: 500, step: 10, defaultValue: 100 },
    { key: 'L_mH', label: { en: 'Inductance (L)',  ko: '인덕턴스 (L)' },   unit: 'mH', min: 10, max: 500, step: 10, defaultValue: 100 },
    { key: 'C_uF', label: { en: 'Capacitance (C)', ko: '커패시턴스 (C)' }, unit: 'μF', min: 1,  max: 100, step: 1,  defaultValue: 10  },
    { key: 'freq', label: { en: 'Frequency (f)',   ko: '주파수 (f)' },     unit: 'Hz', min: 10, max: 500, step: 10, defaultValue: 100 },
  ],
  Scene: RLCLandscapeScene,
  Scene2D: RLCPhasor2DScene,
}

export default concept
