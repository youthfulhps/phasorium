import { useRef, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { setOpacity, useStoryReveal } from '../../components/scene/storyReveal'
import { Marker } from '../../components/scene/Marker'
import { TrailLine } from '../../components/scene/TrailLine'
import { FloorCaption, PotentialRuler, useRiseIn } from '../../components/scene/PotentialLandscape'
import { Resistor, setResistorHeat } from '../../components/scene/Resistor'
import type { ConceptMeta } from '../../types/concept'

// Slider ranges (also used to normalize the visuals)
const V_MAX = 24 // [V]
const R_MIN = 2 // [Ω]
const I_MAX = V_MAX / R_MIN // 12 A
const P_MAX = V_MAX * I_MAX // 288 W

// Pure physics: Ohm's law and Joule heating. Worked example: V = 12 V, R = 6 Ω → I = 2 A, P = 24 W.
function solve({ V = 12, R = 6 }: Record<string, number>) {
  const I = V / R
  const P = V * I // = I²R = V²/R
  return { I, P }
}

function useSliderValues() {
  const valuesRef = useRef<Record<string, number>>(useSliderStore.getState().values)
  useEffect(
    () => useSliderStore.subscribe((s) => s.values, (v) => { valuesRef.current = v }),
    []
  )
  return valuesRef
}

// ── Narration steps drive what exists on screen. Each reveal eases 0 → 1 so nothing pops in. ──
const REV_HEIGHT = 0 // the battery lifting the loop to its potential, and the voltage ruler
const REV_FLOW = 1 // the sheen travelling round the loop, i.e. the current
const REV_RESISTOR = 2 // the glass tube in the C→D drop
const REV_HEAT = 3 // the filament inside it, i.e. the power turning into heat
const REV_COUNT = 4

/** Which reveals the sentence at `step` asks for. Step 0 is the flat loop, so nothing yet. */
function revealTargets(step: number, out: Float32Array) {
  out[REV_HEIGHT] = step >= 1 ? 1 : 0
  out[REV_FLOW] = step >= 2 ? 1 : 0
  out[REV_RESISTOR] = step >= 3 ? 1 : 0
  out[REV_HEAT] = step >= 6 ? 1 : 0
}

// ── 3D: the circuit as a potential landscape (height = electric potential) ──
// Loop in the XZ plane, walked in the direction of conventional current:
// A→B battery (potential rises by V), B→C wire at V, C→D resistor (drops by V), D→A wire at 0.
const BASE_Y = -0.8
const HEIGHT_PER_VOLT = 2 / V_MAX // 24 V → 2 units
const RULER_VOLTS = [0, 12, 24]
const A = new THREE.Vector3(-1.8, 0, 1.2)
const B = new THREE.Vector3(-1.8, 0, -1.2)
const C = new THREE.Vector3(1.0, 0, -1.2)
const D = new THREE.Vector3(1.0, 0, 1.2)
const LEN_AB = A.distanceTo(B)
const LEN_BC = B.distanceTo(C)
const LEN_CD = C.distanceTo(D)
const LOOP_LEN = 2 * (LEN_AB + LEN_BC)

const PATH_POINTS = 161
const FLOW_SPEED = 0.3 // scene units per second per amp (on-screen, not drift velocity)
const SHEEN_BANDS = 14 // bands along the loop in TrailLine's shader

/** Point at arc-length fraction s ∈ [0, 1) along the loop, with height set by the potential. */
function pathPoint(s: number, height: number, out: THREE.Vector3) {
  let d = s * LOOP_LEN
  if (d < LEN_AB) return out.lerpVectors(A, B, d / LEN_AB).setY(BASE_Y + height * (d / LEN_AB))
  d -= LEN_AB
  if (d < LEN_BC) return out.lerpVectors(B, C, d / LEN_BC).setY(BASE_Y + height)
  d -= LEN_BC
  if (d < LEN_CD) return out.lerpVectors(C, D, d / LEN_CD).setY(BASE_Y + height * (1 - d / LEN_CD))
  d -= LEN_CD
  return out.lerpVectors(D, A, d / LEN_BC).setY(BASE_Y)
}

const PATH_INIT: [number, number, number][] = Array.from({ length: PATH_POINTS }, () => [0, 0, 0])
const DROP_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]
const FOOTPRINT: [number, number, number][] = [A, B, C, D, A].map((p) => [p.x, BASE_Y, p.z])
const X_AXIS = new THREE.Vector3(1, 0, 0)
// Scratch objects reused every frame (no allocations in useFrame)
const tmp = new THREE.Vector3()
const top = new THREE.Vector3()
const dir = new THREE.Vector3()

function OhmsLawScene() {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  const rise = useRiseIn()
  const reveal = useStoryReveal(revealTargets, REV_COUNT)
  const pathBuf = useMemo(() => new Float32Array(PATH_POINTS * 3), [])
  const pathRef = useRef<any>(null)
  const dropB = useRef<any>(null)
  const dropC = useRef<any>(null)
  const dropBuf = useMemo(() => new Float32Array(6), [])
  const resistorRef = useRef<THREE.Group>(null)
  const pLabelPos = useRef<THREE.Group>(null)
  const vLabelPos = useRef<THREE.Group>(null)
  const vLabel = useRef<HTMLSpanElement>(null)
  const iLabel = useRef<HTMLSpanElement>(null)
  const pLabel = useRef<HTMLSpanElement>(null)

  useFrame(() => {
    const { V = 12, R = 6 } = valuesRef.current
    const { I, P } = solve({ V, R })
    const height = V * HEIGHT_PER_VOLT * rise.current * reveal[REV_HEIGHT]

    // Potential profile along the loop
    for (let i = 0; i < PATH_POINTS; i++) {
      pathPoint(i / (PATH_POINTS - 1), height, tmp)
      pathBuf[i * 3] = tmp.x
      pathBuf[i * 3 + 1] = tmp.y
      pathBuf[i * 3 + 2] = tmp.z
    }
    pathRef.current?.geometry.setPositions(pathBuf)

    // Vertical guides from the high wire down to the footprint, so height reads as height
    for (const [ref, p] of [[dropB, B], [dropC, C]] as const) {
      dropBuf[0] = p.x; dropBuf[1] = BASE_Y; dropBuf[2] = p.z
      dropBuf[3] = p.x; dropBuf[4] = BASE_Y + height; dropBuf[5] = p.z
      ref.current?.geometry.setPositions(dropBuf)
      ref.current?.computeLineDistances()
    }

    // Resistor: a glass tube along the C→D drop; thinner for higher R, filament thickens with P
    top.set(C.x, BASE_Y + height, C.z)
    tmp.set(D.x, BASE_Y, D.z)
    dir.subVectors(tmp, top)
    const length = dir.length()
    const radius = 0.16 * Math.sqrt(R_MIN / R)
    const g = resistorRef.current
    if (g) {
      g.position.addVectors(top, tmp).multiplyScalar(0.5)
      g.quaternion.setFromUnitVectors(X_AXIS, dir.normalize())
      const show = reveal[REV_RESISTOR]
      g.scale.set(length * 0.55 * show, radius * show, radius * show)
    }
    setResistorHeat(g, Math.sqrt(P / P_MAX) * 1.4 * reveal[REV_HEAT]) // √: filament area tracks power
    pLabelPos.current?.position.set(C.x + 0.6, BASE_Y + height / 2, 0)

    // Live readouts
    vLabelPos.current?.position.set(A.x - 0.5, BASE_Y + height / 2, (A.z + B.z) / 2)
    if (vLabel.current) vLabel.current.textContent = `V = ${V.toFixed(1)} V`
    if (iLabel.current) iLabel.current.textContent = `I = ${I.toFixed(2)} A`
    if (pLabel.current) pLabel.current.textContent = `P = ${P.toFixed(1)} W`
    setOpacity(vLabel.current, reveal[REV_HEIGHT])
    setOpacity(iLabel.current, reveal[REV_FLOW])
    setOpacity(pLabel.current, reveal[REV_HEAT])
  })

  const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums'

  return (
    <>
      {/* Camera and orbit target share an x offset to the right, which slides the circuit
          left out from under the narration floating on the right. */}
      <PerspectiveCamera makeDefault position={[3.9, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} target={[0.9, 0, 0]} />

      <Line points={FOOTPRINT} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      <FloorCaption position={[D.x + 0.2, BASE_Y, D.z + 0.45]} />
      <PotentialRuler
        x={B.x - 0.4}
        z={B.z}
        baseY={BASE_Y}
        volts={RULER_VOLTS}
        unitsPerVolt={() => HEIGHT_PER_VOLT * rise.current * reveal[REV_HEIGHT]}
      />
      {/* The current itself: a liquid loop whose height is the potential and whose sheen flows at a speed ∝ I */}
      <TrailLine
        ref={pathRef}
        points={PATH_INIT}
        lineWidth={4}
        color={c.current}
        fade={false}
        flowSpeed={() =>
          (solve(valuesRef.current).I * FLOW_SPEED * SHEEN_BANDS * reveal[REV_FLOW]) / LOOP_LEN
        }
      />
      <Line ref={dropB} points={DROP_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.06} gapSize={0.05} />
      <Line ref={dropC} points={DROP_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.06} gapSize={0.05} />


      <Resistor ref={resistorRef} />
      <group ref={pLabelPos}>
        <Html center>
          <span ref={pLabel} className={labelClass} style={{ color: c.power }} />
        </Html>
      </group>

      <group ref={vLabelPos}>
        <Html center>
          <span ref={vLabel} className={`${labelClass} text-foreground/85`} />
        </Html>
      </group>
      <Html position={[(A.x + D.x) / 2, BASE_Y - 0.25, A.z]} center>
        <span ref={iLabel} className={labelClass} style={{ color: c.current }} />
      </Html>
    </>
  )
}

// ── 2D: the V–I plane. Slope of the line = R, area of the rectangle = P ──
const X0 = -3.5
const Y0 = -1.8
const X_PER_AMP = 7 / I_MAX
const Y_PER_VOLT = 3.6 / V_MAX
const AXES: [number, number, number][] = [
  [X0, Y0 + 3.8, 0], [X0, Y0, 0], [X0 + 7.3, Y0, 0],
]
const LINE_INIT: [number, number, number][] = [[X0, Y0, 0], [X0, Y0, 0]]

function OhmsLaw2DScene() {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  const lineBuf = useMemo(() => new Float32Array(6), [])
  const lineRef = useRef<any>(null)
  const dotRef = useRef<THREE.Group>(null)
  const rectRef = useRef<THREE.Mesh>(null)
  const labelPos = useRef<THREE.Group>(null)
  const label = useRef<HTMLSpanElement>(null)

  useFrame(() => {
    const { V = 12, R = 6 } = valuesRef.current
    const { I, P } = solve({ V, R })
    const x = X0 + I * X_PER_AMP
    const y = Y0 + V * Y_PER_VOLT

    // V = I·R through the origin, drawn across the whole plot so its slope reads as R
    const vAtEdge = Math.min(I_MAX * R, V_MAX)
    lineBuf[0] = X0; lineBuf[1] = Y0; lineBuf[2] = 0
    lineBuf[3] = X0 + (vAtEdge / R) * X_PER_AMP; lineBuf[4] = Y0 + vAtEdge * Y_PER_VOLT; lineBuf[5] = 0
    lineRef.current?.geometry.setPositions(lineBuf)

    dotRef.current?.position.set(x, y, 0.01)
    const w = Math.max(x - X0, 1e-3)
    const h = Math.max(y - Y0, 1e-3)
    rectRef.current?.position.set(X0 + w / 2, Y0 + h / 2, -0.01)
    rectRef.current?.scale.set(w, h, 1)

    labelPos.current?.position.set(x, y, 0)
    if (label.current) label.current.textContent = `${I.toFixed(2)} A · ${V.toFixed(1)} V → P = ${P.toFixed(1)} W`
  })

  return (
    <>
      <Line points={AXES} lineWidth={1} color={c.guide} />
      <mesh ref={rectRef}>
        <planeGeometry args={[1, 1]} />
        <meshBasicMaterial color={c.power} transparent opacity={0.28} depthWrite={false} />
      </mesh>
      <Line ref={lineRef} points={LINE_INIT} lineWidth={2.5} color={c.resistance} />
      <Marker ref={dotRef} color={c.marker} />
      <group ref={labelPos}>
        <Html position={[0.2, 0.25, 0]}>
          <span
            ref={label}
            className="pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground"
          />
        </Html>
      </group>
      <Html position={[X0 + 7.3, Y0 - 0.2, 0]} center>
        <span className="pointer-events-none whitespace-nowrap text-[11px] text-muted-foreground">I (A)</span>
      </Html>
      <Html position={[X0 - 0.3, Y0 + 3.8, 0]} center>
        <span className="pointer-events-none whitespace-nowrap text-[11px] text-muted-foreground">V (V)</span>
      </Html>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'ohms-law-power',
  title: { en: "Ohm's Law & Power", ko: '옴의 법칙과 전력' },
  subject: 'circuits',
  order: 2,
  description: {
    en: `## Ohm's Law and Power

**Voltage is height, current is flow, resistance is the bottleneck.**

The circuit is drawn as a potential landscape: height = electric potential.
- Height of the loop: potential — the battery (left ramp) lifts charge by V, the resistor (right ramp) lets it back down
- Cyan liquid line: the current; the faster its sheen flows, the larger I
- Glass tube: the resistor; thinner means larger R
- Amber filament: it thickens inside the resistor as power P turns into heat

**Ohm's law**: V = I × R, so I = V ÷ R

**Power**: P = V × I = I² × R = V² ÷ R

Raise V: the loop rises, the flow speeds up, and the glow grows with V². Raise R: the height stays, but the flow slows and the glow fades.

In 2D, the gray line is V = I × R (its slope is R) and the amber rectangle's area is P.`,
    ko: `## 옴의 법칙과 전력

**전압은 높이, 전류는 흐름, 저항은 병목입니다.**

회로를 전위 지형으로 그렸습니다. 높이 = 전위입니다.
- 회로의 높이: 전위 — 배터리(왼쪽 오르막)가 전하를 V만큼 끌어올리고, 저항(오른쪽 내리막)에서 다시 내려옵니다
- 시안 액체 선: 전류. 광택이 빨리 흐를수록 I가 큽니다
- 유리관: 저항. 가늘수록 R이 큽니다
- 앰버 필라멘트: 전력 P가 열로 바뀔수록 저항 안에서 굵어집니다

**옴의 법칙**: V = I × R, 즉 I = V ÷ R

**전력**: P = V × I = I² × R = V² ÷ R

V를 올리면 회로가 높아지고 흐름이 빨라지며 빛은 V²에 비례해 커집니다. R을 올리면 높이는 그대로지만 흐름이 느려지고 빛이 약해집니다.

2D에서 회색 선은 V = I × R(기울기가 R)이고, 앰버 직사각형의 넓이가 P입니다.`,
  },
  story: [
    {
      set: { V: 12, R: 6 },
      text: {
        en: 'A loop for a charge to run around. Flat. Completely flat. And on flat ground, nothing goes anywhere.',
        ko: '전하가 한 바퀴 돌 수 있는 길입니다. 평평합니다. 완전히 평평합니다. 평평한 땅에서는 아무것도 흐르지 않습니다.',
      },
    },
    {
      set: { V: 12, R: 6 },
      text: {
        en: 'Drop a battery in. One side lifts. That is the entire job of a battery — carry charge up to a height. That height is the voltage. 12 V.',
        ko: '배터리를 넣습니다. 한쪽이 들립니다. 배터리가 하는 일은 이게 전부입니다. 전하를 높은 곳으로 퍼 올리는 것. 그 높이가 전압입니다. 12 V.',
      },
    },
    {
      text: {
        en: 'Now there is a hill, so the charge rolls back down. That rolling is the current. Watch the sheen travel — that speed is how much current you have.',
        ko: '언덕이 생겼으니 전하는 다시 굴러 내려옵니다. 그 굴러 내려오는 것이 전류입니다. 광택이 지나가는 속도를 보세요. 그 속도가 전류의 크기입니다.',
      },
    },
    {
      set: { V: 12, R: 6 },
      text: {
        en: 'But it does not fall free. A narrow tube sits in the way — the resistor. 12 V through 6 Ω gives 2 A. Height divided by bottleneck. `I = V ÷ R` That is all Ohm’s law is.',
        ko: '하지만 그냥 떨어지진 않습니다. 길목에 가느다란 관이 하나 끼워져 있습니다. 저항입니다. 12 V를 6 Ω에 흘리면 2 A. 높이 ÷ 병목. `I = V ÷ R` 옴의 법칙은 이게 전부입니다.',
      },
    },
    {
      set: { V: 12, R: 12 },
      text: {
        en: 'Squeeze the tube. 6 Ω becomes 12 Ω. The height never moved — and the current drops to 1 A. Thinner tube, slower flow. You can watch both happen.',
        ko: '관을 조입니다. 6 Ω이 12 Ω이 됩니다. 높이는 손도 대지 않았는데 전류는 1 A로 떨어집니다. 관이 가늘어지고, 흐름이 느려집니다. 둘 다 눈에 보입니다.',
      },
    },
    {
      set: { V: 24, R: 12 },
      text: {
        en: 'Fine — raise the hill instead. 24 V. Back to 2 A. Current goes up with the height and down with the bottleneck. There is nothing else in it.',
        ko: '그럼 언덕을 높입니다. 24 V. 다시 2 A입니다. 전류는 높이를 따라 오르고 병목을 따라 내립니다. 그 외엔 아무것도 없습니다.',
      },
    },
    {
      set: { V: 24, R: 12 },
      text: {
        en: 'So where did the height the charge gave up actually go? Into heat, inside the resistor. That is the amber filament swelling in the tube. `P = V × I = 24 × 2 = 48 W`',
        ko: '그럼 전하가 내려오며 잃은 높이는 어디로 갔을까요. 저항 속에서 열이 됐습니다. 관 안에서 굵어지는 앰버색 필라멘트, 저게 그 열입니다. `P = V × I = 24 × 2 = 48 W`',
      },
    },
    {
      set: { V: 12, R: 12 },
      text: {
        en: 'Halve the voltage. The current halves with it — and power is their product, so it falls to a quarter. 48 W → 12 W. Heat follows the square. `P = V² ÷ R`',
        ko: '전압을 절반으로 내립니다. 전류도 같이 절반이 됩니다. 전력은 둘의 곱이니 4분의 1로 떨어집니다. 48 W가 12 W로. 열은 제곱을 따라갑니다. `P = V² ÷ R`',
      },
    },
    {
      set: { V: 12, R: 6 },
      text: {
        en: '`V = I × R` `P = V × I = I² × R = V² ÷ R` Three ways of writing one sentence. The sliders are yours now.',
        ko: '`V = I × R` `P = V × I = I² × R = V² ÷ R` 같은 문장을 세 가지로 쓴 것뿐입니다. 이제 슬라이더를 직접 돌려 보세요.',
      },
    },
  ],
  sliderParams: [
    { key: 'V', label: { en: 'Voltage (V)', ko: '전압 (V)' }, unit: 'V', min: 1, max: V_MAX, step: 0.5, defaultValue: 12 },
    { key: 'R', label: { en: 'Resistance (R)', ko: '저항 (R)' }, unit: 'Ω', min: R_MIN, max: 24, step: 0.5, defaultValue: 6 },
  ],
  Scene: OhmsLawScene,
  Scene2D: OhmsLaw2DScene,
}

export default concept
