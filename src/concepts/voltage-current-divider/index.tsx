import { useRef, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import type { LineMaterial } from 'three-stdlib'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { useLocaleStore } from '../../lib/i18n'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import { FloorCaption, PotentialRuler, useRiseIn } from '../../components/scene/PotentialLandscape'
import { Resistor, setResistorHeat } from '../../components/scene/Resistor'
import type { ConceptMeta } from '../../types/concept'

const V_MAX = 24 // [V]
const RT_MIN = 2 // [Ω]
const P_MAX = (V_MAX * V_MAX) / RT_MIN

// Pure physics. α is the tap position measured from ground, so R2 (tap → ground) = α·Rt.
// Worked example: V = 12 V, Rt = 12 Ω, α = 25 % → R1 = 9 Ω, R2 = 3 Ω, Vout = 3 V (= αV),
// and in parallel I = 5.33 A, I1 = 1.33 A, I2 = 4 A, with I1 ÷ I = α.
function solve({ V = 12, Rt = 12, alphaPct = 25 }: Record<string, number>) {
  const a = alphaPct / 100
  const R1 = (1 - a) * Rt // source → tap
  const R2 = a * Rt // tap → ground
  const I1 = V / R1
  const I2 = V / R2
  return {
    V, Rt, a, R1, R2,
    V1: V * (R1 / Rt), // voltage divider: each resistor takes its own share
    Vout: V * (R2 / Rt), // = α·V
    I1, I2, I: I1 + I2, // current divider: each branch takes the *other* resistor's share
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

const fmt = (x: number, unit: string) => `${x.toFixed(x < 10 ? 2 : 1)} ${unit}`
const pct = (x: number) => `${Math.round(x * 100)}%`
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'

// ── 3D: a potentiometer whose tap slides, next to the same two resistors in parallel ──
const BASE_Y = -0.6
const HEIGHT_PER_VOLT = 1.6 / V_MAX
const RULER_VOLTS = [0, 12, 24]
const Z_BACK = -1.0
const Z_FRONT = 1.0
const R_Z: [number, number] = [-0.6, 0.6]
const SEG_PTS = 20

type XZ = [number, number]
// Divider loop: source up the left side, then R1 and R2 along the back wire with the tap between them
const SX0 = -2.6
const SX1 = -0.9
const TAP_FROM = SX0 + 0.3
const TAP_TO = SX1 - 0.3
const DIVIDER_XZ: XZ[] = [
  [SX0, Z_FRONT], [SX0, Z_BACK], // source
  [TAP_FROM, Z_BACK], [0, Z_BACK], [TAP_TO, Z_BACK], // R1 · tap · R2 (tap x is rewritten each frame)
  [SX1, Z_BACK], [SX1, Z_FRONT], [SX0, Z_FRONT],
]
const TAP_INDEX = 3
// The same two resistors in parallel
const PX0 = -0.4
const PXM = 0.5
const PX1 = 1.4
const MAIN_XZ: XZ[] = [[PXM, Z_FRONT], [PX0, Z_FRONT], [PX0, Z_BACK], [PXM, Z_BACK]]
const BRANCH1_XZ: XZ[] = [[PXM, Z_BACK], [PX1, Z_BACK], [PX1, R_Z[0]], [PX1, R_Z[1]], [PX1, Z_FRONT], [PXM, Z_FRONT]]
const BRANCH2_XZ: XZ[] = [[PXM, Z_BACK], [PXM, R_Z[0]], [PXM, R_Z[1]], [PXM, Z_FRONT]]

const dividerV = new Float32Array(DIVIDER_XZ.length)
const mainV = new Float32Array(MAIN_XZ.length)
const branch1V = new Float32Array(BRANCH1_XZ.length)
const branch2V = new Float32Array(BRANCH2_XZ.length)
const LINE_XZ = [DIVIDER_XZ, MAIN_XZ, BRANCH1_XZ, BRANCH2_XZ]
const LINE_V = [dividerV, mainV, branch1V, branch2V]
const lineI = new Float32Array(4)

const pointCount = (xz: XZ[]) => (xz.length - 1) * SEG_PTS + 1
const initPoints = (xz: XZ[]): [number, number, number][] =>
  Array.from({ length: pointCount(xz) }, () => [0, 0, 0])

function fillPath(buf: Float32Array, xz: XZ[], volts: Float32Array, k: number) {
  let n = 0
  for (let s = 0; s < xz.length - 1; s++) {
    for (let j = 0; j < SEG_PTS + (s === xz.length - 2 ? 1 : 0); j++) {
      const u = j / SEG_PTS
      buf[n++] = xz[s][0] + (xz[s + 1][0] - xz[s][0]) * u
      buf[n++] = BASE_Y + k * (volts[s] + (volts[s + 1] - volts[s]) * u)
      buf[n++] = xz[s][1] + (xz[s + 1][1] - xz[s][1]) * u
    }
  }
}

const X_AXIS = new THREE.Vector3(1, 0, 0)
const p0 = new THREE.Vector3()
const p1 = new THREE.Vector3()
const dir = new THREE.Vector3()
const unit = new THREE.Vector3()

function placeResistor(
  g: THREE.Group | null, tag: THREE.Group | null,
  a: XZ, va: number, b: XZ, vb: number, k: number, R: number, P: number
) {
  p0.set(a[0], BASE_Y + k * va, a[1])
  p1.set(b[0], BASE_Y + k * vb, b[1])
  dir.subVectors(p1, p0)
  if (g) {
    g.position.addVectors(p0, p1).multiplyScalar(0.5)
    g.quaternion.setFromUnitVectors(X_AXIS, unit.copy(dir).normalize())
    const r = 0.12 * Math.sqrt(RT_MIN / Math.max(R, 0.2)) + 0.03
    g.scale.set(dir.length(), r, r)
  }
  setResistorHeat(g, Math.sqrt(P / P_MAX) * 1.6)
  tag?.position.set((p0.x + p1.x) / 2, Math.max(p0.y, p1.y) + 0.3, (p0.z + p1.z) / 2)
}

const widthFor = (I: number) => Math.min(1.2 + 1.6 * Math.sqrt(I), 8)
const flowFor = (I: number) => Math.min(0.6 * I, 8)

const TAP_DROP_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]

function DividerScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const rise = useRiseIn()
  const bufs = useMemo(() => LINE_XZ.map((xz) => new Float32Array(pointCount(xz) * 3)), [])
  const inits = useMemo(() => LINE_XZ.map(initPoints), [])
  const dropBuf = useMemo(() => new Float32Array(6), [])
  const lineRefs = useRef<any[]>([])
  const dropRef = useRef<any>(null)
  const resRefs = useRef<(THREE.Group | null)[]>([]) // divider R1, R2, parallel R1, R2
  const tagRefs = useRef<(THREE.Group | null)[]>([]) // 4 resistors + tap
  const textRefs = useRef<(HTMLSpanElement | null)[]>([]) // 4 resistors, tap, divider summary, parallel summary
  const tapRef = useRef<THREE.Group>(null)
  const flows = useRef([0, 0, 0, 0])
  const labelledFor = useRef<unknown>(null)

  useFrame(() => {
    const s = solve(valuesRef.current)
    const { V, R1, R2, Vout } = s
    const k = HEIGHT_PER_VOLT * rise.current

    // The tap slides along the resistor: R1 takes (1 − α) of the span, R2 takes α
    const tapX = TAP_FROM + (1 - s.a) * (TAP_TO - TAP_FROM)
    DIVIDER_XZ[TAP_INDEX][0] = tapX
    dividerV[1] = dividerV[2] = V
    dividerV[TAP_INDEX] = Vout
    mainV[2] = mainV[3] = V
    branch1V[0] = branch1V[1] = branch1V[2] = V
    branch2V[0] = branch2V[1] = V

    lineI[0] = V / s.Rt // one current through the divider
    lineI[1] = s.I; lineI[2] = s.I1; lineI[3] = s.I2
    for (let l = 0; l < 4; l++) {
      const line = lineRefs.current[l]
      if (!line) continue
      fillPath(bufs[l], LINE_XZ[l], LINE_V[l], k)
      line.geometry.setPositions(bufs[l])
      ;(line.material as LineMaterial).linewidth = widthFor(lineI[l])
      flows.current[l] = flowFor(lineI[l])
    }

    // Tap marker and its dashed drop to the floor: its height is Vout
    const tapY = BASE_Y + k * Vout
    tapRef.current?.position.set(tapX, tapY, Z_BACK)
    dropBuf[0] = tapX; dropBuf[1] = BASE_Y; dropBuf[2] = Z_BACK
    dropBuf[3] = tapX; dropBuf[4] = tapY; dropBuf[5] = Z_BACK
    dropRef.current?.geometry.setPositions(dropBuf)
    dropRef.current?.computeLineDistances()
    // Keep the tap label off the resistor labels, which sit on the back wire
    tagRefs.current[4]?.position.set(tapX, tapY + 0.15, Z_BACK + 0.55)

    const Id = V / s.Rt
    placeResistor(resRefs.current[0], tagRefs.current[0], DIVIDER_XZ[2], V, DIVIDER_XZ[3], Vout, k, R1, Id * s.V1)
    placeResistor(resRefs.current[1], tagRefs.current[1], DIVIDER_XZ[3], Vout, DIVIDER_XZ[4], 0, k, R2, Id * Vout)
    placeResistor(resRefs.current[2], tagRefs.current[2], BRANCH1_XZ[2], V, BRANCH1_XZ[3], 0, k, R1, V * s.I1)
    placeResistor(resRefs.current[3], tagRefs.current[3], BRANCH2_XZ[1], V, BRANCH2_XZ[2], 0, k, R2, V * s.I2)

    if (labelledFor.current !== valuesRef.current && textRefs.current[6]) {
      labelledFor.current = valuesRef.current
      const ko = locale === 'ko'
      const texts = [
        `R1 ${fmt(R1, 'Ω')} · ${fmt(s.V1, 'V')}`,
        `R2 ${fmt(R2, 'Ω')} · ${fmt(Vout, 'V')}`,
        `R1 ${fmt(R1, 'Ω')} · ${fmt(s.I1, 'A')}`,
        `R2 ${fmt(R2, 'Ω')} · ${fmt(s.I2, 'A')}`,
        `Vout ${fmt(Vout, 'V')}`,
        `${ko ? '전압 분배' : 'Voltage divider'} · Vout : V1 = R2 : R1 = ${pct(s.a)} : ${pct(1 - s.a)}`,
        `${ko ? '전류 분배' : 'Current divider'} · I1 : I2 = R2 : R1 = ${pct(s.a)} : ${pct(1 - s.a)}`,
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  const footprints = useMemo<[number, number, number][][]>(() => [
    [[SX0, BASE_Y, Z_FRONT], [SX0, BASE_Y, Z_BACK], [SX1, BASE_Y, Z_BACK], [SX1, BASE_Y, Z_FRONT], [SX0, BASE_Y, Z_FRONT]],
    [[PX0, BASE_Y, Z_FRONT], [PX0, BASE_Y, Z_BACK], [PX1, BASE_Y, Z_BACK], [PX1, BASE_Y, Z_FRONT], [PX0, BASE_Y, Z_FRONT]],
    [[PXM, BASE_Y, Z_BACK], [PXM, BASE_Y, Z_FRONT]],
  ], [])

  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />

      {footprints.map((pts, i) => (
        <Line key={i} points={pts} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      ))}
      <FloorCaption position={[-0.5, BASE_Y, Z_FRONT + 0.75]} />
      <PotentialRuler x={SX0 - 0.4} z={Z_BACK} baseY={BASE_Y} volts={RULER_VOLTS} unitsPerVolt={() => HEIGHT_PER_VOLT * rise.current} />

      {inits.map((pts, l) => (
        <TrailLine
          key={l}
          ref={(line) => { lineRefs.current[l] = line }}
          points={pts}
          color={c.current}
          fade={false}
          flowSpeed={() => flows.current[l]}
        />
      ))}

      <Line ref={dropRef} points={TAP_DROP_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.06} gapSize={0.05} />
      <Marker ref={tapRef} color={c.marker} size={0.07} />

      {[0, 1, 2, 3].map((r) => (
        <Resistor key={r} ref={(g) => { resRefs.current[r] = g }} />
      ))}
      {[0, 1, 2, 3, 4].map((t) => (
        <group key={t} ref={(g) => { tagRefs.current[t] = g }}>
          <Html center>
            <span ref={(el) => { textRefs.current[t] = el }} className={labelClass} />
          </Html>
        </group>
      ))}

      <Html position={[(SX0 + SX1) / 2, BASE_Y - 0.1, Z_FRONT + 0.35]} center>
        <span ref={(el) => { textRefs.current[5] = el }} className={labelClass} />
      </Html>
      <Html position={[(PX0 + PX1) / 2, BASE_Y - 0.1, Z_FRONT + 0.35]} center>
        <span ref={(el) => { textRefs.current[6] = el }} className={labelClass} />
      </Html>
    </>
  )
}

// ── 2D: the two splits as 100 % bars, so the swapped ratios sit right above each other ──
const BAR_X0 = -3.2
const BAR_W = 6.4
const BAR_H = 0.62
const BAR_Y = [0.75, -0.75] // voltage bar, current bar
const splits = new Float32Array(2) // first-segment fraction of each bar

function DividerBars2DScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const segRefs = useRef<(THREE.Mesh | null)[]>([]) // V1, Vout, I1, I2
  const tagRefs = useRef<(THREE.Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<unknown>(null)

  useFrame(() => {
    const s = solve(valuesRef.current)
    // Voltage: R1 takes (1 − α) — its own share. Current: branch 1 takes α — the other one's share.
    splits[0] = 1 - s.a
    splits[1] = s.a
    for (let b = 0; b < 2; b++) {
      const w1 = Math.max(BAR_W * splits[b], 1e-3)
      const w2 = Math.max(BAR_W - w1, 1e-3)
      segRefs.current[b * 2]?.position.set(BAR_X0 + w1 / 2, BAR_Y[b], 0)
      segRefs.current[b * 2]?.scale.set(w1, BAR_H, 1)
      segRefs.current[b * 2 + 1]?.position.set(BAR_X0 + w1 + w2 / 2, BAR_Y[b], 0)
      segRefs.current[b * 2 + 1]?.scale.set(w2, BAR_H, 1)
      // Labels go under the bar: text on top of a filled segment can match its color
      const labelY = BAR_Y[b] - BAR_H / 2 - 0.22
      tagRefs.current[b * 2]?.position.set(BAR_X0 + w1 / 2, labelY, 0.01)
      tagRefs.current[b * 2 + 1]?.position.set(BAR_X0 + w1 + w2 / 2, labelY, 0.01)
    }

    if (labelledFor.current !== valuesRef.current && textRefs.current[5]) {
      labelledFor.current = valuesRef.current
      const ko = locale === 'ko'
      const texts = [
        `V1 ${fmt(s.V1, 'V')} · ${pct(1 - s.a)}`,
        `Vout ${fmt(s.Vout, 'V')} · ${pct(s.a)}`,
        `I1 ${fmt(s.I1, 'A')} · ${pct(s.a)}`,
        `I2 ${fmt(s.I2, 'A')} · ${pct(1 - s.a)}`,
        `${ko ? '전압 분배' : 'Voltage divider'} · V ${fmt(s.V, 'V')}`,
        `${ko ? '전류 분배' : 'Current divider'} · I ${fmt(s.I, 'A')}`,
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  const segColors = [c.voltage, c.resistance, c.current, c.current]
  const segOpacity = [0.9, 0.45, 0.9, 0.4]

  return (
    <>
      {[0, 1, 2, 3].map((i) => (
        <mesh key={i} ref={(m) => { segRefs.current[i] = m }}>
          <planeGeometry args={[1, 1]} />
          <meshBasicMaterial color={segColors[i]} transparent opacity={segOpacity[i]} depthWrite={false} />
        </mesh>
      ))}
      {[0, 1, 2, 3].map((i) => (
        <group key={i} ref={(g) => { tagRefs.current[i] = g }}>
          <Html center>
            <span ref={(el) => { textRefs.current[i] = el }} className={labelClass} />
          </Html>
        </group>
      ))}
      {[0, 1].map((b) => (
        <Html key={b} position={[BAR_X0, BAR_Y[b] + BAR_H / 2 + 0.25, 0]}>
          <span ref={(el) => { textRefs.current[4 + b] = el }} className={labelClass} />
        </Html>
      ))}
    </>
  )
}

const concept: ConceptMeta = {
  id: 'voltage-current-divider',
  title: { en: 'Voltage & Current Dividers', ko: '전압 분배·전류 분배' },
  subject: 'circuits',
  order: 4,
  description: {
    en: `## Voltage and Current Dividers

**A resistor takes its own share of the voltage, but the other one's share of the current.**

Left: a potentiometer as a potential landscape. The tap slides along the resistor, and its height is the output voltage.
- Bright dot: the tap; the dashed line drops to the 0 V floor, so its height reads as Vout
- Move the tap slider: R1 and R2 change length, and Vout follows the tap

Right: the same two resistors, now in parallel. The current splits the other way, and the wider, faster line is the larger current.

**Voltage divider**: Vout = V × R2 ÷ (R1 + R2) = α × V
**Current divider**: I1 = I × R2 ÷ (R1 + R2), I2 = I × R1 ÷ (R1 + R2)

Only the ratio matters: change the total resistance and the split stays put, while every current scales.

In 2D: both splits as 100 % bars, one above the other, so you can see the shares swap.`,
    ko: `## 전압 분배·전류 분배

**저항은 전압은 자기 몫만큼, 전류는 상대 몫만큼 가져갑니다.**

왼쪽: 가변저항을 전위 지형으로 그렸습니다. 탭이 저항 위를 미끄러지고, 탭의 높이가 곧 출력 전압입니다.
- 밝은 점: 탭. 점선이 0 V 바닥까지 내려와서 높이가 Vout임을 보여줍니다
- 탭 위치 슬라이더를 움직이면 R1과 R2의 길이가 바뀌고 Vout이 따라옵니다

오른쪽: 같은 두 저항을 병렬로 연결한 모습입니다. 전류는 반대 비율로 갈라지고, 굵고 빠른 선이 더 큰 전류입니다.

**전압 분배**: Vout = V × R2 ÷ (R1 + R2) = α × V
**전류 분배**: I1 = I × R2 ÷ (R1 + R2), I2 = I × R1 ÷ (R1 + R2)

비율만이 중요합니다. 전체 저항을 바꿔도 분배 비율은 그대로이고 전류만 함께 변합니다.

2D: 두 분배를 100% 막대 두 개로 위아래에 놓았습니다. 몫이 서로 뒤바뀌는 것이 보입니다.`,
  },
  sliderParams: [
    { key: 'V',        label: { en: 'Voltage (V)',      ko: '전압 (V)' },        unit: 'V', min: 1,       max: V_MAX, step: 0.5, defaultValue: 12 },
    { key: 'Rt',       label: { en: 'Total R (R1 + R2)', ko: '전체 저항 (R1 + R2)' }, unit: 'Ω', min: RT_MIN, max: 40,    step: 1,   defaultValue: 12 },
    { key: 'alphaPct', label: { en: 'Tap position (α)',  ko: '탭 위치 (α)' },     unit: '%', min: 5,       max: 95,    step: 5,   defaultValue: 25 },
  ],
  Scene: DividerScene,
  Scene2D: DividerBars2DScene,
}

export default concept
