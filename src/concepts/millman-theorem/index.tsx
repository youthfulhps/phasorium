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

// Slider ranges, also used to normalize the visuals
const V_MAX = 24 // [V]
const R_MIN = 1 // [Ω]
const R_MAX = 20 // [Ω]
const P_MAX = 180 // [W] hottest branch reachable inside the slider ranges

// The third branch is fixed: a 0 V source is just a resistance to ground, so it contributes
// conductance to the denominator with a zero numerator term.
const V3 = 0 // [V]
const R3 = 6 // [Ω]

const BRANCHES = 3
const srcV = new Float32Array(BRANCHES)
const srcR = new Float32Array(BRANCHES)
const srcI = new Float32Array(BRANCHES)
const state = { Vn: 0, sumI: 0, sumG: 0 }

// Millman: V_node = Σ(Vk ÷ Rk) ÷ Σ(1 ÷ Rk), branch current Ik = (Vk − V_node) ÷ Rk, and ΣIk = 0.
// Worked example: V1 = 12 V, R1 = 2 Ω, V2 = 6 V, R2 = 3 Ω, V3 = 0 V, R3 = 6 Ω →
// numerator 6 + 2 + 0 = 8, denominator 0.5 + 0.3333 + 0.1667 = 1, so V_node = 8 V.
// I1 = +2.0000 A into the node, I2 = −0.6667 A, I3 = −1.3333 A, ΣIk = 0 (that is KCL).
// Branch powers Ik² × Rk: 8.0000 W, 1.3333 W, 10.6667 W.
function solve({ V1 = 12, R1 = 2, V2 = 6, R2 = 3 }: Record<string, number>) {
  srcV[0] = V1; srcV[1] = V2; srcV[2] = V3
  srcR[0] = R1; srcR[1] = R2; srcR[2] = R3
  let num = 0
  let den = 0
  for (let k = 0; k < BRANCHES; k++) {
    num += srcV[k] / srcR[k]
    den += 1 / srcR[k]
  }
  const Vn = num / den
  let sum = 0
  for (let k = 0; k < BRANCHES; k++) {
    srcI[k] = (srcV[k] - Vn) / srcR[k]
    sum += srcI[k]
  }
  state.Vn = Vn
  state.sumI = sum
  state.sumG = den
  return state
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
/** Signed current: + flows into the common node, − flows out of it. */
const fmtI = (x: number) => `${x >= 0 ? '+' : '−'}${fmt(Math.abs(x), 'A')}`
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const axisLabelClass = 'pointer-events-none whitespace-nowrap text-[11px] text-muted-foreground'

// ── 3D: three source branches meeting at one node, drawn as a potential landscape ──
const BASE_Y = -0.6
const HEIGHT_PER_VOLT = 1.6 / V_MAX
const RULER_VOLTS = [0, 12, 24]
const Z_GND = -1.7 // the source's − terminal, on the floor at 0 V
const Z_SRC = -1.3 // the source's + terminal, lifted to Vk
const Z_RS = -0.9 // resistor Rk starts
const Z_RE = -0.1 // resistor Rk ends, already at the node height
const Z_NODE = 0.8
const BX = [-2.2, -0.9, 0.4] // one x per branch
const X_NODE = BX[1]
const SEG_PTS = 20

type XZ = [number, number]
const BRANCH_XZ: XZ[][] = BX.map((x) => [
  [x, Z_GND], [x, Z_SRC], [x, Z_RS], [x, Z_RE], [X_NODE, Z_NODE],
])
// Potential at each waypoint, rewritten in place every frame
const BRANCH_V = BRANCH_XZ.map((xz) => new Float32Array(xz.length))

const pointCount = (xz: XZ[]) => (xz.length - 1) * SEG_PTS + 1
const initPoints = (xz: XZ[]): [number, number, number][] =>
  Array.from({ length: pointCount(xz) }, () => [0, 0, 0])

/** Sample a waypoint polyline; potential varies linearly along each segment (flat on wires). */
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

/** Seat a resistor between two waypoints; its filament thickens with power. */
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
    const r = 0.12 * Math.sqrt(R_MIN / Math.max(R, 0.2)) + 0.03 // thinner for larger R
    g.scale.set(dir.length(), r, r)
  }
  setResistorHeat(g, Math.sqrt(P / P_MAX) * 1.6)
  tag?.position.set((p0.x + p1.x) / 2, Math.max(p0.y, p1.y) + 0.32, (p0.z + p1.z) / 2)
}

/** Current as line weight: wider for more current (px), clamped to stay readable. */
const widthFor = (I: number) => Math.min(1.4 + 1.7 * Math.sqrt(Math.abs(I)), 8)
/** Sheen bands per second; the sign carries the direction, so a branch below the node flows backwards. */
const flowFor = (I: number) => Math.max(Math.min(0.7 * I, 8), -8)
const TWO_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]

function MillmanScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const rise = useRiseIn()
  const bufs = useMemo(() => BRANCH_XZ.map((xz) => new Float32Array(pointCount(xz) * 3)), [])
  const inits = useMemo(() => BRANCH_XZ.map(initPoints), [])
  const levelBuf = useMemo(() => new Float32Array(6), [])
  const lineRefs = useRef<any[]>([])
  const levelRef = useRef<any>(null)
  const resRefs = useRef<(THREE.Group | null)[]>([]) // R1, R2, R3
  const tagRefs = useRef<(THREE.Group | null)[]>([]) // 3 branches, node
  const textRefs = useRef<(HTMLSpanElement | null)[]>([]) // 3 branches, node, KCL sum
  const srcRefs = useRef<(THREE.Group | null)[]>([]) // source-height dots
  const nodeRef = useRef<THREE.Group>(null)
  const flows = useRef([0, 0, 0])
  const labelledFor = useRef<unknown>(null)

  useFrame(() => {
    const s = solve(valuesRef.current)
    const Vn = s.Vn
    const k = HEIGHT_PER_VOLT * rise.current

    for (let b = 0; b < BRANCHES; b++) {
      const volts = BRANCH_V[b]
      volts[1] = volts[2] = srcV[b] // source rise, then a flat wire at Vk
      volts[3] = volts[4] = Vn // out of the resistor at the node height, then a flat wire
      const line = lineRefs.current[b]
      if (line) {
        fillPath(bufs[b], BRANCH_XZ[b], volts, k)
        line.geometry.setPositions(bufs[b])
        ;(line.material as LineMaterial).linewidth = widthFor(srcI[b])
        flows.current[b] = flowFor(srcI[b])
      }
      placeResistor(
        resRefs.current[b], tagRefs.current[b],
        BRANCH_XZ[b][2], srcV[b], BRANCH_XZ[b][3], Vn,
        k, srcR[b], srcI[b] * srcI[b] * srcR[b]
      )
      srcRefs.current[b]?.position.set(BX[b], BASE_Y + k * srcV[b], Z_SRC)
    }

    // The common node: every branch line ends here, and its height is the weighted average
    nodeRef.current?.position.set(X_NODE, BASE_Y + k * Vn, Z_NODE)
    tagRefs.current[3]?.position.set(X_NODE + 1.0, BASE_Y + k * Vn + 0.36, Z_NODE)

    // Node level carried back across the source tops: sources above it push in, sources below draw out
    levelBuf[0] = BX[0] - 0.45; levelBuf[1] = BASE_Y + k * Vn; levelBuf[2] = Z_SRC
    levelBuf[3] = BX[2] + 0.45; levelBuf[4] = levelBuf[1]; levelBuf[5] = Z_SRC
    levelRef.current?.geometry.setPositions(levelBuf)
    levelRef.current?.computeLineDistances()

    if (labelledFor.current !== valuesRef.current && textRefs.current[4]) {
      labelledFor.current = valuesRef.current
      const ko = locale === 'ko'
      const texts = [
        `V1 ${fmt(srcV[0], 'V')} · R1 ${fmt(srcR[0], 'Ω')} · I1 ${fmtI(srcI[0])}`,
        `V2 ${fmt(srcV[1], 'V')} · R2 ${fmt(srcR[1], 'Ω')} · I2 ${fmtI(srcI[1])}`,
        `V3 ${fmt(srcV[2], 'V')} · R3 ${fmt(srcR[2], 'Ω')} · I3 ${fmtI(srcI[2])}`,
        `${ko ? '마디' : 'Node'} ${fmt(Vn, 'V')}`,
        `KCL · ΣI = ${fmt(Math.abs(s.sumI) < 5e-3 ? 0 : s.sumI, 'A')}`,
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  const footprints = useMemo<[number, number, number][][]>(() => {
    const flat = BRANCH_XZ.map((xz) => xz.map(([x, z]): [number, number, number] => [x, BASE_Y, z]))
    flat.push([[BX[0], BASE_Y, Z_GND], [BX[2], BASE_Y, Z_GND]]) // ground rail joining the sources
    return flat
  }, [])

  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />

      {footprints.map((pts, i) => (
        <Line key={i} points={pts} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      ))}
      <FloorCaption position={[X_NODE, BASE_Y, Z_NODE + 1.0]} />
      <PotentialRuler
        x={BX[0] - 0.75}
        z={Z_SRC}
        baseY={BASE_Y}
        volts={RULER_VOLTS}
        unitsPerVolt={() => HEIGHT_PER_VOLT * rise.current}
      />
      <Line ref={levelRef} points={TWO_INIT} lineWidth={1.5} color={c.guide} dashed dashSize={0.1} gapSize={0.07} />

      {inits.map((pts, b) => (
        <TrailLine
          key={b}
          ref={(line) => { lineRefs.current[b] = line }}
          points={pts}
          color={c.current}
          fade={false}
          flowSpeed={() => flows.current[b]}
        />
      ))}

      {[0, 1, 2].map((b) => (
        <group key={b}>
          <Resistor ref={(g) => { resRefs.current[b] = g }} />
          <Marker ref={(g) => { srcRefs.current[b] = g }} color={c.voltage} size={0.05} />
          <group ref={(g) => { tagRefs.current[b] = g }}>
            <Html center>
              <span ref={(el) => { textRefs.current[b] = el }} className={labelClass} />
            </Html>
          </group>
        </group>
      ))}

      <Marker ref={nodeRef} color={c.marker} size={0.07} />
      <group ref={(g) => { tagRefs.current[3] = g }}>
        <Html center>
          <span ref={(el) => { textRefs.current[3] = el }} className={labelClass} />
        </Html>
      </group>
      <Html position={[X_NODE, BASE_Y - 0.12, Z_NODE + 0.4]} center>
        <span ref={(el) => { textRefs.current[4] = el }} className={labelClass} />
      </Html>
    </>
  )
}

// ── 2D: the sources on a voltage line, each weighted by its conductance, balancing at V_node ──
const AX_Y = -0.9 // the voltage axis
const AX_X0 = -3.2
const X_PER_VOLT = 6.4 / V_MAX
const Y_PER_SIEMENS = 2.0 // bar height per 1 S of branch conductance
const BAR_W = 0.2
const TICK_VOLTS = [0, 12, 24]
const AXIS_PTS: [number, number, number][] = [
  [AX_X0 - 0.25, AX_Y, 0], [AX_X0 + V_MAX * X_PER_VOLT + 0.35, AX_Y, 0],
]
const TICK_PTS: [number, number, number][] = [[0, AX_Y, 0], [0, AX_Y - 0.12, 0]]
// A balance fulcrum sitting under the node voltage
const FULCRUM_PTS: [number, number, number][] = [
  [-0.17, -0.3, 0], [0.17, -0.3, 0], [0, 0, 0], [-0.17, -0.3, 0],
]

function MillmanBalance2DScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const barRefs = useRef<(THREE.Mesh | null)[]>([])
  const tagRefs = useRef<(THREE.Group | null)[]>([]) // 3 bars, node
  const textRefs = useRef<(HTMLSpanElement | null)[]>([]) // 3 bars, node, sum
  const dropRef = useRef<any>(null)
  const dropBuf = useMemo(() => new Float32Array(6), [])
  const nodeRef = useRef<THREE.Group>(null)
  const fulcrumRef = useRef<THREE.Group>(null)
  const labelledFor = useRef<unknown>(null)

  useFrame(() => {
    const s = solve(valuesRef.current)

    for (let b = 0; b < BRANCHES; b++) {
      const x = AX_X0 + srcV[b] * X_PER_VOLT
      const h = Math.max((1 / srcR[b]) * Y_PER_SIEMENS, 1e-3)
      barRefs.current[b]?.position.set(x, AX_Y + h / 2, 0)
      barRefs.current[b]?.scale.set(BAR_W, h, 1)
      tagRefs.current[b]?.position.set(x, AX_Y + h + 0.22, 0.01)
    }

    // The balance point: the conductance-weighted average of the source voltages
    const xn = AX_X0 + s.Vn * X_PER_VOLT
    dropBuf[0] = xn; dropBuf[1] = AX_Y; dropBuf[2] = 0
    dropBuf[3] = xn; dropBuf[4] = AX_Y + 1.5; dropBuf[5] = 0
    dropRef.current?.geometry.setPositions(dropBuf)
    dropRef.current?.computeLineDistances()
    nodeRef.current?.position.set(xn, AX_Y, 0.02)
    fulcrumRef.current?.position.set(xn, AX_Y - 0.06, 0)
    tagRefs.current[3]?.position.set(xn, AX_Y - 0.6, 0.01)

    if (labelledFor.current !== valuesRef.current && textRefs.current[4]) {
      labelledFor.current = valuesRef.current
      const ko = locale === 'ko'
      const texts = [
        `V1 ${fmt(srcV[0], 'V')} · 1 ÷ R1 ${(1 / srcR[0]).toFixed(3)} S`,
        `V2 ${fmt(srcV[1], 'V')} · 1 ÷ R2 ${(1 / srcR[1]).toFixed(3)} S`,
        `V3 ${fmt(srcV[2], 'V')} · 1 ÷ R3 ${(1 / srcR[2]).toFixed(3)} S`,
        `${ko ? '마디' : 'Node'} ${fmt(s.Vn, 'V')}`,
        `Σ(Vk ÷ Rk) ${(s.Vn * s.sumG).toFixed(3)} ÷ Σ(1 ÷ Rk) ${s.sumG.toFixed(3)} = ${fmt(s.Vn, 'V')}`,
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  return (
    <>
      <Line points={AXIS_PTS} lineWidth={1} color={c.guide} />
      {TICK_VOLTS.map((v) => (
        <group key={v} position={[AX_X0 + v * X_PER_VOLT, 0, 0]}>
          <Line points={TICK_PTS} lineWidth={1} color={c.guide} />
          <Html position={[0, AX_Y - 0.28, 0]} center>
            <span className={axisLabelClass}>{v} V</span>
          </Html>
        </group>
      ))}

      <Line ref={dropRef} points={TWO_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      {[0, 1, 2].map((b) => (
        <mesh key={b} ref={(m) => { barRefs.current[b] = m }}>
          <planeGeometry args={[1, 1]} />
          <meshBasicMaterial color={c.voltage} transparent opacity={0.7} depthWrite={false} />
        </mesh>
      ))}

      <group ref={fulcrumRef}>
        <Line points={FULCRUM_PTS} lineWidth={1.5} color={c.marker} />
      </group>
      <Marker ref={nodeRef} color={c.marker} size={0.06} />

      {[0, 1, 2, 3].map((t) => (
        <group key={t} ref={(g) => { tagRefs.current[t] = g }}>
          <Html center>
            <span ref={(el) => { textRefs.current[t] = el }} className={labelClass} />
          </Html>
        </group>
      ))}
      <Html position={[AX_X0 + 3.2, AX_Y + 2.6, 0]} center>
        <span ref={(el) => { textRefs.current[4] = el }} className={labelClass} />
      </Html>
      <Html position={[AX_X0 - 0.55, AX_Y + 1.0, 0]} center>
        <span className={axisLabelClass}>1 ÷ R (S)</span>
      </Html>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'millman-theorem',
  title: { en: "Millman's Theorem", ko: '밀만의 정리' },
  subject: 'circuits',
  order: 10,
  description: {
    en: `## Millman's Theorem

**Tie several sources, each behind its own resistance, to one node and that node settles at the conductance-weighted average of their voltages.**

Three branches meeting at one common node, drawn as a potential landscape (height = potential).
- Small dots on the source tops: each branch's own voltage V1, V2, V3
- Bright dot: the common node. The dashed level line carries its height back across the sources, so you can see which ones sit above it and which below
- Cyan liquid line: the branch current. A source above the node pushes current in, one below draws current out, and the sheen runs the other way when it does
- Glass tubes: the branch resistances, with amber filaments showing the power each one dissipates

**Node voltage**: V = (V1 ÷ R1 + V2 ÷ R2 + V3 ÷ R3) ÷ (1 ÷ R1 + 1 ÷ R2 + 1 ÷ R3)
**Branch current**: Ik = (Vk − V) ÷ Rk, and ΣIk = 0, which is just KCL at the node

The weight of a branch is its conductance 1 ÷ Rk, so a small series resistance pulls the node hardest. Branch 3 is a source of 0 V, which is simply a resistance to ground: it adds conductance to the denominator and nothing to the numerator, dragging the node down.

Shrink R1 and watch the node climb toward V1; grow R1 instead and V2 takes over the pull.

In 2D: the three sources as weights on a voltage line, each bar as tall as its conductance, balancing at the node voltage.`,
    ko: `## 밀만의 정리

**각자 저항을 달고 있는 여러 전원을 한 마디에 모으면, 그 마디는 전원 전압의 컨덕턴스 가중평균에 자리잡습니다.**

한 마디에서 만나는 세 갈래를 전위 지형(높이 = 전위)으로 그렸습니다.
- 전원 꼭대기의 작은 점: 각 갈래가 가진 전압 V1, V2, V3
- 밝은 점: 공통 마디. 점선이 마디의 높이를 전원 쪽으로 이어주므로, 어느 전원이 마디보다 위에 있고 어느 것이 아래에 있는지 한눈에 보입니다
- 시안 액체 선: 갈래 전류. 마디보다 높은 전원은 전류를 밀어 넣고, 낮은 전원은 끌어냅니다. 끌어낼 때는 광택도 반대 방향으로 흐릅니다
- 유리관: 갈래 저항. 안쪽 앰버 필라멘트가 각 저항이 소비하는 전력을 나타냅니다

**마디 전압**: V = (V1 ÷ R1 + V2 ÷ R2 + V3 ÷ R3) ÷ (1 ÷ R1 + 1 ÷ R2 + 1 ÷ R3)
**갈래 전류**: Ik = (Vk − V) ÷ Rk, 그리고 ΣIk = 0. 이것이 마디에서의 KCL입니다

갈래의 가중치는 컨덕턴스 1 ÷ Rk입니다. 그래서 직렬 저항이 작을수록 마디를 세게 끌어당깁니다. 3번 갈래는 0 V 전원, 곧 접지로 가는 저항일 뿐입니다. 분모에는 컨덕턴스를 더하고 분자에는 아무것도 더하지 않으니 마디를 아래로 끌어내립니다.

R1을 줄이면 마디가 V1 쪽으로 올라가고, 키우면 V2가 끌어당기는 쪽으로 넘어갑니다.

2D: 세 전원을 전압 축 위의 추로 놓고, 막대 높이를 컨덕턴스로 그렸습니다. 마디 전압이 그 균형점입니다.`,
  },
  sliderParams: [
    { key: 'V1', label: { en: 'Source 1 (V1)', ko: '전원 1 (V1)' }, unit: 'V', min: 0,     max: V_MAX, step: 0.5, defaultValue: 12 },
    { key: 'R1', label: { en: 'Branch R1',     ko: '분기 저항 R1' }, unit: 'Ω', min: R_MIN, max: R_MAX, step: 0.5, defaultValue: 2 },
    { key: 'V2', label: { en: 'Source 2 (V2)', ko: '전원 2 (V2)' }, unit: 'V', min: 0,     max: V_MAX, step: 0.5, defaultValue: 6 },
    { key: 'R2', label: { en: 'Branch R2',     ko: '분기 저항 R2' }, unit: 'Ω', min: R_MIN, max: R_MAX, step: 0.5, defaultValue: 3 },
  ],
  Scene: MillmanScene,
  Scene2D: MillmanBalance2DScene,
}

export default concept
