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

const V = 12 // fixed source [V]
const RG = 50 // galvanometer resistance [Ω]
const R_MIN = 2
const R_MAX = 60
const P_MAX = (V * V) / R_MIN
const BALANCED_A = 0.15e-3 // |Ig| below this reads as a null [A]

// Pure physics: two node equations for the bridge nodes A and B, with the galvanometer loading them.
// Worked example: V = 12 V, R1 = 10 Ω, R2 = 20 Ω, R3 = 15 Ω, Rg = 50 Ω →
// R4 = 24 Ω: VA = 7.94 V, VB = 7.47 V, Ig = 9.34 mA · R4 = 30 Ω (R1·R4 = R2·R3): VA = VB = 8 V, Ig = 0.
/** Bridge voltage only, without building an object — the 2D sweep calls this once per sample. */
function bridgeVg(R1: number, R2: number, R3: number, R4: number) {
  const Gg = 1 / RG
  const a11 = 1 / R1 + 1 / R2 + Gg
  const a22 = 1 / R3 + 1 / R4 + Gg
  const b1 = V / R1
  const b2 = V / R3
  const det = a11 * a22 - Gg * Gg
  return (b1 * a22 + Gg * b2 - (a11 * b2 + Gg * b1)) / det
}

function solve({ R1 = 10, R2 = 20, R3 = 15, R4 = 24 }: Record<string, number>) {
  const Gg = 1 / RG
  const a11 = 1 / R1 + 1 / R2 + Gg
  const a22 = 1 / R3 + 1 / R4 + Gg
  const b1 = V / R1
  const b2 = V / R3
  const det = a11 * a22 - Gg * Gg
  const VA = (b1 * a22 + Gg * b2) / det
  const VB = (a11 * b2 + Gg * b1) / det
  const Ig = (VA - VB) / RG
  return {
    R1, R2, R3, R4, VA, VB, Vg: VA - VB, Ig,
    detune: R1 * R4 - R2 * R3, // zero when balanced
    balanceR4: (R2 * R3) / R1,
    I1: (V - VA) / R1, I2: VA / R2, I3: (V - VB) / R3, I4: VB / R4,
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

const fmtV = (x: number) => `${x.toFixed(2)} V`
const fmtR = (x: number) => `${x.toFixed(x < 10 ? 1 : 0)} Ω`
const fmtI = (a: number) => (Math.abs(a) < 1 ? `${(a * 1e3).toFixed(2)} mA` : `${a.toFixed(2)} A`)
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'

// ── 3D: the bridge as a potential landscape — balance is the bridge lying level ──
const BASE_Y = -0.6
const HEIGHT_PER_VOLT = 1.6 / V
const RULER_VOLTS = [0, 6, 12]
const Z_TOP = -1.2 // top rail (source +)
const Z_BOT = 1.2 // bottom rail (0 V)
const Z_NODE = -0.1 // where the bridge crosses
const SEG_PTS = 16

type XZ = [number, number]
const SRC_X = -2.0
const LX = -0.8 // left branch (R1 over R2)
const RX = 0.8 // right branch (R3 over R4)
const R_TOP: [number, number] = [-1.0, -0.35] // resistor span above a node
const R_BOT: [number, number] = [0.15, 0.85] // resistor span below a node

const TOP_XZ: XZ[] = [[SRC_X, Z_BOT], [SRC_X, Z_TOP], [LX, Z_TOP], [RX, Z_TOP]] // source + top rail
const LEFT_XZ: XZ[] = [
  [LX, Z_TOP], [LX, R_TOP[0]], [LX, R_TOP[1]], [LX, Z_NODE],
  [LX, R_BOT[0]], [LX, R_BOT[1]], [LX, Z_BOT], [SRC_X, Z_BOT],
]
const RIGHT_XZ: XZ[] = [
  [RX, Z_TOP], [RX, R_TOP[0]], [RX, R_TOP[1]], [RX, Z_NODE],
  [RX, R_BOT[0]], [RX, R_BOT[1]], [RX, Z_BOT], [LX, Z_BOT],
]
const BRIDGE_XZ: XZ[] = [[LX, Z_NODE], [RX, Z_NODE]]

const topV = new Float32Array(TOP_XZ.length)
const leftV = new Float32Array(LEFT_XZ.length)
const rightV = new Float32Array(RIGHT_XZ.length)
const bridgeV = new Float32Array(BRIDGE_XZ.length)
const LINE_XZ = [TOP_XZ, LEFT_XZ, RIGHT_XZ, BRIDGE_XZ]
const LINE_V = [topV, leftV, rightV, bridgeV]

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
  a: XZ, va: number, b: XZ, vb: number, k: number, R: number, P: number, tagDx: number
) {
  p0.set(a[0], BASE_Y + k * va, a[1])
  p1.set(b[0], BASE_Y + k * vb, b[1])
  dir.subVectors(p1, p0)
  if (g) {
    g.position.addVectors(p0, p1).multiplyScalar(0.5)
    g.quaternion.setFromUnitVectors(X_AXIS, unit.copy(dir).normalize())
    const r = 0.13 * Math.sqrt(R_MIN / Math.max(R, 0.2)) + 0.03
    g.scale.set(dir.length(), r, r)
  }
  setResistorHeat(g, Math.sqrt(P / P_MAX) * 1.6)
  tag?.position.set((p0.x + p1.x) / 2 + tagDx, (p0.y + p1.y) / 2 + 0.28, (p0.z + p1.z) / 2)
}

const LEVEL_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]
// Resistor endpoints and the per-line current buffer, kept out of the frame loop
const R1_A: XZ = [LX, R_TOP[0]]
const R1_B: XZ = [LX, R_TOP[1]]
const R2_A: XZ = [LX, R_BOT[0]]
const R2_B: XZ = [LX, R_BOT[1]]
const R3_A: XZ = [RX, R_TOP[0]]
const R3_B: XZ = [RX, R_TOP[1]]
const R4_A: XZ = [RX, R_BOT[0]]
const R4_B: XZ = [RX, R_BOT[1]]
const lineI = new Float32Array(4)

function WheatstoneScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const rise = useRiseIn()
  const bufs = useMemo(() => LINE_XZ.map((xz) => new Float32Array(pointCount(xz) * 3)), [])
  const inits = useMemo(() => LINE_XZ.map(initPoints), [])
  const levelBuf = useMemo(() => new Float32Array(6), [])
  const lineRefs = useRef<any[]>([])
  const levelRef = useRef<any>(null)
  const resRefs = useRef<(THREE.Group | null)[]>([])
  const tagRefs = useRef<(THREE.Group | null)[]>([]) // R1..R4, node A, node B, galvanometer
  const textRefs = useRef<(HTMLSpanElement | null)[]>([]) // R1..R4, A, B, Ig, balance note
  const nodeA = useRef<THREE.Group>(null)
  const nodeB = useRef<THREE.Group>(null)
  const galv = useRef<THREE.Group>(null)
  const flows = useRef([0, 0, 0, 0])
  const labelledFor = useRef<unknown>(null)

  useFrame(() => {
    const s = solve(valuesRef.current)
    const k = HEIGHT_PER_VOLT * rise.current

    topV[1] = topV[2] = topV[3] = V
    leftV[0] = leftV[1] = V
    leftV[2] = leftV[3] = leftV[4] = s.VA
    rightV[0] = rightV[1] = V
    rightV[2] = rightV[3] = rightV[4] = s.VB
    bridgeV[0] = s.VA
    bridgeV[1] = s.VB

    lineI[0] = s.I1 + s.I3; lineI[1] = s.I1; lineI[2] = s.I3; lineI[3] = Math.abs(s.Ig)
    for (let l = 0; l < 4; l++) {
      const line = lineRefs.current[l]
      if (!line) continue
      fillPath(bufs[l], LINE_XZ[l], LINE_V[l], k)
      line.geometry.setPositions(bufs[l])
      // The bridge line carries the whole story: it thins to a thread and stops flowing at balance
      const isBridge = l === 3
      ;(line.material as LineMaterial).linewidth = isBridge
        ? 1 + Math.min(6, Math.abs(s.Ig) * 260)
        : 2.6
      flows.current[l] = isBridge ? Math.sign(s.Ig) * Math.min(6, Math.abs(s.Ig) * 220) : Math.min(4, lineI[l] * 4)
    }

    placeResistor(resRefs.current[0], tagRefs.current[0], R1_A, V, R1_B, s.VA, k, s.R1, s.I1 * (V - s.VA), -0.45)
    placeResistor(resRefs.current[1], tagRefs.current[1], R2_A, s.VA, R2_B, 0, k, s.R2, s.I2 * s.VA, -0.45)
    placeResistor(resRefs.current[2], tagRefs.current[2], R3_A, V, R3_B, s.VB, k, s.R3, s.I3 * (V - s.VB), 0.45)
    placeResistor(resRefs.current[3], tagRefs.current[3], R4_A, s.VB, R4_B, 0, k, s.R4, s.I4 * s.VB, 0.45)

    const yA = BASE_Y + k * s.VA
    const yB = BASE_Y + k * s.VB
    nodeA.current?.position.set(LX, yA, Z_NODE)
    nodeB.current?.position.set(RX, yB, Z_NODE)
    galv.current?.position.set((LX + RX) / 2, (yA + yB) / 2, Z_NODE)
    tagRefs.current[4]?.position.set(LX - 0.5, yA + 0.05, Z_NODE)
    tagRefs.current[5]?.position.set(RX + 0.5, yB + 0.05, Z_NODE)
    tagRefs.current[6]?.position.set((LX + RX) / 2, (yA + yB) / 2 + 0.3, Z_NODE)

    // Level reference at node A's height: the gap to node B is the bridge voltage
    levelBuf[0] = LX; levelBuf[1] = yA; levelBuf[2] = Z_NODE
    levelBuf[3] = RX + 0.25; levelBuf[4] = yA; levelBuf[5] = Z_NODE
    levelRef.current?.geometry.setPositions(levelBuf)
    levelRef.current?.computeLineDistances()

    if (labelledFor.current !== valuesRef.current && textRefs.current[7]) {
      labelledFor.current = valuesRef.current
      const ko = locale === 'ko'
      const balanced = Math.abs(s.Ig) < BALANCED_A
      const texts = [
        `R1 ${fmtR(s.R1)}`,
        `R2 ${fmtR(s.R2)}`,
        `R3 ${fmtR(s.R3)}`,
        `R4 ${fmtR(s.R4)}`,
        `A ${fmtV(s.VA)}`,
        `B ${fmtV(s.VB)}`,
        `Ig ${fmtI(s.Ig)}`,
        balanced
          ? `${ko ? '평형' : 'Balanced'} · R1 × R4 = R2 × R3 = ${(s.R1 * s.R4).toFixed(0)}`
          : `${ko ? '불평형' : 'Unbalanced'} · Vg ${fmtV(s.Vg)} · ${ko ? '평형까지' : 'balance at'} R4 = ${fmtR(s.balanceR4)}`,
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  const footprint = useMemo<[number, number, number][]>(() => [
    [SRC_X, BASE_Y, Z_BOT], [SRC_X, BASE_Y, Z_TOP], [RX, BASE_Y, Z_TOP], [RX, BASE_Y, Z_BOT], [SRC_X, BASE_Y, Z_BOT],
  ], [])
  const rungs = useMemo<[number, number, number][][]>(() => [
    [[LX, BASE_Y, Z_TOP], [LX, BASE_Y, Z_BOT]],
    [[LX, BASE_Y, Z_NODE], [RX, BASE_Y, Z_NODE]],
  ], [])

  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />

      <Line points={footprint} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      {rungs.map((pts, i) => (
        <Line key={i} points={pts} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      ))}
      <FloorCaption position={[-0.6, BASE_Y, Z_BOT + 0.7]} />
      <PotentialRuler x={SRC_X - 0.4} z={Z_TOP} baseY={BASE_Y} volts={RULER_VOLTS} unitsPerVolt={() => HEIGHT_PER_VOLT * rise.current} />

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
      {/* Level line at node A's height: node B sitting off it is the imbalance */}
      <Line ref={levelRef} points={LEVEL_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.06} gapSize={0.05} />

      {[0, 1, 2, 3].map((r) => (
        <Resistor key={r} ref={(g) => { resRefs.current[r] = g }} />
      ))}
      <Marker ref={nodeA} color={c.marker} size={0.06} />
      <Marker ref={nodeB} color={c.marker} size={0.06} />
      <Marker ref={galv} color={c.power} size={0.08} />

      {[0, 1, 2, 3, 4, 5, 6].map((t) => (
        <group key={t} ref={(g) => { tagRefs.current[t] = g }}>
          <Html center>
            <span
              ref={(el) => { textRefs.current[t] = el }}
              className={labelClass}
              style={t === 6 ? { color: c.power } : undefined}
            />
          </Html>
        </group>
      ))}
      <Html position={[-0.6, BASE_Y - 0.1, Z_BOT + 0.35]} center>
        <span ref={(el) => { textRefs.current[7] = el }} className={labelClass} />
      </Html>
    </>
  )
}

// ── 2D: sweep R4 and watch the bridge voltage cross zero — the null the instrument looks for ──
const X0 = -3.4
const X_SPAN = 6.8
const Y_MID = 0
const VG_SCALE = 1.6 / 3 // ±3 V fills the plot
const CURVE_PTS = 121
const CURVE_INIT: [number, number, number][] = Array.from({ length: CURVE_PTS }, () => [0, 0, 0])
const AXIS_INIT: [number, number, number][] = [[X0, Y_MID, 0], [X0 + X_SPAN, Y_MID, 0]]
const clampY = (v: number) => Math.max(-1.75, Math.min(1.75, v * VG_SCALE))

function BridgeNull2DScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const curveBuf = useMemo(() => new Float32Array(CURVE_PTS * 3), [])
  const markBuf = useMemo(() => new Float32Array(6), [])
  const curveRef = useRef<any>(null)
  const nullRef = useRef<any>(null)
  const dotRef = useRef<THREE.Group>(null)
  const tagRefs = useRef<(THREE.Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<unknown>(null)

  useFrame(() => {
    const s = solve(valuesRef.current)
    for (let i = 0; i < CURVE_PTS; i++) {
      const r4 = R_MIN + (i / (CURVE_PTS - 1)) * (R_MAX - R_MIN)
      const vg = bridgeVg(s.R1, s.R2, s.R3, r4)
      curveBuf[i * 3] = X0 + ((r4 - R_MIN) / (R_MAX - R_MIN)) * X_SPAN
      curveBuf[i * 3 + 1] = Y_MID + clampY(vg)
      curveBuf[i * 3 + 2] = 0
    }
    curveRef.current?.geometry.setPositions(curveBuf)

    // Vertical marker at the balance point, where the curve crosses zero
    const xb = X0 + ((s.balanceR4 - R_MIN) / (R_MAX - R_MIN)) * X_SPAN
    markBuf[0] = xb; markBuf[1] = Y_MID - 1.8; markBuf[2] = 0
    markBuf[3] = xb; markBuf[4] = Y_MID + 1.8; markBuf[5] = 0
    nullRef.current?.geometry.setPositions(markBuf)
    nullRef.current?.computeLineDistances()

    const x = X0 + ((s.R4 - R_MIN) / (R_MAX - R_MIN)) * X_SPAN
    const y = Y_MID + clampY(s.Vg)
    dotRef.current?.position.set(x, y, 0.01)
    tagRefs.current[0]?.position.set(x, y + 0.3, 0)
    tagRefs.current[1]?.position.set(xb, Y_MID - 2.05, 0)

    if (labelledFor.current !== valuesRef.current && textRefs.current[1]) {
      labelledFor.current = valuesRef.current
      const ko = locale === 'ko'
      const texts = [
        `R4 ${fmtR(s.R4)} · Vg ${fmtV(s.Vg)}`,
        `${ko ? '평형' : 'balance'} R4 = R2 × R3 ÷ R1 = ${fmtR(s.balanceR4)}`,
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  const axisLabel = 'pointer-events-none whitespace-nowrap text-[11px] text-muted-foreground'

  return (
    <>
      <Line points={AXIS_INIT} lineWidth={1} color={c.guide} />
      <Line ref={nullRef} points={LEVEL_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      <Line ref={curveRef} points={CURVE_INIT} lineWidth={2.5} color={c.voltage} />
      <Marker ref={dotRef} color={c.marker} size={0.07} />
      {[0, 1].map((t) => (
        <group key={t} ref={(g) => { tagRefs.current[t] = g }}>
          <Html center>
            <span ref={(el) => { textRefs.current[t] = el }} className={labelClass} />
          </Html>
        </group>
      ))}
      <Html position={[X0 + X_SPAN + 0.25, Y_MID, 0]} center>
        <span className={axisLabel}>R4</span>
      </Html>
      <Html position={[X0 - 0.35, Y_MID + 1.7, 0]} center>
        <span className={axisLabel}>Vg</span>
      </Html>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'wheatstone-bridge',
  title: { en: 'Wheatstone Bridge', ko: '휘트스톤 브리지' },
  subject: 'circuits',
  order: 6,
  description: {
    en: `## Wheatstone Bridge

**Two voltage dividers side by side. When their taps sit at the same height, nothing crosses the bridge.**

The landscape shows why balance means "no current": the bridge is a wire between two taps, and current only crosses it if one end is higher than the other.
- Bright dots: the two taps, A and B. The dashed level line runs from A across to B, so the gap you see is Vg
- Amber dot: the galvanometer on the bridge. The bridge line thickens and flows with |Ig|, and goes still at balance
- Glass tubes: R1 over R2 on the left, R3 over R4 on the right

**Balance**: R1 × R4 = R2 × R3, or equivalently R1 ÷ R2 = R3 ÷ R4
**Unknown resistance**: R4 = R2 × R3 ÷ R1 at the null

The source is fixed at 12 V and the galvanometer has 50 Ω. Slide R4 up to 30 Ω with the defaults and the bridge goes flat: both taps reach 8.00 V and Ig falls to zero. That null is what makes the bridge a precision instrument — you read a ratio, not a meter's scale.

In 2D: the bridge voltage as R4 sweeps, crossing zero exactly at the balance point.`,
    ko: `## 휘트스톤 브리지

**전압 분배기 두 개를 나란히 둔 회로입니다. 두 탭의 높이가 같아지면 다리를 건너는 것이 없습니다.**

전위 지형으로 보면 평형이 왜 "전류 0"인지 바로 보입니다. 다리는 두 탭을 잇는 전선이고, 한쪽이 높아야만 전류가 건너갑니다.
- 밝은 점: 두 탭 A와 B. 점선 수평선이 A에서 B까지 이어져서, 벌어진 간격이 곧 Vg입니다
- 앰버 점: 다리 위의 검류계. 다리 선은 |Ig|만큼 굵어지고 흐르며, 평형에서는 멈춥니다
- 유리관: 왼쪽은 R1 위 R2, 오른쪽은 R3 위 R4

**평형 조건**: R1 × R4 = R2 × R3, 즉 R1 ÷ R2 = R3 ÷ R4
**미지 저항**: 평형점에서 R4 = R2 × R3 ÷ R1

전원은 12 V, 검류계는 50 Ω으로 고정했습니다. 기본값에서 R4를 30 Ω으로 올리면 다리가 수평이 됩니다. 두 탭이 모두 8.00 V가 되고 Ig가 0으로 떨어집니다. 이 영점이 브리지를 정밀 계측기로 만들어 줍니다. 계기 눈금이 아니라 저항의 비를 읽기 때문입니다.

2D: R4를 바꿀 때의 브리지 전압입니다. 평형점에서 정확히 0을 지납니다.`,
  },
  sliderParams: [
    { key: 'R1', label: { en: 'R1 (upper left)',  ko: 'R1 (좌측 위)' },  unit: 'Ω', min: R_MIN, max: R_MAX, step: 1, defaultValue: 10 },
    { key: 'R2', label: { en: 'R2 (lower left)',  ko: 'R2 (좌측 아래)' }, unit: 'Ω', min: R_MIN, max: R_MAX, step: 1, defaultValue: 20 },
    { key: 'R3', label: { en: 'R3 (upper right)', ko: 'R3 (우측 위)' },  unit: 'Ω', min: R_MIN, max: R_MAX, step: 1, defaultValue: 15 },
    { key: 'R4', label: { en: 'R4 (unknown)',     ko: 'R4 (미지 저항)' }, unit: 'Ω', min: R_MIN, max: R_MAX, step: 1, defaultValue: 24 },
  ],
  Scene: WheatstoneScene,
  Scene2D: BridgeNull2DScene,
}

export default concept
