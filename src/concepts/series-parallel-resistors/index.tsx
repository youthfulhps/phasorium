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

// Slider ranges (also used to normalize the visuals)
const V_MAX = 24 // [V]
const R_MIN = 1 // [Ω]
const R_MAX = 20 // [Ω]
const P_MAX = (V_MAX * V_MAX) / R_MIN // hottest possible resistor [W]

// Pure physics. Worked example: V = 12 V, R1 = 4 Ω, R2 = 12 Ω →
// series: Req = 16 Ω, I = 0.75 A, V1 = 3 V, V2 = 9 V · parallel: Req = 3 Ω, I = 4 A, I1 = 3 A, I2 = 1 A.
function solve({ V = 12, R1 = 4, R2 = 12 }: Record<string, number>) {
  const series = R1 + R2
  const parallel = (R1 * R2) / (R1 + R2) // 1/Req = 1/R1 + 1/R2
  const Is = V / series
  return {
    V, R1, R2,
    series: { Req: series, I: Is, V1: Is * R1, V2: Is * R2 },
    parallel: { Req: parallel, I: V / parallel, I1: V / R1, I2: V / R2 },
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
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'

// ── 3D: both circuits as potential landscapes, side by side on the same source ──
const BASE_Y = -0.6
const HEIGHT_PER_VOLT = 1.6 / V_MAX
const RULER_VOLTS = [0, 12, 24]
const Z_BACK = -1.0
const Z_FRONT = 1.0
const R_Z: [number, number] = [-0.6, 0.6] // resistors running front-to-back on a side
const SEG_PTS = 24 // samples per straight segment

type XZ = [number, number]
// Series loop (x ∈ [−2.6, −1.1]): source up the left side, R1 on the back wire, R2 down the right side
const SX0 = -2.6
const SX1 = -1.1
const SERIES_XZ: XZ[] = [
  [SX0, Z_FRONT], [SX0, Z_BACK], // source
  [SX0 + 0.3, Z_BACK], [SX1 - 0.3, Z_BACK], // R1
  [SX1, Z_BACK], [SX1, R_Z[0]], [SX1, R_Z[1]], // R2
  [SX1, Z_FRONT], [SX0, Z_FRONT],
]
// Parallel loop (x ∈ [−0.5, 1.1]): the source feeds junction J1 (back middle); R1 on the right side
// and R2 on the middle rung both drop from J1 to J2 (front middle)
const PX0 = -0.5
const PXM = 0.3
const PX1 = 1.1
const MAIN_XZ: XZ[] = [[PXM, Z_FRONT], [PX0, Z_FRONT], [PX0, Z_BACK], [PXM, Z_BACK]]
const BRANCH1_XZ: XZ[] = [[PXM, Z_BACK], [PX1, Z_BACK], [PX1, R_Z[0]], [PX1, R_Z[1]], [PX1, Z_FRONT], [PXM, Z_FRONT]]
const BRANCH2_XZ: XZ[] = [[PXM, Z_BACK], [PXM, R_Z[0]], [PXM, R_Z[1]], [PXM, Z_FRONT]]

// Potential (volts) at each waypoint, rewritten every frame — preallocated, no per-frame arrays
const seriesV = new Float32Array(SERIES_XZ.length)
const mainV = new Float32Array(MAIN_XZ.length)
const branch1V = new Float32Array(BRANCH1_XZ.length)
const branch2V = new Float32Array(BRANCH2_XZ.length)
const LINE_XZ = [SERIES_XZ, MAIN_XZ, BRANCH1_XZ, BRANCH2_XZ]
const LINE_V = [seriesV, mainV, branch1V, branch2V]
const lineI = new Float32Array(4) // current in each line

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
    const r = 0.13 * Math.sqrt(R_MIN / R) + 0.03 // thinner for larger R
    g.scale.set(dir.length(), r, r)
  }
  setResistorHeat(g, Math.sqrt(P / P_MAX) * 1.6)
  tag?.position.set((p0.x + p1.x) / 2 + 0.35, Math.max(p0.y, p1.y) + 0.15, (p0.z + p1.z) / 2)
}

/** Current as line weight: wider for more current (px), clamped to stay readable. */
const widthFor = (I: number) => Math.min(1.2 + 1.6 * Math.sqrt(I), 8)
/** Sheen bands per second: faster for more current. */
const flowFor = (I: number) => Math.min(0.6 * I, 8)

function SeriesParallelScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const rise = useRiseIn()
  const bufs = useMemo(() => LINE_XZ.map((xz) => new Float32Array(pointCount(xz) * 3)), [])
  const inits = useMemo(() => LINE_XZ.map(initPoints), [])
  const lineRefs = useRef<any[]>([]) // series, parallel main, branch 1, branch 2
  const resRefs = useRef<(THREE.Group | null)[]>([]) // series R1, series R2, parallel R1, parallel R2
  const tagRefs = useRef<(THREE.Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([]) // 4 resistor tags, series total, parallel total
  const flows = useRef([0, 0, 0, 0])
  const labelledFor = useRef<unknown>(null)

  useFrame(() => {
    const s = solve(valuesRef.current)
    const { V, R1, R2 } = s
    const k = HEIGHT_PER_VOLT * rise.current

    // Series: V → (drop V1 across R1) → (drop V2 across R2) → 0
    const V2start = V - s.series.V1
    seriesV[1] = seriesV[2] = V
    seriesV[3] = seriesV[4] = seriesV[5] = V2start
    // Parallel: every branch starts at V and ends at 0
    mainV[2] = mainV[3] = V
    branch1V[0] = branch1V[1] = branch1V[2] = V
    branch2V[0] = branch2V[1] = V

    lineI[0] = s.series.I; lineI[1] = s.parallel.I; lineI[2] = s.parallel.I1; lineI[3] = s.parallel.I2
    for (let l = 0; l < 4; l++) {
      const line = lineRefs.current[l]
      if (!line) continue
      fillPath(bufs[l], LINE_XZ[l], LINE_V[l], k)
      line.geometry.setPositions(bufs[l])
      ;(line.material as LineMaterial).linewidth = widthFor(lineI[l])
      flows.current[l] = flowFor(lineI[l])
    }

    placeResistor(resRefs.current[0], tagRefs.current[0],
      SERIES_XZ[2], V, SERIES_XZ[3], V2start, k, R1, s.series.I * s.series.V1)
    placeResistor(resRefs.current[1], tagRefs.current[1],
      SERIES_XZ[5], V2start, SERIES_XZ[6], 0, k, R2, s.series.I * s.series.V2)
    placeResistor(resRefs.current[2], tagRefs.current[2],
      BRANCH1_XZ[2], V, BRANCH1_XZ[3], 0, k, R1, V * s.parallel.I1)
    placeResistor(resRefs.current[3], tagRefs.current[3],
      BRANCH2_XZ[1], V, BRANCH2_XZ[2], 0, k, R2, V * s.parallel.I2)

    if (labelledFor.current !== valuesRef.current && textRefs.current[5]) {
      labelledFor.current = valuesRef.current
      const ko = locale === 'ko'
      const texts = [
        `R1 · ${fmt(s.series.V1, 'V')} · ${fmt(s.series.I, 'A')}`,
        `R2 · ${fmt(s.series.V2, 'V')} · ${fmt(s.series.I, 'A')}`,
        `R1 · ${fmt(V, 'V')} · ${fmt(s.parallel.I1, 'A')}`,
        `R2 · ${fmt(V, 'V')} · ${fmt(s.parallel.I2, 'A')}`,
        `${ko ? '직렬' : 'Series'} · Req ${fmt(s.series.Req, 'Ω')} · I ${fmt(s.series.I, 'A')}`,
        `${ko ? '병렬' : 'Parallel'} · Req ${fmt(s.parallel.Req, 'Ω')} · I ${fmt(s.parallel.I, 'A')}`,
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  // Relabel immediately when the language changes
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

      {[0, 1, 2, 3].map((r) => (
        <group key={r}>
          <Resistor ref={(g) => { resRefs.current[r] = g }} />
          <group ref={(g) => { tagRefs.current[r] = g }}>
            <Html center>
              <span ref={(el) => { textRefs.current[r] = el }} className={labelClass} />
            </Html>
          </group>
        </group>
      ))}

      <Html position={[(SX0 + SX1) / 2, BASE_Y - 0.1, Z_FRONT + 0.35]} center>
        <span ref={(el) => { textRefs.current[4] = el }} className={labelClass} />
      </Html>
      <Html position={[(PX0 + PX1) / 2, BASE_Y - 0.1, Z_FRONT + 0.35]} center>
        <span ref={(el) => { textRefs.current[5] = el }} className={labelClass} />
      </Html>
    </>
  )
}

// ── 2D: Req as R2 sweeps, with R1 fixed — series always above the larger, parallel below the smaller ──
const X0 = -3.5
const Y0 = -1.8
const X_PER_OHM = 7 / R_MAX
const REQ_MAX = 2 * R_MAX
const Y_PER_OHM = 3.6 / REQ_MAX
const CURVE_PTS = 81
const AXES: [number, number, number][] = [[X0, Y0 + 3.8, 0], [X0, Y0, 0], [X0 + 7.3, Y0, 0]]
const CURVE_INIT: [number, number, number][] = Array.from({ length: CURVE_PTS }, () => [0, 0, 0])
const GUIDE_INIT: [number, number, number][] = [[X0, Y0, 0], [X0 + 7, Y0, 0]]

function ReqPlot2DScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const seriesBuf = useMemo(() => new Float32Array(CURVE_PTS * 3), [])
  const parallelBuf = useMemo(() => new Float32Array(CURVE_PTS * 3), [])
  const guideBuf = useMemo(() => new Float32Array(6), [])
  const seriesRef = useRef<any>(null)
  const parallelRef = useRef<any>(null)
  const guideRef = useRef<any>(null)
  const dotS = useRef<THREE.Group>(null)
  const dotP = useRef<THREE.Group>(null)
  const tagS = useRef<THREE.Group>(null)
  const tagP = useRef<THREE.Group>(null)
  const tagG = useRef<THREE.Group>(null)
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<unknown>(null)

  useFrame(() => {
    const s = solve(valuesRef.current)
    const { R1, R2 } = s
    for (let i = 0; i < CURVE_PTS; i++) {
      const r2 = (i / (CURVE_PTS - 1)) * R_MAX
      const x = X0 + r2 * X_PER_OHM
      seriesBuf[i * 3] = x; seriesBuf[i * 3 + 1] = Y0 + (R1 + r2) * Y_PER_OHM; seriesBuf[i * 3 + 2] = 0
      parallelBuf[i * 3] = x; parallelBuf[i * 3 + 1] = Y0 + ((R1 * r2) / (R1 + r2)) * Y_PER_OHM; parallelBuf[i * 3 + 2] = 0
    }
    seriesRef.current?.geometry.setPositions(seriesBuf)
    parallelRef.current?.geometry.setPositions(parallelBuf)
    // R1 level: the parallel curve approaches it but never reaches it
    guideBuf[0] = X0; guideBuf[1] = Y0 + R1 * Y_PER_OHM; guideBuf[2] = 0
    guideBuf[3] = X0 + 7; guideBuf[4] = guideBuf[1]; guideBuf[5] = 0
    guideRef.current?.geometry.setPositions(guideBuf)
    guideRef.current?.computeLineDistances()

    const x = X0 + R2 * X_PER_OHM
    dotS.current?.position.set(x, Y0 + s.series.Req * Y_PER_OHM, 0.01)
    dotP.current?.position.set(x, Y0 + s.parallel.Req * Y_PER_OHM, 0.01)
    tagS.current?.position.set(x, Y0 + s.series.Req * Y_PER_OHM + 0.25, 0)
    tagP.current?.position.set(x, Y0 + s.parallel.Req * Y_PER_OHM - 0.25, 0)
    tagG.current?.position.set(X0 + 7.4, guideBuf[1], 0)

    if (labelledFor.current !== valuesRef.current && textRefs.current[2]) {
      labelledFor.current = valuesRef.current
      const ko = locale === 'ko'
      const texts = [
        `${ko ? '직렬' : 'Series'} ${fmt(s.series.Req, 'Ω')}`,
        `${ko ? '병렬' : 'Parallel'} ${fmt(s.parallel.Req, 'Ω')}`,
        `R1 = ${fmt(R1, 'Ω')}`,
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  const axisLabel = 'pointer-events-none whitespace-nowrap text-[11px] text-muted-foreground'

  return (
    <>
      <Line points={AXES} lineWidth={1} color={c.guide} />
      <Line ref={guideRef} points={GUIDE_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      <Line ref={seriesRef} points={CURVE_INIT} lineWidth={2.5} color={c.voltage} />
      <Line ref={parallelRef} points={CURVE_INIT} lineWidth={2.5} color={c.resistance} />
      <Marker ref={dotS} color={c.marker} size={0.06} />
      <Marker ref={dotP} color={c.marker} size={0.06} />
      {[tagS, tagP, tagG].map((ref, t) => (
        <group key={t} ref={ref}>
          <Html center>
            <span ref={(el) => { textRefs.current[t] = el }} className={labelClass} />
          </Html>
        </group>
      ))}
      <Html position={[X0 + 7.3, Y0 - 0.25, 0]} center>
        <span className={axisLabel}>R2 (Ω)</span>
      </Html>
      <Html position={[X0 - 0.35, Y0 + 3.8, 0]} center>
        <span className={axisLabel}>Req (Ω)</span>
      </Html>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'series-parallel-resistors',
  title: { en: 'Series & Parallel Resistors', ko: '직렬·병렬 합성저항' },
  subject: 'circuits',
  order: 3,
  description: {
    en: `## Series and Parallel Resistors

**Series shares the current and splits the voltage. Parallel shares the voltage and splits the current.**

Two circuits on the same source, drawn as potential landscapes (height = potential).
- Left, series: one path, so one current; the height comes down in two steps, V1 and V2
- Right, parallel: both branches start at the full height V; the current splits, more of it through the smaller resistor
- Cyan liquid line: the current; wider and faster means more current
- Glass tube: a resistor, thinner for larger R; its amber filament thickens with the power turned into heat

**Series**: Req = R1 + R2, the same I everywhere, V1 : V2 = R1 : R2
**Parallel**: 1 ÷ Req = 1 ÷ R1 + 1 ÷ R2, the same V on each, I1 : I2 = R2 : R1

Series Req is always larger than the larger resistor; parallel Req is always smaller than the smaller one. So the parallel circuit draws more current and more power from the same source.

In 2D: Req as R2 sweeps with R1 fixed.`,
    ko: `## 직렬·병렬 합성저항

**직렬은 전류를 함께 쓰고 전압을 나눕니다. 병렬은 전압을 함께 쓰고 전류를 나눕니다.**

같은 전원에 연결한 두 회로를 전위 지형(높이 = 전위)으로 그렸습니다.
- 왼쪽, 직렬: 길이 하나라 전류도 하나. 높이가 V1, V2 두 계단으로 내려옵니다
- 오른쪽, 병렬: 두 갈래 모두 같은 높이 V에서 출발합니다. 전류가 갈라지고, 저항이 작은 쪽으로 더 많이 흐릅니다
- 시안 액체 선: 전류. 굵고 빠를수록 전류가 큽니다
- 유리관: 저항. R이 클수록 가늘고, 열로 바뀌는 전력이 클수록 안쪽 앰버 필라멘트가 굵어집니다

**직렬**: Req = R1 + R2, 어디서나 같은 I, V1 : V2 = R1 : R2
**병렬**: 1 ÷ Req = 1 ÷ R1 + 1 ÷ R2, 각 저항에 같은 V, I1 : I2 = R2 : R1

직렬 Req는 항상 큰 저항보다 크고, 병렬 Req는 항상 작은 저항보다 작습니다. 그래서 같은 전원에서 병렬 회로가 더 많은 전류와 전력을 끌어갑니다.

2D: R1을 고정하고 R2를 바꿀 때의 Req입니다.`,
  },
  sliderParams: [
    { key: 'V',  label: { en: 'Voltage (V)',      ko: '전압 (V)' },   unit: 'V', min: 1,     max: V_MAX, step: 0.5, defaultValue: 12 },
    { key: 'R1', label: { en: 'Resistance (R1)',  ko: '저항 (R1)' },  unit: 'Ω', min: R_MIN, max: R_MAX, step: 0.5, defaultValue: 4 },
    { key: 'R2', label: { en: 'Resistance (R2)',  ko: '저항 (R2)' },  unit: 'Ω', min: R_MIN, max: R_MAX, step: 0.5, defaultValue: 12 },
  ],
  Scene: SeriesParallelScene,
  Scene2D: ReqPlot2DScene,
}

export default concept
