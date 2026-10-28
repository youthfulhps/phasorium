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
const R2_FIXED = 4 // [Ω] R2 is not a slider, so the two source arms stay comparable
const P_MAX = (V_MAX * V_MAX) / R_MIN // hottest possible resistor [W]

// Which source is alive in each of the three landscapes: V1 alone, V2 alone, both.
const MODE_V1 = 0
const MODE_V2 = 1
const MODE_BOTH = 2
const SRC_ON = [[1, 0], [0, 1], [1, 1]]

type Part = { V1: number; V2: number; VA: number; I1: number; I2: number; I3: number; R1: number; R3: number }
const mkPart = (): Part => ({ V1: 0, V2: 0, VA: 0, I1: 0, I2: 0, I3: 0, R1: 0, R3: 0 })
// Reused every frame so useFrame never allocates; each mode owns one slot.
const PARTS: Part[] = [mkPart(), mkPart(), mkPart()]

// Pure physics: one node A, fed by V1 through R1 and V2 through R2, draining through R3.
// Node analysis: VA = (V1 ÷ R1 + V2 ÷ R2) ÷ (1 ÷ R1 + 1 ÷ R2 + 1 ÷ R3)
// Worked example: V1 = 12 V, V2 = 6 V, R1 = 4 Ω, R2 = 4 Ω, R3 = 4 Ω →
// VA1 = 4 V (V2 shorted), VA2 = 2 V (V1 shorted), VA = 6 V, and VA1 + VA2 = VA exactly.
// At that point I1 = 1.5 A, I2 = 0 A (node A already sits at V2), I3 = 1.5 A.
function solve({ V1 = 12, V2 = 6, R1 = 4, R3 = 4 }: Record<string, number>, mode: number): Part {
  const p = PARTS[mode]
  p.R1 = R1
  p.R3 = R3
  p.V1 = V1 * SRC_ON[mode][0]
  p.V2 = V2 * SRC_ON[mode][1]
  p.VA = (p.V1 / R1 + p.V2 / R2_FIXED) / (1 / R1 + 1 / R2_FIXED + 1 / R3)
  p.I1 = (p.V1 - p.VA) / R1
  p.I2 = (p.V2 - p.VA) / R2_FIXED
  p.I3 = p.VA / R3
  return p
}

function useSliderValues() {
  const valuesRef = useRef<Record<string, number>>(useSliderStore.getState().values)
  useEffect(
    () => useSliderStore.subscribe((s) => s.values, (v) => { valuesRef.current = v }),
    []
  )
  return valuesRef
}

const fmt = (x: number, unit: string) => `${Math.abs(x) < 10 ? x.toFixed(2) : x.toFixed(1)} ${unit}`
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const captionClass = 'pointer-events-none whitespace-nowrap text-[10px] text-muted-foreground'

// ── 3D: the same circuit as three potential landscapes, whose node heights add up ──
const BASE_Y = -0.9
const HEIGHT_PER_VOLT = 1.5 / V_MAX
const RULER_VOLTS = [0, 12, 24]
const Z_BACK = -0.7
const Z_FRONT = 0.7 // the front rail is the 0 V ground return
const SEG_PTS = 16 // samples per straight segment

type XZ = [number, number]
const AF: XZ = [0, Z_FRONT] // ground under node A
const S1F: XZ = [-1.3, Z_FRONT]
const S1B: XZ = [-1.3, Z_BACK]
const R1S: XZ = [-1.05, Z_BACK]
const R1E: XZ = [-0.28, Z_BACK]
const NODE: XZ = [0, Z_BACK] // node A
const R2S: XZ = [0.28, Z_BACK]
const R2E: XZ = [1.05, Z_BACK]
const S2B: XZ = [1.3, Z_BACK]
const S2F: XZ = [1.3, Z_FRONT]
const R3S: XZ = [0, -0.25]
const R3E: XZ = [0, 0.3]

// Each path is drawn in the direction a positive current takes
const PATH_L: XZ[] = [AF, S1F, S1B, R1S, R1E, NODE] // carries I1, source V1 → node
const PATH_R: XZ[] = [AF, S2F, S2B, R2E, R2S, NODE] // carries I2, source V2 → node
const PATH_G: XZ[] = [NODE, R3S, R3E, AF] // carries I3, node → ground
const PATHS = [PATH_L, PATH_R, PATH_G]

const pointCount = (xz: XZ[]) => (xz.length - 1) * SEG_PTS + 1
const initPoints = (xz: XZ[]): [number, number, number][] =>
  Array.from({ length: pointCount(xz) }, () => [0, 0, 0])
const PATH_INITS = PATHS.map(initPoints)

/** Sample a waypoint polyline into `buf`; the panel sits at (ox, oz) and is `s` times full size. */
function fillPath(
  buf: Float32Array, xz: XZ[], volts: Float32Array, k: number, ox: number, oz: number, s: number
) {
  let n = 0
  for (let seg = 0; seg < xz.length - 1; seg++) {
    for (let j = 0; j < SEG_PTS + (seg === xz.length - 2 ? 1 : 0); j++) {
      const u = j / SEG_PTS
      buf[n++] = ox + s * (xz[seg][0] + (xz[seg + 1][0] - xz[seg][0]) * u)
      buf[n++] = BASE_Y + k * (volts[seg] + (volts[seg + 1] - volts[seg]) * u)
      buf[n++] = oz + s * (xz[seg][1] + (xz[seg + 1][1] - xz[seg][1]) * u)
    }
  }
}

const X_AXIS = new THREE.Vector3(1, 0, 0)
const p0 = new THREE.Vector3()
const p1 = new THREE.Vector3()
const dir = new THREE.Vector3()
const unit = new THREE.Vector3()

/** Seat a resistor between two waypoints of a panel; its filament thickens with power. */
function placeResistor(
  g: THREE.Group | null, tag: THREE.Group | null,
  a: XZ, va: number, b: XZ, vb: number,
  k: number, ox: number, oz: number, s: number, R: number, P: number
) {
  p0.set(ox + s * a[0], BASE_Y + k * va, oz + s * a[1])
  p1.set(ox + s * b[0], BASE_Y + k * vb, oz + s * b[1])
  dir.subVectors(p1, p0)
  if (g) {
    g.position.addVectors(p0, p1).multiplyScalar(0.5)
    g.quaternion.setFromUnitVectors(X_AXIS, unit.copy(dir).normalize())
    const r = s * (0.13 * Math.sqrt(R_MIN / Math.max(R, R_MIN)) + 0.04)
    g.scale.set(dir.length(), r, r)
  }
  setResistorHeat(g, Math.sqrt(P / P_MAX) * 1.6)
  tag?.position.set((p0.x + p1.x) / 2, Math.max(p0.y, p1.y) + 0.22 * s + 0.1, (p0.z + p1.z) / 2)
}

/** Current as line weight and sheen speed; the sign flips the sheen so a reversed branch shows it. */
const widthFor = (I: number) => Math.min(1.2 + 1.7 * Math.sqrt(Math.abs(I)), 8)
const flowFor = (I: number) => Math.max(-8, Math.min(0.8 * I, 8))

const DROP_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]

type PanelProps = { ox: number; oz: number; s: number; mode: number; detail?: boolean }

/** One potential landscape of the same circuit, with the sources named by `mode` switched on. */
function Panel({ ox, oz, s, mode, detail = false }: PanelProps) {
  const c = usePhysicsColors()
  const locale = useLocaleStore((state) => state.locale)
  const valuesRef = useSliderValues()
  const rise = useRiseIn()
  const bufs = useMemo(() => PATHS.map((xz) => new Float32Array(pointCount(xz) * 3)), [])
  const volts = useMemo(() => PATHS.map((xz) => new Float32Array(xz.length)), [])
  const dropBuf = useMemo(() => new Float32Array(6), [])
  const currents = useMemo(() => new Float32Array(3), [])
  const flows = useRef([0, 0, 0])
  const lineRefs = useRef<any[]>([])
  const dropRef = useRef<any>(null)
  const resRefs = useRef<(THREE.Group | null)[]>([]) // R1, R2, R3
  const tagRefs = useRef<(THREE.Group | null)[]>([]) // node, R1, R2, R3
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const nodeRef = useRef<THREE.Group>(null)
  const labelledFor = useRef<unknown>(null)

  useFrame(() => {
    const p = solve(valuesRef.current, mode)
    const k = HEIGHT_PER_VOLT * rise.current
    const { V1, V2, VA } = p

    // Height of every waypoint, in volts. A shorted source simply stays flat on the floor.
    volts[0][2] = volts[0][3] = V1
    volts[0][4] = volts[0][5] = VA
    volts[1][2] = volts[1][3] = V2
    volts[1][4] = volts[1][5] = VA
    volts[2][0] = volts[2][1] = VA

    currents[0] = p.I1
    currents[1] = p.I2
    currents[2] = p.I3
    for (let l = 0; l < 3; l++) {
      const line = lineRefs.current[l]
      if (!line) continue
      fillPath(bufs[l], PATHS[l], volts[l], k, ox, oz, s)
      line.geometry.setPositions(bufs[l])
      ;(line.material as LineMaterial).linewidth = widthFor(currents[l]) * (detail ? 1 : 0.75)
      flows.current[l] = flowFor(currents[l])
    }

    placeResistor(resRefs.current[0], tagRefs.current[1], R1S, V1, R1E, VA, k, ox, oz, s, p.R1, p.I1 * p.I1 * p.R1)
    placeResistor(resRefs.current[1], tagRefs.current[2], R2E, V2, R2S, VA, k, ox, oz, s, R2_FIXED, p.I2 * p.I2 * R2_FIXED)
    placeResistor(resRefs.current[2], tagRefs.current[3], R3S, VA, R3E, 0, k, ox, oz, s, p.R3, p.I3 * p.I3 * p.R3)
    // R3 runs under the node, so nudge its label toward the viewer
    if (tagRefs.current[3]) tagRefs.current[3].position.z += 0.35 * s

    // Node A: the height that has to add up across the three panels
    const ny = BASE_Y + k * VA
    nodeRef.current?.position.set(ox, ny, oz + s * Z_BACK)
    tagRefs.current[0]?.position.set(ox, ny + 0.26, oz + s * Z_BACK - 0.05)
    dropBuf[0] = ox; dropBuf[1] = BASE_Y; dropBuf[2] = oz + s * Z_BACK
    dropBuf[3] = ox; dropBuf[4] = ny; dropBuf[5] = oz + s * Z_BACK
    dropRef.current?.geometry.setPositions(dropBuf)
    dropRef.current?.computeLineDistances()

    if (labelledFor.current !== valuesRef.current && textRefs.current[0]) {
      labelledFor.current = valuesRef.current
      const name = mode === MODE_V1 ? 'VA1' : mode === MODE_V2 ? 'VA2' : 'VA'
      textRefs.current[0]!.textContent = `${name} ${fmt(VA, 'V')}`
      if (detail) {
        const texts = [
          `R1 ${fmt(p.R1, 'Ω')} · ${fmt(p.I1, 'A')}`,
          `R2 ${fmt(R2_FIXED, 'Ω')} · ${fmt(p.I2, 'A')}`,
          `R3 ${fmt(p.R3, 'Ω')} · ${fmt(p.I3, 'A')}`,
        ]
        texts.forEach((text, t) => { if (textRefs.current[t + 1]) textRefs.current[t + 1]!.textContent = text })
      }
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  const footprint = useMemo<[number, number, number][]>(() => [
    [ox + s * S1F[0], BASE_Y, oz + s * S1F[1]],
    [ox + s * S1B[0], BASE_Y, oz + s * S1B[1]],
    [ox + s * S2B[0], BASE_Y, oz + s * S2B[1]],
    [ox + s * S2F[0], BASE_Y, oz + s * S2F[1]],
    [ox + s * S1F[0], BASE_Y, oz + s * S1F[1]],
  ], [ox, oz, s])
  const rung = useMemo<[number, number, number][]>(() => [
    [ox, BASE_Y, oz + s * Z_BACK],
    [ox, BASE_Y, oz + s * Z_FRONT],
  ], [ox, oz, s])

  const ko = locale === 'ko'
  const caption = mode === MODE_V1
    ? (ko ? 'V1만 · V2는 단락' : 'V1 alone · V2 shorted')
    : mode === MODE_V2
      ? (ko ? 'V2만 · V1은 단락' : 'V2 alone · V1 shorted')
      : (ko ? '두 전원 모두' : 'Both sources on')
  const tagCount = detail ? 4 : 1

  return (
    <>
      <Line points={footprint} lineWidth={1} color={c.guide} dashed dashSize={0.07} gapSize={0.055} />
      <Line points={rung} lineWidth={1} color={c.guide} dashed dashSize={0.07} gapSize={0.055} />
      <Line ref={dropRef} points={DROP_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.06} gapSize={0.05} />

      {PATH_INITS.map((pts, l) => (
        <TrailLine
          key={l}
          ref={(line) => { lineRefs.current[l] = line }}
          points={pts}
          color={c.current}
          fade={false}
          flowSpeed={() => flows.current[l]}
        />
      ))}

      {[0, 1, 2].map((r) => (
        <Resistor key={r} ref={(g) => { resRefs.current[r] = g }} />
      ))}

      <Marker ref={nodeRef} color={c.marker} size={detail ? 0.07 : 0.05} />

      {Array.from({ length: tagCount }, (_, t) => (
        <group key={t} ref={(g) => { tagRefs.current[t] = g }}>
          <Html center>
            <span ref={(el) => { textRefs.current[t] = el }} className={labelClass} />
          </Html>
        </group>
      ))}

      <Html position={[ox, BASE_Y - 0.08, oz + s * Z_FRONT + 0.3]} center>
        <span className={captionClass}>{caption}</span>
      </Html>
    </>
  )
}

// Two small partial landscapes on the left, the full one large on the right
const PANEL_A = { ox: -1.85, oz: -0.95, s: 0.52, mode: MODE_V1 }
const PANEL_B = { ox: -1.85, oz: 0.95, s: 0.52, mode: MODE_V2 }
const PANEL_SUM = { ox: 0.65, oz: 0, s: 0.85, mode: MODE_BOTH }

function SuperpositionScene() {
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const rise = useRiseIn()
  const sumRef = useRef<HTMLSpanElement | null>(null)
  const labelledFor = useRef<unknown>(null)

  useFrame(() => {
    if (labelledFor.current === valuesRef.current || !sumRef.current) return
    labelledFor.current = valuesRef.current
    const a = solve(valuesRef.current, MODE_V1).VA
    const b = solve(valuesRef.current, MODE_V2).VA
    const total = solve(valuesRef.current, MODE_BOTH).VA
    sumRef.current.textContent = `VA1 ${fmt(a, 'V')} + VA2 ${fmt(b, 'V')} = VA ${fmt(total, 'V')}`
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />

      <Panel {...PANEL_A} />
      <Panel {...PANEL_B} />
      <Panel {...PANEL_SUM} detail />

      <FloorCaption position={[0.65, BASE_Y, 1.55]} />
      <PotentialRuler
        x={1.95}
        z={0.5}
        baseY={BASE_Y}
        volts={RULER_VOLTS}
        unitsPerVolt={() => HEIGHT_PER_VOLT * rise.current}
      />

      <Html position={[-0.6, BASE_Y + 1.55, 0]} center>
        <span ref={sumRef} className={labelClass} />
      </Html>
    </>
  )
}

// ── 2D: VA1 and VA2 as bars, stacked, beside the node voltage of the full circuit ──
const Y0 = -1.35 // 0 V baseline
const V_SCALE = 2.8 / V_MAX
const BAR_W = 0.62
const X_A = -2.6 // VA1
const X_B = -1.4 // VA2
const X_STACK = 0.3 // VA1 stacked on VA2
const X_TOTAL = 1.8 // VA from the full circuit
const BASE_INIT: [number, number, number][] = [[X_A - 0.8, Y0, 0], [X_TOTAL + 0.8, Y0, 0]]
const TIE_INIT: [number, number, number][] = [[X_STACK, Y0, 0], [X_TOTAL, Y0, 0]]
const BAR_X = [X_A, X_B, X_STACK, X_STACK, X_TOTAL]
const BAR_OPACITY = [0.85, 0.45, 0.85, 0.45, 0.85]

function SuperpositionBars2DScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const barRefs = useRef<(THREE.Mesh | null)[]>([])
  const tagRefs = useRef<(THREE.Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const tieRef = useRef<any>(null)
  const tieBuf = useMemo(() => new Float32Array(6), [])
  const labelledFor = useRef<unknown>(null)

  useFrame(() => {
    const a = solve(valuesRef.current, MODE_V1).VA
    const b = solve(valuesRef.current, MODE_V2).VA
    const total = solve(valuesRef.current, MODE_BOTH).VA
    const ha = Math.max(a * V_SCALE, 1e-3)
    const hb = Math.max(b * V_SCALE, 1e-3)
    const ht = Math.max(total * V_SCALE, 1e-3)

    // VA1 alone, VA2 alone, then VA2 stacked on VA1, then VA measured on the full circuit
    barRefs.current[0]?.position.set(X_A, Y0 + ha / 2, 0)
    barRefs.current[0]?.scale.set(BAR_W, ha, 1)
    barRefs.current[1]?.position.set(X_B, Y0 + hb / 2, 0)
    barRefs.current[1]?.scale.set(BAR_W, hb, 1)
    barRefs.current[2]?.position.set(X_STACK, Y0 + ha / 2, 0)
    barRefs.current[2]?.scale.set(BAR_W, ha, 1)
    barRefs.current[3]?.position.set(X_STACK, Y0 + ha + hb / 2, 0)
    barRefs.current[3]?.scale.set(BAR_W, hb, 1)
    barRefs.current[4]?.position.set(X_TOTAL, Y0 + ht / 2, 0)
    barRefs.current[4]?.scale.set(BAR_W, ht, 1)

    tagRefs.current[0]?.position.set(X_A, Y0 + ha + 0.22, 0.01)
    tagRefs.current[1]?.position.set(X_B, Y0 + hb + 0.22, 0.01)
    tagRefs.current[2]?.position.set(X_STACK, Y0 + ha + hb + 0.22, 0.01)
    tagRefs.current[3]?.position.set(X_TOTAL, Y0 + ht + 0.22, 0.01)

    // The tie line is horizontal only because the two heights really are equal
    tieBuf[0] = X_STACK; tieBuf[1] = Y0 + ha + hb; tieBuf[2] = 0
    tieBuf[3] = X_TOTAL; tieBuf[4] = Y0 + ht; tieBuf[5] = 0
    tieRef.current?.geometry.setPositions(tieBuf)
    tieRef.current?.computeLineDistances()

    if (labelledFor.current !== valuesRef.current && textRefs.current[3]) {
      labelledFor.current = valuesRef.current
      const texts = [
        `VA1 ${fmt(a, 'V')}`,
        `VA2 ${fmt(b, 'V')}`,
        `VA1 + VA2 ${fmt(a + b, 'V')}`,
        `VA ${fmt(total, 'V')}`,
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  const ko = locale === 'ko'

  return (
    <>
      <Line points={BASE_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      <Line ref={tieRef} points={TIE_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      {BAR_X.map((x, i) => (
        <mesh key={i} ref={(m) => { barRefs.current[i] = m }} position={[x, Y0, 0]}>
          <planeGeometry args={[1, 1]} />
          <meshBasicMaterial color={c.voltage} transparent opacity={BAR_OPACITY[i]} depthWrite={false} />
        </mesh>
      ))}
      {[0, 1, 2, 3].map((t) => (
        <group key={t} ref={(g) => { tagRefs.current[t] = g }}>
          <Html center>
            <span ref={(el) => { textRefs.current[t] = el }} className={labelClass} />
          </Html>
        </group>
      ))}
      <Html position={[(X_A + X_B) / 2, Y0 + 0.3, 0]} center>
        <span className={labelClass}>+</span>
      </Html>
      <Html position={[(X_B + X_STACK) / 2, Y0 + 0.3, 0]} center>
        <span className={labelClass}>=</span>
      </Html>
      <Html position={[(X_STACK + X_TOTAL) / 2, Y0 + 0.3, 0]} center>
        <span className={labelClass}>=</span>
      </Html>
      <Html position={[X_A, Y0 - 0.3, 0]} center>
        <span className={captionClass}>{ko ? 'V1만' : 'V1 alone'}</span>
      </Html>
      <Html position={[X_B, Y0 - 0.3, 0]} center>
        <span className={captionClass}>{ko ? 'V2만' : 'V2 alone'}</span>
      </Html>
      <Html position={[X_STACK, Y0 - 0.3, 0]} center>
        <span className={captionClass}>{ko ? '부분 응답의 합' : 'partial responses added'}</span>
      </Html>
      <Html position={[X_TOTAL, Y0 - 0.3, 0]} center>
        <span className={captionClass}>{ko ? '두 전원 모두' : 'both sources on'}</span>
      </Html>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'superposition',
  title: { en: 'Superposition', ko: '중첩의 원리' },
  subject: 'circuits',
  order: 8,
  description: {
    en: `## Superposition

**In a linear circuit every source acts as if the others were not there, and the separate responses simply add.**

Node A is fed by V1 through R1 and by V2 through R2, and drains to ground through R3 (R2 is fixed at 4 Ω). The same circuit is drawn three times as a potential landscape, where height = potential.
- Back left, small: V1 alone, with V2 replaced by a short, so its arm lies flat on the floor and node A rests at VA1
- Front left, small: V2 alone, with V1 shorted, so node A rests at VA2
- Right, large: both sources on, and node A rests at exactly VA1 + VA2
- Cyan liquid line: the current; wider and faster means more current, and the sheen runs backwards when a branch feeds the node instead of draining it
- Glass tube: a resistor, thinner for larger R, with an amber filament showing the power it turns into heat

**Node A**: VA = (V1 ÷ R1 + V2 ÷ R2) ÷ (1 ÷ R1 + 1 ÷ R2 + 1 ÷ R3)
**Superposition**: VA1 + VA2 = VA, where VA1 is solved with V2 = 0 and VA2 with V1 = 0

Raise V1 and watch what does **not** move: the V2 landscape ignores it completely, while VA1 and the total rise together. The resistance sliders change all three at once, because they belong to every version of the circuit.

At the defaults node A already sits at 6 V, exactly where V2 is, so R2 carries no current. Nudge V2 either way and that branch starts to flow, one direction or the other.

Superposition holds for voltage and current, never for power: power goes as the square of the response, so the two partial powers do not add.

In 2D: VA1 and VA2 as bars, stacked, beside the node voltage measured on the full circuit.`,
    ko: `## 중첩의 원리

**선형 회로에서 각 전원은 다른 전원이 없는 것처럼 작용하고, 따로 구한 응답을 그냥 더하면 됩니다.**

마디 A는 R1을 거쳐 V1에서, R2를 거쳐 V2에서 전류를 받고, R3를 통해 접지로 흘려보냅니다 (R2는 4 Ω 고정). 같은 회로를 전위 지형(높이 = 전위)으로 세 번 그렸습니다.
- 왼쪽 뒤, 작은 것: V1만 살리고 V2는 단락. 그쪽 팔은 바닥에 눕고, 마디 A는 VA1 높이에 섭니다
- 왼쪽 앞, 작은 것: V2만 살리고 V1은 단락. 마디 A는 VA2 높이에 섭니다
- 오른쪽, 큰 것: 두 전원을 모두 켠 회로. 마디 A는 정확히 VA1 + VA2 높이에 섭니다
- 시안 액체 선: 전류. 굵고 빠를수록 전류가 크고, 그 가지가 마디로 전류를 밀어 넣을 때는 광택이 반대로 흐릅니다
- 유리관: 저항. R이 클수록 가늘고, 열로 바뀌는 전력이 클수록 안쪽 앰버 필라멘트가 굵어집니다

**마디 A**: VA = (V1 ÷ R1 + V2 ÷ R2) ÷ (1 ÷ R1 + 1 ÷ R2 + 1 ÷ R3)
**중첩**: VA1 + VA2 = VA. 여기서 VA1은 V2 = 0으로, VA2는 V1 = 0으로 푼 값입니다

V1을 올릴 때 **움직이지 않는 것**을 보세요. V2 지형은 전혀 반응하지 않고, VA1과 합계만 함께 올라갑니다. 저항 슬라이더는 어느 회로에나 들어 있으므로 세 지형을 한꺼번에 바꿉니다.

기본값에서는 마디 A가 이미 6 V, 즉 V2와 같은 높이에 있어서 R2에는 전류가 흐르지 않습니다. V2를 어느 쪽으로든 살짝 움직이면 그 가지가 한 방향으로 흐르기 시작합니다.

중첩은 전압과 전류에는 성립하지만 전력에는 성립하지 않습니다. 전력은 응답의 제곱이라서 부분 전력끼리는 더해지지 않습니다.

2D: VA1과 VA2를 막대로 쌓아 올리고, 전체 회로에서 직접 잰 마디 전압과 나란히 놓았습니다.`,
  },
  sliderParams: [
    { key: 'V1', label: { en: 'Source 1 (V1)', ko: '전원 1 (V1)' }, unit: 'V', min: 0, max: V_MAX, step: 0.5, defaultValue: 12 },
    { key: 'V2', label: { en: 'Source 2 (V2)', ko: '전원 2 (V2)' }, unit: 'V', min: 0, max: V_MAX, step: 0.5, defaultValue: 6 },
    { key: 'R1', label: { en: 'Resistance (R1)', ko: '저항 (R1)' }, unit: 'Ω', min: R_MIN, max: R_MAX, step: 0.5, defaultValue: 4 },
    { key: 'R3', label: { en: 'Resistance (R3)', ko: '저항 (R3)' }, unit: 'Ω', min: R_MIN, max: R_MAX, step: 0.5, defaultValue: 4 },
  ],
  Scene: SuperpositionScene,
  Scene2D: SuperpositionBars2DScene,
}

export default concept
