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
const R_MIN = 1 // [Ω]
const P_MAX = (V_MAX * V_MAX) / R_MIN

// Pure physics for a two-mesh ladder: source V, R1 in series to node A, then R2 and R3 from A to ground.
// Node view: one unknown VA, one KCL equation.  Mesh view: two unknowns Im1, Im2, two KVL equations.
// Worked example: V = 12 V, R1 = 2 Ω, R2 = 6 Ω, R3 = 3 Ω → R2 ∥ R3 = 2 Ω, VA = 6 V,
// I1 = 3 A, I2 = 1 A, I3 = 2 A; Im1 = 3 A, Im2 = 2 A, shared branch Im1 − Im2 = 1 A = I2.
// Verified against a 2x2 mesh matrix solve: Im1 = I1, Im2 = I3, Im1 − Im2 = I2 for every R, V.
function solve({ V = 12, R1 = 2, R2 = 6, R3 = 3 }: Record<string, number>) {
  const Rp = (R2 * R3) / (R2 + R3)
  const VA = (V * Rp) / (R1 + Rp) // node equation: (VA − V) ÷ R1 + VA ÷ R2 + VA ÷ R3 = 0
  const I1 = (V - VA) / R1
  const I2 = VA / R2
  const I3 = VA / R3
  // Mesh currents, both taken clockwise: Im1 rides the source loop, Im2 the right loop
  return { V, R1, R2, R3, Rp, VA, I1, I2, I3, Im1: I1, Im2: I3, Ishared: I1 - I3, VR1: V - VA }
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

// ── 3D: one landscape, two bookkeeping layers ──
// Above: the potential landscape with branch currents — the node view.
// On the floor: two circulating loops inside the circuit outline — the mesh view.
const BASE_Y = -0.6
const HEIGHT_PER_VOLT = 1.6 / V_MAX
const RULER_VOLTS = [0, 12, 24]
const Z_BACK = -1.0
const Z_FRONT = 1.0
const SEG_PTS = 16
const INSET = 0.22 // how far the mesh loops sit inside the circuit outline

type XZ = [number, number]
const SRC_X = -2.4
const AX = -0.5 // node A and the shared branch
const BX = 1.3
const SRC_F: XZ = [SRC_X, Z_FRONT]
const SRC_B: XZ = [SRC_X, Z_BACK]
const R1S: XZ = [-2.1, Z_BACK]
const R1E: XZ = [-1.1, Z_BACK]
const NODE: XZ = [AX, Z_BACK]
const R2S: XZ = [AX, -0.35]
const R2E: XZ = [AX, 0.35]
const GND_A: XZ = [AX, Z_FRONT]
const TOP_B: XZ = [BX, Z_BACK]
const R3S: XZ = [BX, -0.35]
const R3E: XZ = [BX, 0.35]
const GND_B: XZ = [BX, Z_FRONT]

const TRUNK_XZ: XZ[] = [SRC_F, SRC_B, R1S, R1E, NODE] // carries I1
const SHARED_XZ: XZ[] = [NODE, R2S, R2E, GND_A] // the shared branch, carries I2
const RIGHT_XZ: XZ[] = [NODE, TOP_B, R3S, R3E, GND_B, GND_A] // carries I3
const RET_XZ: XZ[] = [GND_A, SRC_F] // carries I1 back to the source

const trunkV = new Float32Array(TRUNK_XZ.length)
const sharedV = new Float32Array(SHARED_XZ.length)
const rightV = new Float32Array(RIGHT_XZ.length)
const retV = new Float32Array(RET_XZ.length)
const LINE_XZ = [TRUNK_XZ, SHARED_XZ, RIGHT_XZ, RET_XZ]
const LINE_V = [trunkV, sharedV, rightV, retV]
const lineI = new Float32Array(4)

// The two mesh loops, listed in their clockwise traversal order so the sheen circulates that way.
// They straddle the shared rung at x = AX, running antiparallel there — that is the difference.
const MESH1_XZ: XZ[] = [
  [SRC_X + INSET, Z_FRONT - INSET], [SRC_X + INSET, Z_BACK + INSET],
  [AX - INSET, Z_BACK + INSET], [AX - INSET, Z_FRONT - INSET], [SRC_X + INSET, Z_FRONT - INSET],
]
const MESH2_XZ: XZ[] = [
  [AX + INSET, Z_BACK + INSET], [BX - INSET, Z_BACK + INSET],
  [BX - INSET, Z_FRONT - INSET], [AX + INSET, Z_FRONT - INSET], [AX + INSET, Z_BACK + INSET],
]
const MESH_XZ = [MESH1_XZ, MESH2_XZ]
const meshI = new Float32Array(2)

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

/** Same walk as fillPath but pinned to the floor: the mesh loops are bookkeeping, not potential. */
function fillFlat(buf: Float32Array, xz: XZ[], y: number) {
  let n = 0
  for (let s = 0; s < xz.length - 1; s++) {
    for (let j = 0; j < SEG_PTS + (s === xz.length - 2 ? 1 : 0); j++) {
      const u = j / SEG_PTS
      buf[n++] = xz[s][0] + (xz[s + 1][0] - xz[s][0]) * u
      buf[n++] = y
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
    const r = 0.12 * Math.sqrt(R_MIN / Math.max(R, 0.2)) + 0.03
    g.scale.set(dir.length(), r, r)
  }
  setResistorHeat(g, Math.sqrt(P / P_MAX) * 1.6)
  tag?.position.set((p0.x + p1.x) / 2 + tagDx, Math.max(p0.y, p1.y) + 0.28, (p0.z + p1.z) / 2)
}

const widthFor = (I: number) => Math.min(1.4 + 1.7 * Math.sqrt(Math.abs(I)), 8)
const flowFor = (I: number) => Math.min(0.7 * Math.abs(I), 8)
const MESH_Y = BASE_Y + 0.012 // a hair above the floor outline so the loops read as their own layer

function NodeMeshScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const rise = useRiseIn()
  const bufs = useMemo(() => LINE_XZ.map((xz) => new Float32Array(pointCount(xz) * 3)), [])
  const inits = useMemo(() => LINE_XZ.map(initPoints), [])
  const meshBufs = useMemo(() => MESH_XZ.map((xz) => new Float32Array(pointCount(xz) * 3)), [])
  const meshInits = useMemo(() => MESH_XZ.map(initPoints), [])
  const lineRefs = useRef<any[]>([])
  const meshRefs = useRef<any[]>([])
  const resRefs = useRef<(THREE.Group | null)[]>([]) // R1, R2, R3
  const tagRefs = useRef<(THREE.Group | null)[]>([]) // R1, R2, R3, node A, KCL, Im1, Im2, shared
  const textRefs = useRef<(HTMLSpanElement | null)[]>([]) // the 8 tags above, then the floor caption
  const nodeRef = useRef<THREE.Group>(null)
  const sharedRef = useRef<THREE.Group>(null)
  const flows = useRef([0, 0, 0, 0])
  const meshFlows = useRef([0, 0])
  const labelledFor = useRef<unknown>(null)

  useFrame(() => {
    const s = solve(valuesRef.current)
    const { V, VA } = s
    const k = HEIGHT_PER_VOLT * rise.current

    trunkV[1] = trunkV[2] = V
    trunkV[3] = trunkV[4] = VA
    sharedV[0] = sharedV[1] = VA
    rightV[0] = rightV[1] = rightV[2] = VA

    lineI[0] = s.I1; lineI[1] = s.I2; lineI[2] = s.I3; lineI[3] = s.I1
    for (let l = 0; l < 4; l++) {
      const line = lineRefs.current[l]
      if (!line) continue
      fillPath(bufs[l], LINE_XZ[l], LINE_V[l], k)
      line.geometry.setPositions(bufs[l])
      ;(line.material as LineMaterial).linewidth = widthFor(lineI[l])
      flows.current[l] = flowFor(lineI[l])
    }

    // The mesh layer: two closed loops, each carrying its own loop current
    meshI[0] = s.Im1; meshI[1] = s.Im2
    for (let m = 0; m < 2; m++) {
      const loop = meshRefs.current[m]
      if (!loop) continue
      fillFlat(meshBufs[m], MESH_XZ[m], MESH_Y)
      loop.geometry.setPositions(meshBufs[m])
      ;(loop.material as LineMaterial).linewidth = widthFor(meshI[m])
      meshFlows.current[m] = flowFor(meshI[m])
    }

    placeResistor(resRefs.current[0], tagRefs.current[0], R1S, V, R1E, VA, k, s.R1, s.I1 * s.VR1, 0)
    placeResistor(resRefs.current[1], tagRefs.current[1], R2S, VA, R2E, 0, k, s.R2, s.I2 * VA, -0.75)
    placeResistor(resRefs.current[2], tagRefs.current[2], R3S, VA, R3E, 0, k, s.R3, s.I3 * VA, 0.55)

    // Node A: the single unknown of the node view, read straight off the ruler as a height
    const yA = BASE_Y + k * VA
    nodeRef.current?.position.set(AX, yA, Z_BACK)
    tagRefs.current[3]?.position.set(AX - 0.55, yA + 0.2, Z_BACK)
    tagRefs.current[4]?.position.set(AX + 1.0, yA + 0.62, Z_BACK - 0.1)

    // The shared rung, where the mesh view needs the difference of the two loop currents
    sharedRef.current?.position.set(AX, MESH_Y, 0.6)
    tagRefs.current[5]?.position.set((SRC_X + AX) / 2, MESH_Y + 0.06, 0.1)
    tagRefs.current[6]?.position.set((AX + BX) / 2 + 0.1, MESH_Y + 0.06, 0.1)
    tagRefs.current[7]?.position.set(AX, MESH_Y + 0.06, 0.95)

    if (labelledFor.current !== valuesRef.current && textRefs.current[8]) {
      labelledFor.current = valuesRef.current
      const ko = locale === 'ko'
      const texts = [
        `R1 ${fmt(s.R1, 'Ω')}`,
        `R2 ${fmt(s.R2, 'Ω')}`,
        `R3 ${fmt(s.R3, 'Ω')}`,
        `A ${fmt(VA, 'V')}`,
        `KCL · I1 ${fmt(s.I1, 'A')} = I2 ${fmt(s.I2, 'A')} + I3 ${fmt(s.I3, 'A')}`,
        `Im1 ${fmt(s.Im1, 'A')}`,
        `Im2 ${fmt(s.Im2, 'A')}`,
        `Im1 − Im2 = ${fmt(s.Ishared, 'A')} = I2`,
        ko
          ? `마디 해석 · 미지수 1개 VA   ·   망 해석 · 미지수 2개 Im1, Im2`
          : `Node view · 1 unknown VA   ·   Mesh view · 2 unknowns Im1, Im2`,
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  const footprint = useMemo<[number, number, number][]>(() => [
    [SRC_X, BASE_Y, Z_FRONT], [SRC_X, BASE_Y, Z_BACK], [BX, BASE_Y, Z_BACK],
    [BX, BASE_Y, Z_FRONT], [SRC_X, BASE_Y, Z_FRONT],
  ], [])
  const rung = useMemo<[number, number, number][]>(() => [
    [AX, BASE_Y, Z_BACK], [AX, BASE_Y, Z_FRONT],
  ], [])

  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />

      <Line points={footprint} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      <Line points={rung} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      <FloorCaption position={[AX, BASE_Y, Z_FRONT + 1.3]} />
      <PotentialRuler x={SRC_X - 0.4} z={Z_BACK} baseY={BASE_Y} volts={RULER_VOLTS} unitsPerVolt={() => HEIGHT_PER_VOLT * rise.current} />

      {/* Node view: branch currents climbing the potential landscape */}
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

      {/* Mesh view: the same circuit as two circulating loops, flat on the floor */}
      {meshInits.map((pts, m) => (
        <TrailLine
          key={`m${m}`}
          ref={(loop) => { meshRefs.current[m] = loop }}
          points={pts}
          color={c.current}
          fade={false}
          flowSpeed={() => meshFlows.current[m]}
        />
      ))}

      {[0, 1, 2].map((r) => (
        <Resistor key={r} ref={(g) => { resRefs.current[r] = g }} />
      ))}

      <Marker ref={nodeRef} color={c.marker} size={0.06} />
      <Marker ref={sharedRef} color={c.marker} size={0.05} />

      {[0, 1, 2, 3, 4, 5, 6, 7].map((t) => (
        <group key={t} ref={(g) => { tagRefs.current[t] = g }}>
          <Html center>
            <span
              ref={(el) => { textRefs.current[t] = el }}
              className={labelClass}
              style={t === 7 ? { color: c.current } : undefined}
            />
          </Html>
        </group>
      ))}
      <Html position={[AX, BASE_Y - 0.1, Z_FRONT + 0.75]} center>
        <span ref={(el) => { textRefs.current[8] = el }} className={labelClass} />
      </Html>
    </>
  )
}

// ── 2D: the two sets of books side by side, meeting on the shared branch ──
const Y0 = -0.9 // 0 A baseline
const FULL_H = 1.5 // I1 always fills this, so both panels share one honest scale
const BAR_W = 0.6
const NODE_X1 = -3.0 // I1 arriving at A
const NODE_X2 = -1.6 // I2 stacked on I3, leaving A
const MESH_X1 = 0.7
const MESH_X2 = 1.85
const MESH_X3 = 3.0
const DIVIDER_X = -0.2
const AXIS_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]

function NodeMeshBars2DScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const barRefs = useRef<(THREE.Mesh | null)[]>([]) // I1, I3, I2, Im1, Im2, Im1 − Im2
  const tagRefs = useRef<(THREE.Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const baseRef = useRef<any>(null)
  const divRef = useRef<any>(null)
  const baseBuf = useMemo(() => new Float32Array(6), [])
  const divBuf = useMemo(() => new Float32Array(6), [])
  const labelledFor = useRef<unknown>(null)

  useFrame(() => {
    const s = solve(valuesRef.current)
    const scale = FULL_H / Math.max(s.I1, 1e-6) // ratios stay true; I1 is always the tallest bar
    const h1 = Math.max(s.I1 * scale, 1e-3)
    const h2 = Math.max(s.I2 * scale, 1e-3)
    const h3 = Math.max(s.I3 * scale, 1e-3)
    const hm1 = Math.max(s.Im1 * scale, 1e-3)
    const hm2 = Math.max(s.Im2 * scale, 1e-3)
    const hd = Math.max(s.Ishared * scale, 1e-3)

    // Node view: one bar in, two stacked out — the KCL sum at A
    barRefs.current[0]?.position.set(NODE_X1, Y0 + h1 / 2, 0)
    barRefs.current[0]?.scale.set(BAR_W, h1, 1)
    barRefs.current[1]?.position.set(NODE_X2, Y0 + h3 / 2, 0)
    barRefs.current[1]?.scale.set(BAR_W, h3, 1)
    barRefs.current[2]?.position.set(NODE_X2, Y0 + h3 + h2 / 2, 0)
    barRefs.current[2]?.scale.set(BAR_W, h2, 1)
    tagRefs.current[0]?.position.set(NODE_X1, Y0 + h1 + 0.24, 0.01)
    tagRefs.current[1]?.position.set(NODE_X2, Y0 + h3 + h2 + 0.24, 0.01)
    tagRefs.current[2]?.position.set(NODE_X2 + 0.85, Y0 + h3 + h2 / 2, 0.01)

    // Mesh view: two loop currents, and the shared branch as their difference
    barRefs.current[3]?.position.set(MESH_X1, Y0 + hm1 / 2, 0)
    barRefs.current[3]?.scale.set(BAR_W, hm1, 1)
    barRefs.current[4]?.position.set(MESH_X2, Y0 + hm2 / 2, 0)
    barRefs.current[4]?.scale.set(BAR_W, hm2, 1)
    barRefs.current[5]?.position.set(MESH_X3, Y0 + hd / 2, 0)
    barRefs.current[5]?.scale.set(BAR_W, hd, 1)
    tagRefs.current[3]?.position.set(MESH_X1, Y0 + hm1 + 0.24, 0.01)
    tagRefs.current[4]?.position.set(MESH_X2, Y0 + hm2 + 0.24, 0.01)
    tagRefs.current[5]?.position.set(MESH_X3, Y0 + hd + 0.24, 0.01)

    baseBuf[0] = -3.6; baseBuf[1] = Y0; baseBuf[2] = 0
    baseBuf[3] = 3.7; baseBuf[4] = Y0; baseBuf[5] = 0
    baseRef.current?.geometry.setPositions(baseBuf)
    baseRef.current?.computeLineDistances()
    divBuf[0] = DIVIDER_X; divBuf[1] = Y0 - 0.35; divBuf[2] = 0
    divBuf[3] = DIVIDER_X; divBuf[4] = 1.2; divBuf[5] = 0
    divRef.current?.geometry.setPositions(divBuf)
    divRef.current?.computeLineDistances()

    if (labelledFor.current !== valuesRef.current && textRefs.current[8]) {
      labelledFor.current = valuesRef.current
      const ko = locale === 'ko'
      const texts = [
        `I1 ${fmt(s.I1, 'A')}`,
        `I2 + I3 ${fmt(s.I2 + s.I3, 'A')}`,
        `I2 ${fmt(s.I2, 'A')}`,
        `Im1 ${fmt(s.Im1, 'A')}`,
        `Im2 ${fmt(s.Im2, 'A')}`,
        `Im1 − Im2 ${fmt(s.Ishared, 'A')}`,
        ko ? `마디 해석 · A에서 KCL` : `Node view · KCL at A`,
        ko ? `망 해석 · 망전류 2개` : `Mesh view · two loop currents`,
        ko
          ? `공유 가지 · I2 ${fmt(s.I2, 'A')} = Im1 − Im2 ${fmt(s.Ishared, 'A')}`
          : `shared branch · I2 ${fmt(s.I2, 'A')} = Im1 − Im2 ${fmt(s.Ishared, 'A')}`,
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  // I2 and the mesh difference are the same physical current, so they carry the same weight
  const barOpacity = [0.85, 0.4, 0.85, 0.6, 0.6, 0.85]

  return (
    <>
      <Line ref={baseRef} points={AXIS_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      <Line ref={divRef} points={AXIS_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.06} gapSize={0.06} />
      {barOpacity.map((opacity, i) => (
        <mesh key={i} ref={(m) => { barRefs.current[i] = m }}>
          <planeGeometry args={[1, 1]} />
          <meshBasicMaterial color={c.current} transparent opacity={opacity} depthWrite={false} />
        </mesh>
      ))}
      {[0, 1, 2, 3, 4, 5].map((t) => (
        <group key={t} ref={(g) => { tagRefs.current[t] = g }}>
          <Html center>
            <span
              ref={(el) => { textRefs.current[t] = el }}
              className={labelClass}
              style={t === 2 || t === 5 ? { color: c.current } : undefined}
            />
          </Html>
        </group>
      ))}
      <Html position={[-2.3, 1.45, 0]} center>
        <span ref={(el) => { textRefs.current[6] = el }} className={labelClass} />
      </Html>
      <Html position={[1.85, 1.45, 0]} center>
        <span ref={(el) => { textRefs.current[7] = el }} className={labelClass} />
      </Html>
      <Html position={[0, -1.6, 0]} center>
        <span
          ref={(el) => { textRefs.current[8] = el }}
          className={labelClass}
          style={{ color: c.current }}
        />
      </Html>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'node-mesh-analysis',
  title: { en: 'Node & Mesh Analysis', ko: '마디 해석법·망 해석법' },
  subject: 'circuits',
  order: 11,
  description: {
    en: `## Node and Mesh Analysis

**One circuit, two sets of books: name the node heights and write KCL, or name the loop currents and write KVL. The answer is the same.**

The circuit is V with R1 in series to node A, then R2 and R3 down to ground — two meshes, one unknown node.
- The landscape above the floor is the **node view**: node A is the only unknown, and you read it off the ruler as a height. The current lines meeting at A are its KCL sum
- The two loops on the floor are the **mesh view**: Im1 rides the source loop, Im2 the right loop. They run side by side along the shared rung, in opposite directions
- Bright dot on the rung: the shared branch. It carries Im1 − Im2, not Im1 — that is the step learners drop

**Node view**: (VA − V) ÷ R1 + VA ÷ R2 + VA ÷ R3 = 0, so VA = V × (R2 ∥ R3) ÷ (R1 + R2 ∥ R3)
**Mesh view**: R1 × Im1 + R2 × (Im1 − Im2) = V and R2 × (Im2 − Im1) + R3 × Im2 = 0
**They agree**: Im1 = I1, Im2 = I3, and Im1 − Im2 = I2 = VA ÷ R2

At V = 12 V, R1 = 2 Ω, R2 = 6 Ω, R3 = 3 Ω: R2 ∥ R3 = 2 Ω, VA = 6 V, I1 = 3 A, I2 = 1 A, I3 = 2 A, and the mesh books give Im1 = 3 A, Im2 = 2 A, shared branch 3 − 2 = 1 A. Take R3 up to 20 Ω and Im2 collapses to 0.42 A while Im1 falls only to 1.81 A, so the gap between the two loops widens to 1.40 A — and the node view reads exactly that as I2.

In 2D: the node books on the left, the mesh books on the right, with the shared branch reading the same on both sides.`,
    ko: `## 마디 해석법과 망 해석법

**회로는 하나인데 장부는 둘입니다. 마디 전위를 미지수로 놓고 KCL을 쓰거나, 망전류를 미지수로 놓고 KVL을 씁니다. 답은 같습니다.**

전원 V에 R1이 직렬로 마디 A까지 이어지고, 거기서 R2와 R3가 접지로 내려갑니다. 망은 둘, 미지 마디는 하나입니다.
- 바닥 위로 솟은 지형이 **마디 해석**입니다. 미지수는 마디 A 하나뿐이고, 눈금자에서 높이로 바로 읽힙니다. A에서 만나는 전류 선들이 KCL 합입니다
- 바닥에 놓인 두 고리가 **망 해석**입니다. Im1은 전원 쪽 망, Im2는 오른쪽 망을 돕니다. 공유 가지에서는 둘이 나란히, 서로 반대 방향으로 흐릅니다
- 공유 가지 위의 밝은 점: 이 가지가 나르는 전류는 Im1이 아니라 Im1 − Im2입니다. 학습자가 가장 자주 빠뜨리는 대목입니다

**마디 해석**: (VA − V) ÷ R1 + VA ÷ R2 + VA ÷ R3 = 0, 따라서 VA = V × (R2 ∥ R3) ÷ (R1 + R2 ∥ R3)
**망 해석**: R1 × Im1 + R2 × (Im1 − Im2) = V, R2 × (Im2 − Im1) + R3 × Im2 = 0
**두 답은 일치합니다**: Im1 = I1, Im2 = I3, 그리고 Im1 − Im2 = I2 = VA ÷ R2

V = 12 V, R1 = 2 Ω, R2 = 6 Ω, R3 = 3 Ω일 때 R2 ∥ R3 = 2 Ω, VA = 6 V, I1 = 3 A, I2 = 1 A, I3 = 2 A입니다. 망 장부로는 Im1 = 3 A, Im2 = 2 A이고 공유 가지는 3 − 2 = 1 A입니다. R3를 20 Ω까지 올리면 Im2는 0.42 A로 주저앉는 반면 Im1은 1.81 A까지만 내려가서, 두 망전류의 차는 1.40 A로 벌어집니다. 마디 장부가 I2로 읽는 값이 바로 그것입니다.

2D: 왼쪽은 마디 장부, 오른쪽은 망 장부입니다. 공유 가지 값은 양쪽에서 똑같이 나옵니다.`,
  },
  sliderParams: [
    { key: 'V',  label: { en: 'Source (V)',            ko: '전원 전압 (V)' },      unit: 'V', min: 1,     max: V_MAX, step: 0.5, defaultValue: 12 },
    { key: 'R1', label: { en: 'Series R1',             ko: '직렬 저항 R1' },       unit: 'Ω', min: R_MIN, max: 20,    step: 0.5, defaultValue: 2 },
    { key: 'R2', label: { en: 'Shared branch R2',      ko: '공유 가지 저항 R2' },  unit: 'Ω', min: R_MIN, max: 20,    step: 0.5, defaultValue: 6 },
    { key: 'R3', label: { en: 'Right mesh R3',         ko: '오른쪽 망 저항 R3' },  unit: 'Ω', min: R_MIN, max: 20,    step: 0.5, defaultValue: 3 },
  ],
  Scene: NodeMeshScene,
  Scene2D: NodeMeshBars2DScene,
}

export default concept
