import { useRef, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import type { LineMaterial } from 'three-stdlib'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { useLocaleStore } from '../../lib/i18n'
import { TrailLine } from '../../components/scene/TrailLine'
import { setLineOpacity, setOpacity, useSettledLabels, useStoryReveal } from '../../components/scene/storyReveal'
import { Marker } from '../../components/scene/Marker'
import { FloorCaption, PotentialRuler, useRiseIn } from '../../components/scene/PotentialLandscape'
import { Resistor, setResistorHeat } from '../../components/scene/Resistor'
import type { ConceptMeta } from '../../types/concept'

const V_MAX = 24 // [V]
const R_MIN = 1 // [Ω]
const P_MAX = (V_MAX * V_MAX) / R_MIN

// Pure physics for one source, R1 in series, then R2 and R3 in parallel to ground.
// Worked example: V = 12 V, R1 = 2 Ω, R2 = 6 Ω, R3 = 3 Ω →
// I1 = 3 A, node = 6 V, I2 = 1 A, I3 = 2 A; KCL: 3 = 1 + 2, KVL: 12 − 6 − 6 = 0.
//
// `w3` scales R3's conductance, so a narration step before the road splits solves the circuit that
// is actually on screen: w3 = 0 opens the branch, leaving V across R1 and R2 alone (1.5 A, node 9 V).
function solve({ V = 12, R1 = 2, R2 = 6, R3 = 3 }: Record<string, number>, w3 = 1) {
  const Rp = 1 / (1 / R2 + w3 / R3)
  const I1 = V / (R1 + Rp)
  const Vn = I1 * Rp // node potential
  // I3 carries the same weight, so KCL still reads I1 = I2 + I3 while the branch fades in
  return { V, R1, R2, R3, I1, I2: Vn / R2, I3: (Vn / R3) * w3, Vn, VR1: I1 * R1 }
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
const REV_HEIGHT = 0 // the battery's lift: the landscape's height and the voltage ruler
const REV_FLOW = 1 // the sheen travelling the wires, i.e. the current
const REV_B3 = 2 // the second branch — and with it R3's conductance, so the solve changes too
const REV_KCL = 3 // the node dot and its current readout
const REV_WALK = 4 // the walker, its drop line, and the KVL readouts
const REV_COUNT = 5

/** Which reveals the sentence at `step` asks for. Step 0 is the flat loop, so nothing yet. */
function revealTargets(step: number, out: Float32Array) {
  out[REV_HEIGHT] = step >= 1 ? 1 : 0
  out[REV_FLOW] = step >= 2 ? 1 : 0
  out[REV_B3] = step >= 3 ? 1 : 0
  out[REV_KCL] = step >= 4 ? 1 : 0
  out[REV_WALK] = step >= 6 ? 1 : 0
}

const fmt = (x: number, unit: string) => `${x.toFixed(x < 10 ? 2 : 1)} ${unit}`
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'

// ── 3D: the network as a potential landscape, with a walker that adds up one loop ──
const BASE_Y = -0.6
const HEIGHT_PER_VOLT = 1.6 / V_MAX
const RULER_VOLTS = [0, 12, 24]
const Z_BACK = -1.0
const Z_FRONT = 1.0
const R_Z: [number, number] = [-0.6, 0.6]
const SEG_PTS = 20
const WALK_SPEED = 1.3 // scene units per second

type XZ = [number, number]
const A_F: XZ = [-2.2, Z_FRONT]
const A_B: XZ = [-2.2, Z_BACK]
const R1S: XZ = [-1.9, Z_BACK]
const R1E: XZ = [-1.0, Z_BACK]
const NODE: XZ = [-0.5, Z_BACK]
const R2S: XZ = [-0.5, R_Z[0]]
const R2E: XZ = [-0.5, R_Z[1]]
const J2: XZ = [-0.5, Z_FRONT]
const B3_TOP: XZ = [1.2, Z_BACK]
const R3S: XZ = [1.2, R_Z[0]]
const R3E: XZ = [1.2, R_Z[1]]
const B3_BOT: XZ = [1.2, Z_FRONT]

const TRUNK_XZ: XZ[] = [A_F, A_B, R1S, R1E, NODE] // carries I1
const B2_XZ: XZ[] = [NODE, R2S, R2E, J2] // carries I2
const B3_XZ: XZ[] = [NODE, B3_TOP, R3S, R3E, B3_BOT, J2] // carries I3
const RET_XZ: XZ[] = [J2, A_F] // carries I1 back to the source

const trunkV = new Float32Array(TRUNK_XZ.length)
const b2V = new Float32Array(B2_XZ.length)
const b3V = new Float32Array(B3_XZ.length)
const retV = new Float32Array(RET_XZ.length)
const LINE_XZ = [TRUNK_XZ, B2_XZ, B3_XZ, RET_XZ]
const LINE_V = [trunkV, b2V, b3V, retV]
const lineI = new Float32Array(4)

// Loop 1 for the KVL walk: source → R1 → node → R2 → back to the source's − terminal
const WALK_XZ: XZ[] = [A_F, A_B, R1S, R1E, NODE, R2S, R2E, J2, A_F]
const walkV = new Float32Array(WALK_XZ.length)
const WALK_SEG_LEN = WALK_XZ.slice(0, -1).map((p, i) =>
  Math.hypot(WALK_XZ[i + 1][0] - p[0], WALK_XZ[i + 1][1] - p[1])
)
const WALK_LEN = WALK_SEG_LEN.reduce((a, b) => a + b, 0)

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
const walkPos = new THREE.Vector3()

/** Position (and potential) at distance d along the KVL loop; out.y holds the running sum in volts. */
function walkAt(d: number, out: THREE.Vector3) {
  let rest = ((d % WALK_LEN) + WALK_LEN) % WALK_LEN
  for (let s = 0; s < WALK_SEG_LEN.length; s++) {
    if (rest > WALK_SEG_LEN[s] && s < WALK_SEG_LEN.length - 1) {
      rest -= WALK_SEG_LEN[s]
      continue
    }
    const u = WALK_SEG_LEN[s] > 0 ? Math.min(rest / WALK_SEG_LEN[s], 1) : 0
    out.x = WALK_XZ[s][0] + (WALK_XZ[s + 1][0] - WALK_XZ[s][0]) * u
    out.z = WALK_XZ[s][1] + (WALK_XZ[s + 1][1] - WALK_XZ[s][1]) * u
    out.y = walkV[s] + (walkV[s + 1] - walkV[s]) * u
    return out
  }
  return out
}

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
    const r = 0.12 * Math.sqrt(R_MIN / Math.max(R, 0.2)) + 0.03
    g.scale.set(dir.length(), r, r)
  }
  setResistorHeat(g, Math.sqrt(P / P_MAX) * 1.6)
  tag?.position.set((p0.x + p1.x) / 2, Math.max(p0.y, p1.y) + 0.3, (p0.z + p1.z) / 2)
}

const widthFor = (I: number) => Math.min(1.4 + 1.7 * Math.sqrt(I), 8)
const flowFor = (I: number) => Math.min(0.7 * I, 8)
const DROP_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]

function KirchhoffScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const rise = useRiseIn()
  const labels = useSettledLabels(revealTargets, REV_COUNT, [locale])
  const reveal = useStoryReveal(revealTargets, REV_COUNT)
  const bufs = useMemo(() => LINE_XZ.map((xz) => new Float32Array(pointCount(xz) * 3)), [])
  const inits = useMemo(() => LINE_XZ.map(initPoints), [])
  const dropBuf = useMemo(() => new Float32Array(6), [])
  const lineRefs = useRef<any[]>([])
  const dropRef = useRef<any>(null)
  const resRefs = useRef<(THREE.Group | null)[]>([]) // R1, R2, R3
  const tagRefs = useRef<(THREE.Group | null)[]>([]) // R1, R2, R3, node, walker
  const textRefs = useRef<(HTMLSpanElement | null)[]>([]) // 3 resistors, node KCL, walker Σ, KVL summary
  const nodeRef = useRef<THREE.Group>(null)
  const walkerRef = useRef<THREE.Group>(null)
  const flows = useRef([0, 0, 0, 0])
  const sumText = useRef('')

  useFrame(({ clock }) => {
    const w3 = reveal[REV_B3]
    const s = solve(valuesRef.current, w3)
    const { V, Vn } = s
    const k = HEIGHT_PER_VOLT * rise.current * reveal[REV_HEIGHT]

    trunkV[1] = trunkV[2] = V
    trunkV[3] = trunkV[4] = Vn
    b2V[0] = b2V[1] = Vn
    b3V[0] = b3V[1] = b3V[2] = Vn
    walkV[1] = walkV[2] = V
    walkV[3] = walkV[4] = walkV[5] = Vn

    lineI[0] = s.I1; lineI[1] = s.I2; lineI[2] = s.I3; lineI[3] = s.I1
    for (let l = 0; l < 4; l++) {
      const line = lineRefs.current[l]
      if (!line) continue
      fillPath(bufs[l], LINE_XZ[l], LINE_V[l], k)
      line.geometry.setPositions(bufs[l])
      ;(line.material as LineMaterial).linewidth = widthFor(lineI[l])
      if (l === 2) setLineOpacity(line, w3)
      flows.current[l] = flowFor(lineI[l]) * reveal[REV_FLOW]
    }

    const heat = reveal[REV_FLOW] // the tubes stay cold until the sentence that names the current
    placeResistor(resRefs.current[0], tagRefs.current[0], R1S, V, R1E, Vn, k, s.R1, s.I1 * s.VR1 * heat)
    placeResistor(resRefs.current[1], tagRefs.current[1], R2S, Vn, R2E, 0, k, s.R2, s.I2 * Vn * heat)
    // R2 runs right under the node, so nudge its label toward the viewer
    if (tagRefs.current[1]) tagRefs.current[1].position.z += 0.45
    placeResistor(resRefs.current[2], tagRefs.current[2], R3S, Vn, R3E, 0, k, s.R3, s.I3 * Vn * heat)
    resRefs.current[2]?.scale.multiplyScalar(w3) // grows out of nothing when its sentence opens the branch

    // The node where the current splits: KCL lives here
    nodeRef.current?.position.set(NODE[0], BASE_Y + k * Vn, NODE[1])
    nodeRef.current?.scale.setScalar(reveal[REV_KCL])
    tagRefs.current[3]?.position.set(NODE[0] + 0.9, BASE_Y + k * Vn + 0.62, NODE[1] - 0.1)

    // The walker: its height is the running sum of rises and drops, so it is back at 0 V after a lap
    walkAt(clock.getElapsedTime() * WALK_SPEED, walkPos)
    const running = walkPos.y
    const wy = BASE_Y + k * running
    walkerRef.current?.position.set(walkPos.x, wy, walkPos.z)
    walkerRef.current?.scale.setScalar(reveal[REV_WALK])
    dropBuf[0] = walkPos.x; dropBuf[1] = BASE_Y; dropBuf[2] = walkPos.z
    dropBuf[3] = walkPos.x; dropBuf[4] = wy; dropBuf[5] = walkPos.z
    dropRef.current?.geometry.setPositions(dropBuf)
    dropRef.current?.computeLineDistances()
    setLineOpacity(dropRef.current, reveal[REV_WALK])
    tagRefs.current[4]?.position.set(walkPos.x - 0.35, wy + 0.22, walkPos.z + 0.25)

    // Every readout fades with the thing it labels
    setOpacity(textRefs.current[0], reveal[REV_HEIGHT])
    setOpacity(textRefs.current[1], reveal[REV_HEIGHT])
    setOpacity(textRefs.current[2], Math.min(reveal[REV_HEIGHT], w3))
    setOpacity(textRefs.current[3], reveal[REV_KCL])
    setOpacity(textRefs.current[4], reveal[REV_WALK])
    setOpacity(textRefs.current[5], reveal[REV_WALK])

    const next = `Σ ${fmt(running, 'V')}`
    if (sumText.current !== next && textRefs.current[4]) {
      sumText.current = next
      textRefs.current[4]!.textContent = next
    }

    const w = textRefs.current[5] ? labels.settled(valuesRef.current) : null
    if (w) {
      const q = solve(valuesRef.current, w[REV_B3])
      const texts = [
        `R1 ${fmt(q.R1, 'Ω')} · ${fmt(q.VR1, 'V')}`,
        `R2 ${fmt(q.R2, 'Ω')} · ${fmt(q.I2, 'A')}`,
        `R3 ${fmt(q.R3, 'Ω')} · ${fmt(q.I3, 'A')}`,
        `KCL · I1 ${fmt(q.I1, 'A')} = I2 ${fmt(q.I2, 'A')} + I3 ${fmt(q.I3, 'A')}`,
        sumText.current,
        `KVL · +${fmt(q.V, 'V')} − ${fmt(q.VR1, 'V')} − ${fmt(q.Vn, 'V')} = ${fmt(0, 'V')}`,
      ]
      texts.forEach((text, t) => { if (t !== 4 && textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  const footprint = useMemo<[number, number, number][]>(() => [
    [A_F[0], BASE_Y, A_F[1]], [A_B[0], BASE_Y, A_B[1]], [B3_TOP[0], BASE_Y, B3_TOP[1]],
    [B3_BOT[0], BASE_Y, B3_BOT[1]], [A_F[0], BASE_Y, A_F[1]],
  ], [])
  const rung = useMemo<[number, number, number][]>(() => [
    [NODE[0], BASE_Y, NODE[1]], [J2[0], BASE_Y, J2[1]],
  ], [])

  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />

      <Line points={footprint} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      <Line points={rung} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      <FloorCaption position={[-0.5, BASE_Y, Z_FRONT + 0.75]} />
      <PotentialRuler x={A_F[0] - 0.4} z={Z_BACK} baseY={BASE_Y} volts={RULER_VOLTS} unitsPerVolt={() => HEIGHT_PER_VOLT * rise.current * reveal[REV_HEIGHT]} />

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

      {[0, 1, 2].map((r) => (
        <Resistor key={r} ref={(g) => { resRefs.current[r] = g }} />
      ))}

      <Marker ref={nodeRef} color={c.marker} size={0.06} />
      <Line ref={dropRef} points={DROP_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.06} gapSize={0.05} />
      <Marker ref={walkerRef} color={c.power} size={0.08} />

      {[0, 1, 2, 3, 4].map((t) => (
        <group key={t} ref={(g) => { tagRefs.current[t] = g }}>
          <Html center>
            <span
              ref={(el) => { textRefs.current[t] = el }}
              className={labelClass}
              style={t === 4 ? { color: c.power } : undefined}
            />
          </Html>
        </group>
      ))}
      <Html position={[-0.5, BASE_Y - 0.1, Z_FRONT + 1.3]} center>
        <span ref={(el) => { textRefs.current[5] = el }} className={labelClass} />
      </Html>
    </>
  )
}

// ── 2D: KVL as a waterfall that lands on zero, and KCL as one bar that equals two ──
const WF_X0 = -3.3
const WF_W = 0.9
const WF_GAP = 0.35
const WF_Y0 = -0.3 // 0 V baseline for the waterfall
const V_SCALE = 2.2 / V_MAX
const KCL_X = 1.6
const KCL_W = 0.55
const I_SCALE = 1.6 / 6 // 6 A fills the bar
const steps = new Float32Array(4) // running sum after each KVL step
const barOpacity = [0.85, 0.5, 0.5, 0.85, 0.85, 0.45] // 3 waterfall bars, then I1, I2, I3

function KirchhoffBars2DScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const barRefs = useRef<(THREE.Mesh | null)[]>([]) // 3 waterfall bars, I1, I2, I3
  const tagRefs = useRef<(THREE.Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const baseRef = useRef<any>(null)
  const baseBuf = useMemo(() => new Float32Array(6), [])
  const labels = useSettledLabels(revealTargets, REV_COUNT, [locale])
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(() => {
    const w3 = reveal[REV_B3]
    const s = solve(valuesRef.current, w3)
    // Running sum: 0 → +V → −VR1 → −Vn (back to 0)
    steps[0] = 0
    steps[1] = s.V
    steps[2] = s.V - s.VR1
    steps[3] = s.V - s.VR1 - s.Vn

    for (let i = 0; i < 3; i++) {
      const from = steps[i] * V_SCALE
      const to = steps[i + 1] * V_SCALE
      const x = WF_X0 + i * (WF_W + WF_GAP) + WF_W / 2
      const h = Math.max(Math.abs(to - from), 1e-3)
      barRefs.current[i]?.position.set(x, WF_Y0 + (from + to) / 2, 0)
      barRefs.current[i]?.scale.set(WF_W, h, 1)
      tagRefs.current[i]?.position.set(x, WF_Y0 + Math.max(from, to) + 0.25, 0.01)
    }
    baseBuf[0] = WF_X0 - 0.3; baseBuf[1] = WF_Y0; baseBuf[2] = 0
    baseBuf[3] = WF_X0 + 3 * (WF_W + WF_GAP) + 0.2; baseBuf[4] = WF_Y0; baseBuf[5] = 0
    baseRef.current?.geometry.setPositions(baseBuf)
    baseRef.current?.computeLineDistances()
    setLineOpacity(baseRef.current, reveal[REV_HEIGHT])

    // KCL: one bar of I1 beside I2 stacked on I3
    const h1 = Math.max(s.I1 * I_SCALE, 1e-3)
    const h2 = Math.max(s.I2 * I_SCALE, 1e-3)
    const h3 = Math.max(s.I3 * I_SCALE, 1e-3)
    barRefs.current[3]?.position.set(KCL_X, WF_Y0 + h1 / 2, 0)
    barRefs.current[3]?.scale.set(KCL_W, h1, 1)
    barRefs.current[4]?.position.set(KCL_X + 0.9, WF_Y0 + h3 + h2 / 2, 0)
    barRefs.current[4]?.scale.set(KCL_W, h2, 1)
    barRefs.current[5]?.position.set(KCL_X + 0.9, WF_Y0 + h3 / 2, 0)
    barRefs.current[5]?.scale.set(KCL_W, h3, 1)
    tagRefs.current[3]?.position.set(KCL_X, WF_Y0 + h1 + 0.25, 0.01)
    tagRefs.current[4]?.position.set(KCL_X + 0.9, WF_Y0 + h3 + h2 + 0.25, 0.01)
    tagRefs.current[5]?.position.set(KCL_X + 1.75, WF_Y0 + h3 / 2, 0.01)

    // The waterfall arrives with the battery's lift, the drops once current flows, and the two
    // branch bars once the road splits; each caption waits for the law it states.
    const fades = [
      reveal[REV_HEIGHT], reveal[REV_FLOW], reveal[REV_FLOW],
      reveal[REV_FLOW], w3, w3,
    ]
    fades.forEach((v, i) => {
      const m = barRefs.current[i]?.material as THREE.Material | undefined
      if (m) m.opacity = barOpacity[i] * v
      setOpacity(textRefs.current[i], v)
    })
    setOpacity(textRefs.current[6], reveal[REV_WALK])
    setOpacity(textRefs.current[7], reveal[REV_KCL])

    const w = textRefs.current[6] ? labels.settled(valuesRef.current) : null
    if (w) {
      const q = solve(valuesRef.current, w[REV_B3])
      const ko = locale === 'ko'
      const texts = [
        `+V ${fmt(q.V, 'V')}`,
        `− R1 ${fmt(q.VR1, 'V')}`,
        `− R2 ${fmt(q.Vn, 'V')}`,
        `I1 ${fmt(q.I1, 'A')}`,
        `I2 + I3 ${fmt(q.I2 + q.I3, 'A')}`,
        `I3 ${fmt(q.I3, 'A')} · I2 ${fmt(q.I2, 'A')}`,
        `${ko ? 'KVL · 한 바퀴 합' : 'KVL · once around'} = ${fmt(0, 'V')}`,
        `${ko ? 'KCL · 들어온 만큼 나간다' : 'KCL · in equals out'}`,
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  const barColors = [c.voltage, c.resistance, c.resistance, c.current, c.current, c.current]

  return (
    <>
      <Line ref={baseRef} points={DROP_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      {barColors.map((color, i) => (
        <mesh key={i} ref={(m) => { barRefs.current[i] = m }}>
          <planeGeometry args={[1, 1]} />
          <meshBasicMaterial color={color} transparent opacity={barOpacity[i]} depthWrite={false} />
        </mesh>
      ))}
      {[0, 1, 2, 3, 4, 5].map((t) => (
        <group key={t} ref={(g) => { tagRefs.current[t] = g }}>
          <Html center>
            <span ref={(el) => { textRefs.current[t] = el }} className={labelClass} />
          </Html>
        </group>
      ))}
      <Html position={[WF_X0 + 1.2, WF_Y0 - 0.45, 0]} center>
        <span ref={(el) => { textRefs.current[6] = el }} className={labelClass} />
      </Html>
      <Html position={[KCL_X + 0.45, WF_Y0 - 0.45, 0]} center>
        <span ref={(el) => { textRefs.current[7] = el }} className={labelClass} />
      </Html>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'kirchhoff-laws',
  title: { en: "Kirchhoff's Laws (KCL, KVL)", ko: '키르히호프 법칙 (KCL, KVL)' },
  subject: 'circuits',
  order: 5,
  description: {
    en: `## Kirchhoff's Current and Voltage Laws

**Charge is not stored at a node, and a lap around any loop brings you back to the same height.**

One source, R1 in series, then R2 and R3 in parallel, drawn as a potential landscape.
- Bright dot: the node where the current splits. The wide line arriving equals the two narrower lines leaving — that is KCL
- Amber walker: it walks one loop, and its height is the running sum of every rise and drop. Back at the start it stands at 0 V — that is KVL
- Glass tubes: the resistors, with amber filaments showing the power each one dissipates

**KCL**: the currents into a node equal the currents out, I1 = I2 + I3
**KVL**: around any closed loop the voltages sum to zero, V − I1 × R1 − I2 × R2 = 0

Both laws hold whatever you do with the sliders: raise R3 and its branch gives up current to R2, while the node itself simply moves to a new height.

In 2D: KVL as a waterfall that lands back on zero, and KCL as one bar matching two stacked ones.`,
    ko: `## 키르히호프 전류·전압 법칙

**전하는 마디에 쌓이지 않고, 폐로를 한 바퀴 돌면 같은 높이로 돌아옵니다.**

전원 하나에 R1이 직렬, 그 뒤에 R2와 R3가 병렬인 회로를 전위 지형으로 그렸습니다.
- 밝은 점: 전류가 갈라지는 마디. 들어오는 굵은 선 하나가 나가는 두 선과 같습니다. 이것이 KCL입니다
- 앰버 보행자: 폐로를 한 바퀴 걷습니다. 보행자의 높이가 지금까지 더한 상승과 강하의 합입니다. 출발점에 돌아오면 0 V에 서게 됩니다. 이것이 KVL입니다
- 유리관: 저항. 안쪽 앰버 필라멘트가 각 저항이 소비하는 전력을 나타냅니다

**KCL**: 마디로 들어오는 전류의 합은 나가는 전류의 합과 같습니다. I1 = I2 + I3
**KVL**: 폐로를 한 바퀴 돌면 전압의 합이 0입니다. V − I1 × R1 − I2 × R2 = 0

슬라이더를 어떻게 움직여도 두 법칙은 그대로입니다. R3를 키우면 그쪽 전류를 R2에 넘겨주고, 마디는 새로운 높이로 옮겨갈 뿐입니다.

2D: KVL은 0으로 되돌아오는 워터폴, KCL은 막대 하나가 두 막대의 합과 같은 모습입니다.`,
  },
  story: [
    {
      set: { V: 12, R1: 2, R2: 6, R3: 3 },
      text: {
        en: 'One road, two bottlenecks, and a charge that has to end up back where it started. Right now it is dead flat — so nothing moves.',
        ko: '길 하나, 병목 둘, 그리고 출발한 자리로 돌아와야 하는 전하 하나. 지금은 완전히 평평합니다. 그래서 아무것도 움직이지 않습니다.',
      },
    },
    {
      set: { V: 12, R1: 2, R2: 6, R3: 3 },
      text: {
        en: 'Battery in. One corner lifts to 12 V. Now there is a hill — and downhill is the only direction a charge knows.',
        ko: '배터리를 넣습니다. 한쪽 구석이 12 V로 들립니다. 언덕이 생겼습니다. 전하가 아는 방향은 내리막 하나뿐입니다.',
      },
    },
    {
      set: { V: 12, R1: 2, R2: 6, R3: 3 },
      text: {
        en: 'Down it rolls: 1.5 A, through R1 and out through R2. One road, one current. Boring. Let’s break it.',
        ko: '굴러 내려옵니다. 1.5 A, R1을 지나 R2로 빠져나갑니다. 길 하나, 전류 하나. 심심하죠. 부숴봅시다.',
      },
    },
    {
      set: { V: 12, R1: 2, R2: 6, R3: 3 },
      text: {
        en: 'A second branch opens — R3, straight down to ground. And look: the total current jumps to 3 A. Two roads are easier than one.',
        ko: '두 번째 갈림길이 열립니다. R3, 접지로 직행. 그리고 보세요. 전체 전류가 3 A로 뛰었습니다. 길이 둘이면 하나보다 쉬우니까요.',
      },
    },
    {
      set: { V: 12, R1: 2, R2: 6, R3: 3 },
      text: {
        en: 'Here is the whole trick: the junction stores nothing. 3 A walks in, 1 A turns left, 2 A turns right. Add them up. `I1 = I2 + I3` That is KCL.',
        ko: '핵심은 이것뿐입니다. 마디는 아무것도 쌓아두지 않습니다. 3 A가 들어오고, 1 A가 왼쪽, 2 A가 오른쪽으로 나갑니다. 더해 보세요. `I1 = I2 + I3` 이게 KCL입니다.',
      },
    },
    {
      set: { V: 12, R1: 2, R2: 6, R3: 6 },
      text: {
        en: 'Squeeze R3 to 6 Ω and it hands its share over: 1.2 A each, 2.4 A total. The split moved. The sum never did.',
        ko: 'R3를 6 Ω으로 조이면 자기 몫을 넘겨줍니다. 각각 1.2 A, 합은 2.4 A. 나뉘는 비율은 바뀌었지만 합은 한 번도 어긋나지 않았습니다.',
      },
    },
    {
      set: { V: 12, R1: 2, R2: 6, R3: 3 },
      text: {
        en: 'Second law, and it is even simpler. Follow one charge the whole way around. The amber dot’s height is everything it has gained and lost so far.',
        ko: '두 번째 법칙은 더 간단합니다. 전하 하나를 끝까지 따라가 봅시다. 앰버색 점의 높이가 지금까지 얻고 잃은 것의 총합입니다.',
      },
    },
    {
      set: { V: 12, R1: 2, R2: 6, R3: 3 },
      text: {
        en: 'Up 12 V through the battery. Down 6 V across R1. Down 6 V across R2. Every drop is just current times resistance.',
        ko: '배터리에서 12 V 상승. R1에서 6 V 강하. R2에서 6 V 강하. 모든 강하는 그냥 전류 × 저항입니다.',
      },
    },
    {
      set: { V: 12, R1: 2, R2: 6, R3: 3 },
      text: {
        en: 'And it lands exactly where it set off. It has to — that is the same point. `+12 − 6 − 6 = 0` Walk any loop you like; the voltages cancel. That is KVL.',
        ko: '그리고 출발한 자리에 정확히 내려앉습니다. 당연합니다. 같은 지점이니까요. `+12 − 6 − 6 = 0` 어떤 폐로를 돌아도 전압은 상쇄됩니다. 이게 KVL입니다.',
      },
    },
    {
      set: { V: 12, R1: 2, R2: 6, R3: 3 },
      text: {
        en: 'Two sentences, and every circuit you will ever solve falls out of them. `ΣI(in) = ΣI(out)` `ΣV(loop) = 0` The sliders are yours.',
        ko: '문장 두 개. 앞으로 풀 모든 회로가 여기서 나옵니다. `ΣI(들어옴) = ΣI(나감)` `ΣV(폐로) = 0` 이제 슬라이더는 당신 것입니다.',
      },
    },
  ],
  sliderParams: [
    { key: 'V',  label: { en: 'Voltage (V)',     ko: '전압 (V)' },   unit: 'V', min: 1,     max: V_MAX, step: 0.5, defaultValue: 12 },
    { key: 'R1', label: { en: 'Series R1',       ko: '직렬 저항 R1' }, unit: 'Ω', min: R_MIN, max: 20,    step: 0.5, defaultValue: 2 },
    { key: 'R2', label: { en: 'Branch R2',       ko: '분기 저항 R2' }, unit: 'Ω', min: R_MIN, max: 20,    step: 0.5, defaultValue: 6 },
    { key: 'R3', label: { en: 'Branch R3',       ko: '분기 저항 R3' }, unit: 'Ω', min: R_MIN, max: 20,    step: 0.5, defaultValue: 3 },
  ],
  Scene: KirchhoffScene,
  Scene2D: KirchhoffBars2DScene,
}

export default concept
