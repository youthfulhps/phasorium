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
import { setLineOpacity, setOpacity, useStoryReveal } from '../../components/scene/storyReveal'
import type { ConceptMeta } from '../../types/concept'

// Slider ranges, also used to normalize heights, line weights and the 2D axes
const V_MAX = 24 // [V]
const R_MIN = 2 // [Ω] for R1 and R2
const R_MAX = 20 // [Ω]
const RL_MIN = 1 // [Ω]
const I_MAX = V_MAX / R_MIN // largest possible short-circuit current, 12 A
const P_MAX = (V_MAX * V_MAX) / RL_MIN // hottest possible resistor [W]

// Pure physics. Worked example: V = 12 V, R1 = 4 Ω, R2 = 12 Ω, RL = 6 Ω →
// Vth = 12 × 12 ÷ 16 = 9 V, Rth = 4 × 12 ÷ 16 = 3 Ω, In = 9 ÷ 3 = 3 A,
// IL = 9 ÷ (3 + 6) = 1 A, VL = 9 × 6 ÷ 9 = 6 V.
// Cross-check on the untouched network: R2 ∥ RL = 4 Ω, Itot = 12 ÷ 8 = 1.5 A,
// VL = 1.5 × 4 = 6 V, IL = 1 A — identical, which is the whole point.
function solve({ V = 12, R1 = 4, R2 = 12, RL = 6 }: Record<string, number>) {
  const Vth = (V * R2) / (R1 + R2) // open-circuit terminal voltage
  const Rth = (R1 * R2) / (R1 + R2) // source killed: R1 ∥ R2
  const In = Vth / Rth // short-circuit current, equals V ÷ R1
  const IL = Vth / (Rth + RL)
  const VL = IL * RL
  return {
    V, R1, R2, RL, Vth, Rth, In, IL, VL,
    I2: VL / R2, // through R2 in the original network
    Itot: (V - VL) / R1, // drawn from the real source
  }
}

// ── Narration steps drive what exists on screen. Each reveal eases 0 → 1 so nothing pops in. ──
// Step 0 is the real network on its own: the equivalent is the punchline, not the premise.
const REV_LABEL = 0 // the original network's own readouts
const REV_TERM = 1 // the terminal dots and the VL / IL the load actually sees
const REV_THEV = 2 // the whole Thévenin side: its loop, its two parts, its readouts
const REV_LEVEL = 3 // the dashed line joining the two terminals at one height
const REV_NORTON = 4 // the Norton reading of the same box
const REV_COUNT = 5

function revealTargets(step: number, out: Float32Array) {
  out[REV_LABEL] = step >= 1 ? 1 : 0
  out[REV_TERM] = step >= 2 ? 1 : 0
  out[REV_THEV] = step >= 4 ? 1 : 0
  out[REV_LEVEL] = step >= 5 ? 1 : 0
  out[REV_NORTON] = step >= 8 ? 1 : 0
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
const fmtA = (x: number) => `${x.toFixed(2)} A`
const fmtR = (x: number) => `${x.toFixed(x < 10 ? 1 : 0)} Ω`
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const captionClass = 'pointer-events-none whitespace-nowrap text-[10px] text-muted-foreground'
const axisLabelClass = 'pointer-events-none whitespace-nowrap text-[11px] text-muted-foreground'

// ── 3D: the real network beside its Thévenin equivalent, both as potential landscapes ──
const BASE_Y = -0.6
const HEIGHT_PER_VOLT = 1.6 / V_MAX
const RULER_VOLTS = [0, 12, 24]
const Z_BACK = -1.0
const Z_FRONT = 1.0
const RZ: [number, number] = [-0.45, 0.45] // resistor span on a rung running front-to-back
const SEG_PTS = 16

type XZ = [number, number]
// Original network (left): source up the left side, R1 along the back rail, then the terminal node
// feeds two rungs down to the 0 V front rail — R2 on the middle rung, the load RL on the right.
const OX_S = -2.9
const OX_T = -1.85 // terminal node
const OX_L = -1.0
const R1_A: XZ = [OX_S + 0.3, Z_BACK]
const R1_B: XZ = [OX_T - 0.3, Z_BACK]
const ORIG_MAIN_XZ: XZ[] = [[OX_T, Z_FRONT], [OX_S, Z_FRONT], [OX_S, Z_BACK], R1_A, R1_B, [OX_T, Z_BACK]]
const ORIG_R2_XZ: XZ[] = [[OX_T, Z_BACK], [OX_T, RZ[0]], [OX_T, RZ[1]], [OX_T, Z_FRONT]]
const ORIG_RL_XZ: XZ[] = [
  [OX_T, Z_BACK], [OX_L, Z_BACK], [OX_L, RZ[0]], [OX_L, RZ[1]], [OX_L, Z_FRONT], [OX_T, Z_FRONT],
]
// Thévenin equivalent (right): one source, one resistance, the same load
const TX_S = -0.1
const TX_L = 1.1
const RTH_A: XZ = [TX_S + 0.3, Z_BACK]
const RTH_B: XZ = [TX_L - 0.3, Z_BACK]
const THEV_XZ: XZ[] = [
  [TX_L, Z_FRONT], [TX_S, Z_FRONT], [TX_S, Z_BACK], RTH_A, RTH_B,
  [TX_L, Z_BACK], [TX_L, RZ[0]], [TX_L, RZ[1]], [TX_L, Z_FRONT],
]

// Potential (volts) at each waypoint, rewritten every frame — preallocated, no per-frame arrays
const origMainV = new Float32Array(ORIG_MAIN_XZ.length)
const origR2V = new Float32Array(ORIG_R2_XZ.length)
const origRLV = new Float32Array(ORIG_RL_XZ.length)
const thevV = new Float32Array(THEV_XZ.length)
const LINE_XZ = [ORIG_MAIN_XZ, ORIG_R2_XZ, ORIG_RL_XZ, THEV_XZ]
const LINE_V = [origMainV, origR2V, origRLV, thevV]
const lineI = new Float32Array(4)

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
  a: XZ, va: number, b: XZ, vb: number, k: number, R: number, P: number, tagDx: number
) {
  p0.set(a[0], BASE_Y + k * va, a[1])
  p1.set(b[0], BASE_Y + k * vb, b[1])
  dir.subVectors(p1, p0)
  if (g) {
    g.position.addVectors(p0, p1).multiplyScalar(0.5)
    g.quaternion.setFromUnitVectors(X_AXIS, unit.copy(dir).normalize())
    const r = 0.13 * Math.sqrt(RL_MIN / Math.max(R, 0.2)) + 0.03 // thinner for larger R
    g.scale.set(dir.length(), r, r)
  }
  setResistorHeat(g, Math.sqrt(P / P_MAX) * 1.6)
  tag?.position.set((p0.x + p1.x) / 2 + tagDx, (p0.y + p1.y) / 2 + 0.26, (p0.z + p1.z) / 2)
}

/** Current as line weight: wider for more current (px), clamped to stay readable. */
const widthFor = (I: number) => Math.min(1.2 + 1.6 * Math.sqrt(I), 8)
/** Sheen bands per second: faster for more current. */
const flowFor = (I: number) => Math.min(0.6 * I, 8)

const LEVEL_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]

function TheveninScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const rise = useRiseIn()
  const bufs = useMemo(() => LINE_XZ.map((xz) => new Float32Array(pointCount(xz) * 3)), [])
  const inits = useMemo(() => LINE_XZ.map(initPoints), [])
  const levelBuf = useMemo(() => new Float32Array(6), [])
  const lineRefs = useRef<any[]>([])
  const levelRef = useRef<any>(null)
  const resRefs = useRef<(THREE.Group | null)[]>([]) // R1, R2, RL left, Rth, RL right
  const tagRefs = useRef<(THREE.Group | null)[]>([]) // 5 resistor tags, 2 terminal readouts, 2 source tags
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const termL = useRef<THREE.Group>(null)
  const termR = useRef<THREE.Group>(null)
  const flows = useRef([0, 0, 0, 0])
  const labelledFor = useRef<unknown>(null)
  const footRefs = useRef<any[]>([])
  const thevCaption = useRef<HTMLSpanElement>(null)
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(() => {
    const s = solve(valuesRef.current)
    const k = HEIGHT_PER_VOLT * rise.current

    // Original: 0 → V up the source, down to the terminal potential VL across R1, then to 0 in each rung
    origMainV[2] = origMainV[3] = s.V
    origMainV[4] = origMainV[5] = s.VL
    origR2V[0] = origR2V[1] = s.VL
    origRLV[0] = origRLV[1] = origRLV[2] = s.VL
    // Thévenin: 0 → Vth up the source, down to the same VL across Rth, then to 0 across RL
    thevV[2] = thevV[3] = s.Vth
    thevV[4] = thevV[5] = thevV[6] = s.VL

    lineI[0] = s.Itot; lineI[1] = s.I2; lineI[2] = s.IL; lineI[3] = s.IL
    for (let l = 0; l < 4; l++) {
      const line = lineRefs.current[l]
      if (!line) continue
      fillPath(bufs[l], LINE_XZ[l], LINE_V[l], k)
      line.geometry.setPositions(bufs[l])
      ;(line.material as LineMaterial).linewidth = widthFor(lineI[l])
      flows.current[l] = flowFor(lineI[l])
      if (l === 3) setLineOpacity(line, reveal[REV_THEV])
    }

    placeResistor(resRefs.current[0], tagRefs.current[0], R1_A, s.V, R1_B, s.VL, k, s.R1, s.Itot * s.Itot * s.R1, 0)
    placeResistor(resRefs.current[1], tagRefs.current[1],
      ORIG_R2_XZ[1], s.VL, ORIG_R2_XZ[2], 0, k, s.R2, s.I2 * s.I2 * s.R2, -0.4)
    placeResistor(resRefs.current[2], tagRefs.current[2],
      ORIG_RL_XZ[2], s.VL, ORIG_RL_XZ[3], 0, k, s.RL, s.IL * s.IL * s.RL, 0.4)
    placeResistor(resRefs.current[3], tagRefs.current[3], RTH_A, s.Vth, RTH_B, s.VL, k, s.Rth, s.IL * s.IL * s.Rth, 0)
    placeResistor(resRefs.current[4], tagRefs.current[4],
      THEV_XZ[6], s.VL, THEV_XZ[7], 0, k, s.RL, s.IL * s.IL * s.RL, 0.4)

    // Both terminals sit at exactly the same height; the dashed level line between them says so
    const yT = BASE_Y + k * s.VL
    termL.current?.position.set(OX_T, yT, Z_BACK)
    termR.current?.position.set(TX_L, yT, Z_BACK)
    levelBuf[0] = OX_T; levelBuf[1] = yT; levelBuf[2] = Z_BACK
    levelBuf[3] = TX_L; levelBuf[4] = yT; levelBuf[5] = Z_BACK
    levelRef.current?.geometry.setPositions(levelBuf)
    levelRef.current?.computeLineDistances()

    tagRefs.current[5]?.position.set(OX_T, yT + 0.3, Z_BACK)
    tagRefs.current[6]?.position.set(TX_L + 0.1, yT + 0.3, Z_BACK)
    tagRefs.current[7]?.position.set(OX_S - 0.05, BASE_Y + k * s.V + 0.24, Z_BACK)
    tagRefs.current[8]?.position.set(TX_S - 0.05, BASE_Y + k * s.Vth + 0.24, Z_BACK)

    // The equivalent is built in front of the learner, so everything on its side shares one weight
    const wT = reveal[REV_THEV]
    resRefs.current[3]?.scale.multiplyScalar(wT)
    resRefs.current[4]?.scale.multiplyScalar(wT)
    setLineOpacity(footRefs.current[2], wT)
    setLineOpacity(levelRef.current, reveal[REV_LEVEL])
    termL.current?.scale.setScalar(reveal[REV_TERM])
    termR.current?.scale.setScalar(reveal[REV_TERM] * wT)
    setOpacity(thevCaption.current, wT)

    const fades = [
      reveal[REV_LABEL], reveal[REV_LABEL], reveal[REV_LABEL], wT, wT,
      reveal[REV_TERM], reveal[REV_TERM] * wT, reveal[REV_LABEL], wT, reveal[REV_NORTON],
    ]
    fades.forEach((v, t) => setOpacity(textRefs.current[t], v))

    if (labelledFor.current !== valuesRef.current && textRefs.current[9]) {
      labelledFor.current = valuesRef.current
      const ko = locale === 'ko'
      const terminal = `VL ${fmtV(s.VL)} · IL ${fmtA(s.IL)}`
      const texts = [
        `R1 ${fmtR(s.R1)}`,
        `R2 ${fmtR(s.R2)}`,
        `RL ${fmtR(s.RL)}`,
        `Rth ${fmtR(s.Rth)}`,
        `RL ${fmtR(s.RL)}`,
        terminal,
        terminal,
        `V ${fmtV(s.V)}`,
        `Vth ${fmtV(s.Vth)}`,
        `${ko ? '노턴' : 'Norton'} · In ${fmtA(s.In)} ∥ Rn ${fmtR(s.Rth)}`,
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  const footprints = useMemo<[number, number, number][][]>(() => [
    [[OX_S, BASE_Y, Z_FRONT], [OX_S, BASE_Y, Z_BACK], [OX_L, BASE_Y, Z_BACK], [OX_L, BASE_Y, Z_FRONT], [OX_S, BASE_Y, Z_FRONT]],
    [[OX_T, BASE_Y, Z_BACK], [OX_T, BASE_Y, Z_FRONT]],
    [[TX_S, BASE_Y, Z_FRONT], [TX_S, BASE_Y, Z_BACK], [TX_L, BASE_Y, Z_BACK], [TX_L, BASE_Y, Z_FRONT], [TX_S, BASE_Y, Z_FRONT]],
  ], [])
  const ko = locale === 'ko'

  return (
    <>
      {/* Aimed right of centre, so the equivalent's terminal readout clears the narration panel */}
      <PerspectiveCamera makeDefault position={[3, 2, 5.4]} fov={50} />
      <OrbitControls enablePan={false} target={[0.8, 0, 0]} />

      {footprints.map((pts, i) => (
        <Line
          key={i}
          ref={(line) => { footRefs.current[i] = line }}
          points={pts}
          lineWidth={1}
          color={c.guide}
          dashed
          dashSize={0.08}
          gapSize={0.06}
        />
      ))}
      <FloorCaption position={[-0.9, BASE_Y, Z_FRONT + 0.8]} />
      <PotentialRuler x={OX_S - 0.4} z={Z_BACK} baseY={BASE_Y} volts={RULER_VOLTS} unitsPerVolt={() => HEIGHT_PER_VOLT * rise.current} />

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
      {/* The two terminals at one height: the load cannot tell the circuits apart */}
      <Line ref={levelRef} points={LEVEL_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.07} gapSize={0.05} />

      {[0, 1, 2, 3, 4].map((r) => (
        <Resistor key={r} ref={(g) => { resRefs.current[r] = g }} />
      ))}
      <Marker ref={termL} color={c.marker} size={0.06} />
      <Marker ref={termR} color={c.marker} size={0.06} />

      {[0, 1, 2, 3, 4, 5, 6, 7, 8].map((t) => (
        <group key={t} ref={(g) => { tagRefs.current[t] = g }}>
          <Html center>
            <span ref={(el) => { textRefs.current[t] = el }} className={labelClass} />
          </Html>
        </group>
      ))}

      <Html position={[(OX_S + OX_L) / 2, BASE_Y - 0.12, Z_FRONT + 0.4]} center>
        <span className={captionClass}>{ko ? '원래 회로망' : 'Original network'}</span>
      </Html>
      <Html position={[(TX_S + TX_L) / 2, BASE_Y - 0.12, Z_FRONT + 0.4]} center>
        <span ref={thevCaption} className={captionClass}>{ko ? '테브난 등가' : 'Thévenin equivalent'}</span>
      </Html>
      {/* Norton stays a readout: same Rth, the source read as a current instead of a voltage */}
      <Html position={[(TX_S + TX_L) / 2, BASE_Y - 0.38, Z_FRONT + 0.4]} center>
        <span ref={(el) => { textRefs.current[9] = el }} className={labelClass} style={{ color: c.current }} />
      </Html>
    </>
  )
}

// ── 2D: the terminal i–v line v = Vth − i × Rth, with both intercepts and the operating point ──
const X0 = -3.2 // i = 0
const Y0 = -1.7 // v = 0
const X_SPAN = 6.4 // 0 … I_MAX
const Y_SPAN = 3.4 // 0 … V_MAX
const X_PER_AMP = X_SPAN / I_MAX
const Y_PER_VOLT = Y_SPAN / V_MAX
const IV_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]
const AXES: [number, number, number][] = [[X0, Y0 + Y_SPAN + 0.25, 0], [X0, Y0, 0], [X0 + X_SPAN + 0.3, Y0, 0]]

function TerminalIV2DScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const ivBuf = useMemo(() => new Float32Array(6), [])
  const loadBuf = useMemo(() => new Float32Array(6), [])
  const ivRef = useRef<any>(null)
  const loadRef = useRef<any>(null)
  const dotV = useRef<THREE.Group>(null)
  const dotI = useRef<THREE.Group>(null)
  const dotOp = useRef<THREE.Group>(null)
  const tagRefs = useRef<(THREE.Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<unknown>(null)
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(() => {
    const s = solve(valuesRef.current)
    // One straight line from the open-circuit point (0, Vth) to the short-circuit point (In, 0)
    const yOC = Y0 + s.Vth * Y_PER_VOLT
    const xSC = X0 + s.In * X_PER_AMP
    ivBuf[0] = X0; ivBuf[1] = yOC; ivBuf[2] = 0
    ivBuf[3] = xSC; ivBuf[4] = Y0; ivBuf[5] = 0
    ivRef.current?.geometry.setPositions(ivBuf)

    // Load line v = i × RL through the origin, cut where it leaves the plot box
    const iEnd = Math.min(I_MAX, V_MAX / s.RL)
    loadBuf[0] = X0; loadBuf[1] = Y0; loadBuf[2] = 0
    loadBuf[3] = X0 + iEnd * X_PER_AMP; loadBuf[4] = Y0 + iEnd * s.RL * Y_PER_VOLT; loadBuf[5] = 0
    loadRef.current?.geometry.setPositions(loadBuf)
    loadRef.current?.computeLineDistances()

    const xOp = X0 + s.IL * X_PER_AMP
    const yOp = Y0 + s.VL * Y_PER_VOLT
    dotV.current?.position.set(X0, yOC, 0.01)
    dotI.current?.position.set(xSC, Y0, 0.01)
    dotOp.current?.position.set(xOp, yOp, 0.02)
    tagRefs.current[0]?.position.set(X0 + 0.55, yOC + 0.22, 0)
    tagRefs.current[1]?.position.set(xSC + 0.05, Y0 - 0.28, 0)
    tagRefs.current[2]?.position.set(xOp + 0.9, yOp + 0.26, 0)

    // The terminal line arrives with the equivalent it describes; the operating point with the load
    const wT = reveal[REV_THEV]
    setLineOpacity(ivRef.current, wT)
    dotV.current?.scale.setScalar(wT)
    dotI.current?.scale.setScalar(wT)
    setOpacity(textRefs.current[0], wT)
    setOpacity(textRefs.current[1], wT)
    setLineOpacity(loadRef.current, reveal[REV_TERM])
    dotOp.current?.scale.setScalar(reveal[REV_TERM])
    setOpacity(textRefs.current[2], reveal[REV_TERM])

    if (labelledFor.current !== valuesRef.current && textRefs.current[2]) {
      labelledFor.current = valuesRef.current
      const ko = locale === 'ko'
      const texts = [
        `${ko ? '개방' : 'open'} · Vth ${fmtV(s.Vth)}`,
        `${ko ? '단락' : 'short'} · In ${fmtA(s.In)}`,
        `RL ${fmtR(s.RL)} · VL ${fmtV(s.VL)} · IL ${fmtA(s.IL)}`,
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  return (
    <>
      <Line points={AXES} lineWidth={1} color={c.guide} />
      <Line ref={loadRef} points={IV_INIT} lineWidth={1.5} color={c.resistance} dashed dashSize={0.1} gapSize={0.07} />
      <Line ref={ivRef} points={IV_INIT} lineWidth={3} color={c.voltage} />
      <Marker ref={dotV} color={c.marker} size={0.06} />
      <Marker ref={dotI} color={c.marker} size={0.06} />
      <Marker ref={dotOp} color={c.marker} size={0.08} />
      {[0, 1, 2].map((t) => (
        <group key={t} ref={(g) => { tagRefs.current[t] = g }}>
          <Html center>
            <span ref={(el) => { textRefs.current[t] = el }} className={labelClass} />
          </Html>
        </group>
      ))}
      <Html position={[X0 + X_SPAN + 0.35, Y0 - 0.28, 0]} center>
        <span className={axisLabelClass}>i (A)</span>
      </Html>
      <Html position={[X0 - 0.35, Y0 + Y_SPAN + 0.25, 0]} center>
        <span className={axisLabelClass}>v (V)</span>
      </Html>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'thevenin-norton',
  title: { en: 'Thévenin & Norton Equivalents', ko: '테브난·노턴 등가회로' },
  subject: 'circuits',
  order: 7,
  description: {
    en: `## Thévenin and Norton Equivalents

**Seen from two terminals, any linear source network is one source and one resistance — and the load cannot tell the difference.**

Left, the real network: a source V behind R1, with R2 across the terminals. Right, its Thévenin equivalent: Vth behind Rth, feeding the same load RL.
- Bright dots: the two terminals. The dashed level line joins them, and it stays level because both sit at the same VL
- Cyan liquid line: the current. The load branch is equally wide and equally fast on both sides
- Glass tubes: resistors, thinner for larger R; the amber filament thickens with the heat

**Thévenin**: Vth = V × R2 ÷ (R1 + R2), Rth = R1 ∥ R2 = R1 × R2 ÷ (R1 + R2)
**Norton**: In = Vth ÷ Rth, Rn = Rth — the same box, the source read as a current
**With a load**: IL = Vth ÷ (Rth + RL), VL = Vth × RL ÷ (Rth + RL)

Defaults V = 12 V, R1 = 4 Ω, R2 = 12 Ω give Vth = 9 V, Rth = 3 Ω, In = 3 A; at RL = 6 Ω both circuits settle at VL = 6.00 V and IL = 1.00 A. Move V, R1 or R2 and the two landscapes change together, never apart. RL moves the operating point along the line without touching the line itself.

In 2D: the terminal characteristic v = Vth − i × Rth. Its v intercept is the open-circuit Vth, its i intercept is the short-circuit In, and the dashed load line v = i × RL picks the operating point. That one line is the equivalence.`,
    ko: `## 테브난·노턴 등가회로

**두 단자에서 보면 어떤 선형 회로망도 전원 하나와 저항 하나입니다. 부하는 둘을 구별하지 못합니다.**

왼쪽은 실제 회로망입니다. 전원 V 뒤에 R1이 있고 단자에는 R2가 걸려 있습니다. 오른쪽은 그 테브난 등가입니다. Vth 뒤에 Rth가 있고 같은 부하 RL에 전력을 공급합니다.
- 밝은 점: 두 단자. 점선 수평선이 둘을 잇는데, 양쪽 VL이 같으니 언제나 수평입니다
- 시안 액체 선: 전류. 부하 가지는 양쪽에서 굵기도 속도도 똑같습니다
- 유리관: 저항. R이 클수록 가늘고, 열이 클수록 안쪽 앰버 필라멘트가 굵어집니다

**테브난**: Vth = V × R2 ÷ (R1 + R2), Rth = R1 ∥ R2 = R1 × R2 ÷ (R1 + R2)
**노턴**: In = Vth ÷ Rth, Rn = Rth. 같은 상자를 전류원으로 바꿔 읽은 것뿐입니다
**부하를 달면**: IL = Vth ÷ (Rth + RL), VL = Vth × RL ÷ (Rth + RL)

기본값 V = 12 V, R1 = 4 Ω, R2 = 12 Ω이면 Vth = 9 V, Rth = 3 Ω, In = 3 A입니다. RL = 6 Ω에서 두 회로 모두 VL = 6.00 V, IL = 1.00 A로 똑같이 자리잡습니다. V, R1, R2를 움직이면 두 지형이 늘 함께 바뀌고 따로 놀지 않습니다. RL은 직선은 그대로 둔 채 동작점만 직선 위에서 옮깁니다.

2D: 단자 특성 v = Vth − i × Rth입니다. v축 절편이 개방 전압 Vth, i축 절편이 단락 전류 In이고, 점선 부하선 v = i × RL이 동작점을 정합니다. 이 직선 하나가 곧 등가입니다.`,
  },
  story: [
    {
      set: { V: 12, R1: 4, R2: 12, RL: 6 },
      text: {
        en: 'A source, two resistors, a load hanging off two terminals. Fine. Now picture a hundred of these behind a wall, and you only get to touch the two terminals.',
        ko: '전원 하나, 저항 둘, 단자 두 개에 매달린 부하. 좋습니다. 이제 벽 뒤에 이런 게 백 개 있고, 당신이 만질 수 있는 건 단자 두 개뿐이라고 생각해 보세요.',
      },
    },
    {
      set: { V: 12, R1: 4, R2: 12, RL: 6 },
      text: {
        en: 'Put numbers on it. 12 V behind a 4 Ω, a 12 Ω across the terminals, and a 6 Ω load.',
        ko: '값을 붙입니다. 12 V 뒤에 4 Ω, 단자에 12 Ω, 그리고 부하 6 Ω.',
      },
    },
    {
      set: { V: 12, R1: 4, R2: 12, RL: 6 },
      text: {
        en: 'But the load does not know any of that. It knows two numbers: 6.00 V across it, 1.00 A through it. That is the entire relationship.',
        ko: '하지만 부하는 그걸 하나도 모릅니다. 부하가 아는 건 두 숫자뿐입니다. 걸린 전압 6.00 V, 흐르는 전류 1.00 A. 관계는 이게 전부입니다.',
      },
    },
    {
      set: { V: 12, R1: 4, R2: 12, RL: 6 },
      text: {
        en: 'So Thévenin asked the obvious question: if only two numbers get out, can we fake the whole wall with two parts? Take the load off and measure the terminal voltage. Short the terminals and measure the current. Two measurements, done.',
        ko: '그래서 테브난은 당연한 질문을 던졌습니다. 나오는 숫자가 둘뿐이라면, 벽 전체를 부품 두 개로 흉내 낼 수 있지 않을까? 부하를 떼고 단자 전압을 재고, 단자를 단락시켜 전류를 잽니다. 측정 두 번이면 끝입니다.',
      },
    },
    {
      set: { V: 12, R1: 4, R2: 12, RL: 6 },
      text: {
        en: 'And there it is. One source, one resistance. `Vth = V × R2 ÷ (R1 + R2) = 9 V` `Rth = R1 ∥ R2 = 3 Ω`',
        ko: '그리고 이게 나옵니다. 전원 하나, 저항 하나. `Vth = V × R2 ÷ (R1 + R2) = 9 V` `Rth = R1 ∥ R2 = 3 Ω`',
      },
    },
    {
      set: { V: 12, R1: 4, R2: 12, RL: 6 },
      text: {
        en: 'Hang the same load on it and read both terminals: 6.00 V, 1.00 A. Identical. The dashed line joining them is dead level — spin the view and you will see it. The load cannot tell which circuit it is bolted to.',
        ko: '같은 부하를 달고 두 단자를 읽어 봅니다. 6.00 V, 1.00 A. 똑같습니다. 둘을 잇는 점선은 완벽히 수평입니다. 시점을 돌려 보면 보입니다. 부하는 자기가 어느 회로에 붙어 있는지 구별하지 못합니다.',
      },
    },
    {
      set: { V: 12, R1: 4, R2: 6, RL: 6 },
      text: {
        en: 'Go on, break it. Drop R2 to 6 Ω. The left landscape shifts — and the right one shifts with it, to Vth 7.20 V and Rth 2.4 Ω. The two terminals stay level.',
        ko: '한번 깨뜨려 보세요. R2를 6 Ω으로 내립니다. 왼쪽 지형이 바뀌고, 오른쪽도 따라 바뀝니다. Vth 7.20 V, Rth 2.4 Ω. 두 단자는 여전히 나란합니다.',
      },
    },
    {
      set: { V: 12, R1: 4, R2: 12, RL: 2 },
      text: {
        en: 'Now change only the load — 2 Ω. The operating point slides to 1.80 A and 3.60 V, but Vth and Rth do not move at all. The equivalent describes the wall, not whatever you hang on it.',
        ko: '이번엔 부하만 바꿉니다. 2 Ω. 동작점이 1.80 A, 3.60 V로 미끄러지는데 Vth와 Rth는 꿈쩍도 하지 않습니다. 등가회로는 벽을 설명하는 것이지, 거기 뭘 매달든 상관하지 않습니다.',
      },
    },
    {
      set: { V: 12, R1: 4, R2: 12, RL: 6 },
      text: {
        en: 'Same box, read the other way round: call it a current source with that resistance in parallel and you have Norton. `In = Vth ÷ Rth = 9 ÷ 3 = 3 A` Thévenin and Norton are two faces of one card.',
        ko: '같은 상자를 반대로 읽어 봅니다. 전류원에 그 저항을 병렬로 붙이면 노턴이 됩니다. `In = Vth ÷ Rth = 9 ÷ 3 = 3 A` 테브난과 노턴은 카드 한 장의 앞뒤입니다.',
      },
    },
    {
      set: { V: 12, R1: 4, R2: 12, RL: 6 },
      text: {
        en: '`Vth = V × R2 ÷ (R1 + R2)` `Rth = R1 ∥ R2` However ugly the network, seen from two terminals it is one source and one resistance. That is why nobody solves the whole wall twice. The sliders are yours.',
        ko: '`Vth = V × R2 ÷ (R1 + R2)` `Rth = R1 ∥ R2` 회로망이 아무리 험악해도 두 단자에서 보면 전원 하나와 저항 하나입니다. 아무도 벽 전체를 두 번 풀지 않는 이유입니다. 이제 슬라이더는 당신 것입니다.',
      },
    },
  ],
  sliderParams: [
    { key: 'V',  label: { en: 'Source voltage (V)',  ko: '전원 전압 (V)' },  unit: 'V', min: 2,      max: V_MAX, step: 0.5, defaultValue: 12 },
    { key: 'R1', label: { en: 'Series (R1)',         ko: '직렬 저항 (R1)' }, unit: 'Ω', min: R_MIN,  max: R_MAX, step: 0.5, defaultValue: 4 },
    { key: 'R2', label: { en: 'Shunt (R2)',          ko: '병렬 저항 (R2)' }, unit: 'Ω', min: R_MIN,  max: R_MAX, step: 0.5, defaultValue: 12 },
    { key: 'RL', label: { en: 'Load (RL)',           ko: '부하 저항 (RL)' }, unit: 'Ω', min: RL_MIN, max: R_MAX, step: 0.5, defaultValue: 6 },
  ],
  Scene: TheveninScene,
  Scene2D: TerminalIV2DScene,
}

export default concept
