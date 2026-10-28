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
import { Resistor, setResistorHeat } from '../../components/scene/Resistor'
import type { ConceptMeta } from '../../types/concept'

const R_MIN = 5 // [Ω]
const R_MAX = 60 // [Ω]
const I_TEST = 1 // the probe drives 1 A through the pair it is clipped to [A]
const P_MAX = I_TEST * I_TEST * R_MAX // hottest a single resistor can get under the test current [W]

// Terminals, in the cyclic order the probe walks: 0 = A, 1 = B, 2 = C.
// Δ edge k joins terminal k to terminal k+1, so edge 0 = Rab, edge 1 = Rbc, edge 2 = Rca.
// Y arm k is the spoke from the star point out to terminal k.
const PAIR_NAMES = ['A–B', 'B–C', 'C–A']
const EDGE_NAMES = ['Rab', 'Rbc', 'Rca']
const ARM_NAMES = ['Ra', 'Rb', 'Rc']

// Shared physics buffers: filled by solve(), read by both scenes. Typed arrays so a frame
// allocates nothing.
const EDGE = new Float32Array(3) // Δ resistances [Ω]
const ARM = new Float32Array(3) // Y resistances [Ω]
const PAIR_D = new Float32Array(3) // terminal-pair resistance measured on the Δ [Ω]
const PAIR_Y = new Float32Array(3) // the same pair measured on the Y [Ω]

// Δ→Y: Ra = (Rab·Rca) ÷ Σ, Rb = (Rab·Rbc) ÷ Σ, Rc = (Rbc·Rca) ÷ Σ, with Σ = Rab + Rbc + Rca.
// Y→Δ: Rab = (RaRb + RbRc + RcRa) ÷ Rc, and the same pattern for Rbc (÷ Ra) and Rca (÷ Rb).
// Worked example (verified): Rab = 10 Ω, Rbc = 20 Ω, Rca = 30 Ω, Σ = 60 Ω →
// Ra = 5 Ω, Rb = 3.333 Ω, Rc = 10 Ω; converting back gives exactly 10 ÷ 20 ÷ 30 Ω again.
// Terminal pairs then agree on both networks: A–B 8.333 Ω, B–C 13.333 Ω, C–A 15 Ω.
// Balanced case Rab = Rbc = Rca = 30 Ω → Ra = Rb = Rc = 10 Ω, so RΔ = 3 · RY.
function solve({ Rab = 10, Rbc = 20, Rca = 30 }: Record<string, number>) {
  EDGE[0] = Rab
  EDGE[1] = Rbc
  EDGE[2] = Rca
  const sum = Rab + Rbc + Rca
  for (let k = 0; k < 3; k++) {
    // Each Y arm is the product of the two Δ resistors meeting at that terminal, over Σ
    ARM[k] = (EDGE[k] * EDGE[(k + 2) % 3]) / sum
  }
  for (let k = 0; k < 3; k++) {
    // Δ seen at the pair: the direct edge in parallel with the other two in series
    PAIR_D[k] = (EDGE[k] * (sum - EDGE[k])) / sum
    // Y seen at the same pair: two arms in series, the third carries nothing
    PAIR_Y[k] = ARM[k] + ARM[(k + 1) % 3]
  }
  return sum
}

function useSliderValues() {
  const valuesRef = useRef<Record<string, number>>(useSliderStore.getState().values)
  useEffect(
    () => useSliderStore.subscribe((s) => s.values, (v) => { valuesRef.current = v }),
    []
  )
  return valuesRef
}

const fmtR = (x: number) => `${x.toFixed(x < 10 ? 2 : 1)} Ω`
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const captionClass = 'pointer-events-none whitespace-nowrap text-[11px] text-muted-foreground'

/** Tube radius: thinner for larger R, clamped so a tiny Y arm cannot swallow its spoke. */
const radiusFor = (R: number) => Math.min(0.17, 0.13 * Math.sqrt(R_MIN / Math.max(R, 0.2)) + 0.03)
/** Line weight in px from the current it carries. */
const widthFor = (I: number) => 1.1 + 2.4 * Math.sqrt(Math.abs(I))

// ── 3D: the Δ and its Y twin side by side, each with its own probe reading the same pair ──
const NET_Y = -0.3 // both networks lie flat in this plane
const TRI_X = [0, 0.693, -0.693] // A, B, C — equilateral, A at the back
const TRI_Z = [-0.8, 0.4, 0.4]
const DELTA_CX = -1.45
const WYE_CX = 0.55
const METER_Y = NET_Y + 1.15
const SEG = 0.55 // fraction of a wire the resistor body covers
const EDGE_PTS = 12
const LEAD_PTS = 10

type Point = [number, number, number]

const straight = (
  ax: number, ay: number, az: number, bx: number, by: number, bz: number, n: number
): Point[] =>
  Array.from({ length: n }, (_, i) => {
    const u = i / (n - 1)
    return [ax + (bx - ax) * u, ay + (by - ay) * u, az + (bz - az) * u] as Point
  })

// Wire paths, fixed geometry: only their weight and flow change with the test current.
const DELTA_PATHS: Point[][] = [0, 1, 2].map((k) =>
  straight(
    DELTA_CX + TRI_X[k], NET_Y, TRI_Z[k],
    DELTA_CX + TRI_X[(k + 1) % 3], NET_Y, TRI_Z[(k + 1) % 3],
    EDGE_PTS
  )
)
const WYE_PATHS: Point[][] = [0, 1, 2].map((k) =>
  straight(WYE_CX, NET_Y, 0, WYE_CX + TRI_X[k], NET_Y, TRI_Z[k], EDGE_PTS)
)
const WIRE_PATHS: Point[][] = [...DELTA_PATHS, ...WYE_PATHS]
const LEAD_INIT: Point[] = Array.from({ length: LEAD_PTS }, () => [0, 0, 0] as Point)

// Resistor bodies: the middle SEG of each wire, as [ax, az, bx, bz] — Δ edges then Y arms.
const RES_ENDS = new Float32Array(6 * 4)
for (let k = 0; k < 3; k++) {
  const lo = (1 - SEG) / 2
  const hi = 1 - lo
  const ex0 = DELTA_CX + TRI_X[k]
  const ez0 = TRI_Z[k]
  const ex1 = DELTA_CX + TRI_X[(k + 1) % 3]
  const ez1 = TRI_Z[(k + 1) % 3]
  RES_ENDS[k * 4] = ex0 + (ex1 - ex0) * lo
  RES_ENDS[k * 4 + 1] = ez0 + (ez1 - ez0) * lo
  RES_ENDS[k * 4 + 2] = ex0 + (ex1 - ex0) * hi
  RES_ENDS[k * 4 + 3] = ez0 + (ez1 - ez0) * hi
  const i = k + 3
  RES_ENDS[i * 4] = WYE_CX + TRI_X[k] * lo
  RES_ENDS[i * 4 + 1] = TRI_Z[k] * lo
  RES_ENDS[i * 4 + 2] = WYE_CX + TRI_X[k] * hi
  RES_ENDS[i * 4 + 3] = TRI_Z[k] * hi
}

const X_AXIS = new THREE.Vector3(1, 0, 0)
const p0 = new THREE.Vector3()
const p1 = new THREE.Vector3()
const dir = new THREE.Vector3()
const unit = new THREE.Vector3()

/** Seat a resistor body on wire `i`; thinner for larger R, filament thicker with power. */
function placeResistor(g: THREE.Group | null, i: number, R: number, P: number) {
  if (g) {
    p0.set(RES_ENDS[i * 4], NET_Y, RES_ENDS[i * 4 + 1])
    p1.set(RES_ENDS[i * 4 + 2], NET_Y, RES_ENDS[i * 4 + 3])
    dir.subVectors(p1, p0)
    g.position.addVectors(p0, p1).multiplyScalar(0.5)
    g.quaternion.setFromUnitVectors(X_AXIS, unit.copy(dir).normalize())
    const r = radiusFor(R)
    g.scale.set(dir.length(), r, r)
  }
  setResistorHeat(g, Math.sqrt(P / P_MAX) * 1.6)
}

// Probe walk: two tips one terminal apart, stepping A→B→C→A around both networks at once.
const CYCLE = 4.8 // s for a full lap
const DWELL = 0.72 // fraction of each step the probe rests on a pair
const easeInOutCubic = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2)

const edgeI = new Float32Array(3) // Δ branch currents, signed along each path [A]
const armI = new Float32Array(3) // Y branch currents, signed outward from the star point [A]
const flows = new Float32Array(10) // 3 Δ edges, 3 Y arms, 4 probe leads

function DeltaWyeScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const leadBufs = useMemo(
    () => [0, 1, 2, 3].map(() => new Float32Array(LEAD_PTS * 3)),
    []
  )
  const wireRefs = useRef<any[]>([]) // 3 Δ edges, then 3 Y arms
  const leadRefs = useRef<any[]>([]) // Δ in, Δ out, Y in, Y out
  const resRefs = useRef<(THREE.Group | null)[]>([])
  const tipRefs = useRef<(THREE.Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([]) // 6 resistor values, 2 meter readouts
  const labelledFor = useRef<unknown>(null)
  const shownPair = useRef(-1)

  useFrame(({ clock }) => {
    const sum = solve(valuesRef.current)

    // Where the probe is: tip 1 walks terminal k → k+1, tip 2 stays one terminal ahead
    const phase = ((clock.getElapsedTime() % CYCLE) / CYCLE) * 3
    const k = Math.floor(phase)
    const step = phase - k
    const u = step < DWELL ? 0 : easeInOutCubic((step - DWELL) / (1 - DWELL))
    const pair = u < 0.5 ? k : (k + 1) % 3

    // Δ: the test current splits between the direct edge and the two in series with it
    const direct = (sum - EDGE[pair]) / sum
    const around = EDGE[pair] / sum
    edgeI[pair] = I_TEST * direct
    edgeI[(pair + 1) % 3] = -I_TEST * around
    edgeI[(pair + 2) % 3] = -I_TEST * around
    // Y: the whole test current runs in one arm and out the other; the third arm is dead
    armI[pair] = -I_TEST
    armI[(pair + 1) % 3] = I_TEST
    armI[(pair + 2) % 3] = 0

    for (let i = 0; i < 6; i++) {
      const delta = i < 3
      const I = delta ? edgeI[i] : armI[i - 3]
      const R = delta ? EDGE[i] : ARM[i - 3]
      const wire = wireRefs.current[i]
      if (wire) {
        ;(wire.material as LineMaterial).linewidth = widthFor(I)
        flows[i] = I * 3
      }
      placeResistor(resRefs.current[i], i, R, I * I * R)
    }

    // Probe tips slide between terminals; the leads follow them down from each meter
    for (let n = 0; n < 4; n++) {
      const wye = n > 1
      const cx = wye ? WYE_CX : DELTA_CX
      const from = (n % 2 === 0 ? k : k + 1) % 3
      const to = (from + 1) % 3
      const tx = cx + TRI_X[from] + (TRI_X[to] - TRI_X[from]) * u
      const tz = TRI_Z[from] + (TRI_Z[to] - TRI_Z[from]) * u
      tipRefs.current[n]?.position.set(tx, NET_Y + 0.02, tz)
      const buf = leadBufs[n]
      for (let i = 0; i < LEAD_PTS; i++) {
        const f = i / (LEAD_PTS - 1)
        buf[i * 3] = cx + (tx - cx) * f
        buf[i * 3 + 1] = METER_Y + (NET_Y - METER_Y) * f
        buf[i * 3 + 2] = tz * f // the meter sits at z = 0 above its network
      }
      leadRefs.current[n]?.geometry.setPositions(buf)
      // Lead points meter → terminal, so the entry lead flows forward and the exit lead back
      flows[6 + n] = n % 2 === 0 ? I_TEST * 3 : -I_TEST * 3
    }

    if (labelledFor.current !== valuesRef.current && textRefs.current[5]) {
      labelledFor.current = valuesRef.current
      for (let i = 0; i < 3; i++) {
        const edgeEl = textRefs.current[i]
        if (edgeEl) edgeEl.textContent = `${EDGE_NAMES[i]} ${fmtR(EDGE[i])}`
        const armEl = textRefs.current[i + 3]
        if (armEl) armEl.textContent = `${ARM_NAMES[i]} ${fmtR(ARM[i])}`
      }
      shownPair.current = -1
    }
    if (shownPair.current !== pair && textRefs.current[7]) {
      shownPair.current = pair
      const name = PAIR_NAMES[pair]
      textRefs.current[6]!.textContent = `Δ · R(${name}) = ${fmtR(PAIR_D[pair])}`
      textRefs.current[7]!.textContent = `Y · R(${name}) = ${fmtR(PAIR_Y[pair])}`
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  // Faint outline around each network: from outside, only the three terminals exist
  const boxes = useMemo<Point[][]>(
    () => [DELTA_CX, WYE_CX].map((cx): Point[] => [
      [cx - 0.95, NET_Y, -1.0], [cx + 0.95, NET_Y, -1.0],
      [cx + 0.95, NET_Y, 0.7], [cx - 0.95, NET_Y, 0.7], [cx - 0.95, NET_Y, -1.0],
    ]),
    []
  )

  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />

      {boxes.map((pts, i) => (
        <Line key={i} points={pts} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      ))}

      {WIRE_PATHS.map((pts, i) => (
        <TrailLine
          key={i}
          ref={(line) => { wireRefs.current[i] = line }}
          points={pts}
          color={c.current}
          fade={false}
          flowSpeed={() => flows[i]}
        />
      ))}
      {[0, 1, 2, 3].map((n) => (
        <TrailLine
          key={n}
          ref={(line) => { leadRefs.current[n] = line }}
          points={LEAD_INIT}
          color={c.current}
          fade={false}
          lineWidth={1.8}
          flowSpeed={() => flows[6 + n]}
        />
      ))}

      {[0, 1, 2, 3, 4, 5].map((i) => (
        <Resistor key={i} ref={(g) => { resRefs.current[i] = g }} />
      ))}

      {/* The three shared terminals, on both networks, plus the Y star point */}
      {[DELTA_CX, WYE_CX].map((cx) =>
        TRI_X.map((x, i) => (
          <group key={`${cx}-${i}`} position={[cx + x, NET_Y, TRI_Z[i]]}>
            <Marker color={c.marker} size={0.055} />
          </group>
        ))
      )}
      <group position={[WYE_CX, NET_Y, 0]}>
        <Marker color={c.resistance} size={0.05} />
      </group>
      {[0, 1, 2, 3].map((n) => (
        <Marker key={n} ref={(g) => { tipRefs.current[n] = g }} color={c.current} size={0.075} />
      ))}
      {[DELTA_CX, WYE_CX].map((cx) => (
        <group key={cx} position={[cx, METER_Y, 0]}>
          <Marker color={c.marker} size={0.07} />
        </group>
      ))}

      {/* Resistance readouts: positions are fixed, only the numbers move */}
      {[0, 1, 2].map((i) => (
        <Html
          key={`d${i}`}
          position={[
            DELTA_CX + ((TRI_X[i] + TRI_X[(i + 1) % 3]) / 2) * 1.5,
            NET_Y + 0.22,
            ((TRI_Z[i] + TRI_Z[(i + 1) % 3]) / 2) * 1.5,
          ]}
          center
        >
          <span ref={(el) => { textRefs.current[i] = el }} className={labelClass} />
        </Html>
      ))}
      {[0, 1, 2].map((i) => (
        <Html key={`y${i}`} position={[WYE_CX + TRI_X[i] * 0.68, NET_Y + 0.24, TRI_Z[i] * 0.68]} center>
          <span ref={(el) => { textRefs.current[i + 3] = el }} className={labelClass} />
        </Html>
      ))}
      {[DELTA_CX, WYE_CX].map((cx, n) => (
        <Html key={cx} position={[cx, METER_Y + 0.28, 0]} center>
          <span ref={(el) => { textRefs.current[6 + n] = el }} className={labelClass} />
        </Html>
      ))}

      {/* Terminal letters and network names never change */}
      {[DELTA_CX, WYE_CX].map((cx) =>
        ['A', 'B', 'C'].map((name, i) => (
          <Html key={`${cx}${name}`} position={[cx + TRI_X[i] * 1.38, NET_Y + 0.08, TRI_Z[i] * 1.38]} center>
            <span className={labelClass}>{name}</span>
          </Html>
        ))
      )}
      <Html position={[DELTA_CX, NET_Y - 0.12, 1.15]} center>
        <span className={captionClass}>Δ</span>
      </Html>
      <Html position={[WYE_CX, NET_Y - 0.12, 1.15]} center>
        <span className={captionClass}>Y</span>
      </Html>
    </>
  )
}

// ── 2D: both schematics flat, with every resistance and all three terminal pairs listed ──
const D2_CX = -2.95
const W2_CX = -0.95
const C2_Y = 0.3
const T2_X = [0, 0.693, -0.693]
const T2_Y = [0.8, -0.4, -0.4]
const vx2 = (cx: number, i: number) => cx + T2_X[i]
const vy2 = (i: number) => C2_Y + T2_Y[i]

const WIRE_2D: Point[][] = [
  ...[0, 1, 2].map((k): Point[] => [
    [vx2(D2_CX, k), vy2(k), 0],
    [vx2(D2_CX, (k + 1) % 3), vy2((k + 1) % 3), 0],
  ]),
  ...[0, 1, 2].map((k): Point[] => [
    [W2_CX, C2_Y, 0],
    [vx2(W2_CX, k), vy2(k), 0],
  ]),
]
const TABLE_X = 2.2
const ROW_Y = [0.55, 0.05, -0.45]

function DeltaWye2DScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const wireRefs = useRef<any[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([]) // 3 Δ, 3 Y, 3 rows, 1 sum
  const labelledFor = useRef<unknown>(null)

  useFrame(() => {
    const sum = solve(valuesRef.current)
    for (let i = 0; i < 6; i++) {
      const wire = wireRefs.current[i]
      if (wire) {
        // Same convention as the tubes in 3D: a larger resistance draws thinner
        ;(wire.material as LineMaterial).linewidth = 1.2 + 3 * Math.sqrt(R_MIN / Math.max(i < 3 ? EDGE[i] : ARM[i - 3], 0.2))
      }
    }
    if (labelledFor.current === valuesRef.current || !textRefs.current[9]) return
    labelledFor.current = valuesRef.current
    for (let i = 0; i < 3; i++) {
      const edgeEl = textRefs.current[i]
      if (edgeEl) edgeEl.textContent = `${EDGE_NAMES[i]} ${fmtR(EDGE[i])}`
      const armEl = textRefs.current[i + 3]
      if (armEl) armEl.textContent = `${ARM_NAMES[i]} ${fmtR(ARM[i])}`
      const rowEl = textRefs.current[i + 6]
      if (rowEl) rowEl.textContent = `${PAIR_NAMES[i]}   Δ ${fmtR(PAIR_D[i])}   =   Y ${fmtR(PAIR_Y[i])}`
    }
    textRefs.current[9]!.textContent = `Σ = Rab + Rbc + Rca = ${fmtR(sum)}`
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  const ko = locale === 'ko'

  return (
    <>
      {WIRE_2D.map((pts, i) => (
        <Line
          key={i}
          ref={(line) => { wireRefs.current[i] = line }}
          points={pts}
          lineWidth={2.5}
          color={c.resistance}
        />
      ))}

      {[D2_CX, W2_CX].map((cx) =>
        [0, 1, 2].map((i) => (
          <group key={`${cx}-${i}`} position={[vx2(cx, i), vy2(i), 0.01]}>
            <Marker color={c.marker} size={0.06} />
          </group>
        ))
      )}
      <group position={[W2_CX, C2_Y, 0.01]}>
        <Marker color={c.resistance} size={0.05} />
      </group>

      {[D2_CX, W2_CX].map((cx) =>
        ['A', 'B', 'C'].map((name, i) => (
          <Html key={`${cx}${name}`} position={[cx + T2_X[i] * 1.35, C2_Y + T2_Y[i] * 1.35, 0]} center>
            <span className={labelClass}>{name}</span>
          </Html>
        ))
      )}

      {[0, 1, 2].map((i) => (
        <Html
          key={`d${i}`}
          position={[
            D2_CX + ((T2_X[i] + T2_X[(i + 1) % 3]) / 2) * 1.45,
            C2_Y + ((T2_Y[i] + T2_Y[(i + 1) % 3]) / 2) * 1.45,
            0,
          ]}
          center
        >
          <span ref={(el) => { textRefs.current[i] = el }} className={labelClass} />
        </Html>
      ))}
      {[0, 1, 2].map((i) => (
        <Html key={`y${i}`} position={[W2_CX + T2_X[i] * 0.62, C2_Y + T2_Y[i] * 0.62 + 0.18, 0]} center>
          <span ref={(el) => { textRefs.current[i + 3] = el }} className={labelClass} />
        </Html>
      ))}

      <Html position={[D2_CX, C2_Y - 1.5, 0]} center>
        <span className={captionClass}>Δ</span>
      </Html>
      <Html position={[W2_CX, C2_Y - 1.5, 0]} center>
        <span className={captionClass}>Y</span>
      </Html>
      <Html position={[(D2_CX + W2_CX) / 2, C2_Y - 1.85, 0]} center>
        <span ref={(el) => { textRefs.current[9] = el }} className={labelClass} />
      </Html>

      <Html position={[TABLE_X, 1.15, 0]} center>
        <span className={captionClass}>
          {ko ? '단자쌍에서 본 저항' : 'Resistance seen at each terminal pair'}
        </span>
      </Html>
      {ROW_Y.map((y, i) => (
        <Html key={y} position={[TABLE_X, y, 0]} center>
          <span ref={(el) => { textRefs.current[i + 6] = el }} className={labelClass} />
        </Html>
      ))}
      <Html position={[TABLE_X, -1.1, 0]} center>
        <span className={captionClass}>
          {ko ? '세 저항이 같으면 RΔ = 3 · RY' : 'All three equal: RΔ = 3 · RY'}
        </span>
      </Html>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'delta-wye',
  title: { en: 'Δ–Y Transformation', ko: 'Δ–Y 변환' },
  subject: 'circuits',
  order: 20,
  description: {
    en: `## Δ–Y Transformation

**A three-terminal triangle and a three-terminal star can be swapped freely: from outside the terminals A, B, C, nothing can tell them apart.**

Two networks, one dashed box each, the same three terminals.
- Cyan liquid lines: the probe's 1 A test current. Width and flow follow the current, so you can see it split in the Δ and see the third Y arm sit dead
- Glass tubes: the resistors, thinner for larger R; the amber filament thickens with the power that resistor turns into heat
- Bright dots: the terminals A, B, C. The cyan dots are the probe, walking A→B→C→A so every pair gets checked

**Δ→Y**: Ra = (Rab × Rca) ÷ Σ, Rb = (Rab × Rbc) ÷ Σ, Rc = (Rbc × Rca) ÷ Σ, with Σ = Rab + Rbc + Rca
**Y→Δ**: Rab = (Ra × Rb + Rb × Rc + Rc × Ra) ÷ Rc, and the same pattern for Rbc (÷ Ra) and Rca (÷ Rb)
**Balanced**: all three equal gives RΔ = 3 × RY

With the defaults Rab = 10 Ω, Rbc = 20 Ω, Rca = 30 Ω the star comes out Ra = 5 Ω, Rb = 3.33 Ω, Rc = 10 Ω, and both meters read 8.33 Ω at A–B, 13.33 Ω at B–C, 15.00 Ω at C–A. The two readouts never disagree, whatever you do to the sliders. That is what makes the swap legal, and it is how a bridge network that is neither series nor parallel becomes solvable: replace one Δ with its Y and the rest collapses.

Set all three sliders to 30 Ω to see the balanced case, where every Y arm is 10 Ω.

In 2D: both schematics flat, with all six resistances and all three terminal pairs side by side.`,
    ko: `## Δ–Y 변환

**세 단자를 가진 삼각형과 세 단자를 가진 성형은 서로 바꿔 끼울 수 있습니다. 단자 A, B, C 바깥에서는 둘을 구별할 방법이 없습니다.**

점선 상자 안에 회로가 하나씩, 단자는 똑같이 셋입니다.
- 시안 액체 선: 탐침이 흘리는 1 A 시험 전류. 굵기와 흐름이 전류를 따라가므로, Δ에서 전류가 갈라지는 것과 Y의 나머지 한 가지가 죽어 있는 것이 보입니다
- 유리관: 저항. R이 클수록 가늘고, 열로 바뀌는 전력이 클수록 안쪽 앰버 필라멘트가 굵어집니다
- 밝은 점: 단자 A, B, C. 시안 점은 탐침이며 A→B→C→A로 돌면서 모든 쌍을 확인합니다

**Δ→Y**: Ra = (Rab × Rca) ÷ Σ, Rb = (Rab × Rbc) ÷ Σ, Rc = (Rbc × Rca) ÷ Σ, 여기서 Σ = Rab + Rbc + Rca
**Y→Δ**: Rab = (Ra × Rb + Rb × Rc + Rc × Ra) ÷ Rc, Rbc는 ÷ Ra, Rca는 ÷ Rb로 같은 꼴입니다
**평형**: 세 저항이 같으면 RΔ = 3 × RY

기본값 Rab = 10 Ω, Rbc = 20 Ω, Rca = 30 Ω이면 성형은 Ra = 5 Ω, Rb = 3.33 Ω, Rc = 10 Ω이 되고, 두 계기는 A–B에서 8.33 Ω, B–C에서 13.33 Ω, C–A에서 15.00 Ω으로 같은 값을 읽습니다. 슬라이더를 어떻게 움직여도 두 값은 어긋나지 않습니다. 그래서 교체가 허용되고, 직렬도 병렬도 아닌 브리지 회로가 풀립니다. Δ 하나를 Y로 바꾸면 나머지가 직렬·병렬로 정리됩니다.

슬라이더 셋을 모두 30 Ω으로 맞추면 평형인 경우가 됩니다. 이때 Y의 각 가지는 10 Ω입니다.

2D: 두 회로도를 평면에 펼치고, 여섯 개의 저항값과 세 단자쌍의 저항을 나란히 보여줍니다.`,
  },
  sliderParams: [
    { key: 'Rab', label: { en: 'Δ side A–B (Rab)', ko: 'Δ 변 A–B (Rab)' }, unit: 'Ω', min: R_MIN, max: R_MAX, step: 1, defaultValue: 10 },
    { key: 'Rbc', label: { en: 'Δ side B–C (Rbc)', ko: 'Δ 변 B–C (Rbc)' }, unit: 'Ω', min: R_MIN, max: R_MAX, step: 1, defaultValue: 20 },
    { key: 'Rca', label: { en: 'Δ side C–A (Rca)', ko: 'Δ 변 C–A (Rca)' }, unit: 'Ω', min: R_MIN, max: R_MAX, step: 1, defaultValue: 30 },
  ],
  Scene: DeltaWyeScene,
  Scene2D: DeltaWye2DScene,
}

export default concept
