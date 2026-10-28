import { useRef, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import type { Group } from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import { useLocaleStore } from '../../lib/i18n'
import {
  setLineOpacity, setOpacity, useStoryReveal,
} from '../../components/scene/storyReveal'
import type { ConceptMeta } from '../../types/concept'

/*
 * Pure physics of the magnetic force, the two faces of one law.
 *   moving charge : F = q·v×B,  magnitude q·v·B·sinθ, always perpendicular to both v and B
 *   circular path : r = m·v ÷ (q·B),  T = 2π·m ÷ (q·B)   (speed v never changes, so no work is done)
 *   straight wire : F = B·I·L·sinθ,  θ = angle between the current and the field
 * Constants: electron charge magnitude q = 1.602176634e-19 C, mass m = 9.1093837015e-31 kg.
 *
 * Worked example, verified with node -e:
 *   B = 0.5 T, I = 10 A, L = 20 cm, θ = 90°  ->  F = 0.5 × 10 × 0.2 × 1    = 1.0000 N
 *   same wire at θ = 30°                     ->  F = 0.5 × 10 × 0.2 × 0.5  = 0.5000 N
 *   same wire at θ = 0°                      ->  F = 0                     = 0.0000 N
 *   electron at v = 1e6 m÷s, B = 0.1 T       ->  F = q·v·B = 1.602e-14 N
 *   electron at v = 1e6 m÷s, B = 0.01 T      ->  r = m·v ÷ (q·B) = 5.686e-4 m
 *   over the slider range at v = 1e6 m÷s:
 *     B = 0.25 T -> r = 22.74 µm, T = 142.9 ps, F = 40.1 fN
 *     B = 0.50 T -> r = 11.37 µm, T =  71.4 ps, F = 80.1 fN
 *     B = 1.00 T -> r =  5.69 µm, T =  35.7 ps, F = 160.2 fN
 *   and r·(q·B ÷ m) = 1.00e6 m÷s at every B, the check that the speed is untouched.
 * One module-scope record holds the result so no frame allocates.
 */
const QE = 1.602176634e-19 // elementary charge [C]
const ME = 9.1093837015e-31 // electron mass [kg]
const V_CHARGE = 1e6 // the charge's fixed speed [m/s]

const S = {
  B: 0.5,
  I: 10,
  L: 0.2,
  thetaDeg: 90,
  theta: Math.PI / 2,
  sinT: 1,
  F: 1,
  r: 1.137e-5,
  T: 7.14e-11,
  Fq: 8.011e-14,
}

function solve(v: Record<string, number>) {
  const B = v.B_T ?? 0.5
  const I = v.I_A ?? 10
  const L = (v.L_cm ?? 20) / 100 // cm -> m
  const thetaDeg = v.theta_deg ?? 90
  const theta = (thetaDeg * Math.PI) / 180
  S.B = B
  S.I = I
  S.L = L
  S.thetaDeg = thetaDeg
  S.theta = theta
  S.sinT = Math.sin(theta)
  S.F = B * I * L * S.sinT
  S.r = (ME * V_CHARGE) / (QE * B)
  S.T = (2 * Math.PI * ME) / (QE * B)
  S.Fq = QE * V_CHARGE * B
  return S
}

function useSliderValues() {
  const valuesRef = useRef<Record<string, number>>(useSliderStore.getState().values)
  useEffect(
    () => useSliderStore.subscribe((s) => s.values, (v) => { valuesRef.current = v }),
    []
  )
  return valuesRef
}

const fmtN = (f: number) => (f >= 10 ? `${f.toFixed(1)} N` : `${f.toFixed(2)} N`)
// Scene labels read in the foreground color; only current (cyan) and power (amber) tint their text
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const smallClass = 'pointer-events-none whitespace-nowrap text-[10px] font-medium tabular-nums text-foreground/70'

const L10N = {
  en: {
    noWork: 'F ⟂ v always, so the speed never changes — the magnetic force does no work',
    turns: 'it only turns the velocity',
    axis: 'angle θ between I and B',
    peak: 'θ = 90°, force at its largest',
    zero: 'θ = 0°, wire along B, no force',
  },
  ko: {
    noWork: 'F는 언제나 v에 수직이므로 속력은 변하지 않습니다 — 자기력은 일을 하지 않습니다',
    turns: '속도의 방향만 바꿉니다',
    axis: 'I와 B 사이의 각 θ',
    peak: 'θ = 90°, 힘이 가장 큽니다',
    zero: 'θ = 0°, 도체가 B와 나란하면 힘이 없습니다',
  },
} satisfies Record<string, Record<string, string>>

// ── Narration: the scene is two faces of one law, so it is revealed one face at a time ──
const REV_CHARGE = 0 // the orbit, the charge, its v and F arrows, and the r / T / F readouts
const REV_NOWORK = 1 // the two lines that spell out why the speed cannot change
const REV_WIRE = 2 // the current, the force on it, and the 2D sine curve
const REV_THETA = 3 // the angle arc, its readout, and the 2D plot's two angle notes
const REV_COUNT = 4

function revealTargets(step: number, out: Float32Array) {
  out[REV_CHARGE] = step >= 1 ? 1 : 0
  out[REV_NOWORK] = step >= 4 ? 1 : 0
  out[REV_WIRE] = step >= 5 ? 1 : 0
  out[REV_THETA] = step >= 7 ? 1 : 0
}

// ── Geometry, shared by both scenes ──
// The field points along −Z, into the screen, the same field over the whole scene. The charge
// circles in the X–Y plane, perpendicular to it. The wire tilts in the X–Z plane, so the angle θ
// it makes with the field is a real angle on screen and the force it feels is always straight up.
const ORBIT_CX = -1.75
const ORBIT_CY = -0.05
const B_REF = 0.5 // T, the worked example's field
const R_AT_REF = 0.78 // scene units drawn for the orbit radius at B_REF
const OMEGA_AT_REF = 1.1 // on-screen rad/s at B_REF, slowed hard from the real 2π ÷ T
const rOf = (B: number) => (R_AT_REF * B_REF) / B // r ∝ 1 ÷ B, exactly as the formula says
const omegaOf = (B: number) => (OMEGA_AT_REF * B) / B_REF // ω ∝ B, so r·ω stays constant

const WX = 0.5 // wire midpoint, where the three directions meet
const WY = -0.3
const L_REF_CM = 20
const HALF_AT_REF = 0.5 // drawn half-length at L_REF_CM
const halfOf = (Lcm: number) => (HALF_AT_REF * Lcm) / L_REF_CM
const F_SCALE = 0.55 // scene units per newton, linear against the worked example
const F_FULL = 1.4 // the force arrow clips here while its readout keeps climbing
const FLOW_REF = 2.4 // sheen bands per second at I = 10 A
// The two arrows at the charge are the payoff, so they are drawn as long as the orbit allows:
// a fixed length at the wide orbits, shrinking with the radius once the circle gets tight, so
// the centripetal arrow never shoots past the centre.
const ARROW_V = 0.62 // drawn length of the velocity arrow at the charge
const ARROW_F = 0.5 // drawn length of the force arrow at the charge
const clip = (x: number) => (x > F_FULL ? F_FULL : x)

// The uniform field: lines along −Z with a cone at the far end. The first one runs through the
// wire midpoint, so it doubles as the B leg of the B–I–F triad.
// A negative z shifts a point to the right on screen (the camera sits at +X +Z), so the field's
// depth is kept shallow and its rightmost line well inside the floating description card.
const B_Z0 = 0.6
const B_Z1 = -0.8
const B_AT: [number, number][] = [
  [WX, WY],
  [-2.0, 0.95],
  [-0.95, 0.95],
  [-2.0, -1.0],
  [-0.95, -1.0],
  [1.25, 0.5],
]
const bLine = (x: number, y: number): [number, number, number][] => [[x, y, B_Z0], [x, y, B_Z1]]

// ── Scratch buffers, filled and handed straight to a geometry so no frame allocates ──
const SEG = new Float32Array(6)
const ARC_N = 32
const ARC = new Float32Array(ARC_N * 3)
const CIRC_N = 72
const CIRC = new Float32Array(CIRC_N * 3)
const WIRE_N = 24
const WIRE = new Float32Array(WIRE_N * 3)
const TRAIL_N = 170
const TRAIL = new Float32Array(TRAIL_N * 3)
const TRAIL_SPAN = 1.5 * Math.PI // how much of the orbit the trail remembers

const SEG_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]
const zeros = (n: number): [number, number, number][] =>
  Array.from({ length: n }, (): [number, number, number] => [0, 0, 0])
const ARC_INIT = zeros(ARC_N)
const CIRC_INIT = zeros(CIRC_N)
const WIRE_INIT = zeros(WIRE_N)
const TRAIL_INIT = zeros(TRAIL_N)

function setSeg(line: any, x0: number, y0: number, z0: number, x1: number, y1: number, z1: number) {
  SEG[0] = x0; SEG[1] = y0; SEG[2] = z0
  SEG[3] = x1; SEG[4] = y1; SEG[5] = z1
  line?.geometry.setPositions(SEG)
  line?.computeLineDistances()
}

const put = (g: Group | null | undefined, x: number, y: number, z: number) => {
  g?.position.set(x, y, z)
}

// ── 3D: one field, two faces of the same law ──
function LorentzScene() {
  const c = usePhysicsColors()
  const t10n = L10N[useLocaleStore((s) => s.locale)]
  const valuesRef = useSliderValues()
  const circleRef = useRef<any>(null)
  const radiusRef = useRef<any>(null)
  const trailRef = useRef<any>(null)
  const wireRef = useRef<any>(null)
  const arcRef = useRef<any>(null)
  const wireForceRef = useRef<any>(null)
  const chargeDot = useRef<Group>(null)
  const vArrow = useRef<any>(null)
  const vCone = useRef<Group>(null)
  const fArrow = useRef<any>(null)
  const fCone = useRef<Group>(null)
  const wireHead = useRef<Group>(null)
  const wireForceCone = useRef<Group>(null)
  const iTag = useRef<Group>(null)
  const fTag = useRef<Group>(null)
  const thetaTag = useRef<Group>(null)
  const vTag = useRef<Group>(null)
  const chargeFTag = useRef<Group>(null)
  // 0 wire force readout, 1 angle readout, 2 orbit radius readout, 3 charge force readout,
  // then the bare glyphs and notes, which need refs of their own because drei's <Html> is a DOM
  // overlay: scaling or hiding its parent group does nothing to the span.
  // 4 v, 5 F at the charge, 6 and 7 the no-work notes, 8 I at the wire head, 9 θ on the arc
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<Record<string, number> | null>(null)
  const flow = useRef(FLOW_REF)
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(({ clock }) => {
    const time = clock.getElapsedTime()
    const s = solve(valuesRef.current)

    // ── the moving charge: a circle whose radius shrinks as B grows, run at ω ∝ B ──
    const r = rOf(s.B)
    const a = omegaOf(s.B) * time
    const cx = ORBIT_CX + r * Math.cos(a)
    const cy = ORBIT_CY + r * Math.sin(a)

    for (let i = 0; i < CIRC_N; i++) {
      const u = (i / (CIRC_N - 1)) * 2 * Math.PI
      CIRC[i * 3] = ORBIT_CX + r * Math.cos(u)
      CIRC[i * 3 + 1] = ORBIT_CY + r * Math.sin(u)
      CIRC[i * 3 + 2] = 0
    }
    circleRef.current?.geometry.setPositions(CIRC)
    circleRef.current?.computeLineDistances()

    for (let i = 0; i < TRAIL_N; i++) {
      const u = a - (1 - i / (TRAIL_N - 1)) * TRAIL_SPAN
      TRAIL[i * 3] = ORBIT_CX + r * Math.cos(u)
      TRAIL[i * 3 + 1] = ORBIT_CY + r * Math.sin(u)
      TRAIL[i * 3 + 2] = 0
    }
    trailRef.current?.geometry.setPositions(TRAIL)

    setSeg(radiusRef.current, ORBIT_CX, ORBIT_CY, 0, cx, cy, 0)
    put(chargeDot.current, cx, cy, 0)

    // v is the tangent, F = q·v×B points straight at the centre. Two arrows at a right angle,
    // every frame, at every radius: that is the whole point of the cross product.
    const aV = Math.min(ARROW_V, 0.8 * r)
    const aF = Math.min(ARROW_F, 0.62 * r)
    const vx = -Math.sin(a)
    const vy = Math.cos(a)
    setSeg(vArrow.current, cx, cy, 0, cx + aV * vx, cy + aV * vy, 0)
    put(vCone.current, cx + (aV + 0.07) * vx, cy + (aV + 0.07) * vy, 0)
    if (vCone.current) vCone.current.rotation.z = a
    const fx = -Math.cos(a)
    const fy = -Math.sin(a)
    setSeg(fArrow.current, cx, cy, 0, cx + aF * fx, cy + aF * fy, 0)
    put(fCone.current, cx + (aF + 0.07) * fx, cy + (aF + 0.07) * fy, 0)
    if (fCone.current) fCone.current.rotation.z = a + Math.PI / 2
    // Each tag rides beside its own arrow, never beyond its head, so neither label can drift
    // across the circle when the orbit tightens.
    put(vTag.current, cx + (aV + 0.24) * vx, cy + (aV + 0.24) * vy, 0)
    put(
      chargeFTag.current,
      cx + 0.6 * aF * fx - 0.3 * vx,
      cy + 0.6 * aF * fy - 0.3 * vy,
      0
    )

    // ── the current-carrying wire: the same law, now summed over a whole current ──
    // The wire lies along û = (sinθ, 0, −cosθ), so θ = 0 lays it along the field and θ = 90°
    // puts it square across it. F = I·L×B then always points along +Y, with length B·I·L·sinθ.
    const ux = Math.sin(s.theta)
    const uz = -Math.cos(s.theta)
    const half = halfOf(s.L * 100)
    for (let i = 0; i < WIRE_N; i++) {
      const p = -1 + (2 * i) / (WIRE_N - 1)
      WIRE[i * 3] = WX + ux * half * p
      WIRE[i * 3 + 1] = WY
      WIRE[i * 3 + 2] = uz * half * p
    }
    wireRef.current?.geometry.setPositions(WIRE)
    flow.current = (FLOW_REF * s.I) / 10

    put(wireHead.current, WX + ux * (half + 0.09), WY, uz * (half + 0.09))
    if (wireHead.current) wireHead.current.rotation.y = Math.PI / 2 - s.theta
    put(iTag.current, WX + ux * (half + 0.34), WY + 0.22, uz * (half + 0.34))

    const fLen = clip(F_SCALE * s.F)
    setSeg(wireForceRef.current, WX, WY, 0, WX, WY + fLen, 0)
    put(wireForceCone.current, WX, WY + fLen + 0.07, 0)
    put(fTag.current, WX, WY + Math.max(fLen, 0.45) + 0.3, 0)

    // The arc from the field direction round to the wire: θ itself, collapsing to nothing at 0°
    for (let i = 0; i < ARC_N; i++) {
      const u = (s.theta * i) / (ARC_N - 1)
      ARC[i * 3] = WX + 0.3 * Math.sin(u)
      ARC[i * 3 + 1] = WY
      ARC[i * 3 + 2] = -0.3 * Math.cos(u)
    }
    arcRef.current?.geometry.setPositions(ARC)
    arcRef.current?.computeLineDistances()
    // Only the bare θ glyph rides the arc; its numbers live on a fixed line under the wire, well
    // clear of the I label at the wire head.
    const hu = s.theta / 2
    put(thetaTag.current, WX + 0.45 * Math.sin(hu), WY - 0.24, -0.45 * Math.cos(hu))

    // ── the narration's fades: every frame, since the weights ease between steps ──
    const wCharge = reveal[REV_CHARGE]
    for (const line of [circleRef, radiusRef, trailRef, vArrow, fArrow]) {
      setLineOpacity(line.current, wCharge)
    }
    chargeDot.current?.scale.setScalar(wCharge)
    vCone.current?.scale.setScalar(wCharge)
    fCone.current?.scale.setScalar(wCharge)
    setOpacity(textRefs.current[2], wCharge)
    setOpacity(textRefs.current[3], wCharge)
    setOpacity(textRefs.current[4], wCharge)
    setOpacity(textRefs.current[5], wCharge)
    setOpacity(textRefs.current[6], reveal[REV_NOWORK])
    setOpacity(textRefs.current[7], reveal[REV_NOWORK])

    const wWire = reveal[REV_WIRE]
    setLineOpacity(wireRef.current, wWire)
    setLineOpacity(wireForceRef.current, wWire)
    wireHead.current?.scale.setScalar(wWire)
    // A zero-length arrow still drew a full-size head, which flatly contradicts the step that
    // lays the wire along the field, so the head shrinks away with the arrow.
    wireForceCone.current?.scale.setScalar(wWire * Math.min(1, fLen / 0.12))
    setOpacity(textRefs.current[0], wWire)
    setOpacity(textRefs.current[8], wWire)

    const wTheta = reveal[REV_THETA]
    setLineOpacity(arcRef.current, wTheta)
    setOpacity(textRefs.current[1], wTheta)
    setOpacity(textRefs.current[9], wTheta)

    // Readouts are rebuilt only when the slider record itself changes, never per frame
    if (labelledFor.current !== valuesRef.current && textRefs.current[3]) {
      labelledFor.current = valuesRef.current
      if (textRefs.current[0]) {
        textRefs.current[0]!.textContent = `F = B·I·L·sinθ = ${fmtN(s.F)}`
      }
      if (textRefs.current[1]) {
        textRefs.current[1]!.textContent = `θ = ${s.thetaDeg.toFixed(0)}°, sinθ = ${s.sinT.toFixed(2)}`
      }
      if (textRefs.current[2]) {
        textRefs.current[2]!.textContent =
          `r = m·v ÷ (q·B) = ${(s.r * 1e6).toFixed(2)} µm, T = 2π·m ÷ (q·B) = ${(s.T * 1e12).toFixed(1)} ps`
      }
      textRefs.current[3]!.textContent =
        `F = q·v·B = ${(s.Fq * 1e15).toFixed(1)} fN · v = 1 × 10⁶ m÷s`
    }
  })

  return (
    <>
      {/* Aimed right of centre so the orbit clears the narration panel on the right */}
      <PerspectiveCamera makeDefault position={[3, 2, 5.6]} fov={50} />
      <OrbitControls enablePan={false} target={[0.55, 0, 0]} />

      {/* Background, lights, grid, and bloom come from the shared SceneStage */}
      {/* One uniform field over the whole scene, pointing away from the viewer */}
      {B_AT.map(([x, y], i) => (
        <group key={i}>
          <Line points={bLine(x, y)} lineWidth={1.5} color={c['b-field']} />
          <mesh position={[x, y, B_Z1]} rotation={[-Math.PI / 2, 0, 0]}>
            <coneGeometry args={[0.07, 0.2, 12]} />
            <meshBasicMaterial color={c['b-field']} />
          </mesh>
        </group>
      ))}

      {/* Face one: a free charge, pushed sideways for ever, so it circles */}
      <Line
        ref={circleRef}
        points={CIRC_INIT}
        lineWidth={1}
        color={c.guide}
        dashed
        dashSize={0.1}
        gapSize={0.08}
      />
      <Line
        ref={radiusRef}
        points={SEG_INIT}
        lineWidth={1}
        color={c.guide}
        dashed
        dashSize={0.08}
        gapSize={0.07}
      />
      <TrailLine ref={trailRef} points={TRAIL_INIT} lineWidth={2.5} color={c.voltage} />
      <Line ref={vArrow} points={SEG_INIT} lineWidth={2} color={c.voltage} />
      <group ref={vCone}>
        <mesh>
          <coneGeometry args={[0.06, 0.17, 12]} />
          <meshBasicMaterial color={c.voltage} />
        </mesh>
      </group>
      <Line ref={fArrow} points={SEG_INIT} lineWidth={2} color={c.marker} />
      <group ref={fCone}>
        <mesh>
          <coneGeometry args={[0.06, 0.17, 12]} />
          <meshBasicMaterial color={c.marker} />
        </mesh>
      </group>
      <Marker ref={chargeDot} color={c.marker} size={0.075} />

      {/* Face two: the same push on a whole current, which is how a motor turns */}
      <TrailLine
        ref={wireRef}
        points={WIRE_INIT}
        lineWidth={3}
        color={c.current}
        fade={false}
        flowSpeed={() => flow.current}
      />
      <group ref={wireHead}>
        <mesh rotation={[0, 0, -Math.PI / 2]}>
          <coneGeometry args={[0.07, 0.2, 12]} />
          <meshBasicMaterial color={c.current} />
        </mesh>
      </group>
      <Line ref={wireForceRef} points={SEG_INIT} lineWidth={2.5} color={c.marker} />
      <group ref={wireForceCone}>
        <mesh>
          <coneGeometry args={[0.08, 0.22, 12]} />
          <meshBasicMaterial color={c.marker} />
        </mesh>
      </group>
      <Line
        ref={arcRef}
        points={ARC_INIT}
        lineWidth={1}
        color={c.guide}
        dashed
        dashSize={0.06}
        gapSize={0.05}
      />

      <Html position={[-0.95, 1.17, B_Z1]} center>
        <span className={labelClass}>B</span>
      </Html>
      <group ref={iTag}>
        <Html center>
          <span
            ref={(el) => { textRefs.current[8] = el }}
            className={labelClass}
            style={{ color: c.current }}
          >
            I
          </span>
        </Html>
      </group>
      <group ref={fTag}>
        <Html center>
          <span ref={(el) => { textRefs.current[0] = el }} className={labelClass} />
        </Html>
      </group>
      <group ref={thetaTag}>
        <Html center>
          <span ref={(el) => { textRefs.current[9] = el }} className={smallClass}>θ</span>
        </Html>
      </group>
      <Html position={[WX, WY - 0.82, 0]} center>
        <span ref={(el) => { textRefs.current[1] = el }} className={smallClass} />
      </Html>
      <group ref={vTag}>
        <Html center>
          <span ref={(el) => { textRefs.current[4] = el }} className={smallClass}>v</span>
        </Html>
      </group>
      <group ref={chargeFTag}>
        <Html center>
          <span ref={(el) => { textRefs.current[5] = el }} className={smallClass}>F</span>
        </Html>
      </group>
      <Html position={[ORBIT_CX, 1.88, 0]} center>
        <span ref={(el) => { textRefs.current[2] = el }} className={labelClass} />
      </Html>
      <Html position={[ORBIT_CX, 1.66, 0]} center>
        <span ref={(el) => { textRefs.current[3] = el }} className={smallClass} />
      </Html>
      <Html position={[ORBIT_CX, -1.78, 0]} center>
        <span ref={(el) => { textRefs.current[6] = el }} className={smallClass}>{t10n.noWork}</span>
      </Html>
      <Html position={[ORBIT_CX, -1.98, 0]} center>
        <span ref={(el) => { textRefs.current[7] = el }} className={smallClass}>{t10n.turns}</span>
      </Html>
    </>
  )
}

// ── 2D: the sinθ factor on its own, the force against the angle ──
// The whole curve is B·I·L·sinθ. The marker sits at the angle the slider is set to, so moving
// that slider walks the marker along a fixed sine, and moving B, I or L lifts the whole curve.
const X2_0 = -3.4
const W2 = 6.8
const AX_Y = -1.35
const CURVE_N = 181 // one point per degree of θ, 0° to 180°
const CURVE = new Float32Array(CURVE_N * 3)
const CURVE_INIT = zeros(CURVE_N)
const xOfDeg = (deg: number) => X2_0 + (deg / 180) * W2
const AXIS2: [number, number, number][] = [[X2_0, AX_Y, 0], [X2_0 + W2, AX_Y, 0]]
const TICKS = [0, 90, 180]

function LorentzForce2DScene() {
  const c = usePhysicsColors()
  const t10n = L10N[useLocaleStore((s) => s.locale)]
  const valuesRef = useSliderValues()
  const curveRef = useRef<any>(null)
  const dropRef = useRef<any>(null)
  const peakRef = useRef<any>(null)
  const dot = useRef<Group>(null)
  // 0 force readout, 1 formula readout, 2 the peak note, 3 the zero note
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<Record<string, number> | null>(null)
  const tickXs = useMemo(() => TICKS.map(xOfDeg), [])
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(() => {
    const s = solve(valuesRef.current)
    const peak = s.B * s.I * s.L // the force at θ = 90°

    for (let i = 0; i < CURVE_N; i++) {
      const deg = (i / (CURVE_N - 1)) * 180
      CURVE[i * 3] = xOfDeg(deg)
      CURVE[i * 3 + 1] = AX_Y + clip(F_SCALE * peak * Math.sin((deg * Math.PI) / 180))
      CURVE[i * 3 + 2] = 0
    }
    curveRef.current?.geometry.setPositions(CURVE)

    const x = xOfDeg(s.thetaDeg)
    const y = AX_Y + clip(F_SCALE * s.F)
    dot.current?.position.set(x, y, 0.02)
    setSeg(dropRef.current, x, AX_Y, 0, x, y, 0)
    setSeg(peakRef.current, xOfDeg(90), AX_Y, 0, xOfDeg(90), AX_Y + clip(F_SCALE * peak), 0)

    // The sine factor only means anything once there is a wire to feel it, so the whole plot
    // waits for that step; the axis and its ticks stay, as the frame the curve will arrive in.
    const wWire = reveal[REV_WIRE]
    setLineOpacity(curveRef.current, wWire)
    setLineOpacity(dropRef.current, wWire)
    setLineOpacity(peakRef.current, wWire)
    dot.current?.scale.setScalar(wWire)
    setOpacity(textRefs.current[0], wWire)
    setOpacity(textRefs.current[1], wWire)
    setOpacity(textRefs.current[2], reveal[REV_THETA])
    setOpacity(textRefs.current[3], reveal[REV_THETA])

    if (labelledFor.current !== valuesRef.current && textRefs.current[1]) {
      labelledFor.current = valuesRef.current
      if (textRefs.current[0]) {
        textRefs.current[0]!.textContent =
          `θ = ${s.thetaDeg.toFixed(0)}°, sinθ = ${s.sinT.toFixed(2)}, F = ${fmtN(s.F)}`
      }
      textRefs.current[1]!.textContent =
        `F = B·I·L·sinθ · θ = 90° → ${fmtN(peak)}`
    }
  })

  return (
    <>
      <Line points={AXIS2} lineWidth={1} color={c.guide} />
      <Line
        ref={peakRef}
        points={SEG_INIT}
        lineWidth={1}
        color={c.guide}
        dashed
        dashSize={0.09}
        gapSize={0.07}
      />
      <Line
        ref={dropRef}
        points={SEG_INIT}
        lineWidth={1}
        color={c.guide}
        dashed
        dashSize={0.08}
        gapSize={0.06}
      />
      <TrailLine ref={curveRef} points={CURVE_INIT} lineWidth={3} color={c.marker} />
      <Marker ref={dot} color={c.marker} size={0.07} />

      {TICKS.map((deg, i) => (
        <Html key={deg} position={[tickXs[i], AX_Y - 0.22, 0]} center>
          <span className={smallClass}>{`${deg}°`}</span>
        </Html>
      ))}
      <Html position={[0, AX_Y - 0.52, 0]} center>
        <span className={smallClass}>{t10n.axis}</span>
      </Html>
      <Html position={[0, 1.88, 0]} center>
        <span ref={(el) => { textRefs.current[0] = el }} className={labelClass} />
      </Html>
      <Html position={[0, 1.62, 0]} center>
        <span ref={(el) => { textRefs.current[1] = el }} className={smallClass} />
      </Html>
      <Html position={[xOfDeg(90), 0.72, 0]} center>
        <span ref={(el) => { textRefs.current[2] = el }} className={smallClass}>{t10n.peak}</span>
      </Html>
      <Html position={[X2_0 + 0.55, AX_Y + 0.3, 0]} center>
        <span ref={(el) => { textRefs.current[3] = el }} className={smallClass}>{t10n.zero}</span>
      </Html>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'lorentz-force',
  title: { en: "Lorentz Force & Fleming's Rules", ko: '로렌츠 힘과 플레밍 법칙' },
  subject: 'electromagnetics',
  order: 6,
  description: {
    en: `## Lorentz Force and Fleming's Rules

**A magnetic field never pushes a charge forward or backward — it pushes it sideways, square across both the velocity and the field.**

- The deep cool lines with cones: the uniform field B, pointing away from the viewer. The same field fills the whole scene
- Left, the bright curve: the path of a single charge moving across that field, with its velocity v drawn as the tangent arrow
- Left, the foreground arrow at the charge: the force F, aimed straight at the centre of the circle, always at a right angle to v
- Right, the cyan line: a wire carrying a current I, the sheen running the way the current flows
- Right, the foreground arrow: the force F on that wire, always straight up, square across both I and B
- The faint dashed arc at the wire: the angle θ between the current and the field

**Moving charge**: F = q·v×B, magnitude q·v·B·sinθ
**Circular path**: r = m·v ÷ (q·B), period T = 2π·m ÷ (q·B)
**Current-carrying wire**: F = B·I·L·sinθ

Both lines are the same law. A current is charge in motion, so adding up q·v×B over every carrier in a length L of wire gives I·L×B, and the wire gets pushed. That push is the motor principle, and it is the only reason a motor turns.

The cross product is the part learners get wrong, so watch the two arrows at the charge. They stay at a right angle for ever. Because the force is always perpendicular to the motion, it can never add to the speed: it only bends the path. **The magnetic force does no work.** A charge in a magnetic field goes round in a circle at an unchanging speed, which is exactly what a cyclotron or a mass spectrometer relies on.

**Fleming's left hand** gives the motor direction: thumb for the force, first finger for the field, second finger for the current. **Fleming's right hand** gives the generator direction, with the thumb now the motion you impose and the second finger the current you induce. Same geometry, two uses.

Worked example: B = 0.5 T, I = 10 A, L = 20 cm, θ = 90° give F = 0.5 × 10 × 0.2 = 1.00 N. Tilt the wire to θ = 30° and only sin 30° = 0.5 of it survives, so F = 0.50 N. Lay the wire along the field at θ = 0° and the force disappears completely. For a single electron at v = 1 × 10⁶ m÷s in B = 0.1 T the force is tiny, 1.6 × 10⁻¹⁴ N, yet the electron is so light that in B = 0.01 T it still circles with a radius of only 5.69 × 10⁻⁴ m.

Raise B and two things happen at once: the orbit tightens, because r = m·v ÷ (q·B), and the charge goes round faster, because T = 2π·m ÷ (q·B) shrinks by the same factor. The two cancel, so the speed on the path is untouched — the force did no work. Raise I or L and the force on the wire grows in proportion, and the current's sheen runs faster with I. Pull θ down toward 0° and the force fades to nothing even though B, I and L have not changed at all.

In 2D: the force plotted against θ, which is just the sine factor on its own. The marker sits at the angle you have set. The curve clips at full scale while the readout keeps climbing.`,
    ko: `## 로렌츠 힘과 플레밍 법칙

**자기장은 전하를 앞이나 뒤로 밀지 않습니다. 속도와 자기장 양쪽에 모두 수직인 옆 방향으로 밀어냅니다.**

- 원뿔이 달린 짙은 한색 직선: 보는 사람에게서 멀어지는 방향의 균일 자기장 B. 장면 전체가 같은 자기장입니다
- 왼쪽 밝은 곡선: 그 자기장을 가로질러 움직이는 전하 하나의 경로이고, 접선 화살표가 속도 v입니다
- 왼쪽 전하에 붙은 전경색 화살표: 힘 F. 원의 중심을 똑바로 향하며, 언제나 v와 직각입니다
- 오른쪽 청록색 선: 전류 I가 흐르는 도체. 광택이 흐르는 방향이 전류의 방향입니다
- 오른쪽 전경색 화살표: 그 도체가 받는 힘 F. 언제나 위를 향하고, I와 B 양쪽에 수직입니다
- 도체 쪽의 희미한 점선 호: 전류와 자기장이 이루는 각 θ

**움직이는 전하**: F = q·v×B, 크기는 q·v·B·sinθ
**원운동 반지름**: r = m·v ÷ (q·B), 주기 T = 2π·m ÷ (q·B)
**전류가 흐르는 도체**: F = B·I·L·sinθ

두 식은 같은 법칙입니다. 전류는 움직이는 전하이므로, 길이 L인 도체 안의 모든 전하에 대해 q·v×B를 더하면 I·L×B가 되고, 도체가 밀립니다. 이 밀림이 전동기의 원리이며, 전동기가 도는 이유는 오직 이것뿐입니다.

학습자가 가장 많이 틀리는 부분이 벡터곱이므로, 전하에 달린 두 화살표를 보세요. 둘은 끝까지 직각을 유지합니다. 힘이 항상 운동 방향에 수직이기 때문에 속력을 더해 줄 수는 없고, 경로만 휘게 합니다. **자기력은 일을 하지 않습니다.** 자기장 속의 전하는 속력이 변하지 않는 원운동을 하며, 사이클로트론과 질량분석기가 바로 이 성질에 의존합니다.

**플레밍의 왼손 법칙**은 전동기의 방향을 줍니다. 엄지가 힘, 검지가 자기장, 중지가 전류입니다. **플레밍의 오른손 법칙**은 발전기의 방향을 주며, 이때 엄지는 외부에서 가한 운동이고 중지는 유도된 전류입니다. 기하는 같고 쓰임만 둘입니다.

계산 예: B = 0.5 T, I = 10 A, L = 20 cm, θ = 90°이면 F = 0.5 × 10 × 0.2 = 1.00 N입니다. 도체를 θ = 30°로 기울이면 sin 30° = 0.5만 남아 F = 0.50 N이 됩니다. θ = 0°로 자기장과 나란히 누이면 힘은 완전히 사라집니다. v = 1 × 10⁶ m÷s로 움직이는 전자 하나가 B = 0.1 T에서 받는 힘은 1.6 × 10⁻¹⁴ N으로 아주 작지만, 전자가 워낙 가벼워서 B = 0.01 T에서도 반지름이 5.69 × 10⁻⁴ m밖에 되지 않는 원을 그립니다.

B를 올리면 두 가지가 동시에 일어납니다. r = m·v ÷ (q·B)이므로 궤도가 작아지고, T = 2π·m ÷ (q·B)이므로 같은 비율로 더 빨리 돕니다. 두 효과가 정확히 상쇄되어 경로 위의 속력은 그대로입니다. 힘이 일을 하지 않았기 때문입니다. I나 L을 올리면 도체가 받는 힘이 비례해서 커지고, 전류의 광택도 I에 따라 빨라집니다. θ를 0°쪽으로 내리면 B, I, L이 전혀 변하지 않았는데도 힘이 0으로 사그라듭니다.

2D: θ에 대한 힘의 그래프, 즉 sinθ 인자만 따로 떼어 본 모습입니다. 표시점은 지금 설정한 각에 놓입니다. 곡선은 화면 끝에서 잘리지만 수치는 계속 올라갑니다.`,
  },
  story: [
    {
      set: { B_T: 0.5, I_A: 10, L_cm: 20, theta_deg: 90 },
      text: {
        en: 'These parallel lines are one uniform magnetic field, the same everywhere in the scene, pointing away from you into the screen.',
        ko: '이 평행선들이 균일한 자기장 B입니다. 장면 어디서나 세기가 같고, 보는 사람에게서 멀어지는 방향으로 깔려 있습니다.',
      },
    },
    {
      set: { B_T: 0.5, I_A: 10, L_cm: 20, theta_deg: 90 },
      text: {
        en: 'Drop one charge into it at a million metres per second. The field never pushes it forward or back; it pushes it sideways, square across both the velocity and the field. `F = q·v×B`',
        ko: '여기에 전하 하나를 초속 100만 미터로 던져 넣습니다. 자기장은 전하를 앞으로도 뒤로도 밀지 않습니다. 속도와 자기장 양쪽에 수직인 옆으로만 밉니다. `F = q·v×B`',
      },
    },
    {
      set: { B_T: 0.5, I_A: 10, L_cm: 20, theta_deg: 90 },
      text: {
        en: 'A sideways push that never lets up keeps bending the path until it closes into a circle. At 0.5 T the radius is 11.37 µm and one lap takes 71.4 ps. `r = m·v ÷ (q·B)`',
        ko: '쉬지 않고 옆으로만 미니 경로가 계속 휘어 결국 원으로 닫힙니다. 0.5 T에서 반지름은 11.37 µm, 한 바퀴에 71.4 ps 걸립니다. `r = m·v ÷ (q·B)`',
      },
    },
    {
      set: { B_T: 1, I_A: 10, L_cm: 20, theta_deg: 90 },
      text: {
        en: 'Double the field and the orbit tightens to half, 5.69 µm. But the lap time halves too, down to 35.7 ps, so the speed along the path is still exactly a million metres per second.',
        ko: '자기장을 두 배로 올리면 궤도는 절반인 5.69 µm로 좁아집니다. 그런데 한 바퀴 도는 시간도 절반인 35.7 ps가 되므로, 경로 위의 속력은 여전히 정확히 초속 100만 미터입니다.',
      },
    },
    {
      set: { B_T: 1, I_A: 10, L_cm: 20, theta_deg: 90 },
      text: {
        en: 'It could not come out any other way. The force sits at a right angle to the motion every single frame, so it has no component along the path to speed the charge up. A magnetic field bends, and never does work.',
        ko: '다르게 나올 수가 없습니다. 힘이 매 순간 운동 방향과 직각이라, 경로 방향 성분이 하나도 없어 속력을 올려 줄 수가 없습니다. 자기장은 휘게만 하고, 일은 절대 하지 않습니다.',
      },
    },
    {
      set: { B_T: 0.5, I_A: 10, L_cm: 20, theta_deg: 90 },
      text: {
        en: 'Now swap the single charge for a current. A current is just charge in motion, so adding that same sideways push over every carrier in 20 cm of wire pushes the whole wire: 1.00 N at 0.5 T and 10 A. `F = B·I·L·sinθ`',
        ko: '이제 전하 하나를 전류로 바꿉니다. 전류는 결국 움직이는 전하이니, 20 cm 도체 속 모든 전하가 받는 그 옆 방향 힘을 더하면 도체 전체가 밀립니다. 0.5 T, 10 A에서 1.00 N. `F = B·I·L·sinθ`',
      },
    },
    {
      set: { B_T: 0.5, I_A: 20, L_cm: 20, theta_deg: 90 },
      text: {
        en: 'Push 20 A instead and the force doubles to 2.00 N. This is the entire motor principle: a wire in a field, shoved sideways.',
        ko: '전류를 20 A로 올리면 힘도 두 배인 2.00 N이 됩니다. 전동기의 원리는 이것이 전부입니다. 자기장 속 도체가 옆으로 밀리는 것.',
      },
    },
    {
      set: { B_T: 0.5, I_A: 20, L_cm: 20, theta_deg: 60 },
      text: {
        en: 'The angle matters, though. Tilt the wire to 60° from the field and only sin 60° of it is still crossing the field, so the force drops to 1.73 N.',
        ko: '다만 각이 중요합니다. 도체를 자기장에서 60° 기울이면 자기장을 가로지르는 몫이 sin 60°로 줄어들어, 힘은 1.73 N으로 떨어집니다.',
      },
    },
    {
      set: { B_T: 0.5, I_A: 20, L_cm: 20, theta_deg: 0 },
      text: {
        en: 'Lay it flat along the field at 0° and the force vanishes completely: 0.00 N, with B, I and L every bit as big as before. Cross the field or feel nothing.',
        ko: '자기장과 완전히 나란히 0°로 누이면 힘이 아예 사라집니다. B도 I도 L도 그대로인데 0.00 N. 가로질러야 힘이 생깁니다.',
      },
    },
    {
      set: { B_T: 0.5, I_A: 20, L_cm: 20, theta_deg: 90 },
      text: {
        en: 'Thumb for the force, first finger for the field, second finger for the current: Fleming\'s left hand reads a motor. Switch to the right hand and the same geometry reads a generator. `F = I·L×B`',
        ko: '엄지는 힘, 검지는 자기장, 중지는 전류. 플레밍의 왼손은 전동기를 읽습니다. 오른손으로 바꿔 쥐면 같은 기하가 발전기를 읽습니다. `F = I·L×B`',
      },
    },
  ],
  sliderParams: [
    { key: 'B_T',       label: { en: 'Flux Density (B)',      ko: '자속밀도 (B)' },    unit: 'T',  min: 0.25, max: 1,  step: 0.05, defaultValue: 0.5 },
    { key: 'I_A',       label: { en: 'Current (I)',           ko: '전류 (I)' },        unit: 'A',  min: 2,    max: 20, step: 1,    defaultValue: 10 },
    { key: 'L_cm',      label: { en: 'Conductor Length (L)',  ko: '도체 길이 (L)' },   unit: 'cm', min: 5,    max: 40, step: 5,    defaultValue: 20 },
    { key: 'theta_deg', label: { en: 'Angle I to B (θ)',      ko: 'I와 B의 각 (θ)' },  unit: '°',  min: 0,    max: 90, step: 5,    defaultValue: 90 },
  ],
  Scene: LorentzScene,
  Scene2D: LorentzForce2DScene,
}

export default concept
