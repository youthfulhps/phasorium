import { useRef, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import { DoubleSide } from 'three'
import type { Group, MeshBasicMaterial } from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { setLineOpacity, setOpacity, useStoryReveal } from '../../components/scene/storyReveal'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import type { ConceptMeta } from '../../types/concept'

// Pure physics. A flat loop of area A spinning at ω in a uniform field B links a flux
//   Φ = B·A·cos(ωt)
// and Faraday's law turns the *slope* of that flux, not the flux itself, into a voltage:
//   e = −N·(dΦ ÷ dt) = N·B·A·ω·sin(ωt),  peak = N·B·A·ω
// The minus sign is Lenz's law: the induced current opposes the change that produced it.
//
// Worked example, verified with node -e:
//   N = 100 turns, B = 0.5 T, A = 0.01 m² (= 100 cm²), f = 60 Hz
//     ω      = 2π·60    = 376.991 rad/s
//     Φ peak = B·A      = 0.005000 Wb = 5.00 mWb
//     e peak = N·B·A·ω  = 188.4956 V   (rms 188.4956 ÷ sqrt(2) = 133.286 V)
//   Non-rotating check: N = 50 with ΔΦ = 0.02 Wb in Δt = 0.1 s gives
//     e = N·(ΔΦ ÷ Δt) = 50 × 0.02 ÷ 0.1 = 10.000 V
//
// One module-scope record holds the result so no frame allocates.
const S = { N: 100, B: 0.5, A: 0.01, f: 60, omega: 376.991, phiPeak: 0.005, emfPeak: 188.4956 }

function solve(v: Record<string, number>) {
  const N = v.N ?? 100
  const B = v.B_T ?? 0.5
  const A = (v.A_cm2 ?? 100) / 1e4 // cm² -> m²
  const f = v.f_Hz ?? 60
  S.N = N
  S.B = B
  S.A = A
  S.f = f
  S.omega = 2 * Math.PI * f
  S.phiPeak = B * A
  S.emfPeak = N * S.phiPeak * S.omega
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

// ── Narration steps drive what exists on screen. Each reveal eases 0 → 1 so nothing pops in. ──
const REV_PHI = 0 // the flux trace
const REV_EMF = 1 // the EMF trace beneath it
const REV_SPIN = 2 // multiplies ω, so step 0 holds the coil still facing the field
const REV_I = 3 // the induced-current label on the ring
const REV_GUIDE = 4 // 2D only: the dashed verticals on the flux zero crossings
const REV_COUNT = 5

/** Which reveals the sentence at `step` asks for. Step 0 is a still coil, so no traces yet. */
function revealTargets(step: number, out: Float32Array) {
  out[REV_PHI] = step >= 1 ? 1 : 0
  out[REV_EMF] = step >= 2 ? 1 : 0
  out[REV_SPIN] = step >= 3 ? 1 : 0
  out[REV_I] = step >= 5 ? 1 : 0
  out[REV_GUIDE] = step >= 4 ? 1 : 0
}

const fmtPhi = (wb: number) => `${(wb * 1000).toFixed(2)} mWb`
const fmtV = (v: number) => (v >= 100 ? `${v.toFixed(0)} V` : `${v.toFixed(1)} V`)
// Labels read in the foreground color; only the two accents (current, power) tint their text,
// so the one cyan label here is the induced current and everything else is plain foreground.
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'

// ── Display scaling, shared by both scenes ──
// Each trace is drawn linear in its own quantity against the worked example above, so every
// slider moves the amplitude by exactly the factor the formula says. The EMF trace clips at
// full scale while its readout keeps climbing.
const PHI_REF = 0.005 // Wb, the worked example's flux peak
const EMF_REF = 188.4956 // V, the worked example's EMF peak
const PHI_AMP = 0.32 // scene units drawn for PHI_REF
const EMF_AMP = 0.26 // scene units drawn for EMF_REF
const FULL = 0.72 // band half-height: a trace clips here
const PHI_Y = 1.0 // flux band center
const EMF_Y = -1.0 // EMF band center
const clip = (y: number) => (y > FULL ? FULL : y < -FULL ? -FULL : y)

// A real alternator turns far too fast to watch, so the loop is slowed: SPIN rad/s on screen
// stands for the real ω at F_REF, and the speed slider still changes how fast it turns.
const SPIN = 1.1
const F_REF = 60
const WINDOW = 11 // seconds of history a trace holds
const spinOf = (f: number) => (SPIN * f) / F_REF
const FLOW = 2.6 // sheen bands per second at the worked example's peak EMF

// ── 3D: the loop spinning in the field, with the flux it links and the EMF it makes ──
const LX = -1.5 // loop center on X
const R_SCENE = 0.8 // drawn loop radius at A_REF_CM2
const A_REF_CM2 = 100
const LOOP_N = 97
// Unit loop in the local Y–Z plane, so the group's local +X is the loop's normal and
// rotation.y is the angle between that normal and the field.
const LOOP: [number, number, number][] = Array.from(
  { length: LOOP_N },
  (_, i): [number, number, number] => {
    const u = (i / (LOOP_N - 1)) * 2 * Math.PI
    return [0, Math.cos(u), Math.sin(u)]
  }
)

// The uniform field: straight lines along +X threading the loop, each capped with an arrowhead
const B_X0 = -2.7
const B_X1 = -0.3
const B_OFF: [number, number][] = [[0, 0], [0.62, 0], [-0.62, 0], [0, 0.62], [0, -0.62]]
const bLine = (y: number, z: number): [number, number, number][] => [[B_X0, y, z], [B_X1, y, z]]
const AXLE: [number, number, number][] = [[LX, -1.15, 0], [LX, 1.15, 0]]

const TRACE_N = 220
const TRACE_X = 0.9 // the traces stand in the plane x = TRACE_X, history receding toward −Z
const TRACE_Z = 4
const TRACE_INIT: [number, number, number][] = Array.from(
  { length: TRACE_N },
  (_, i): [number, number, number] => [TRACE_X, 0, -(1 - i / (TRACE_N - 1)) * TRACE_Z]
)
const zeroAxis = (y: number): [number, number, number][] => [
  [TRACE_X, y, -TRACE_Z],
  [TRACE_X, y, 0],
]

function FaradayLoopScene() {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  const reveal = useStoryReveal(revealTargets, REV_COUNT)
  const theta = useRef(0)
  const phiBuf = useMemo(() => new Float32Array(TRACE_N * 3), [])
  const emfBuf = useMemo(() => new Float32Array(TRACE_N * 3), [])
  const phiLine = useRef<any>(null)
  const emfLine = useRef<any>(null)
  const phiAxis = useRef<any>(null)
  const emfAxis = useRef<any>(null)
  const iLabel = useRef<HTMLSpanElement>(null)
  const loopRef = useRef<Group>(null)
  const discRef = useRef<MeshBasicMaterial>(null)
  const phiDot = useRef<Group>(null)
  const emfDot = useRef<Group>(null)
  // 0 flux readout, 1 EMF readout, 2 ω readout
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const flow = useRef(0)
  const labelledFor = useRef<Record<string, number> | null>(null)

  useFrame((_, delta) => {
    const s = solve(valuesRef.current)
    const revPhi = reveal[REV_PHI]
    const revEmf = reveal[REV_EMF]
    // ω is scaled by the reveal, so the coil is held still until the sentence says to turn it.
    const om = spinOf(s.f) * reveal[REV_SPIN]
    // Integrated rather than om·t, so changing the speed leaves the head of the wave where it
    // is instead of jumping its phase. Capped so a backgrounded tab does not skip a turn.
    theta.current += om * Math.min(delta, 0.1)
    const th0 = theta.current
    const phiAmp = (PHI_AMP * s.phiPeak) / PHI_REF
    const emfAmp = (EMF_AMP * s.emfPeak) / EMF_REF

    // The loop: area sets its radius, the angle between its normal and the field sets the flux
    const r = R_SCENE * Math.sqrt((s.A * 1e4) / A_REF_CM2)
    if (loopRef.current) {
      loopRef.current.rotation.y = th0
      loopRef.current.scale.set(1, r, r)
    }
    // The face shades with the flux it links, brightest when the normal points along the field
    if (discRef.current) {
      discRef.current.opacity = 0.04 + 0.34 * Math.abs(Math.cos(th0)) * (s.B / 0.8)
    }
    // e > 0 drives the induced current one way round the loop and e < 0 the other, so the
    // sheen reverses exactly when the flux slope changes sign. That reversal is Lenz's law.
    flow.current = FLOW * Math.sin(th0) * Math.min(2, s.emfPeak / EMF_REF)

    for (let i = 0; i < TRACE_N; i++) {
      const frac = i / (TRACE_N - 1)
      const z = -(1 - frac) * TRACE_Z
      const th = th0 - om * (1 - frac) * WINDOW
      phiBuf[i * 3] = TRACE_X
      phiBuf[i * 3 + 1] = PHI_Y + clip(phiAmp * Math.cos(th))
      phiBuf[i * 3 + 2] = z
      emfBuf[i * 3] = TRACE_X
      emfBuf[i * 3 + 1] = EMF_Y + clip(emfAmp * Math.sin(th))
      emfBuf[i * 3 + 2] = z
    }
    phiLine.current?.geometry.setPositions(phiBuf)
    emfLine.current?.geometry.setPositions(emfBuf)
    phiDot.current?.position.set(TRACE_X, PHI_Y + clip(phiAmp * Math.cos(th0)), 0)
    emfDot.current?.position.set(TRACE_X, EMF_Y + clip(emfAmp * Math.sin(th0)), 0)

    setLineOpacity(phiLine.current, revPhi)
    setLineOpacity(emfLine.current, revEmf)
    setLineOpacity(phiAxis.current, revPhi)
    setLineOpacity(emfAxis.current, revEmf)
    phiDot.current?.scale.setScalar(revPhi)
    emfDot.current?.scale.setScalar(revEmf)
    setOpacity(textRefs.current[0], revPhi)
    setOpacity(textRefs.current[1], revEmf)
    setOpacity(textRefs.current[2], reveal[REV_SPIN])
    setOpacity(iLabel.current, reveal[REV_I])

    if (labelledFor.current !== valuesRef.current && textRefs.current[2]) {
      labelledFor.current = valuesRef.current
      if (textRefs.current[0]) {
        textRefs.current[0]!.textContent = `Φ = B·A·cos(ωt) · Φm ${fmtPhi(s.phiPeak)}`
      }
      if (textRefs.current[1]) {
        textRefs.current[1]!.textContent = `e = N·B·A·ω·sin(ωt) · em ${fmtV(s.emfPeak)}`
      }
      textRefs.current[2]!.textContent = `ω = 2πf = ${s.omega.toFixed(0)} rad/s`
    }
  })

  return (
    <>
      {/* Camera and orbit target share an x offset to the right, which slides the alternator
          left out from under the narration floating on the right. */}
      <PerspectiveCamera makeDefault position={[4, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} target={[1, 0, 0]} />

      {/* Background, lights, grid, and bloom come from the shared SceneStage */}
      {/* The uniform field the loop turns in */}
      {B_OFF.map(([y, z], i) => (
        <group key={i}>
          <Line points={bLine(y, z)} lineWidth={1.5} color={c['b-field']} />
          <mesh position={[B_X1, y, z]} rotation={[0, 0, -Math.PI / 2]}>
            <coneGeometry args={[0.07, 0.2, 12]} />
            <meshBasicMaterial color={c['b-field']} />
          </mesh>
        </group>
      ))}
      <Line points={AXLE} lineWidth={1} color={c.guide} />

      {/* The spinning loop: a cyan ring carrying the induced current, filled with its flux */}
      <group position={[LX, 0, 0]} ref={loopRef}>
        <TrailLine
          points={LOOP}
          lineWidth={3}
          color={c.current}
          fade={false}
          flowSpeed={() => flow.current}
        />
        <mesh rotation={[0, Math.PI / 2, 0]}>
          <circleGeometry args={[1, 48]} />
          <meshBasicMaterial
            ref={discRef}
            color={c['b-field']}
            transparent
            opacity={0.2}
            depthWrite={false}
            side={DoubleSide}
          />
        </mesh>
      </group>

      {/* Two traces on one time axis: the flux, and the EMF its slope produces */}
      <Line ref={phiAxis} points={zeroAxis(PHI_Y)} lineWidth={1} color={c.guide} />
      <Line ref={emfAxis} points={zeroAxis(EMF_Y)} lineWidth={1} color={c.guide} />
      <TrailLine ref={phiLine} points={TRACE_INIT} lineWidth={2} color={c['b-field']} />
      <TrailLine ref={emfLine} points={TRACE_INIT} lineWidth={3} color={c.voltage} />
      <Marker ref={phiDot} color={c.marker} size={0.06} />
      <Marker ref={emfDot} color={c.marker} size={0.07} />

      <Html position={[B_X0, 1.1, 0]} center>
        <span className={labelClass}>B →</span>
      </Html>
      <Html position={[LX, 1.3, 0]} center>
        <span ref={iLabel} className={labelClass} style={{ color: c.current }}>i</span>
      </Html>
      <Html position={[LX, -1.45, 0]} center>
        <span ref={(el) => { textRefs.current[2] = el }} className={labelClass} />
      </Html>
      <Html position={[TRACE_X, PHI_Y + 0.95, 0]} center>
        <span ref={(el) => { textRefs.current[0] = el }} className={labelClass} />
      </Html>
      <Html position={[TRACE_X, EMF_Y - 0.95, 0]} center>
        <span ref={(el) => { textRefs.current[1] = el }} className={labelClass} />
      </Html>
    </>
  )
}

// ── 2D: the same two waves stacked on one time axis, with the quarter cycle marked ──
// The dashed verticals sit exactly where the flux crosses zero. Read down one of them: the flux
// is at nothing, its slope is steepest, and the EMF is at its peak. Read down a flux peak
// instead and the EMF is nothing. That is the whole lesson, e = −N·(dΦ ÷ dt).
const X2_0 = -4
const W2 = 8
const FLAT_N = 240
const FLAT_INIT: [number, number, number][] = Array.from(
  { length: FLAT_N },
  (_, i): [number, number, number] => [X2_0 + (i / (FLAT_N - 1)) * W2, 0, 0]
)
const flatAxis = (y: number): [number, number, number][] => [[X2_0, y, 0], [X2_0 + W2, y, 0]]
const GUIDE_TOP = 1.92
const GUIDE_BOT = -1.92
const GUIDE_INIT: [number, number, number][] = [[0, GUIDE_BOT, 0], [0, GUIDE_TOP, 0]]

function FaradayWaves2DScene() {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  const reveal = useStoryReveal(revealTargets, REV_COUNT)
  const theta = useRef(0)
  const phiBuf = useMemo(() => new Float32Array(FLAT_N * 3), [])
  const emfBuf = useMemo(() => new Float32Array(FLAT_N * 3), [])
  const guideBufs = useMemo(() => [new Float32Array(6), new Float32Array(6)], [])
  const phiLine = useRef<any>(null)
  const emfLine = useRef<any>(null)
  const phiAxis = useRef<any>(null)
  const emfAxis = useRef<any>(null)
  const sideLabels = useRef<(HTMLSpanElement | null)[]>([])
  const tagLabel = useRef<HTMLSpanElement>(null)
  const guideRefs = useRef<any[]>([])
  const phiDot = useRef<Group>(null)
  const emfDot = useRef<Group>(null)
  const tagRef = useRef<Group>(null)
  // 0 flux readout, 1 EMF readout
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<Record<string, number> | null>(null)

  useFrame((_, delta) => {
    const s = solve(valuesRef.current)
    const revPhi = reveal[REV_PHI]
    const revEmf = reveal[REV_EMF]
    const revGuide = reveal[REV_GUIDE]
    const om = spinOf(s.f) * reveal[REV_SPIN]
    theta.current += om * Math.min(delta, 0.1)
    const th0 = theta.current
    const phiAmp = (PHI_AMP * s.phiPeak) / PHI_REF
    const emfAmp = (EMF_AMP * s.emfPeak) / EMF_REF

    for (let i = 0; i < FLAT_N; i++) {
      const frac = i / (FLAT_N - 1)
      const x = X2_0 + frac * W2
      const th = th0 - om * (1 - frac) * WINDOW
      phiBuf[i * 3] = x
      phiBuf[i * 3 + 1] = PHI_Y + clip(phiAmp * Math.cos(th))
      phiBuf[i * 3 + 2] = 0
      emfBuf[i * 3] = x
      emfBuf[i * 3 + 1] = EMF_Y + clip(emfAmp * Math.sin(th))
      emfBuf[i * 3 + 2] = 0
    }
    phiLine.current?.geometry.setPositions(phiBuf)
    emfLine.current?.geometry.setPositions(emfBuf)
    phiDot.current?.position.set(X2_0 + W2, PHI_Y + clip(phiAmp * Math.cos(th0)), 0.02)
    emfDot.current?.position.set(X2_0 + W2, EMF_Y + clip(emfAmp * Math.sin(th0)), 0.02)

    setLineOpacity(phiLine.current, revPhi)
    setLineOpacity(emfLine.current, revEmf)
    setLineOpacity(phiAxis.current, revPhi)
    setLineOpacity(emfAxis.current, revEmf)
    phiDot.current?.scale.setScalar(revPhi)
    emfDot.current?.scale.setScalar(revEmf)
    setOpacity(sideLabels.current[0], revPhi)
    setOpacity(sideLabels.current[1], revEmf)
    setOpacity(textRefs.current[0], revPhi)
    setOpacity(textRefs.current[1], revEmf)
    setOpacity(tagLabel.current, revGuide)

    // The flux crosses zero at ωτ = π ÷ 2 + kπ, which is where the EMF peaks. Mark the two most
    // recent crossings; a crossing that has scrolled off collapses to a zero-length segment.
    const kNow = Math.floor((th0 - Math.PI / 2) / Math.PI)
    let tagX = X2_0
    for (let j = 0; j < 2; j++) {
      const thX = Math.PI / 2 + (kNow - j) * Math.PI
      // A still coil has no crossings to mark, and om = 0 would divide by zero
      const frac = om > 1e-4 ? 1 - (th0 - thX) / (om * WINDOW) : -1
      const inView = revGuide > 0.02 && frac >= 0 && frac <= 1
      const x = inView ? X2_0 + frac * W2 : X2_0
      const buf = guideBufs[j]
      buf[0] = x
      buf[1] = inView ? GUIDE_BOT : GUIDE_TOP
      buf[2] = 0
      buf[3] = x
      buf[4] = GUIDE_TOP
      buf[5] = 0
      guideRefs.current[j]?.geometry.setPositions(buf)
      guideRefs.current[j]?.computeLineDistances()
      if (j === 0 && inView) tagX = x
    }
    tagRef.current?.position.set(tagX, 0, 0)

    if (labelledFor.current !== valuesRef.current && textRefs.current[1]) {
      labelledFor.current = valuesRef.current
      if (textRefs.current[0]) {
        textRefs.current[0]!.textContent = `Φ = B·A·cos(ωt) · Φm ${fmtPhi(s.phiPeak)}`
      }
      textRefs.current[1]!.textContent = `e = −N·(dΦ ÷ dt) = N·B·A·ω·sin(ωt) · em ${fmtV(s.emfPeak)}`
    }
  })

  return (
    <>
      <Line ref={phiAxis} points={flatAxis(PHI_Y)} lineWidth={1} color={c.guide} />
      <Line ref={emfAxis} points={flatAxis(EMF_Y)} lineWidth={1} color={c.guide} />
      {[0, 1].map((j) => (
        <Line
          key={j}
          ref={(l) => { guideRefs.current[j] = l }}
          points={GUIDE_INIT}
          lineWidth={1}
          color={c.guide}
          dashed
          dashSize={0.09}
          gapSize={0.07}
        />
      ))}

      <TrailLine ref={phiLine} points={FLAT_INIT} lineWidth={2.5} color={c['b-field']} />
      <TrailLine ref={emfLine} points={FLAT_INIT} lineWidth={3} color={c.voltage} />
      <Marker ref={phiDot} color={c.marker} size={0.06} />
      <Marker ref={emfDot} color={c.marker} size={0.07} />

      <Html position={[X2_0 - 0.28, PHI_Y, 0]} center>
        <span ref={(el) => { sideLabels.current[0] = el }} className={labelClass}>Φ</span>
      </Html>
      <Html position={[X2_0 - 0.28, EMF_Y, 0]} center>
        <span ref={(el) => { sideLabels.current[1] = el }} className={labelClass}>e</span>
      </Html>
      <group ref={tagRef}>
        <Html center>
          <span ref={tagLabel} className={labelClass}>Φ = 0 → |e| max</span>
        </Html>
      </group>
      <Html position={[0, 1.95, 0]} center>
        <span ref={(el) => { textRefs.current[0] = el }} className={labelClass} />
      </Html>
      <Html position={[0, -1.95, 0]} center>
        <span ref={(el) => { textRefs.current[1] = el }} className={labelClass} />
      </Html>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'faraday-induction',
  title: { en: "Faraday's Law & Lenz's Law", ko: '전자기 유도와 렌츠의 법칙' },
  subject: 'electromagnetics',
  order: 2,
  description: {
    en: `## Faraday's Law and Lenz's Law

**Flux does not make a voltage — changing flux does.**

- The deep cool lines with arrowheads: the uniform magnetic field B the loop turns in
- The shaded face inside the loop: the flux Φ it links right now, brightest when the loop faces the field and empty when it lies edge-on
- The cyan ring: the induced current i. Watch its sheen reverse every half turn
- The deep cool trace: the flux Φ over time
- The bright trace: the induced EMF e over time

**Faraday's law**: e = −N·(dΦ ÷ dt)
**Spinning loop**: Φ = B·A·cos(ωt), so e = N·B·A·ω·sin(ωt)
**Peak EMF**: em = N·B·A·ω, with ω = 2πf

The minus sign is **Lenz's law**: the induced current always flows the way that opposes the change which caused it. So the current reverses exactly when the flux slope changes sign, which is why it is drawn flowing one way round the loop and then the other.

Look at where the two traces line up. The EMF peaks exactly where the flux crosses zero, because that is where the flux is changing fastest, and the EMF is zero at the flux peaks, where the flux is momentarily flat. The EMF is the slope of the flux, a quarter cycle ahead of it.

Worked example: N = 100 turns, B = 0.5 T, A = 100 cm² and f = 60 Hz give ω = 377 rad/s, a flux peak of 5.00 mWb, and a peak EMF of 188.5 V. The simplest non-rotating case is the same law without the trigonometry: 50 turns and a flux change of 0.02 Wb in 0.1 s give 50 × 0.02 ÷ 0.1 = 10 V.

This is an alternator. Spin a coil in a field and a sinusoid comes out, with no switching of any kind.

Raise N, B or A and the EMF trace grows while the flux timing stays put. Raise the speed and the EMF grows too, because ω appears in the peak, and the loop visibly turns faster. The rotation is slowed for viewing; the readout shows the real ω. The EMF trace stops at full scale while its number keeps climbing.

In 2D: both waves on one time axis, with dashed verticals on the flux zero crossings. Read down one of them and the quarter cycle shift is unmistakable.`,
    ko: `## 패러데이 전자기 유도와 렌츠의 법칙

**자속이 전압을 만드는 것이 아니라, 변하는 자속이 전압을 만듭니다.**

- 화살표가 달린 짙은 한색 직선: 코일이 회전하는 균일 자기장 B
- 코일 안쪽의 음영: 지금 코일이 쇄교하는 자속 Φ. 코일이 자기장을 정면으로 마주할 때 가장 밝고, 나란히 누우면 비어 버립니다
- 청록색 고리: 유도 전류 i. 반 바퀴마다 흐르는 방향이 뒤집히는 것을 보세요
- 짙은 한색 파형: 시간에 따른 자속 Φ
- 밝은 파형: 시간에 따른 유도 기전력 e

**패러데이 법칙**: e = −N·(dΦ ÷ dt)
**회전하는 코일**: Φ = B·A·cos(ωt)이므로 e = N·B·A·ω·sin(ωt)
**최대 기전력**: em = N·B·A·ω, 여기서 ω = 2πf

음의 부호가 바로 **렌츠의 법칙**입니다. 유도 전류는 언제나 자신을 만들어 낸 변화를 방해하는 방향으로 흐릅니다. 그래서 자속의 기울기가 부호를 바꾸는 순간 전류도 방향을 바꾸고, 고리를 한쪽으로 돌다가 반대쪽으로 도는 모습으로 그려집니다.

두 파형이 어디서 맞아떨어지는지 보세요. 기전력은 자속이 0을 지나는 바로 그 지점에서 최대가 됩니다. 그곳이 자속이 가장 빠르게 변하는 지점이기 때문입니다. 반대로 자속이 최대인 지점에서는 자속이 순간적으로 평평하므로 기전력이 0입니다. 기전력은 자속의 기울기이고, 자속보다 1/4 주기 앞섭니다.

계산 예: N = 100회, B = 0.5 T, A = 100 cm², f = 60 Hz이면 ω = 377 rad/s, 자속 최대 5.00 mWb, 최대 기전력 188.5 V입니다. 회전이 없는 가장 단순한 경우도 같은 법칙입니다. 50회 코일에서 자속이 0.1초 동안 0.02 Wb 변하면 50 × 0.02 ÷ 0.1 = 10 V입니다.

이것이 교류 발전기입니다. 자기장 속에서 코일을 돌리기만 하면, 어떤 전환 장치도 없이 정현파가 나옵니다.

N, B, A를 올리면 기전력 파형이 커지지만 자속의 타이밍은 그대로입니다. 회전 속도를 올려도 기전력이 커집니다. ω가 최댓값에 들어 있기 때문이며, 코일이 눈에 보이게 빨리 돕니다. 보기 좋게 회전을 느리게 했고, 수치는 실제 ω를 보여 줍니다. 기전력 파형은 화면 끝에서 잘리지만 수치는 계속 올라갑니다.

2D: 두 파형을 하나의 시간축에 올리고, 자속이 0을 지나는 곳에 점선을 세웠습니다. 점선을 따라 내려가 보면 1/4 주기 차이가 분명하게 보입니다.`,
  },
  story: [
    {
      set: { N: 100, B_T: 0.5, A_cm2: 100, f_Hz: 40 },
      text: {
        en: 'A coil, sitting still in a magnetic field. The field is pouring straight through it, full strength. And the voltage coming out is zero.',
        ko: '자기장 속에 코일 하나가 멈춰 서 있습니다. 자기장은 코일을 정면으로, 가득 통과하고 있습니다. 그런데 나오는 전압은 0입니다.',
      },
    },
    {
      set: { N: 100, B_T: 0.5, A_cm2: 100, f_Hz: 40 },
      text: {
        en: 'Plot the flux and you get a flat line. `Φ = B × A = 5.00 mWb` A big number. It just never changes.',
        ko: '자속을 그려 보면 평평한 직선입니다. `Φ = B × A = 5.00 mWb` 큰 값입니다. 다만 절대 변하지 않습니다.',
      },
    },
    {
      text: {
        en: 'And underneath it, the voltage. Zero. Flux does not make a voltage. Changing flux makes a voltage. That one line is the whole law.',
        ko: '그 아래가 기전력입니다. 0입니다. 자속이 전압을 만드는 게 아닙니다. 변하는 자속이 전압을 만듭니다. 이 한 줄이 법칙의 전부입니다.',
      },
    },
    {
      text: {
        en: 'So turn it. Facing the field, edge-on, facing the other way — and the linked flux swings with it. The voltage wakes up.',
        ko: '그래서 돌립니다. 자기장을 정면으로 마주했다가, 나란히 눕고, 반대로 마주합니다. 쇄교 자속이 그에 맞춰 출렁입니다. 그 순간 기전력이 깨어납니다.',
      },
    },
    {
      text: {
        en: 'Line the two waves up. Flux at its peak — voltage zero. Flux crossing zero — voltage at its peak. The voltage is not the flux. It is the flux’s slope.',
        ko: '두 파형을 나란히 놓아 보세요. 자속이 최대일 때 기전력은 0. 자속이 0을 지날 때 기전력이 최대. 기전력은 자속이 아닙니다. 자속의 기울기입니다.',
      },
    },
    {
      text: {
        en: 'Watch the sheen on the ring flip every half turn. The induced current always runs the way that fights the change that created it. That is the minus sign, talking.',
        ko: '고리 위의 광택이 반 바퀴마다 방향을 뒤집습니다. 유도 전류는 언제나 자신을 만들어 낸 변화를 거스르는 쪽으로 흐릅니다. 공식 앞의 마이너스가 하는 말이 이겁니다.',
      },
    },
    {
      set: { N: 100, B_T: 0.5, A_cm2: 100, f_Hz: 80 },
      text: {
        en: 'Spin it twice as fast. The flux peak does not budge — still 5.00 mWb. But it gets there in half the time, so the slope doubles. 125.7 V → 251.3 V.',
        ko: '두 배로 빨리 돌립니다. 자속의 최댓값은 꿈쩍도 않습니다. 여전히 5.00 mWb. 그런데 거기까지 가는 시간이 절반이니 기울기가 두 배. 125.7 V가 251.3 V로.',
      },
    },
    {
      set: { N: 140, B_T: 0.5, A_cm2: 100, f_Hz: 40 },
      text: {
        en: 'Speed back to where it was. Add turns instead: 100 → 140. Every turn links the same flux and they stack. 125.7 V → 175.9 V. Exactly 1.4×. Exactly the turns ratio.',
        ko: '속도는 원래대로. 대신 권수를 100회에서 140회로 올립니다. 모든 권선이 같은 자속을 쇄교하고, 그만큼 쌓입니다. 125.7 V가 175.9 V로. 정확히 1.4배. 권수가 늘어난 비율 그대로입니다.',
      },
    },
    {
      set: { N: 100, B_T: 0.5, A_cm2: 100, f_Hz: 60 },
      text: {
        en: '`e = −N × (dΦ ÷ dt)` For a spinning coil, `e = N × B × A × ω, ω = 2πf` Peak 188.5 V here. Spin a coil in a field and a sine wave falls out. No switches, no electronics. That is an alternator.',
        ko: '`e = −N × (dΦ ÷ dt)` 회전하는 코일이면 `e = N × B × A × ω, ω = 2πf` 지금 설정에서 최대 188.5 V. 자기장 속에서 코일을 돌리기만 하면 정현파가 떨어져 나옵니다. 스위치도, 전자 회로도 없이. 이게 교류 발전기입니다.',
      },
    },
  ],
  sliderParams: [
    { key: 'N',      label: { en: 'Number of Turns (N)',   ko: '권수 (N)' },          min: 50, max: 140,  step: 10,   defaultValue: 100 },
    { key: 'B_T',    label: { en: 'Flux Density (B)',      ko: '자속밀도 (B)' },      unit: 'T',   min: 0.3, max: 0.75, step: 0.05, defaultValue: 0.5 },
    { key: 'A_cm2',  label: { en: 'Loop Area (A)',         ko: '코일 면적 (A)' },     unit: 'cm²', min: 60,  max: 140,  step: 10,   defaultValue: 100 },
    { key: 'f_Hz',   label: { en: 'Rotation Speed (f)',    ko: '회전 속도 (f)' },     unit: 'Hz',  min: 40,  max: 80,   step: 5,    defaultValue: 60 },
  ],
  Scene: FaradayLoopScene,
  Scene2D: FaradayWaves2DScene,
}

export default concept
