import { useRef, useEffect, useMemo, type Ref } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import type { Group } from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import { setLineOpacity, setOpacity, useStoryReveal } from '../../components/scene/storyReveal'
import type { ConceptMeta } from '../../types/concept'

const SQRT3 = Math.sqrt(3)
const TWO_PI = Math.PI * 2
const THIRD = TWO_PI / 3 // 120° between phases
const DISPLAY_OMEGA = 1.2 // on-screen rad/s (~5.2 s per turn): a real 60 Hz rotation is unwatchable

// Pure physics, phase A as the reference. Peak value Vm = sqrt(2)·V_phase, so
// va = Vm·sin(ωt), vb = Vm·sin(ωt − 120°), vc = Vm·sin(ωt + 120°), and va + vb + vc = 0 at every t.
// Y: V_line = sqrt(3)·V_phase with I_line = I_phase.  Δ: I_line = sqrt(3)·I_phase with V_line = V_phase.
// P = sqrt(3)·V_line·I_line·cosθ = 3·V_phase·I_phase·cosθ.
// Verified with node -e:
// V_phase = 220 V -> V_line = sqrt(3) × 220 = 381.0512 V
// max |va + vb + vc| over 2000 samples spanning 10 cycles = 1.364e-12 V (zero to machine precision)
// I_line = 10 A, θ = 36.87° (cosθ = 0.800) -> P = sqrt(3) × 381.0512 × 10 × 0.8 = 5279.99 W ≈ 5280 W
//   and the cross-check 3 × 220 × 10 × 0.8 = 5279.99 W agrees
// |Va − Vb| ÷ |Va| = 1.732051 at an angle of exactly +30.000° (the sqrt(3) triangle)
// slider default θ = 37° -> cosθ = 0.79864, P = 5271.0 W
function solve({ Vp = 220, IL = 10, theta_deg = 37 }: Record<string, number>) {
  const theta = (theta_deg * Math.PI) / 180
  const Vline = SQRT3 * Vp
  return { Vp, Vline, IL, theta, P: SQRT3 * Vline * IL * Math.cos(theta) }
}

function useSliderValues() {
  const valuesRef = useRef<Record<string, number>>(useSliderStore.getState().values)
  useEffect(
    () => useSliderStore.subscribe((s) => s.values, (v) => { valuesRef.current = v }),
    []
  )
  return valuesRef
}

const fmtV = (v: number) => `${v.toFixed(0)} V`
const fmtA = (a: number) => `${a.toFixed(1)} A`
const fmtW = (w: number) => (w >= 1000 ? `${(w / 1000).toFixed(2)} kW` : `${w.toFixed(0)} W`)
// Labels read in the foreground color; only the accents (current, power) tint their text
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const legendClass = 'pointer-events-none whitespace-nowrap text-[10px] font-medium tabular-nums text-foreground/70 leading-4'

// Shared scratch: written and handed straight to a geometry, so no frame ever allocates
const SEG = new Float32Array(6)
function setSeg(line: any, x0: number, y0: number, x1: number, y1: number) {
  SEG[0] = x0; SEG[1] = y0; SEG[2] = 0
  SEG[3] = x1; SEG[4] = y1; SEG[5] = 0
  line?.geometry.setPositions(SEG)
  line?.computeLineDistances()
}
const put = (g: Group | null | undefined, x: number, y: number) => { g?.position.set(x, y, 0) }

const SEG_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]
// A, B, C — three neutral tones at different lightness, told apart by their labels
const PHASE_OFFSET = [0, -THIRD, THIRD]

// ── Narration steps drive what exists on screen. Each reveal eases 0 → 1 so nothing pops in. ──
const REV_BC = 0 // phases B and C, so step 0 is one rotating voltage on its own
const REV_SUM = 1 // their running sum: the straight core, the dot on the origin, the caption
const REV_LINE = 2 // the line voltages closing a triangle between the phasor tips
const REV_LEGEND = 3 // the Y / Δ pair
const REV_I = 4 // the load current
const REV_P = 5 // the power readout
const REV_COUNT = 6

/** Which reveals the sentence at `step` asks for. Step 0 is a single phase and nothing else. */
function revealTargets(step: number, out: Float32Array) {
  out[REV_BC] = step >= 1 ? 1 : 0
  out[REV_SUM] = step >= 2 ? 1 : 0
  out[REV_LINE] = step >= 4 ? 1 : 0
  out[REV_LEGEND] = step >= 6 ? 1 : 0
  out[REV_I] = step >= 7 ? 1 : 0
  out[REV_P] = step >= 8 ? 1 : 0
}

/** The two connections side by side, in symbols only. Static text, fixed spot. */
function YDeltaLegend({ position, elRef }: { position: [number, number, number]; elRef?: Ref<HTMLDivElement> }) {
  return (
    <group position={position}>
      <Html center>
        <div ref={elRef} className={legendClass}>
          <div>Y &nbsp; VL = √3·VP &nbsp; IL = IP</div>
          <div>Δ &nbsp; IL = √3·IP &nbsp; VL = VP</div>
        </div>
      </Html>
    </group>
  )
}

// ── 3D: a three-strand braid around a dead-straight core ──
// Each phase winds back in time as a helix of the same radius, 120° apart; the running sum of the
// three is the straight line through their middle, so "always zero" is a shape, not a claim.
// In the z = 0 face the three phasors star out from the origin and their tips close a triangle —
// each side of that triangle is a line voltage, sqrt(3) times as long as a phase voltage.
const HELIX_N = 130
const TURNS = 2
const DEPTH = 4.2
const OX = -0.9 // braid axis, kept clear of the floating description card on the right
const V_SCALE = 0.0048 // scene units per volt: 380 V -> 1.82
const I_SCALE = 0.055 // scene units per amp: 30 A -> 1.65
const HELIX_INIT: [number, number, number][] = Array.from(
  { length: HELIX_N },
  (): [number, number, number] => [0, 0, 0]
)

function ThreePhaseBraidScene() {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  const helixBufs = useMemo(
    () => [new Float32Array(HELIX_N * 3), new Float32Array(HELIX_N * 3), new Float32Array(HELIX_N * 3)],
    []
  )
  const sumBuf = useMemo(() => new Float32Array(HELIX_N * 3), [])
  const helixRefs = useRef<any[]>([])
  const sumTrailRef = useRef<any>(null)
  // 0 Va, 1 Vb, 2 Vc, 3 Vab, 4 Vbc, 5 Vca, 6 Ia
  const lineRefs = useRef<any[]>([])
  // 0 A, 1 B, 2 C, 3 Vab, 4 Ia
  const tagRefs = useRef<(Group | null)[]>([])
  // 0 Vab, 1 Ia, 2 P
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const sumDotRef = useRef<Group>(null)
  const iTipRef = useRef<Group>(null)
  const labelledFor = useRef<Record<string, number> | null>(null)
  // The A / B / C letters beside the phasors, which fade with the phasors they name
  const tagElRefs = useRef<(HTMLSpanElement | null)[]>([])
  const captionRef = useRef<HTMLSpanElement>(null)
  const legendRef = useRef<HTMLDivElement>(null)
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(({ clock }) => {
    const s = solve(valuesRef.current)
    const wt = clock.getElapsedTime() * DISPLAY_OMEGA
    const r = s.Vp * V_SCALE

    // Three helices and, sample by sample, the sum of the three — which never leaves the axis
    for (let i = 0; i < HELIX_N; i++) {
      const back = 1 - i / (HELIX_N - 1) // 0 at the head (now), 1 at the oldest sample
      const z = -back * DEPTH
      const ang = wt - back * TURNS * TWO_PI
      let sx = 0
      let sy = 0
      for (let p = 0; p < 3; p++) {
        const x = r * Math.cos(ang + PHASE_OFFSET[p])
        const y = r * Math.sin(ang + PHASE_OFFSET[p])
        const buf = helixBufs[p]
        buf[i * 3] = OX + x
        buf[i * 3 + 1] = y
        buf[i * 3 + 2] = z
        sx += x
        sy += y
      }
      sumBuf[i * 3] = OX + sx
      sumBuf[i * 3 + 1] = sy
      sumBuf[i * 3 + 2] = z
    }
    for (let p = 0; p < 3; p++) helixRefs.current[p]?.geometry.setPositions(helixBufs[p])
    sumTrailRef.current?.geometry.setPositions(sumBuf)

    // The z = 0 face: the star of three phasors and the triangle their tips close
    const ax = OX + r * Math.cos(wt)
    const ay = r * Math.sin(wt)
    const bx = OX + r * Math.cos(wt - THIRD)
    const by = r * Math.sin(wt - THIRD)
    const cx = OX + r * Math.cos(wt + THIRD)
    const cy = r * Math.sin(wt + THIRD)
    setSeg(lineRefs.current[0], OX, 0, ax, ay)
    setSeg(lineRefs.current[1], OX, 0, bx, by)
    setSeg(lineRefs.current[2], OX, 0, cx, cy)
    setSeg(lineRefs.current[3], bx, by, ax, ay) // Vab = Va − Vb, tip of B to tip of A
    setSeg(lineRefs.current[4], cx, cy, bx, by) // Vbc
    setSeg(lineRefs.current[5], ax, ay, cx, cy) // Vca

    // The load current, lagging Va by θ
    const iLen = s.IL * I_SCALE
    const ia = wt - s.theta
    const ix = OX + iLen * Math.cos(ia)
    const iy = iLen * Math.sin(ia)
    setSeg(lineRefs.current[6], OX, 0, ix, iy)
    iTipRef.current?.position.set(ix, iy, 0)

    // The sum of the three front phasors, placed where the arithmetic puts it: the origin
    sumDotRef.current?.position.set(ax + bx + cx - 2 * OX, ay + by + cy, 0)

    put(tagRefs.current[0], (OX + ax) / 2, ay / 2)
    put(tagRefs.current[1], (OX + bx) / 2, by / 2)
    put(tagRefs.current[2], (OX + cx) / 2, cy / 2)
    put(tagRefs.current[3], (ax + bx) / 2, (ay + by) / 2)
    // Three quarters out along the arrow, so the readout clears the fat sum dot on the origin
    put(tagRefs.current[4], ix * 0.75 + OX * 0.25, iy * 0.75)

    // Values only move with the sliders, so the text is rebuilt only then
    if (labelledFor.current !== valuesRef.current && textRefs.current[2]) {
      labelledFor.current = valuesRef.current
      if (textRefs.current[0]) {
        textRefs.current[0]!.textContent = `Vab = √3 × ${fmtV(s.Vp)} = ${fmtV(s.Vline)}`
      }
      if (textRefs.current[1]) textRefs.current[1]!.textContent = `Ia ${fmtA(s.IL)}`
      textRefs.current[2]!.textContent = `P = √3·VL·IL·cosθ = ${fmtW(s.P)}`
    }

    const wBC = reveal[REV_BC]
    const wSum = reveal[REV_SUM]
    const wLine = reveal[REV_LINE]
    const wI = reveal[REV_I]
    setLineOpacity(helixRefs.current[1], wBC)
    setLineOpacity(helixRefs.current[2], wBC)
    setLineOpacity(lineRefs.current[1], wBC)
    setLineOpacity(lineRefs.current[2], wBC)
    setLineOpacity(sumTrailRef.current, wSum)
    for (let i = 3; i <= 5; i++) setLineOpacity(lineRefs.current[i], wLine)
    setLineOpacity(lineRefs.current[6], wI)
    sumDotRef.current?.scale.setScalar(wSum)
    iTipRef.current?.scale.setScalar(wI)
    setOpacity(tagElRefs.current[1], wBC)
    setOpacity(tagElRefs.current[2], wBC)
    setOpacity(captionRef.current, wSum)
    setOpacity(legendRef.current, reveal[REV_LEGEND])
    setOpacity(textRefs.current[0], wLine)
    setOpacity(textRefs.current[1], wI)
    setOpacity(textRefs.current[2], reveal[REV_P])
  })

  const segs: { color: string; width: number; dashed?: boolean }[] = [
    { color: c.voltage, width: 3 },
    { color: c.resistance, width: 3 },
    { color: c.inductance, width: 3 },
    { color: c.voltage, width: 2.5 },
    { color: c.guide, width: 1.5, dashed: true },
    { color: c.guide, width: 1.5, dashed: true },
    { color: c.current, width: 2.5 },
  ]
  const tags: { offset: [number, number, number]; text?: string; accent?: string }[] = [
    { offset: [0, 0.2, 0], text: 'A' },
    { offset: [0, -0.2, 0], text: 'B' },
    { offset: [-0.2, 0.08, 0], text: 'C' },
    { offset: [-0.1, 0.26, 0] },
    { offset: [0, -0.24, 0], accent: c.current },
  ]

  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />

      {/* Background, lights, grid, and bloom come from the shared SceneStage */}
      {[c.voltage, c.resistance, c.inductance].map((color, p) => (
        <TrailLine
          key={p}
          ref={(l) => { helixRefs.current[p] = l }}
          points={HELIX_INIT}
          lineWidth={2.5}
          color={color}
        />
      ))}
      {/* The running sum: straight, because it is zero at every instant */}
      <TrailLine ref={sumTrailRef} points={HELIX_INIT} lineWidth={3} color={c.marker} fade={false} />

      {segs.map((sg, i) => (
        <Line
          key={i}
          ref={(l) => { lineRefs.current[i] = l }}
          points={SEG_INIT}
          lineWidth={sg.width}
          color={sg.color}
          dashed={sg.dashed}
          dashSize={0.08}
          gapSize={0.06}
        />
      ))}

      <Marker ref={sumDotRef} color={c.marker} size={0.08} />
      <Marker ref={iTipRef} color={c.current} size={0.055} />

      {tags.map((tag, i) => (
        <group key={i} ref={(g) => { tagRefs.current[i] = g }}>
          <Html position={tag.offset} center>
            {tag.text ? (
              <span ref={(el) => { tagElRefs.current[i] = el }} className={labelClass}>{tag.text}</span>
            ) : (
              <span
                ref={(el) => { textRefs.current[i === 3 ? 0 : 1] = el }}
                className={labelClass}
                style={tag.accent ? { color: tag.accent } : undefined}
              />
            )}
          </Html>
        </group>
      ))}

      <group position={[OX - 0.5, -1.2, 0]}>
        <Html center>
          <span ref={captionRef} className={labelClass}>va + vb + vc = 0</span>
        </Html>
      </group>
      <group position={[OX, -1.75, 0]}>
        <Html center>
          <span
            ref={(el) => { textRefs.current[2] = el }}
            className={labelClass}
            style={{ color: c.power }}
          />
        </Html>
      </group>
      <YDeltaLegend position={[OX, 1.78, 0]} elRef={legendRef} />
    </>
  )
}

// ── 2D: the three waveforms with their sum lying flat on zero, beside the sqrt(3) triangle ──
const W_N = 220
const WX0 = -3.85
const WX1 = 0.15
const W_SPAN = WX1 - WX0
const W_SCALE = 0.0039 // scene units per volt: 380 V -> 1.48
const WAVE_INIT: [number, number, number][] = Array.from(
  { length: W_N },
  (): [number, number, number] => [0, 0, 0]
)
const ZERO_AXIS: [number, number, number][] = [[WX0 - 0.12, 0, 0], [WX1 + 0.12, 0, 0]]
const xOfPhase = (p: number) => WX0 + (p / TWO_PI) * W_SPAN
// The three peaks, one per phase: sin(p + offset) peaks where p + offset = 90°
const PEAK_PHASE = [Math.PI / 2, Math.PI / 2 + THIRD, Math.PI / 2 - THIRD + TWO_PI]

const T_OX = 2.05 // the subtraction triangle sits to the right of the waveforms
const T_OY = -0.1
const T_SCALE = 0.00228 // sqrt(3) × 380 × this = 1.50, the triangle's largest reach
const I2_SCALE = 0.03 // scene units per amp: 30 A -> 0.9

function ThreePhaseWave2DScene() {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  const waveBufs = useMemo(
    () => [new Float32Array(W_N * 3), new Float32Array(W_N * 3), new Float32Array(W_N * 3)],
    []
  )
  const sumBuf = useMemo(() => new Float32Array(W_N * 3), [])
  const waveRefs = useRef<any[]>([])
  const sumLineRef = useRef<any>(null)
  // 0 sweep, 1 Va, 2 Vb, 3 −Vb, 4 Vab, 5 Ia
  const lineRefs = useRef<any[]>([])
  // 0 a, 1 b, 2 c, 3 Va, 4 Vb, 5 −Vb, 6 Vab, 7 Ia
  const tagRefs = useRef<(Group | null)[]>([])
  // 0 Vab, 1 Ia, 2 P
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const dotRefs = useRef<(Group | null)[]>([])
  const sumDotRef = useRef<Group>(null)
  const labelledFor = useRef<Record<string, number> | null>(null)
  const tagElRefs = useRef<(HTMLSpanElement | null)[]>([])
  const captionRef = useRef<HTMLSpanElement>(null)
  const legendRef = useRef<HTMLDivElement>(null)
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(({ clock }) => {
    const s = solve(valuesRef.current)
    const amp = s.Vp * W_SCALE

    // One cycle of each phase, plus their sum sample by sample
    for (let k = 0; k < W_N; k++) {
      const p = (k / (W_N - 1)) * TWO_PI
      const x = xOfPhase(p)
      let sum = 0
      for (let q = 0; q < 3; q++) {
        const y = amp * Math.sin(p + PHASE_OFFSET[q])
        const buf = waveBufs[q]
        buf[k * 3] = x
        buf[k * 3 + 1] = y
        buf[k * 3 + 2] = 0
        sum += y
      }
      sumBuf[k * 3] = x
      sumBuf[k * 3 + 1] = sum
      sumBuf[k * 3 + 2] = 0
    }
    for (let q = 0; q < 3; q++) waveRefs.current[q]?.geometry.setPositions(waveBufs[q])
    sumLineRef.current?.geometry.setPositions(sumBuf)

    // "Now" sweeps across the single cycle and wraps
    const pn = ((clock.getElapsedTime() * DISPLAY_OMEGA) % TWO_PI + TWO_PI) % TWO_PI
    const xn = xOfPhase(pn)
    setSeg(lineRefs.current[0], xn, -1.62, xn, 1.62)
    let sumNow = 0
    for (let q = 0; q < 3; q++) {
      const y = amp * Math.sin(pn + PHASE_OFFSET[q])
      dotRefs.current[q]?.position.set(xn, y, 0)
      put(tagRefs.current[q], xOfPhase(PEAK_PHASE[q]), amp + 0.22)
      sumNow += y
    }
    sumDotRef.current?.position.set(xn, sumNow, 0)

    // The subtraction drawn out: Va, then −Vb hung off its tip, closing onto Vab
    const wt = clock.getElapsedTime() * DISPLAY_OMEGA
    const r = s.Vp * T_SCALE
    const ax = T_OX + r * Math.cos(wt)
    const ay = T_OY + r * Math.sin(wt)
    const bx = T_OX + r * Math.cos(wt - THIRD)
    const by = T_OY + r * Math.sin(wt - THIRD)
    const dx = ax - (bx - T_OX) // Va + (−Vb), i.e. the Vab tip
    const dy = ay - (by - T_OY)
    setSeg(lineRefs.current[1], T_OX, T_OY, ax, ay)
    setSeg(lineRefs.current[2], T_OX, T_OY, bx, by)
    setSeg(lineRefs.current[3], ax, ay, dx, dy)
    setSeg(lineRefs.current[4], T_OX, T_OY, dx, dy)

    const iLen = s.IL * I2_SCALE
    const ia = wt - s.theta
    setSeg(lineRefs.current[5], T_OX, T_OY, T_OX + iLen * Math.cos(ia), T_OY + iLen * Math.sin(ia))

    put(tagRefs.current[3], (T_OX + ax) / 2, (T_OY + ay) / 2)
    put(tagRefs.current[4], (T_OX + bx) / 2, (T_OY + by) / 2)
    put(tagRefs.current[5], (ax + dx) / 2, (ay + dy) / 2)
    put(tagRefs.current[6], (T_OX + dx) / 2, (T_OY + dy) / 2)
    put(tagRefs.current[7], T_OX + iLen * 0.5 * Math.cos(ia), T_OY + iLen * 0.5 * Math.sin(ia))

    if (labelledFor.current !== valuesRef.current && textRefs.current[2]) {
      labelledFor.current = valuesRef.current
      if (textRefs.current[0]) {
        textRefs.current[0]!.textContent = `Vab = √3 × ${fmtV(s.Vp)} = ${fmtV(s.Vline)} · +30°`
      }
      if (textRefs.current[1]) textRefs.current[1]!.textContent = `Ia ${fmtA(s.IL)}`
      textRefs.current[2]!.textContent = `P = √3·VL·IL·cosθ = ${fmtW(s.P)}`
    }

    const wBC = reveal[REV_BC]
    const wSum = reveal[REV_SUM]
    const wLine = reveal[REV_LINE]
    const wI = reveal[REV_I]
    setLineOpacity(waveRefs.current[1], wBC)
    setLineOpacity(waveRefs.current[2], wBC)
    setLineOpacity(sumLineRef.current, wSum)
    setLineOpacity(lineRefs.current[2], wBC) // Vb, the other side of the subtraction
    setLineOpacity(lineRefs.current[3], wLine)
    setLineOpacity(lineRefs.current[4], wLine)
    setLineOpacity(lineRefs.current[5], wI)
    dotRefs.current[1]?.scale.setScalar(wBC)
    dotRefs.current[2]?.scale.setScalar(wBC)
    sumDotRef.current?.scale.setScalar(wSum)
    setOpacity(tagElRefs.current[1], wBC)
    setOpacity(tagElRefs.current[2], wBC)
    setOpacity(tagElRefs.current[4], wBC)
    setOpacity(tagElRefs.current[5], wLine)
    setOpacity(captionRef.current, wSum)
    setOpacity(legendRef.current, reveal[REV_LEGEND])
    setOpacity(textRefs.current[0], wLine)
    setOpacity(textRefs.current[1], wI)
    setOpacity(textRefs.current[2], reveal[REV_P])
  })

  const waveColors = [c.voltage, c.resistance, c.inductance]
  const segs: { color: string; width: number; dashed?: boolean }[] = [
    { color: c.guide, width: 1, dashed: true },
    { color: c.voltage, width: 3 },
    { color: c.resistance, width: 3 },
    { color: c.guide, width: 1.5, dashed: true },
    { color: c.voltage, width: 2.5 },
    { color: c.current, width: 2.5 },
  ]
  const tags: { offset: [number, number, number]; text?: string; accent?: string }[] = [
    { offset: [0, 0, 0], text: 'a' },
    { offset: [0, 0, 0], text: 'b' },
    { offset: [0, 0, 0], text: 'c' },
    { offset: [0, 0.2, 0], text: 'Va' },
    { offset: [0, -0.2, 0], text: 'Vb' },
    { offset: [0.26, 0.04, 0], text: '−Vb' },
    { offset: [-0.1, 0.26, 0] },
    { offset: [0, -0.22, 0], accent: c.current },
  ]

  return (
    <>
      <Line points={ZERO_AXIS} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />

      {waveColors.map((color, q) => (
        <TrailLine
          key={q}
          ref={(l) => { waveRefs.current[q] = l }}
          points={WAVE_INIT}
          lineWidth={2.5}
          color={color}
        />
      ))}
      {/* The sum, drawn from the samples: it lands on the zero line and stays there */}
      <Line ref={sumLineRef} points={WAVE_INIT} lineWidth={3} color={c.marker} />

      {segs.map((sg, i) => (
        <Line
          key={i}
          ref={(l) => { lineRefs.current[i] = l }}
          points={SEG_INIT}
          lineWidth={sg.width}
          color={sg.color}
          dashed={sg.dashed}
          dashSize={0.08}
          gapSize={0.06}
        />
      ))}

      {waveColors.map((color, q) => (
        <Marker key={q} ref={(g) => { dotRefs.current[q] = g }} color={color} size={0.05} />
      ))}
      <Marker ref={sumDotRef} color={c.marker} size={0.075} />

      {tags.map((tag, i) => (
        <group key={i} ref={(g) => { tagRefs.current[i] = g }}>
          <Html position={tag.offset} center>
            {tag.text ? (
              <span ref={(el) => { tagElRefs.current[i] = el }} className={labelClass}>{tag.text}</span>
            ) : (
              <span
                ref={(el) => { textRefs.current[i === 6 ? 0 : 1] = el }}
                className={labelClass}
                style={tag.accent ? { color: tag.accent } : undefined}
              />
            )}
          </Html>
        </group>
      ))}

      <group position={[(WX0 + WX1) / 2, 0.26, 0]}>
        <Html center>
          <span ref={captionRef} className={labelClass}>va + vb + vc = 0</span>
        </Html>
      </group>
      <group position={[T_OX, -1.72, 0]}>
        <Html center>
          <span
            ref={(el) => { textRefs.current[2] = el }}
            className={labelClass}
            style={{ color: c.power }}
          />
        </Html>
      </group>
      <YDeltaLegend position={[-1.85, -1.74, 0]} elRef={legendRef} />
    </>
  )
}

const concept: ConceptMeta = {
  id: 'three-phase',
  title: { en: 'Three-Phase AC', ko: '3상 교류' },
  subject: 'circuits',
  order: 18,
  description: {
    en: `## Three-Phase AC, Y and Δ

**Three sources a third of a cycle apart add to zero at every single instant, which is why a balanced Y system needs no return current in the neutral.**

- A, B, C: the three phase voltages, neutral lines at three different lightnesses, 120° apart and turning together
- The straight core running back through the braid of three helices: their running sum, pinned at zero for all past time
- Vab: a line voltage, built as Va − Vb, sqrt(3) times as long as a phase voltage and 30° ahead of it
- The marker sitting on the origin: the vector sum of the three phasors, which never moves off it
- Cyan liquid line: Ia, the load current, lagging Va by the load angle θ
- Amber text: the three-phase power P

**Instantaneous**: va = Vm·sin(ωt), vb = Vm·sin(ωt − 120°), vc = Vm·sin(ωt + 120°), with Vm = sqrt(2)·V_phase
**The identity**: va + vb + vc = 0 at every t
**Y connection**: V_line = sqrt(3)·V_phase, I_line = I_phase
**Δ connection**: I_line = sqrt(3)·I_phase, V_line = V_phase
**Power**: P = sqrt(3)·V_line·I_line·cosθ = 3·V_phase·I_phase·cosθ

That the sum stays at zero is the whole reason the fourth wire carries nothing: in a balanced Y load the three return currents cancel in the neutral, so the neutral can be thin or absent. Unbalance the load and the neutral immediately starts carrying the difference.

Where does sqrt(3) come from? Subtract two phasors 120° apart and the triangle you draw is isosceles with a 120° apex, so the closing side is 2·cos(30°) = sqrt(3) times a side, and it leans 30° ahead. Y and Δ are that same triangle read two ways: Y stacks two phase voltages into one line voltage, Δ stacks two phase currents into one line current. The legend in the scene keeps both forms side by side.

Raise V_phase and the star, the triangle, and the waveforms all grow together while the sqrt(3) ratio holds. Raise I_line to stretch the current arrow and lift P. Turn θ to swing the current arrow away from Va and watch P fall with cosθ, with no change at all to the voltages.

In 2D: one cycle of the three waveforms with their sum drawn from the same samples, lying dead flat on the zero line, and beside it the subtraction Va − Vb laid out head to tail. Rotation is slowed far below a real 60 Hz so it can be followed; voltages are rms values and all lengths are scaled to fit the frame.`,
    ko: `## 3상 교류, Y 결선과 Δ 결선

**1/3 주기씩 어긋난 세 전원은 모든 순간에 합이 0이 됩니다. 평형 Y 결선의 중성선에 돌아오는 전류가 없는 이유가 바로 이것입니다.**

- A, B, C: 세 상전압. 밝기가 서로 다른 세 가지 중성색 선이며, 120°씩 벌어진 채 함께 돕니다
- 세 가닥 나선 한가운데를 곧게 관통하는 선: 세 상의 순시 합. 지나간 모든 시간 동안 0에 못 박혀 있습니다
- Vab: 선간전압. Va − Vb로 만들어지며, 상전압의 sqrt(3)배 길이에 30° 앞섭니다
- 원점에 놓인 마커: 세 페이저의 벡터 합. 원점을 한 번도 벗어나지 않습니다
- 시안 액체 선: Ia, 부하 전류. 부하각 θ만큼 Va보다 뒤집니다
- 호박색 글자: 3상 전력 P

**순시값**: va = Vm·sin(ωt), vb = Vm·sin(ωt − 120°), vc = Vm·sin(ωt + 120°), Vm = sqrt(2)·V_phase
**항등식**: 모든 t에서 va + vb + vc = 0
**Y 결선**: V_line = sqrt(3)·V_phase, I_line = I_phase
**Δ 결선**: I_line = sqrt(3)·I_phase, V_line = V_phase
**전력**: P = sqrt(3)·V_line·I_line·cosθ = 3·V_phase·I_phase·cosθ

합이 계속 0이라는 사실이 네 번째 전선에 아무것도 흐르지 않는 이유입니다. 평형 Y 부하에서는 세 귀환 전류가 중성선에서 서로 지워지므로 중성선을 가늘게 하거나 아예 생략할 수 있습니다. 부하가 불평형이 되는 순간부터 중성선이 그 차이를 떠안기 시작합니다.

sqrt(3)은 어디서 올까요. 120° 벌어진 두 페이저를 빼면 꼭지각이 120°인 이등변삼각형이 그려지고, 닫히는 변의 길이는 한 변의 2·cos(30°) = sqrt(3)배이며 방향은 30° 앞섭니다. Y와 Δ는 같은 삼각형을 두 가지로 읽은 것입니다. Y는 두 상전압을 쌓아 선간전압을 만들고, Δ는 두 상전류를 쌓아 선전류를 만듭니다. 장면 안의 범례에 두 형태를 나란히 적어 두었습니다.

V_phase를 올리면 별 모양도 삼각형도 파형도 함께 커지며 sqrt(3) 비율은 그대로 유지됩니다. I_line을 올리면 전류 화살표가 길어지고 P가 커집니다. θ를 돌리면 전류 화살표가 Va에서 멀어지고 P가 cosθ를 따라 줄어듭니다. 전압 쪽은 전혀 변하지 않습니다.

2D: 세 파형의 한 주기와, 같은 표본에서 계산한 합을 함께 그립니다. 합은 0선 위에 완전히 평평하게 누워 있습니다. 그 옆에는 Va − Vb를 꼬리에 머리를 잇는 방식으로 펼쳐 두었습니다. 회전은 실제 60 Hz보다 훨씬 느리게 재생하며, 전압은 실횻값이고 길이는 화면에 맞게 축척한 것입니다.`,
  },
  story: [
    {
      set: { Vp: 220, IL: 10, theta_deg: 37 },
      text: {
        en: 'One AC voltage, drawn as a turning arrow. Its tip traces a sine wave. The helix spiralling off behind it is where that tip has already been — one turn, one cycle.',
        ko: '교류 전압 하나를 도는 화살표로 그렸습니다. 끝이 지나간 자리가 정현파입니다. 뒤로 감겨 나가는 나선은 그 끝이 이미 지나온 길입니다. 한 바퀴가 한 주기.',
      },
    },
    {
      text: {
        en: 'Add two more. Same size, each a third of a turn behind the last. Three arrows, 120° apart, turning together forever. Three helices braiding off into the distance.',
        ko: '두 개를 더 얹습니다. 크기는 같고, 각각 1/3 바퀴씩 뒤처집니다. 120°씩 벌어진 화살표 셋이 영원히 함께 돕니다. 나선 세 가닥이 뒤로 엮여 나갑니다.',
      },
    },
    {
      text: {
        en: 'Now add all three up. Every instant. The sum lands on the origin and never leaves it — and its entire history is that dead-straight core running through the middle of the braid. `va + vb + vc = 0`',
        ko: '이제 셋을 더합니다. 매 순간. 합은 원점에 떨어진 뒤 한 번도 벗어나지 않습니다. 그 기록 전체가 나선 세 가닥 한가운데를 곧게 뚫고 가는 저 선입니다. `va + vb + vc = 0`',
      },
    },
    {
      text: {
        en: 'That is why a three-phase line needs no fat wire coming home. In a balanced Y load the three currents meet in the neutral and erase each other, so it carries nothing at all. Unbalance it and the neutral picks up the difference instantly.',
        ko: '3상 선로에 굵은 귀환 전선이 필요 없는 이유가 이겁니다. 평형 Y 부하에서는 세 전류가 중성선에서 만나 서로를 지웁니다. 중성선엔 아무것도 흐르지 않습니다. 불평형이 되는 순간, 중성선이 그 차이를 바로 떠안습니다.',
      },
    },
    {
      set: { Vp: 220, IL: 10, theta_deg: 37 },
      text: {
        en: 'Measure between two lines instead of line to neutral, and you are reading a difference of two phase voltages. Draw it: the three tips close a triangle. `Vab = √3 × 220 = 381 V`',
        ko: '선과 중성점이 아니라 선과 선 사이를 재면, 읽히는 값은 두 상전압의 차입니다. 그려 보면 화살표 셋의 끝이 삼각형을 닫습니다. `Vab = √3 × 220 = 381 V`',
      },
    },
    {
      text: {
        en: 'Where does the √3 come from? Subtract two arrows 120° apart and you have drawn an isosceles triangle with a 120° apex. The closing side is 2 × cos 30° = 1.732 times a side, leaning 30° ahead. There is nothing deeper than that.',
        ko: '√3은 어디서 왔을까요. 120° 벌어진 두 화살표를 빼면 꼭지각 120°짜리 이등변삼각형이 그려집니다. 닫히는 변은 한 변의 2 × cos 30° = 1.732배, 방향은 30° 앞. 그 이상은 없습니다.',
      },
    },
    {
      text: {
        en: 'Y and Δ are that same triangle, read two ways. Y stacks two phase voltages into one line voltage. `VL = √3·VP` Δ stacks two phase currents into one line current. `IL = √3·IP`',
        ko: 'Y와 Δ는 같은 삼각형을 두 방향으로 읽은 것입니다. Y는 상전압 둘을 쌓아 선간전압을 만듭니다. `VL = √3·VP` Δ는 상전류 둘을 쌓아 선전류를 만듭니다. `IL = √3·IP`',
      },
    },
    {
      set: { Vp: 220, IL: 10, theta_deg: 37 },
      text: {
        en: 'Hang a load on it and current flows. But a real load is never pure resistance, so the current arrow does not line up with the voltage. It drags behind by the load angle θ. 37° here.',
        ko: '부하를 겁니다. 전류가 흐릅니다. 그런데 실제 부하는 순수한 저항이 아니라서, 전류 화살표가 전압과 나란하지 않습니다. 부하각 θ만큼 뒤에서 끌려옵니다. 여기선 37°.',
      },
    },
    {
      set: { Vp: 220, IL: 10, theta_deg: 37 },
      text: {
        en: 'Only the slice of current that lines up with the voltage does any work, and that slice is cos θ. `P = √3 × 381 V × 10 A × cos 37° = 5.27 kW`',
        ko: '일을 하는 건 전압과 방향이 맞는 전류 성분뿐입니다. 그 몫이 cos θ입니다. `P = √3 × 381 V × 10 A × cos 37° = 5.27 kW`',
      },
    },
    {
      set: { Vp: 220, IL: 10, theta_deg: 0 },
      text: {
        en: 'Now swing θ to zero. Same voltage, same current, nothing else touched — 6.60 kW. The same wires, 25% more work out of them. `P = √3·VL·IL·cos θ = 3·VP·IP·cos θ` Chasing that last cos θ is the entire job of power-factor correction.',
        ko: 'θ만 0으로 돌려 봅니다. 전압도 전류도 그대로, 다른 건 손도 안 댔습니다. 6.60 kW. 같은 전선으로 25% 더 많은 일을 뽑아냅니다. `P = √3·VL·IL·cos θ = 3·VP·IP·cos θ` 이 마지막 cos θ를 좇는 일, 그게 역률 개선의 전부입니다.',
      },
    },
  ],
  sliderParams: [
    { key: 'Vp',        label: { en: 'Phase Voltage (V_phase)', ko: '상전압 (V_phase)' }, unit: 'V', min: 110, max: 380, step: 10, defaultValue: 220 },
    { key: 'IL',        label: { en: 'Line Current (I_line)',   ko: '선전류 (I_line)' },   unit: 'A', min: 2,   max: 30,  step: 1,  defaultValue: 10  },
    { key: 'theta_deg', label: { en: 'Load Angle (θ)',          ko: '부하각 (θ)' },        unit: '°', min: 0,   max: 60,  step: 1,  defaultValue: 37  },
  ],
  Scene: ThreePhaseBraidScene,
  Scene2D: ThreePhaseWave2DScene,
}

export default concept
