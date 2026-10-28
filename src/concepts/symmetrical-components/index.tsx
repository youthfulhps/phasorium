import { useRef, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import type { Group } from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { useLocaleStore } from '../../lib/i18n'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import type { ConceptMeta } from '../../types/concept'

const TWO_PI = Math.PI * 2
const THIRD = TWO_PI / 3 // 120° between phases
const DEG = Math.PI / 180
const DISPLAY_OMEGA = Math.PI // on-screen rad/s: 0.5 turns per second, slowed far below 60 Hz

// a = 1∠120° = (A_RE, +A_IM); a² = 1∠240° = (A_RE, −A_IM); 1 + a + a² = 0
const A_RE = -0.5
const A_IM = Math.sqrt(3) / 2

// Sequence reactances of the worked single line-to-ground fault, per unit
const X1 = 0.2
const X2 = 0.2
const X0 = 0.1
const E_PU = 1.0
const SLG_SEQ = E_PU / (X1 + X2 + X0) // 2.0 pu, the same in all three sequences
const SLG_IA = 3 * SLG_SEQ // 6.0 pu in the faulted phase

// Layout of `out`, written by solve() so no frame allocates:
//   0..5    Ia, Ib, Ic as (re, im) — the measured, generally unbalanced set
//   6..11   I0, I1, I2 as (re, im) — the three sequence phasors (phase-a members)
//   12..14  |I0|, |I1|, |I2|
//   15      max(|Ia|, |Ib|, |Ic|), used only to normalise the on-screen scale
//   16..21  Ia, Ib, Ic rebuilt from I0, I1, I2 by the inverse transform
const N_OUT = 22

// Symmetrical components, phase a as the reference. With a = 1∠120°:
//   Forward: I0 = (Ia + Ib + Ic) ÷ 3
//            I1 = (Ia + a·Ib + a²·Ic) ÷ 3
//            I2 = (Ia + a²·Ib + a·Ic) ÷ 3
//   Inverse: Ia = I0 + I1 + I2
//            Ib = I0 + a²·I1 + a·I2
//            Ic = I0 + a·I1 + a²·I2
// Verified by replaying solve() with the real slider keys (npx tsx):
//  0. Startup state, phase b down to 0.3 pu (Ia_mag 1, Ib_mag 0.3, Ic_mag 1, Ib_shift 0)
//     -> |I0| = 0.2333, |I1| = 0.7667, |I2| = 0.2333 pu. All three sets are on screen, so the
//     balanced case below is something the learner watches happen rather than starts inside.
//  1. Balanced set, Ia = 1∠0°, Ib = 1∠−120°, Ic = 1∠+120°, one slider move from the startup state
//     (Ia_mag 1, Ib_mag 1, Ic_mag 1, Ib_shift 0)
//     -> |I0| = 0.0000, |I1| = 1.0000, |I2| = 0.0000. Only the positive-sequence set survives.
//  2. Single line-to-ground fault, Ib = Ic = 0, Ia = 6.0 pu
//     (Ia_mag 6, Ib_mag 0, Ic_mag 0, Ib_shift 0)
//     -> I0 = I1 = I2 = Ia ÷ 3 = 2.0000 pu each, all three exactly equal.
//  3. The same fault from sequence impedances, Z1 = Z2 = j0.2, Z0 = j0.1 pu, E = 1.0 pu
//     -> I1 = I2 = I0 = E ÷ (Z1 + Z2 + Z0) = 2.0000 pu and Ia = 3·I1 = 6.0000 pu.
//     A ground fault therefore needs a zero-sequence path: without one, Z0 is infinite,
//     I0 = 0, and the three sequence currents cannot add up to a current in one line.
//  4. Round trip, forward then inverse, swept over every slider combination (816750 sets):
//     max |rebuilt − input| = 1.776e-15 pu, well inside 1e-12. The scene keeps that check live,
//     since the markers are drawn from the inverse and must land on the measured phasors.
function solve(v: Record<string, number>, out: Float64Array) {
  const ma = v.Ia_mag ?? 1
  const mb = v.Ib_mag ?? 1
  const mc = v.Ic_mag ?? 1
  const angB = -THIRD + (v.Ib_shift ?? 0) * DEG

  const ar = ma
  const ai = 0
  const br = mb * Math.cos(angB)
  const bi = mb * Math.sin(angB)
  const cr = mc * A_RE // Ic sits at +120°, so Ic = |Ic|·a
  const ci = mc * A_IM
  out[0] = ar; out[1] = ai
  out[2] = br; out[3] = bi
  out[4] = cr; out[5] = ci

  // a·Ib and a²·Ic for I1
  const aIbR = A_RE * br - A_IM * bi
  const aIbI = A_RE * bi + A_IM * br
  const a2IcR = A_RE * cr + A_IM * ci
  const a2IcI = A_RE * ci - A_IM * cr
  // a²·Ib and a·Ic for I2
  const a2IbR = A_RE * br + A_IM * bi
  const a2IbI = A_RE * bi - A_IM * br
  const aIcR = A_RE * cr - A_IM * ci
  const aIcI = A_RE * ci + A_IM * cr

  const z0r = (ar + br + cr) / 3
  const z0i = (ai + bi + ci) / 3
  const z1r = (ar + aIbR + a2IcR) / 3
  const z1i = (ai + aIbI + a2IcI) / 3
  const z2r = (ar + a2IbR + aIcR) / 3
  const z2i = (ai + a2IbI + aIcI) / 3
  out[6] = z0r; out[7] = z0i
  out[8] = z1r; out[9] = z1i
  out[10] = z2r; out[11] = z2i
  out[12] = Math.hypot(z0r, z0i)
  out[13] = Math.hypot(z1r, z1i)
  out[14] = Math.hypot(z2r, z2i)
  out[15] = Math.max(ma, mb, mc)

  // Inverse transform. The scene draws the chain from out[6..11] and the markers from these,
  // so a marker off its phasor tip would mean the round trip had broken.
  out[16] = z0r + z1r + z2r
  out[17] = z0i + z1i + z2i
  out[18] = z0r + (A_RE * z1r + A_IM * z1i) + (A_RE * z2r - A_IM * z2i)
  out[19] = z0i + (A_RE * z1i - A_IM * z1r) + (A_RE * z2i + A_IM * z2r)
  out[20] = z0r + (A_RE * z1r - A_IM * z1i) + (A_RE * z2r + A_IM * z2i)
  out[21] = z0i + (A_RE * z1i + A_IM * z1r) + (A_RE * z2i - A_IM * z2r)
  return out
}

function useSliderValues() {
  const valuesRef = useRef<Record<string, number>>(useSliderStore.getState().values)
  useEffect(
    () => useSliderStore.subscribe((s) => s.values, (v) => { valuesRef.current = v }),
    []
  )
  return valuesRef
}

const fmtPu = (x: number) => `${x.toFixed(4)} pu`
// Labels read in the foreground color; only the accents (current, power) tint their text
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const legendClass = 'pointer-events-none whitespace-nowrap text-[10px] font-medium tabular-nums text-foreground/70 leading-4'

// Every on-screen string, in both locales. Symbols are identical in en and ko.
const L10N = {
  measured: { en: 'measured phase currents', ko: '측정한 상전류' },
  zero: { en: 'zero-sequence', ko: '영상분' },
  positive: { en: 'positive-sequence', ko: '정상분' },
  negative: { en: 'negative-sequence', ko: '역상분' },
  allInPhase: { en: 'a = b = c, in phase', ko: 'a = b = c, 같은 위상' },
  orderAbc: { en: 'order a-b-c', ko: '순서 a-b-c' },
  orderAcb: { en: 'order a-c-b', ko: '순서 a-c-b' },
  chain: { en: 'tip to tail, landing on the tip of Ia', ko: '꼬리에 머리를 이어, Ia의 끝점에 닿습니다' },
  slg: { en: 'single line-to-ground fault', ko: '1선 지락' },
  fullScale: { en: 'full scale = largest |I_phase|', ko: '최대 눈금 = 가장 큰 |I_phase|' },
  hint: { en: 'drag |Ib| up to 1.0 pu and watch I0 and I2 vanish', ko: '|Ib|를 1.0 pu까지 끌어올리면 I0와 I2가 사라집니다' },
}
const SEQ_NAME = [L10N.zero, L10N.positive, L10N.negative]
const SEQ_NOTE = [L10N.allInPhase, L10N.orderAbc, L10N.orderAcb]
const SEQ_SYMBOL = ['I0', 'I1', 'I2']
const PHASE_LETTER = ['a', 'b', 'c']

// Shared scratch: written and handed straight to a geometry, so no frame ever allocates
const SEG = new Float32Array(6)
function setSeg(line: any, x0: number, y0: number, x1: number, y1: number, z: number) {
  SEG[0] = x0; SEG[1] = y0; SEG[2] = z
  SEG[3] = x1; SEG[4] = y1; SEG[5] = z
  line?.geometry.setPositions(SEG)
  line?.computeLineDistances()
}
const put = (g: Group | null | undefined, x: number, y: number, z: number) => { g?.position.set(x, y, z) }

const SEG_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]

// ── 3D: one unbalanced fan, three balanced triads, and the chain that joins them ──
// The three sequence sets spin counter-clockwise at the same rate; only the order of their
// letters differs. Laid tip to tail from the unbalanced origin, they reach the tip of Ia.
const FIT = 0.65 // scene units given to the longest phase phasor, so the frame always fits
const UO_X = -1.18 // origin of the unbalanced fan, kept clear of the floating card on the right
const UO_Y = 0.3
const TRIAD_X = 0.62
const TRIAD_Y = [1.3, 0.08, -1.14] // zero, positive, negative
const TRIAD_Z = -0.35 // the triads sit a little behind the plane of the measured set
const ZERO_SPREAD = 0.055 // sideways gap between the three identical zero-sequence arrows
const TAG_GAP = 0.15 // how far past an arrow tip its letter sits, along the arrow
const NAME_X = TRIAD_X + 0.7 // sequence names sit just clear of the longest arrow a triad can draw
// Which rotation angle each arrow of each triad uses: 0 = wt, 1 = wt − 120°, 2 = wt + 120°
const SEQ_IDX = [
  [0, 0, 0], // zero: all three arrows identical
  [0, 1, 2], // positive: b lags a, c leads a
  [0, 2, 1], // negative: b leads a, c lags a
]
const ANG_OFF = [0, -THIRD, THIRD]

function SymmetricalComponentsScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const out = useMemo(() => new Float64Array(N_OUT), [])
  const cosS = useMemo(() => new Float64Array(3), [])
  const sinS = useMemo(() => new Float64Array(3), [])
  // 0-2 Ia Ib Ic, 3-11 triad arrows (zero, positive, negative × a, b, c), 12-14 chain
  const lineRefs = useRef<any[]>([])
  // 0-2 a b c of the measured set, 3-5 positive triad, 6-8 negative triad
  const tagRefs = useRef<(Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([]) // |I0|, |I1|, |I2|
  const chainDotRefs = useRef<(Group | null)[]>([])
  const sumDotRef = useRef<Group>(null)
  const labelledFor = useRef<unknown>(null)

  useFrame(({ clock }) => {
    solve(valuesRef.current, out)
    const k = FIT / Math.max(out[15], 1e-6)
    const wt = clock.getElapsedTime() * DISPLAY_OMEGA
    for (let j = 0; j < 3; j++) {
      cosS[j] = Math.cos(wt + ANG_OFF[j])
      sinS[j] = Math.sin(wt + ANG_OFF[j])
    }
    const c0 = cosS[0]
    const s0 = sinS[0]

    // The measured set: each phasor already carries its own angle, so all three share one spin
    for (let p = 0; p < 3; p++) {
      const re = out[p * 2]
      const im = out[p * 2 + 1]
      const vx = k * (re * c0 - im * s0)
      const vy = k * (re * s0 + im * c0)
      setSeg(lineRefs.current[p], UO_X, UO_Y, UO_X + vx, UO_Y + vy, 0)
      // The letter rides just past its own tip, so it can never drift off the arrow it names
      const len = Math.hypot(vx, vy)
      const g = len > 1e-6 ? (len + TAG_GAP) / len : 0
      put(tagRefs.current[p], UO_X + vx * g, UO_Y + vy * g, 0)
    }

    // The three balanced triads
    for (let sq = 0; sq < 3; sq++) {
      const re = out[6 + sq * 2]
      const im = out[7 + sq * 2]
      const oy = TRIAD_Y[sq]
      for (let p = 0; p < 3; p++) {
        const j = SEQ_IDX[sq][p]
        const vx = k * (re * cosS[j] - im * sinS[j])
        const vy = k * (re * sinS[j] + im * cosS[j])
        const len = Math.hypot(vx, vy)
        let sx = TRIAD_X
        let sy = oy
        if (sq === 0 && len > 1e-6) {
          // All three zero-sequence arrows are the same phasor, so they would overlap exactly.
          // Shifting their tails sideways lets all three be seen, still collinear and locked together.
          const off = (p - 1) * ZERO_SPREAD
          sx -= (vy / len) * off
          sy += (vx / len) * off
        }
        setSeg(lineRefs.current[3 + sq * 3 + p], sx, sy, sx + vx, sy + vy, TRIAD_Z)
        if (sq > 0) {
          const g = len > 1e-6 ? (len + TAG_GAP) / len : 0
          put(tagRefs.current[sq * 3 + p], sx + vx * g, sy + vy * g, TRIAD_Z)
        }
      }
    }

    // The decomposition made literal: I0, then I1, then I2, each hung off the last one's tip
    let px = UO_X
    let py = UO_Y
    for (let sq = 0; sq < 3; sq++) {
      const re = out[6 + sq * 2]
      const im = out[7 + sq * 2]
      const nx = px + k * (re * c0 - im * s0)
      const ny = py + k * (re * s0 + im * c0)
      setSeg(lineRefs.current[12 + sq], px, py, nx, ny, 0)
      if (sq < 2) put(chainDotRefs.current[sq], nx, ny, 0)
      px = nx
      py = ny
    }
    // Placed from the inverse transform, not from the chain: it lands on both the chain's end
    // and the tip of Ia, which is the round trip checking itself every frame
    sumDotRef.current?.position.set(
      UO_X + k * (out[16] * c0 - out[17] * s0),
      UO_Y + k * (out[16] * s0 + out[17] * c0),
      0
    )

    // Numbers change only with the sliders, so the text is rebuilt only then
    if (labelledFor.current !== valuesRef.current) {
      labelledFor.current = valuesRef.current
      for (let sq = 0; sq < 3; sq++) {
        const el = textRefs.current[sq]
        if (el) el.textContent = fmtPu(out[12 + sq])
      }
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  const phaseColor = [c.current, c.voltage, c.inductance]

  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />

      {/* Background, lights, grid, and bloom come from the shared SceneStage */}
      {Array.from({ length: 15 }, (_, i) => {
        const chain = i >= 12
        return (
          <Line
            key={i}
            ref={(l) => { lineRefs.current[i] = l }}
            points={SEG_INIT}
            lineWidth={chain ? 2 : i < 3 ? 3 : 2.2}
            color={chain ? c.marker : phaseColor[i < 3 ? i : (i - 3) % 3]}
            dashed={chain}
            dashSize={0.07}
            gapSize={0.05}
          />
        )
      })}

      {[0, 1].map((i) => (
        <Marker key={i} ref={(g) => { chainDotRefs.current[i] = g }} color={c.marker} size={0.045} />
      ))}
      <Marker ref={sumDotRef} color={c.current} size={0.075} />

      {Array.from({ length: 9 }, (_, i) => {
        const p = i % 3
        return (
          <group key={i} ref={(g) => { tagRefs.current[i] = g }}>
            <Html center>
              <span className={labelClass} style={p === 0 ? { color: c.current } : undefined}>
                {PHASE_LETTER[p]}
              </span>
            </Html>
          </group>
        )
      })}

      {/* Each fan keeps a dot on its own origin, so a set that collapses to zero still has
          something to point at instead of leaving its label stranded */}
      <group position={[UO_X, UO_Y, 0]}>
        <Marker color={c.guide} size={0.03} />
      </group>
      {TRIAD_Y.map((y, sq) => (
        <group key={sq} position={[TRIAD_X, y, TRIAD_Z]}>
          <Marker color={c.guide} size={0.03} />
        </group>
      ))}

      <group position={[UO_X, UO_Y + 0.92, 0]}>
        <Html center>
          <span className={labelClass}>{L10N.measured[locale]}</span>
        </Html>
      </group>

      {TRIAD_Y.map((y, sq) => (
        <group key={sq} position={[NAME_X, y, TRIAD_Z]}>
          <Html center>
            <div className={legendClass}>
              <div>{`${SEQ_SYMBOL[sq]} · ${SEQ_NAME[sq][locale]}`}</div>
              <div>
                <span ref={(el) => { textRefs.current[sq] = el }} className={legendClass} />
                {` · ${SEQ_NOTE[sq][locale]}`}
              </div>
            </div>
          </Html>
        </group>
      ))}

      <group position={[UO_X, -0.88, 0]}>
        <Html center>
          <div className={legendClass}>
            <div className={labelClass}>Ia = I0 + I1 + I2</div>
            <div>{L10N.chain[locale]}</div>
          </div>
        </Html>
      </group>

      <group position={[UO_X, -1.58, 0]}>
        <Html center>
          <div className={legendClass}>
            <div>{`${L10N.slg[locale]}: I0 = I1 = I2 = E ÷ (Z1 + Z2 + Z0) = ${SLG_SEQ.toFixed(2)} pu`}</div>
            <div>{`Ia = 3·I1 = ${SLG_IA.toFixed(2)} pu`}</div>
          </div>
        </Html>
      </group>
    </>
  )
}

// ── 2D: the three measured waveforms, and the decomposition as three bars ──
// Chosen over three flattened triads because the unbalance has to be believable first: unequal
// amplitudes and shifted zero crossings are what a meter actually sees, and the bars then give the
// exact split, growing and shrinking under the sliders where lengths in a triad are hard to compare.
const W_N = 200
const WX0 = -3.85
const WX1 = -0.2
const WAVE_FIT = 1.2 // scene units for the largest phase amplitude; the bars share this scale
const BAR_X0 = 0.5
const BAR_Y = [1.0, 0.3, -0.4]
const BAR_FULL = BAR_X0 + WAVE_FIT
const WAVE_INIT: [number, number, number][] = Array.from(
  { length: W_N },
  (): [number, number, number] => [0, 0, 0]
)
const ZERO_AXIS: [number, number, number][] = [[WX0 - 0.12, 0, 0], [WX1 + 0.12, 0, 0]]
const BAR_BASE: [number, number, number][] = [[BAR_X0, -0.75, 0], [BAR_X0, 1.3, 0]]
const BAR_TICK: [number, number, number][] = [[BAR_FULL, -0.75, 0], [BAR_FULL, 1.3, 0]]

function SymmetricalComponents2DScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const out = useMemo(() => new Float64Array(N_OUT), [])
  const waveBufs = useMemo(
    () => [new Float32Array(W_N * 3), new Float32Array(W_N * 3), new Float32Array(W_N * 3)],
    []
  )
  const waveRefs = useRef<any[]>([])
  const lineRefs = useRef<any[]>([]) // 0 sweep, 1-3 bars
  const dotRefs = useRef<(Group | null)[]>([])
  const tagRefs = useRef<(Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<unknown>(null)

  useFrame(({ clock }) => {
    solve(valuesRef.current, out)
    const k = WAVE_FIT / Math.max(out[15], 1e-6)

    // One cycle of each phase: the instantaneous value is the rotating phasor's height
    for (let i = 0; i < W_N; i++) {
      const frac = i / (W_N - 1)
      const th = frac * TWO_PI
      const x = WX0 + frac * (WX1 - WX0)
      const cs = Math.cos(th)
      const sn = Math.sin(th)
      for (let p = 0; p < 3; p++) {
        const buf = waveBufs[p]
        buf[i * 3] = x
        buf[i * 3 + 1] = k * (out[p * 2] * sn + out[p * 2 + 1] * cs)
        buf[i * 3 + 2] = 0
      }
    }
    for (let p = 0; p < 3; p++) waveRefs.current[p]?.geometry.setPositions(waveBufs[p])

    // "Now" sweeps across the cycle and wraps; the dots ride the rebuilt phasors, so seeing them
    // stay on the measured curves is the inverse transform checking itself
    const th = ((clock.getElapsedTime() * DISPLAY_OMEGA) % TWO_PI + TWO_PI) % TWO_PI
    const xn = WX0 + (th / TWO_PI) * (WX1 - WX0)
    const cs = Math.cos(th)
    const sn = Math.sin(th)
    setSeg(lineRefs.current[0], xn, -1.45, xn, 1.45, 0)
    for (let p = 0; p < 3; p++) {
      put(dotRefs.current[p], xn, k * (out[16 + p * 2] * sn + out[17 + p * 2] * cs), 0)
      put(tagRefs.current[p], WX1 + 0.22, k * out[p * 2 + 1], 0)
    }

    for (let sq = 0; sq < 3; sq++) {
      const len = Math.max(out[12 + sq] * k, 1e-3)
      setSeg(lineRefs.current[1 + sq], BAR_X0, BAR_Y[sq], BAR_X0 + len, BAR_Y[sq], 0)
    }

    if (labelledFor.current !== valuesRef.current) {
      labelledFor.current = valuesRef.current
      for (let sq = 0; sq < 3; sq++) {
        const el = textRefs.current[sq]
        if (el) el.textContent = fmtPu(out[12 + sq])
      }
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  const phaseColor = [c.current, c.voltage, c.inductance]

  return (
    <>
      <Line points={ZERO_AXIS} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      <Line points={BAR_BASE} lineWidth={1} color={c.guide} />
      <Line points={BAR_TICK} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />

      {phaseColor.map((color, p) => (
        <TrailLine
          key={p}
          ref={(l) => { waveRefs.current[p] = l }}
          points={WAVE_INIT}
          lineWidth={2.5}
          color={color}
        />
      ))}

      {Array.from({ length: 4 }, (_, i) => (
        <Line
          key={i}
          ref={(l) => { lineRefs.current[i] = l }}
          points={SEG_INIT}
          lineWidth={i === 0 ? 1 : 11}
          color={i === 0 ? c.guide : c.marker}
          dashed={i === 0}
          dashSize={0.08}
          gapSize={0.06}
        />
      ))}

      {phaseColor.map((color, p) => (
        <Marker key={p} ref={(g) => { dotRefs.current[p] = g }} color={color} size={0.05} />
      ))}

      {PHASE_LETTER.map((letter, p) => (
        <group key={letter} ref={(g) => { tagRefs.current[p] = g }}>
          <Html position={[0, 0, 0]} center>
            <span className={labelClass} style={p === 0 ? { color: c.current } : undefined}>
              {letter}
            </span>
          </Html>
        </group>
      ))}

      <group position={[(WX0 + WX1) / 2, 1.62, 0]}>
        <Html center>
          <span className={labelClass}>{L10N.measured[locale]}</span>
        </Html>
      </group>

      {BAR_Y.map((y, sq) => (
        <group key={sq} position={[BAR_X0 + 0.62, y + 0.25, 0]}>
          <Html center>
            <div className={legendClass}>
              <div>{`${SEQ_SYMBOL[sq]} · ${SEQ_NAME[sq][locale]}`}</div>
              <div ref={(el) => { textRefs.current[sq] = el }} className={legendClass} />
            </div>
          </Html>
        </group>
      ))}

      <group position={[BAR_FULL, -0.95, 0]}>
        <Html center>
          <span className={legendClass}>{L10N.fullScale[locale]}</span>
        </Html>
      </group>

      <group position={[BAR_X0 + 0.62, -1.42, 0]}>
        <Html center>
          <div className={legendClass}>
            <div>{L10N.hint[locale]}</div>
            <div>{`${L10N.slg[locale]}: I0 = I1 = I2 = ${SLG_SEQ.toFixed(2)} pu, Ia = ${SLG_IA.toFixed(2)} pu`}</div>
          </div>
        </Html>
      </group>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'symmetrical-components',
  title: { en: 'Symmetrical Components', ko: '대칭좌표법 (영상·정상·역상분)' },
  subject: 'power-systems',
  order: 3,
  description: {
    en: `## Symmetrical Components

**Any three unbalanced phasors are the sum of exactly three balanced sets, and a balanced input leaves only the positive-sequence set alive.**

- Cyan liquid line, labelled a: phase a, the reference phasor and the faulted phase
- Two neutral lines at different lightnesses, labelled b and c: the other two phases
- Three triads down the right: I0, I1, I2, the sequence sets. They are told apart by their labels and their place in the column, never by hue, and all three spin counter-clockwise at the same rate
- I0: three arrows of the same length and direction, drawn side by side so all three can be seen; they are collinear and move as one
- I1 runs a-b-c, I2 runs a-c-b: same spin, reversed order of the letters
- Faint dot at the centre of each fan: that set's origin, so a set which collapses to zero is visibly sitting at zero rather than missing
- Dashed chain from the origin of the measured fan: I0, I1 and I2 laid tip to tail
- Marker at the end of that chain: it sits on the tip of Ia

**With a = 1∠120°**
**Forward**: I0 = (Ia + Ib + Ic) ÷ 3, I1 = (Ia + a·Ib + a²·Ic) ÷ 3, I2 = (Ia + a²·Ib + a·Ic) ÷ 3
**Inverse**: Ia = I0 + I1 + I2, Ib = I0 + a²·I1 + a·I2, Ic = I0 + a·I1 + a²·I2

It opens unbalanced, with phase b down to 0.3 pu, so all three sets are on screen at once: |I1| = 0.7667 pu carrying most of it, |I0| and |I2| at 0.2333 pu each. Now drag |Ib| back up to 1.0 and watch what happens: both of those shrink to 0.0000 pu and their triads collapse onto their origin dots, leaving the positive-sequence set alone. That disappearance is the whole trick, and it is worth triggering yourself rather than reading. A balanced three-phase system is pure positive sequence, so a healthy network can be analysed one sequence at a time, and anything a relay sees in I0 or I2 is unbalance by definition.

Pull |Ib| and |Ic| down to 0 and the measured set collapses to current in phase a alone, the single line-to-ground fault. The arithmetic goes exactly where you can watch it: I0 = I1 = I2 = Ia ÷ 3, three equal thirds laid end to end along one line. Read the other way, the fault current comes out of the sequence network, E ÷ (Z1 + Z2 + Z0): with Z1 = Z2 = j0.2 and Z0 = j0.1 pu at E = 1.0 pu, each sequence carries 2.0000 pu and Ia = 3·I1 = 6.0000 pu. The zero-sequence path is in that sum, so a ground fault cannot happen without one. Remove it and Z0 is infinite, I0 falls to zero, and the three sequence currents can no longer add up to current in a single line.

|Ia| raises the faulted phase on its own, up to the 6 pu of the worked fault, so unbalance grows while b and c stay put. |Ib| runs both ways from where it starts: up to 1.0 pu for the balanced case, down to 0 for the fault. |Ic| shrinks or swells the third phase the same way. Ib phase offset slides phase b away from its −120° spot, which unbalances the angles with every magnitude left untouched. Lengths on screen are normalised so the longest phase phasor always measures the same, and every arrow in the scene shares that one scale, so ratios stay true.

In 2D: one cycle of the three measured currents, with the dots at "now" placed from the phasors rebuilt by the inverse transform, and the split read off as three bars of |I0|, |I1| and |I2| against a full scale of the largest phase. The waveforms come first because unequal amplitudes and shifted zero crossings are what a meter actually sees; bars then give the exact split, which is much easier to compare than arm lengths in a triad. Rotation is slowed to half a turn per second.`,
    ko: `## 대칭좌표법 (영상·정상·역상분)

**불평형한 세 페이저는 언제나 평형한 세 묶음의 합으로 쪼개지며, 입력이 평형하면 정상분만 살아남습니다.**

- a라고 적힌 시안 액체 선: 상 a. 기준 페이저이자 지락이 생긴 상입니다
- 밝기가 다른 두 중성색 선, b와 c: 나머지 두 상
- 오른쪽에 세로로 놓인 세 묶음: I0, I1, I2의 대칭분. 색조가 아니라 이름과 열에서의 위치로 구별하며, 세 묶음 모두 같은 속도로 반시계 방향으로 돕니다
- I0: 길이와 방향이 똑같은 화살표 셋. 겹쳐 보이지 않도록 나란히 그렸을 뿐, 한 직선 위에서 함께 움직입니다
- I1은 a-b-c 순, I2는 a-c-b 순: 회전 방향은 같고 글자 순서만 뒤집힙니다
- 각 부채꼴 중심의 흐린 점: 그 묶음의 원점. 크기가 0으로 줄어든 묶음도 사라진 것이 아니라 0에 앉아 있다는 것이 보입니다
- 측정한 부채꼴의 원점에서 나가는 점선 사슬: I0, I1, I2를 꼬리에 머리를 이어 붙인 것
- 그 사슬 끝의 마커: Ia의 끝점 위에 놓입니다

**a = 1∠120°일 때**
**정변환**: I0 = (Ia + Ib + Ic) ÷ 3, I1 = (Ia + a·Ib + a²·Ic) ÷ 3, I2 = (Ia + a²·Ib + a·Ic) ÷ 3
**역변환**: Ia = I0 + I1 + I2, Ib = I0 + a²·I1 + a·I2, Ic = I0 + a·I1 + a²·I2

처음에는 b상이 0.3 pu로 내려간 불평형 상태로 열립니다. 그래서 세 묶음이 한꺼번에 화면에 있습니다. |I1|이 0.7667 pu로 대부분을 지고, |I0|와 |I2|가 각각 0.2333 pu입니다. 이제 |Ib|를 1.0까지 올려 보세요. 두 값이 0.0000 pu로 줄어들며 두 묶음이 각자의 원점 점으로 주저앉고, 정상분만 남습니다. 바로 이 사라짐이 핵심이고, 글로 읽기보다 직접 일으켜 볼 만한 장면입니다. 평형 3상은 순수한 정상분이므로 건전한 계통은 대칭분마다 따로 떼어 해석할 수 있고, 계전기가 I0나 I2에서 무언가를 읽었다면 그것은 정의상 불평형입니다.

|Ib|와 |Ic|를 0까지 내리면 측정한 세 전류는 상 a 하나만 남습니다. 1선 지락입니다. 계산 결과가 눈에 보이는 자리로 그대로 나옵니다. I0 = I1 = I2 = Ia ÷ 3, 똑같은 1/3 셋이 한 직선 위에 이어 붙습니다. 거꾸로 읽으면 지락전류는 대칭분 회로에서 E ÷ (Z1 + Z2 + Z0)로 나옵니다. E = 1.0 pu에서 Z1 = Z2 = j0.2, Z0 = j0.1 pu이면 각 대칭분이 2.0000 pu를 흘리고 Ia = 3·I1 = 6.0000 pu가 됩니다. 이 합 안에 영상분 경로가 들어 있으므로, 영상분 경로가 없으면 지락 자체가 성립하지 않습니다. 경로를 끊으면 Z0가 무한대가 되어 I0가 0으로 떨어지고, 세 대칭분 전류는 더 이상 한 선의 전류로 합쳐지지 못합니다.

|Ia|는 지락이 난 상만 끌어올립니다. 예제 지락의 6 pu까지 올릴 수 있고, 그동안 b와 c는 가만히 있으므로 불평형이 커집니다. |Ib|는 시작 위치에서 양쪽으로 움직입니다. 1.0 pu까지 올리면 평형, 0까지 내리면 지락입니다. |Ic|도 같은 방식으로 세 번째 상을 줄이거나 늘립니다. Ib 위상 편차는 상 b를 −120° 자리에서 밀어내어, 크기는 전혀 건드리지 않고 각도만 불평형하게 만듭니다. 화면의 길이는 가장 긴 상 페이저가 항상 같은 길이가 되도록 정규화했고, 장면의 모든 화살표가 그 하나의 축척을 공유하므로 비율은 그대로 유지됩니다.

2D: 측정한 세 전류의 한 주기를 그리고, '지금'을 표시하는 점은 역변환으로 되살린 페이저에서 찍습니다. 분해 결과는 |I0|, |I1|, |I2| 세 막대로 읽으며, 최대 눈금은 가장 큰 상 전류입니다. 파형을 앞세운 이유는, 크기가 서로 다르고 영점 통과가 어긋나는 모습이야말로 계측기가 실제로 보는 것이기 때문입니다. 그다음 막대가 정확한 분해 값을 주는데, 이는 묶음 안 화살표 길이를 눈으로 견주는 것보다 훨씬 비교하기 쉽습니다. 회전은 초당 반 바퀴로 늦춰 재생합니다.`,
  },
  sliderParams: [
    { key: 'Ia_mag',   label: { en: 'Phase a Magnitude (|Ia|)', ko: 'a상 크기 (|Ia|)' },      unit: 'pu', min: 0.2, max: 6,   step: 0.2, defaultValue: 1 },
    { key: 'Ib_mag',   label: { en: 'Phase b Magnitude (|Ib|)', ko: 'b상 크기 (|Ib|)' },      unit: 'pu', min: 0,   max: 1.4, step: 0.1, defaultValue: 0.3 },
    { key: 'Ic_mag',   label: { en: 'Phase c Magnitude (|Ic|)', ko: 'c상 크기 (|Ic|)' },      unit: 'pu', min: 0,   max: 1.4, step: 0.1, defaultValue: 1 },
    { key: 'Ib_shift', label: { en: 'Phase b Offset from −120°', ko: 'b상 위상 편차 (−120° 기준)' }, unit: '°', min: -60, max: 60, step: 1, defaultValue: 0 },
  ],
  Scene: SymmetricalComponentsScene,
  Scene2D: SymmetricalComponents2DScene,
}

export default concept
