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

const FREQ = 60 // supply frequency [Hz]
const OMEGA = 2 * Math.PI * FREQ
const SQRT3 = Math.sqrt(3)

// Pure physics. All powers are three-phase totals; the bank is drawn as three units in delta
// across the line voltage V, so each unit carries Qc / 3.
//   Qc = P * (tan θ1 − tan θ2),  θ = arccos(pf)
//   S = P ÷ pf, so with P and V fixed  I2 ÷ I1 = pf1 ÷ pf2  and line loss (∝ I²)
//   falls by 1 − (pf1 ÷ pf2)².
//   C = Qc ÷ (ω V²) as a single-phase equivalent, Qc ÷ (3 ω V²) per delta-connected unit.
// Verified with npx tsx: P = 100 kW, pf1 = 0.80 lagging, pf2 = 0.95, V = 380 V, f = 60 Hz →
//   Q1 = 75.00 kvar, Q2 = 32.87 kvar, Qc = 42.13 kvar,
//   S1 = 125.00 kVA, S2 = 105.26 kVA, I2 ÷ I1 = 0.8421, loss −29.09 %,
//   C = 773.9 µF single-phase equivalent, 258.0 µF per delta unit.
// A target below the starting factor is not a correction, so pf2 is held at pf1 there.
function solve({ P_kW = 100, pf1 = 0.8, pf2 = 0.95, V = 380 }: Record<string, number>) {
  const p1 = Math.min(Math.max(pf1, 0.05), 1)
  const p2 = Math.min(Math.max(pf2, p1), 1)
  const th1 = Math.acos(p1)
  const th2 = Math.acos(p2)
  const P = P_kW
  const Q1 = P * Math.tan(th1)
  const Q2 = P * Math.tan(th2)
  const Qc = Q1 - Q2
  const S1 = P / p1
  const S2 = P / p2
  const vaPerVolt = 1e3 / (SQRT3 * V) // kVA → A on a three-phase feeder
  const var3 = Qc * 1e3 // [var]
  return {
    P,
    pf1: p1,
    pf2: p2,
    th1,
    th2,
    Q1,
    Q2,
    Qc,
    S1,
    S2,
    I1: S1 * vaPerVolt,
    I2: S2 * vaPerVolt,
    iRatio: p1 / p2,
    lossDrop: 1 - (p1 / p2) ** 2,
    C1_uF: (var3 / (OMEGA * V * V)) * 1e6,
    Cd_uF: (var3 / (3 * OMEGA * V * V)) * 1e6,
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

// Labels read in the foreground color; only accent quantities (current, power) tint their text
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'

const L10N = {
  pFixed: { en: 'P unchanged', ko: 'P 그대로' },
  before: { en: 'before', ko: '개선 전' },
  after: { en: 'after', ko: '개선 후' },
  loss: { en: 'loss', ko: '손실' },
  bank: { en: 'per delta unit', ko: 'Δ 결선 각 상' },
  over: { en: 'over-corrected, leading', ko: '과보상, 진상' },
  floor: { en: 'floor S = P', ko: '바닥 S = P' },
  noNeed: { en: 'target not above pf1', ko: '목표가 pf1 이하' },
  target: { en: 'target', ko: '목표' },
  example: { en: 'worked example', ko: '계산 예' },
} as const

const fmtP = (v: number) => `${v.toFixed(0)} kW`
const fmtQ = (v: number) => `${v.toFixed(2)} kvar`
const fmtS = (v: number) => `${v.toFixed(2)} kVA`
const fmtA = (v: number) => `${v.toFixed(v < 100 ? 1 : 0)} A`
const fmtC = (v: number) => `${v.toFixed(1)} µF`
const deg = (rad: number) => `${((rad * 180) / Math.PI).toFixed(1)}°`

// Shared scratch buffers: filled and handed straight to a geometry, which copies them,
// so no frame allocates.
const ARC_N = 24
const SEG = new Float32Array(6)
const ARC = new Float32Array(ARC_N * 3)

function setSeg(line: any, x0: number, y0: number, x1: number, y1: number, z = 0) {
  SEG[0] = x0; SEG[1] = y0; SEG[2] = z
  SEG[3] = x1; SEG[4] = y1; SEG[5] = z
  line?.geometry.setPositions(SEG)
  line?.computeLineDistances()
}

function setArc(line: any, cx: number, cy: number, r: number, a0: number, a1: number) {
  for (let i = 0; i < ARC_N; i++) {
    const a = a0 + ((a1 - a0) * i) / (ARC_N - 1)
    ARC[i * 3] = cx + r * Math.cos(a)
    ARC[i * 3 + 1] = cy + r * Math.sin(a)
    ARC[i * 3 + 2] = 0
  }
  line?.geometry.setPositions(ARC)
  line?.computeLineDistances()
}

const SEG_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]
const ARC_INIT: [number, number, number][] = Array.from(
  { length: ARC_N },
  (): [number, number, number] => [0, 0, 0]
)

// ── 3D: the power triangle with the correction drawn as a subtraction ──
// P lies flat from the origin and never moves when the target factor changes; Q1 stands up from
// its tip, Qc pulls back down from the top of Q1, and the two hypotenuses S1 (dashed) and S2 sit
// side by side so the shrink is literal.
const OX = -1.7
const OY = -0.75
const P_MAX_LEN = 2.6 // scene units at the longest P
const Q_MAX_LEN = 1.5 // scene units at the tallest Q1
const GUIDE_LOW = -1.12
const GUIDE_HIGH = 1.02

// Feeder bars, before and after, normalized so the longer one always spans FEED_LEN.
// Both start at FEED_X, well inside the frame, and a tick there marks the shared left end so the
// ratio of the two lengths can be read off directly.
const FEED_N = 40
const FEED_X = -0.62
const FEED_LEN = 1.95
const FEED_Y1 = 1.62
const FEED_Y2 = 1.12
const FLOW_REF = 4 // sheen bands per second on the before bar
const FEED_INIT: [number, number, number][] = Array.from(
  { length: FEED_N },
  (): [number, number, number] => [0, 0, 0]
)

function CorrectionTriangleScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  // 0 P, 1 Q1, 2 Qc, 3 S1 dashed, 4 S2, 5 θ1 arc, 6 θ2 arc, 7 P-invariance guide, 8 bar start tick
  const lineRefs = useRef<any[]>([])
  const feedRefs = useRef<any[]>([])
  const feedBuf = useMemo(() => new Float32Array(FEED_N * 3), [])
  const plateRef = useRef<Group>(null)
  const tipRefs = useRef<(Group | null)[]>([]) // Q1 tip, Q2 tip
  const tagRefs = useRef<(Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const flows = useMemo(() => [FLOW_REF, FLOW_REF], [])
  const labelledFor = useRef<Record<string, number> | null>(null)

  useFrame(() => {
    const s = solve(valuesRef.current)
    // One scale for P, Q and S, so the triangle's proportions stay true; it depends on P and
    // pf1 only, which is why moving pf2 cannot change the length of P.
    const k = Math.min(P_MAX_LEN / s.P, Q_MAX_LEN / Math.max(s.Q1, 1e-3))
    const px = OX + s.P * k
    const y1 = OY + s.Q1 * k
    const y2 = OY + s.Q2 * k

    setSeg(lineRefs.current[0], OX, OY, px, OY) // P, fixed by the P slider alone
    setSeg(lineRefs.current[1], px, OY, px, y1) // Q1 the load draws, lagging
    setSeg(lineRefs.current[2], px, y1, px, y2, 0.03) // Qc the bank supplies, pulling back down
    setSeg(lineRefs.current[3], OX, OY, px, y1) // S1 before
    setSeg(lineRefs.current[4], OX, OY, px, y2) // S2 after, visibly shorter
    setArc(lineRefs.current[5], OX, OY, 0.46, 0, s.th1)
    setArc(lineRefs.current[6], OX, OY, 0.3, 0, s.th2)
    setSeg(lineRefs.current[7], px, GUIDE_LOW, px, GUIDE_HIGH) // anchored at the tip of P
    setSeg(lineRefs.current[8], FEED_X, FEED_Y2 - 0.14, FEED_X, FEED_Y1 + 0.14) // shared bar start

    tipRefs.current[0]?.position.set(px, y1, 0)
    tipRefs.current[1]?.position.set(px, y2, 0.03)

    // Capacitor plates beside the Qc segment; their area follows C, so the voltage slider
    // changes the bank without touching a single kvar.
    const plateScale = Math.min(Math.max(Math.sqrt(s.Cd_uF / 260), 0.4), 1.7)
    plateRef.current?.position.set(px + 0.34, (y1 + y2) / 2, 0.03)
    plateRef.current?.scale.set(1, plateScale, plateScale)

    // Feeder current before and after: the ratio of the bars is the true ratio of the currents
    for (let b = 0; b < 2; b++) {
      const len = FEED_LEN * (b === 0 ? 1 : s.iRatio)
      const y = b === 0 ? FEED_Y1 : FEED_Y2
      for (let i = 0; i < FEED_N; i++) {
        const f = i / (FEED_N - 1)
        feedBuf[i * 3] = FEED_X + len * f
        feedBuf[i * 3 + 1] = y
        feedBuf[i * 3 + 2] = 0
      }
      const line = feedRefs.current[b]
      line?.geometry.setPositions(feedBuf)
      if (line) line.material.linewidth = 2 + 4 * (b === 0 ? 1 : s.iRatio)
      flows[b] = FLOW_REF * (b === 0 ? 1 : s.iRatio)
      // Label above its own bar, not past the end, so it never sits on the line
      tagRefs.current[b === 0 ? 5 : 6]?.position.set(FEED_X + 0.58, y + 0.23, 0)
    }

    tagRefs.current[0]?.position.set((OX + px) / 2, OY - 0.26, 0)
    tagRefs.current[1]?.position.set(px - 0.62, (OY + y1) / 2, 0)
    tagRefs.current[2]?.position.set(px + 0.95, (y1 + y2) / 2 + 0.26, 0)
    tagRefs.current[3]?.position.set(OX + (px - OX) * 0.4 - 0.1, OY + (y1 - OY) * 0.4 + 0.3, 0)
    tagRefs.current[4]?.position.set(OX + (px - OX) * 0.62, OY + (y2 - OY) * 0.62 - 0.28, 0)
    tagRefs.current[7]?.position.set(px + 0.46, OY - 0.25, 0) // beside the tip of P it annotates
    tagRefs.current[8]?.position.set(px + 0.95, (y1 + y2) / 2 - 0.02, 0)
    tagRefs.current[9]?.position.set(OX + 0.82 * Math.cos(s.th1 / 2), OY + 0.82 * Math.sin(s.th1 / 2), 0)

    // Values only change with the sliders or the locale, so the text is rebuilt only then
    if (labelledFor.current !== valuesRef.current && textRefs.current[9]) {
      labelledFor.current = valuesRef.current
      const L = (key: keyof typeof L10N) => L10N[key][locale]
      // pf2 is held at pf1 below it, so there is nothing to correct and the slider looks dead.
      // Say why in the readout instead of showing a bare zero.
      const qcNote = s.Qc <= 1e-9 ? ` · ${L('noNeed')}` : ''
      const texts = [
        `P ${fmtP(s.P)}`,
        `Q1 ${fmtQ(s.Q1)}`,
        `Qc ${fmtQ(s.Qc)}${qcNote}`,
        `S1 ${fmtS(s.S1)} · ${L('before')}`,
        `S2 ${fmtS(s.S2)} · ${L('after')}`,
        `I1 ${fmtA(s.I1)} · ${L('before')}`,
        `I2 ${fmtA(s.I2)} · ${L('loss')} −${(s.lossDrop * 100).toFixed(2)} %`,
        L('pFixed'),
        `C ${fmtC(s.Cd_uF)} · ${L('bank')}`,
        `θ1 ${deg(s.th1)} → θ2 ${deg(s.th2)}`,
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  const segs: { color: string; width: number; dashed?: boolean; arc?: boolean }[] = [
    { color: c.power, width: 3 },
    { color: c.inductance, width: 2.5 },
    { color: c.capacitance, width: 3.5 },
    { color: c.guide, width: 1.5, dashed: true },
    { color: c.voltage, width: 3 },
    { color: c.guide, width: 1, dashed: true, arc: true },
    { color: c.guide, width: 1.5, arc: true },
    { color: c.guide, width: 1, dashed: true },
    { color: c.guide, width: 1 },
  ]

  // Accent-tinted label text: P and the two feeder currents
  const tagTint: (string | undefined)[] = [
    c.power, undefined, undefined, undefined, undefined,
    c.current, c.current, c.power, undefined, undefined,
  ]

  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />

      {segs.map((sg, i) => (
        <Line
          key={i}
          ref={(l) => { lineRefs.current[i] = l }}
          points={sg.arc ? ARC_INIT : SEG_INIT}
          lineWidth={sg.width}
          color={sg.color}
          dashed={sg.dashed}
          dashSize={0.09}
          gapSize={0.07}
        />
      ))}

      {/* The capacitor bank: two plates whose area follows C */}
      <group ref={plateRef}>
        {[-0.07, 0.07].map((x) => (
          <mesh key={x} position={[x, 0, 0]}>
            <boxGeometry args={[0.025, 0.3, 0.3]} />
            <meshBasicMaterial color={c.capacitance} />
          </mesh>
        ))}
      </group>

      {/* Feeder current before and after correction */}
      {[0, 1].map((b) => (
        <TrailLine
          key={b}
          ref={(l) => { feedRefs.current[b] = l }}
          points={FEED_INIT}
          color={c.current}
          lineWidth={6}
          fade={false}
          flowSpeed={() => flows[b]}
        />
      ))}

      <group ref={(g) => { tipRefs.current[0] = g }}>
        <Marker color={c.marker} size={0.055} />
      </group>
      <group ref={(g) => { tipRefs.current[1] = g }}>
        <Marker color={c.marker} size={0.065} />
      </group>

      {tagTint.map((tint, i) => (
        <group key={i} ref={(g) => { tagRefs.current[i] = g }}>
          <Html center>
            <span
              ref={(el) => { textRefs.current[i] = el }}
              className={labelClass}
              style={tint ? { color: tint } : undefined}
            />
          </Html>
        </group>
      ))}
    </>
  )
}

// ── 2D: what the target factor costs, and where correction stops paying ──
// x is the factor after correction: lagging to the left of unity, leading to the right, so the
// over-correction branch continues past pf2 = 1.00 on the same axis.
const X_U = 0.9 // x of pf2 = 1.00
const X_L = -3.4 // x of pf2 = 0.80 lagging
const X_R = 3.4
const K_X = (X_U - X_L) / 0.2 // scene units per unit of (1 − pf)
const PF_LEAD = 1 - (X_R - X_U) / K_X // factor at the right edge, leading
const TAN_LEAD = Math.tan(Math.acos(PF_LEAD))
const X_95 = X_U - 0.05 * K_X

const N2 = 180
const Y_Q = 0.15 // Qc = 0 line, doubles as the x axis
const H_Q = 1.45
const Y_S = -1.85 // S = P line
const H_S = 1.45
const CURVE_INIT: [number, number, number][] = Array.from(
  { length: N2 },
  (): [number, number, number] => [0, 0, 0]
)

function CorrectionCurve2DScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  // 0 x axis, 1 unity guide, 2 pf2 = 0.95 guide, 3 Qc curve, 4 S2 curve, 5 S = P line, 6 Q1 level
  const lineRefs = useRef<any[]>([])
  const qBuf = useMemo(() => new Float32Array(N2 * 3), [])
  const sBuf = useMemo(() => new Float32Array(N2 * 3), [])
  const dotRefs = useRef<(Group | null)[]>([])
  const tagRefs = useRef<(Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<Record<string, number> | null>(null)

  useFrame(() => {
    const s = solve(valuesRef.current)
    const pfStart = Math.max(0.8, s.pf1) // below pf1 there is nothing to correct
    const xStart = X_U - (1 - pfStart) * K_X
    const tan1 = Math.tan(s.th1)
    const qcMax = s.P * (tan1 + TAN_LEAD) // Qc at the right edge, the largest on screen
    const sMax = s.P / Math.min(pfStart, PF_LEAD)
    const kQ = H_Q / Math.max(qcMax, 1e-6)
    const kS = H_S / Math.max(sMax - s.P, 1e-6)

    for (let i = 0; i < N2; i++) {
      const x = xStart + (X_R - xStart) * (i / (N2 - 1))
      const pf = Math.min(1, 1 - Math.abs(x - X_U) / K_X)
      const th2 = Math.sign(X_U - x) * Math.acos(pf) // positive lagging, negative leading
      const qc = s.P * (tan1 - Math.tan(th2))
      const s2 = s.P / pf
      qBuf[i * 3] = x
      qBuf[i * 3 + 1] = Y_Q + qc * kQ
      qBuf[i * 3 + 2] = 0
      sBuf[i * 3] = x
      sBuf[i * 3 + 1] = Y_S + (s2 - s.P) * kS
      sBuf[i * 3 + 2] = 0
    }
    lineRefs.current[3]?.geometry.setPositions(qBuf)
    lineRefs.current[4]?.geometry.setPositions(sBuf)

    setSeg(lineRefs.current[0], X_L, Y_Q, X_R, Y_Q)
    setSeg(lineRefs.current[1], X_U, Y_S - 0.1, X_U, Y_Q + H_Q + 0.12)
    setSeg(lineRefs.current[2], X_95, Y_S - 0.1, X_95, Y_Q + H_Q + 0.12)
    setSeg(lineRefs.current[5], X_L, Y_S, X_R, Y_S)
    const yQ1 = Y_Q + s.Q1 * kQ // at unity the bank has to supply the whole of Q1
    setSeg(lineRefs.current[6], X_L, yQ1, X_U + 0.25, yQ1)
    tagRefs.current[6]?.position.set(X_L + 0.42, yQ1 + 0.2, 0)

    // Live target on both curves
    const xm = X_U - (1 - s.pf2) * K_X
    const yQm = Y_Q + s.Qc * kQ
    const ySm = Y_S + (s.S2 - s.P) * kS
    dotRefs.current[0]?.position.set(xm, yQm, 0.01)
    dotRefs.current[1]?.position.set(xm, ySm, 0.01)
    tagRefs.current[7]?.position.set(xStart, Y_Q - 0.26, 0)
    tagRefs.current[8]?.position.set(Math.min(xm, 1.4), yQm + 0.26, 0)

    if (labelledFor.current !== valuesRef.current && textRefs.current[8]) {
      labelledFor.current = valuesRef.current
      const L = (key: keyof typeof L10N) => L10N[key][locale]
      // The worked example sits at pf2 = 0.95 whatever the sliders say
      const qc95 = s.P * (tan1 - Math.tan(Math.acos(0.95)))
      const texts = [
        `Qc 0 … ${fmtQ(qcMax)}`,
        `S2 ${fmtS(s.P)} … ${fmtS(sMax)}`,
        'pf2 = 1.00',
        `pf2 0.95 · Qc ${fmtQ(qc95)} · S2 ${fmtS(s.P / 0.95)} · ${L('example')}`,
        L('over'),
        L('floor'),
        `Q1 ${fmtQ(s.Q1)}`,
        `pf2 ${pfStart.toFixed(2)}`,
        s.Qc <= 1e-9
          ? `${L('target')} pf2 ${s.pf2.toFixed(2)} · Qc ${fmtQ(s.Qc)} · ${L('noNeed')}`
          : `${L('target')} pf2 ${s.pf2.toFixed(2)} · Qc ${fmtQ(s.Qc)} · S2 ${fmtS(s.S2)}`,
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  const segs: { color: string; width: number; dashed?: boolean; curve?: boolean }[] = [
    { color: c.guide, width: 1 },
    { color: c.guide, width: 1.5, dashed: true },
    { color: c.guide, width: 1, dashed: true },
    { color: c.capacitance, width: 2.5, curve: true },
    { color: c.voltage, width: 2.5, curve: true },
    { color: c.power, width: 1.5, dashed: true },
    { color: c.guide, width: 1, dashed: true },
  ]

  // Static label spots; indices 6 to 8 follow the sliders and are placed in useFrame
  const tags: { pos?: [number, number, number]; tint?: string }[] = [
    { pos: [X_L + 0.72, Y_Q + H_Q + 0.26, 0] },
    { pos: [X_L + 0.78, Y_S + H_S + 0.26, 0] },
    { pos: [X_U + 0.02, Y_Q + H_Q + 0.34, 0] },
    { pos: [X_95 - 1.35, Y_Q - 0.26, 0] },
    { pos: [2.3, Y_Q + H_Q + 0.34, 0] },
    { pos: [X_L + 0.5, Y_S + 0.2, 0], tint: c.power },
    {},
    {},
    {},
  ]

  return (
    <>
      {segs.map((sg, i) => (
        <Line
          key={i}
          ref={(l) => { lineRefs.current[i] = l }}
          points={sg.curve ? CURVE_INIT : SEG_INIT}
          lineWidth={sg.width}
          color={sg.color}
          dashed={sg.dashed}
          dashSize={0.08}
          gapSize={0.06}
        />
      ))}

      {[0, 1].map((i) => (
        <group key={i} ref={(g) => { dotRefs.current[i] = g }}>
          <Marker color={c.marker} size={0.06} />
        </group>
      ))}

      {tags.map((tg, i) => (
        <group key={i} ref={(g) => { tagRefs.current[i] = g }} position={tg.pos}>
          <Html center>
            <span
              ref={(el) => { textRefs.current[i] = el }}
              className={labelClass}
              style={tg.tint ? { color: tg.tint } : undefined}
            />
          </Html>
        </group>
      ))}
    </>
  )
}

const concept: ConceptMeta = {
  id: 'power-factor-correction',
  title: { en: 'Power Factor Correction', ko: '역률 개선 (콘덴서 용량 산정)' },
  subject: 'installations',
  order: 1,
  description: {
    en: `## Power Factor Correction

**The capacitor supplies the reactive power the load was pulling down the line, so the current drops while the real power P stays exactly the same.**

- Amber, horizontal: the real power P the load consumes. Watch its length while you move the target factor — it never changes, and the dashed guide pinned to its tip proves it
- Light neutral, standing up from the tip of P: Q1, the lagging reactive power the load draws
- Dark neutral, pulling back down from the top of Q1: Qc, the leading reactive power the bank supplies, with its plates drawn beside it
- Dashed faint neutral: S1, the apparent power before correction. Solid strongest neutral: S2, the shorter hypotenuse left after it
- Two cyan liquid bars: the feeder current before and after, in true proportion. Same P, less current

**Reactive power to install**: Qc = P · (tan θ1 − tan θ2), with θ = arccos(pf)
**Apparent power**: S = P ÷ pf, so S1 = P ÷ pf1 and S2 = P ÷ pf2
**Line current**: I = S ÷ (sqrt(3) · V), so I2 ÷ I1 = pf1 ÷ pf2
**Line loss**: proportional to I², so it falls by 1 − (pf1 ÷ pf2)²
**Capacitance at 60 Hz**: C = Qc ÷ (ω · V²) as a single-phase equivalent, or Qc ÷ (3 · ω · V²) for each of three units connected in delta across the line voltage V, which is what the plates stand for

Worked example: P = 100 kW at pf1 = 0.80 lagging, corrected to pf2 = 0.95, at V = 380 V. Q1 = 75.00 kvar and Q2 = 32.87 kvar, so Qc = 42.13 kvar. S falls from 125.00 kVA to 105.26 kVA, the current to 84.21 % of what it was, and the line loss by 29.09 %. The bank is 773.9 µF as a single-phase equivalent, or 258.0 µF per delta-connected unit.

Correction has a limit. Push the target past unity and the bank overshoots: the net reactive power turns leading, S grows again, and the current climbs back up. The 2D view carries that turnaround as the branch to the right of pf2 = 1.00.

Move pf2 and watch Qc eat into Q1 while P holds still. The last few hundredths before unity cost the most kvar, which is why 0.95 is a common place to stop. pf1 sets how much Q there was to cancel, P scales the whole triangle, and V changes only the capacitance in µF, never the kvar.

In 2D: the kvar the bank must supply and the apparent power that results, both against the factor after correction, from 0.80 lagging through unity and on into the over-corrected leading branch. The dashed line at pf2 = 0.95 marks the worked example, and the amber baseline is S = P, the floor S can never go below.`,
    ko: `## 역률 개선 (콘덴서 용량 산정)

**콘덴서가 부하의 무효전력을 대신 공급하면 전류는 줄어들지만, 유효전력 P는 조금도 변하지 않습니다.**

- 앰버 수평선: 부하가 소비하는 유효전력 P. 목표 역률을 움직이는 동안 길이를 보세요. 전혀 변하지 않으며, 끝점에 박아 둔 점선 안내선이 그것을 증명합니다
- P 끝에서 위로 서는 밝은 중성색: 부하가 끌어쓰는 지상 무효전력 Q1
- Q1 끝에서 다시 내려오는 짙은 중성색: 뱅크가 공급하는 진상 무효전력 Qc. 옆에 전극판을 함께 그렸습니다
- 희미한 점선 중성색: 개선 전 피상전력 S1. 가장 진한 실선 중성색: 개선 뒤 남은 더 짧은 빗변 S2
- 시안 액체 막대 둘: 개선 전과 후의 간선 전류를 실제 비율로 그렸습니다. P는 같은데 전류만 줄어듭니다

**필요한 콘덴서 용량**: Qc = P · (tan θ1 − tan θ2), θ = arccos(pf)
**피상전력**: S = P ÷ pf, 즉 S1 = P ÷ pf1, S2 = P ÷ pf2
**선전류**: I = S ÷ (sqrt(3) · V), 따라서 I2 ÷ I1 = pf1 ÷ pf2
**선로 손실**: I²에 비례하므로 1 − (pf1 ÷ pf2)² 만큼 줄어듭니다
**60 Hz에서의 정전용량**: C = Qc ÷ (ω · V²)가 단상 환산값이고, 선간전압 V에 Δ 결선한 세 대로 나누면 각 상은 Qc ÷ (3 · ω · V²)입니다. 장면의 전극판이 그 한 상입니다

계산 예: P = 100 kW, pf1 = 0.80 지상을 V = 380 V에서 pf2 = 0.95로 개선합니다. Q1 = 75.00 kvar, Q2 = 32.87 kvar이므로 Qc = 42.13 kvar입니다. S는 125.00 kVA에서 105.26 kVA로, 전류는 원래의 84.21 %로 줄고, 선로 손실은 29.09 % 감소합니다. 뱅크 용량은 단상 환산 773.9 µF, Δ 결선 각 상 258.0 µF입니다.

개선에도 한계가 있습니다. 목표를 역률 1 너머로 밀면 뱅크가 과보상이 되어 합성 무효전력이 진상으로 넘어가고, S가 다시 커지면서 전류도 올라갑니다. 2D 화면의 pf2 = 1.00 오른쪽 가지가 그 반전을 보여줍니다.

pf2를 움직이면 Qc가 Q1을 깎아 들어가는 동안 P는 꿈쩍도 하지 않습니다. 역률 1 바로 앞의 마지막 몇 백분의 일이 kvar를 가장 많이 먹기 때문에, 보통 0.95에서 멈춥니다. pf1은 상쇄할 Q의 양을 정하고, P는 삼각형 전체 크기를 키우며, V는 kvar는 그대로 둔 채 µF만 바꿉니다.

2D: 뱅크가 공급해야 하는 kvar와 그 결과로 남는 피상전력을 개선 후 역률에 대해 0.80 지상부터 역률 1을 지나 과보상 진상 구간까지 그렸습니다. pf2 = 0.95의 점선이 계산 예 지점이고, 앰버 기준선은 S가 결코 내려갈 수 없는 바닥 S = P입니다.`,
  },
  sliderParams: [
    { key: 'P_kW', label: { en: 'Load real power (P)', ko: '부하 유효전력 (P)' }, unit: 'kW', min: 20, max: 300, step: 10, defaultValue: 100 },
    { key: 'pf1', label: { en: 'Initial power factor (pf1, lagging)', ko: '개선 전 역률 (pf1, 지상)' }, min: 0.6, max: 0.95, step: 0.01, defaultValue: 0.8 },
    { key: 'pf2', label: { en: 'Target power factor (pf2)', ko: '목표 역률 (pf2)' }, min: 0.8, max: 1, step: 0.01, defaultValue: 0.95 },
    { key: 'V', label: { en: 'System voltage (V)', ko: '계통 전압 (V)' }, unit: 'V', min: 200, max: 480, step: 10, defaultValue: 380 },
  ],
  Scene: CorrectionTriangleScene,
  Scene2D: CorrectionCurve2DScene,
}

export default concept
