import { useRef, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { useLocaleStore } from '../../lib/i18n'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import type { ConceptMeta } from '../../types/concept'

const SQRT3 = Math.sqrt(3)
const REF_MVA = 100 // the common study base a utility refers everything to

// Pure physics: the per-unit bases, the per-unit impedance, and the fault it implies.
//   Zbase = Vb² ÷ Sb                 [Ω]
//   Ibase = Sb ÷ (sqrt(3)·Vb)        [A]
//   Zpu   = Z ÷ Zbase                [pu]   so Z = Zpu·Zbase
//   Psc   = Sb ÷ Zpu                 [VA]   short-circuit capacity at the bus
//   Isc   = Ibase ÷ Zpu              [A]    and equally Psc ÷ (sqrt(3)·Vb)
//   base change: Zpu_new = Zpu_old × (Sb_new ÷ Sb_old) × (Vb_old ÷ Vb_new)²
//
// Worked example, verified numerically. Base Sb = 10 MVA, Vb = 22.9 kV:
//   Zbase = 22900² ÷ 10e6               = 52.441 Ω
//   Ibase = 10e6 ÷ (sqrt(3) × 22900)    = 252.12 A
//   Zpu = 0.05 (%Z = 5 %) -> Z = 0.05 × 52.441 = 2.622 Ω
//   Psc = 10 ÷ 0.05                     = 200.0 MVA
//   Isc = 252.12 ÷ 0.05                 = 5042.4 A
//   Isc = 200e6 ÷ (sqrt(3) × 22900)     = 5042.4 A  -- the two routes agree
//   at %Z = 10 %: Psc = 100.0 MVA, Isc = 2521.2 A   -- twice the impedance, half the current
//   referred to 100 MVA: %Z = 5 × (100 ÷ 10) = 50 %, Psc = 100 ÷ 0.50 = 200.0 MVA, unchanged
function solve({ pctZ = 5, Sb_MVA = 10, Vb_kV = 22.9 }: Record<string, number>) {
  const Zpu = pctZ / 100
  const Sb = Sb_MVA * 1e6
  const Vb = Vb_kV * 1e3
  const Zbase = (Vb * Vb) / Sb
  const Ibase = Sb / (SQRT3 * Vb)
  const Psc = Sb / Zpu
  return {
    pctZ,
    Zpu,
    Sb,
    Zbase,
    Ibase,
    Z: Zpu * Zbase,
    Psc,
    Isc: Ibase / Zpu,
    IscFromPsc: Psc / (SQRT3 * Vb), // same number by the other route
    pctZOnRef: pctZ * (REF_MVA / Sb_MVA), // %Z restated on the 100 MVA study base
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

// On-screen wording, bilingual, built once at module scope
const L10N = {
  ohmHead: { en: 'ohmic world — log axis', ko: '옴의 세계 — 로그 축' },
  ohmNote: { en: 'slides with the base', ko: '기준이 바뀌면 통째로 움직임' },
  puHead: { en: 'per-unit world', ko: '퍼유닛 세계' },
  puNote: { en: 'stays put', ko: '제자리에 그대로' },
  gapNote: { en: 'the gap is the per-unit value', ko: '이 간격이 퍼유닛 값' },
  base: { en: 'base', ko: '기준' },
  same: { en: 'same', ko: '동일' },
  plotHead: { en: 'Isc and Psc against %Z', ko: '%Z에 따른 Isc와 Psc' },
  halfNote: { en: 'half the height', ko: '높이는 절반' },
} as const

const fmtOhm = (v: number) =>
  v >= 1000 ? `${(v / 1000).toFixed(2)} kΩ` : v >= 10 ? `${v.toFixed(2)} Ω` : `${v.toFixed(3)} Ω`
const fmtA = (v: number) => (v >= 1000 ? `${v.toFixed(0)} A` : `${v.toFixed(1)} A`)
const fmtMVA = (v: number) => `${(v / 1e6).toFixed(1)} MVA`
const fmtPct = (v: number) => `${v.toFixed(1)} %`

// Scene text reads in the foreground color; only current (cyan) and power (amber) tint their labels
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const accentClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums'
const axisClass = 'pointer-events-none whitespace-nowrap text-[10px] tabular-nums text-muted-foreground'

// Shared scratch, written then handed straight to a geometry, so no frame ever allocates
const SEG = new Float32Array(6)
function setSeg(line: any, x0: number, y0: number, z0: number, x1: number, y1: number, z1: number) {
  SEG[0] = x0; SEG[1] = y0; SEG[2] = z0
  SEG[3] = x1; SEG[4] = y1; SEG[5] = z1
  line?.geometry.setPositions(SEG)
  line?.computeLineDistances()
}

// ── 3D: two rulers for one transformer ──
// Left ruler: ohms on a log axis, because Zbase spans 0.44 Ω (6.6 kV, 100 MVA) to 4.7 kΩ
// (154 kV, 5 MVA). On a log axis the distance between Zbase and Z is log10(1 ÷ Zpu) decades, so
// it depends on %Z alone — move the base and the pair slides without the gap changing.
// Right ruler: the per-unit axis, 0 to 1 pu, linear and dimensionless. It never moves with the base.
const BASE_Y = -1.7
const OHM_X = -1.55
const PU_X = 0.3
const DEC_LO = -3 // 1 mΩ
const DEC_HI = 4 // 10 kΩ
const OHM_H = 3.0
const PER_DEC = OHM_H / (DEC_HI - DEC_LO)
const PU_H = 2.0 // scene units for 1.0 pu

const yOfOhm = (z: number) =>
  BASE_Y + Math.min(Math.max(Math.log10(z) - DEC_LO, 0), DEC_HI - DEC_LO) * PER_DEC
const yOfPu = (p: number) => BASE_Y + p * PU_H

const OHM_AXIS: [number, number, number][] = [[OHM_X, BASE_Y, 0], [OHM_X, BASE_Y + OHM_H, 0]]
const PU_AXIS: [number, number, number][] = [[PU_X, BASE_Y, 0], [PU_X, BASE_Y + PU_H, 0]]
const OHM_TICKS: [number, number, number][][] = Array.from(
  { length: DEC_HI - DEC_LO + 1 },
  (_, i): [number, number, number][] => {
    const y = BASE_Y + i * PER_DEC
    return [[OHM_X - 0.09, y, 0], [OHM_X + 0.09, y, 0]]
  }
)
const OHM_TICK_LABELS: { y: number; text: string }[] = [
  { y: yOfOhm(0.01), text: '0.01 Ω' },
  { y: yOfOhm(1), text: '1 Ω' },
  { y: yOfOhm(100), text: '100 Ω' },
  { y: yOfOhm(10000), text: '10 kΩ' },
]
const PU_TICKS: [number, number, number][][] = [0, 0.5, 1].map(
  (p): [number, number, number][] => [[PU_X - 0.09, yOfPu(p), 0], [PU_X + 0.09, yOfPu(p), 0]]
)
const PU_TICK_LABELS: { y: number; text: string }[] = [
  { y: yOfPu(0), text: '0' },
  { y: yOfPu(0.5), text: '0.5 pu' },
  { y: yOfPu(1), text: '1.0 pu' },
]

// The fault current: a liquid line whose length is literally K ÷ Zpu, so the shorter the per-unit
// segment, the longer and faster it runs. K is set so %Z = 2 % (the stiffest bus here) just fits.
const FAULT_Y = BASE_Y - 0.3
const FAULT_Z = 1.05
const FAULT_K = 0.052 // scene units × pu; 0.052 ÷ 0.02 = 2.6 units at the stiffest bus
const FAULT_PTS = 28
const FAULT_INIT: [number, number, number][] = Array.from(
  { length: FAULT_PTS },
  (): [number, number, number] => [0, 0, 0]
)
const SEG_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]

// Dimension bracket across the two Ω markers, drawn just right of the axis
const BRACKET_X = 0.2 // spine offset from the Ω axis
const VAL_DX = 0.5 // where the Ω value labels are centred, clear of both the axis and its tick labels
const BRACKET_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
const BRACKET = new Float32Array(12)
function setBracket(line: any, x: number, arm: number, y0: number, y1: number) {
  BRACKET[0] = x - arm; BRACKET[1] = y0; BRACKET[2] = 0
  BRACKET[3] = x; BRACKET[4] = y0; BRACKET[5] = 0
  BRACKET[6] = x; BRACKET[7] = y1; BRACKET[8] = 0
  BRACKET[9] = x - arm; BRACKET[10] = y1; BRACKET[11] = 0
  line?.geometry.setPositions(BRACKET)
  line?.computeLineDistances()
}

function PerUnitScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const faultBuf = useMemo(() => new Float32Array(FAULT_PTS * 3), [])
  const faultRef = useRef<any>(null)
  // 0 the %Z gap on the log axis, 1 Zbase tie across, 2 Z tie across
  const lineRefs = useRef<any[]>([])
  const bracketRef = useRef<any>(null)
  // 0 Zbase, 1 Z, 2 Zpu, 3 fault head
  const dotRefs = useRef<(THREE.Group | null)[]>([])
  // 0 Zbase, 1 Z, 2 gap, 3 Zpu, 4 Ibase, 5 Isc, 6 Psc, 7 base change
  const tagRefs = useRef<(THREE.Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<unknown>(null)
  const flow = useRef(0)

  useFrame(() => {
    const s = solve(valuesRef.current)
    const yBase = yOfOhm(s.Zbase)
    const yZ = yOfOhm(s.Z)
    const yPu = yOfPu(s.Zpu)

    // The ohmic pair and the gap between them, plus a dimension bracket measuring that gap, so the
    // "this distance never changes" reading comes across as a shape and not only as numbers
    setSeg(lineRefs.current[0], OHM_X, yZ, 0, OHM_X, yBase, 0)
    setBracket(bracketRef.current, OHM_X + BRACKET_X, 0.12, yZ, yBase)
    // Same two facts, read on the per-unit ruler: Zbase is 1.0 pu, Z is Zpu
    setSeg(lineRefs.current[1], OHM_X, yBase, 0, PU_X, yOfPu(1), 0)
    setSeg(lineRefs.current[2], OHM_X, yZ, 0, PU_X, yPu, 0)

    dotRefs.current[0]?.position.set(OHM_X, yBase, 0)
    dotRefs.current[1]?.position.set(OHM_X, yZ, 0)
    dotRefs.current[2]?.position.set(PU_X, yPu, 0)

    // Fault current, length = FAULT_K ÷ Zpu: the reciprocal of the per-unit segment
    const len = FAULT_K / s.Zpu
    for (let i = 0; i < FAULT_PTS; i++) {
      const f = i / (FAULT_PTS - 1)
      faultBuf[i * 3] = OHM_X + len * f
      faultBuf[i * 3 + 1] = FAULT_Y
      faultBuf[i * 3 + 2] = FAULT_Z
    }
    faultRef.current?.geometry.setPositions(faultBuf)
    dotRefs.current[3]?.position.set(OHM_X + len, FAULT_Y, FAULT_Z)
    flow.current = 0.12 / s.Zpu // sheen bands per second, also ∝ 1 ÷ Zpu

    // Value labels sit to the right of the Ω axis; the decade tick labels own the space to its left
    tagRefs.current[0]?.position.set(OHM_X + VAL_DX, yBase + 0.17, 0)
    tagRefs.current[1]?.position.set(OHM_X + VAL_DX, yZ - 0.17, 0)
    // The per-unit readout is anchored to the bracket it belongs to, just right of its spine
    tagRefs.current[2]?.position.set(OHM_X + BRACKET_X + 0.22, (yBase + yZ) / 2, 0)
    tagRefs.current[3]?.position.set(PU_X + 0.62, yPu - 0.02, 0)
    tagRefs.current[4]?.position.set(OHM_X + len + 0.1, FAULT_Y + 0.46, FAULT_Z)
    tagRefs.current[5]?.position.set(OHM_X + len + 0.1, FAULT_Y + 0.22, FAULT_Z)

    // Readouts move only with the sliders (or the locale), so rebuild the text only then
    if (labelledFor.current !== valuesRef.current && textRefs.current[7]) {
      labelledFor.current = valuesRef.current
      const texts = [
        `Zbase = ${fmtOhm(s.Zbase)}`,
        `Z = ${fmtOhm(s.Z)}`,
        `= ${s.Zpu.toFixed(3)} pu`,
        `Z = ${s.Zpu.toFixed(3)} pu (${fmtPct(s.pctZ)})`,
        `Ibase = ${fmtA(s.Ibase)}`,
        `Isc = Ibase ÷ Zpu = ${fmtA(s.Isc)} (${(1 / s.Zpu).toFixed(1)} pu)`,
        `Psc = Sb ÷ Zpu = ${fmtMVA(s.Psc)}`,
        `${REF_MVA} MVA ${L10N.base[locale]}: %Z = ${fmtPct(s.pctZOnRef)} → Psc = ${fmtMVA(s.Psc)} (${L10N.same[locale]})`,
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  const ties: { color: string; width: number; dashed?: boolean }[] = [
    { color: c.resistance, width: 3 },
    { color: c.guide, width: 1, dashed: true },
    { color: c.guide, width: 1, dashed: true },
  ]
  // 0 Zbase, 1 Z, 2 gap, 3 Zpu are neutral; 4 Ibase and 5 Isc are current; 6 Psc and 7 base change are power
  const tags: { accent?: string }[] = [
    {}, {}, {}, {}, { accent: c.current }, { accent: c.current },
  ]

  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />

      {/* Background, lights, grid, and bloom come from the shared SceneStage */}
      <Line points={OHM_AXIS} lineWidth={1.5} color={c.guide} />
      {/* The whole per-unit base, 0 to 1 pu, for the short Z segment to be read against */}
      <Line points={PU_AXIS} lineWidth={2} color={c.inductance} />
      {OHM_TICKS.map((pts, i) => (
        <Line key={`o${i}`} points={pts} lineWidth={1} color={c.guide} />
      ))}
      {PU_TICKS.map((pts, i) => (
        <Line key={`p${i}`} points={pts} lineWidth={1} color={c.guide} />
      ))}
      {ties.map((t, i) => (
        <Line
          key={i}
          ref={(l) => { lineRefs.current[i] = l }}
          points={SEG_INIT}
          lineWidth={t.width}
          color={t.color}
          dashed={t.dashed}
          dashSize={0.08}
          gapSize={0.06}
        />
      ))}

      {/* Measures the %Z gap: its height depends on %Z alone, so it holds still as the base moves */}
      <Line ref={bracketRef} points={BRACKET_INIT} lineWidth={1.5} color={c.resistance} />

      {/* Short-circuit current at the bus: length and sheen speed both ∝ 1 ÷ Zpu */}
      <TrailLine
        ref={faultRef}
        points={FAULT_INIT}
        lineWidth={4}
        color={c.current}
        fade={false}
        flowSpeed={() => flow.current}
      />
      <group position={[OHM_X, FAULT_Y, FAULT_Z]}>
        <Marker color={c.marker} size={0.05} />
      </group>

      {[0, 1, 2, 3].map((i) => (
        <group key={i} ref={(g) => { dotRefs.current[i] = g }}>
          <Marker color={i === 3 ? c.current : c.marker} size={0.055} />
        </group>
      ))}

      {tags.map((tag, i) => (
        <group key={i} ref={(g) => { tagRefs.current[i] = g }}>
          <Html center>
            <span
              ref={(el) => { textRefs.current[i] = el }}
              className={tag.accent ? accentClass : labelClass}
              style={tag.accent ? { color: tag.accent } : undefined}
            />
          </Html>
        </group>
      ))}

      {/* The base-free facts: the fault capacity, and the same %Z restated on a 100 MVA base */}
      <Html position={[1.0, 0.78, 0]} center>
        <div className="pointer-events-none leading-4" style={{ color: c.power }}>
          <div ref={(el) => { textRefs.current[6] = el }} className={accentClass} />
          <div ref={(el) => { textRefs.current[7] = el }} className={accentClass} />
        </div>
      </Html>

      <Html position={[OHM_X, BASE_Y + OHM_H + 0.26, 0]} center>
        <div className="pointer-events-none text-center leading-4">
          <div className={labelClass}>{L10N.ohmHead[locale]}</div>
          <div className={axisClass}>{L10N.ohmNote[locale]}</div>
          <div className={axisClass}>{L10N.gapNote[locale]}</div>
        </div>
      </Html>
      <Html position={[PU_X + 0.2, BASE_Y + PU_H + 0.26, 0]} center>
        <div className="pointer-events-none text-center leading-4">
          <div className={labelClass}>{L10N.puHead[locale]}</div>
          <div className={axisClass}>{L10N.puNote[locale]}</div>
        </div>
      </Html>
      {OHM_TICK_LABELS.map((t) => (
        <Html key={t.text} position={[OHM_X - 0.44, t.y, 0]} center>
          <span className={axisClass}>{t.text}</span>
        </Html>
      ))}
      {PU_TICK_LABELS.map((t) => (
        <Html key={t.text} position={[PU_X - 0.42, t.y, 0]} center>
          <span className={axisClass}>{t.text}</span>
        </Html>
      ))}
    </>
  )
}

// ── 2D: the hyperbola — Isc and Psc both ∝ 1 ÷ %Z, so one curve carries both ──
// Heights are the fault relative to its value at %Z = 2 %, which makes the curve exactly 1 ÷ %Z and
// puts the 10 % marker at literally half the height of the 5 % marker.
const PCT_MIN = 2
const PCT_MAX = 15
const X2_0 = -3.3
const X2_1 = 3.1
const Y2_0 = -1.55
const PLOT_H2 = 3.1
const xOfPct = (p: number) => X2_0 + ((p - PCT_MIN) / (PCT_MAX - PCT_MIN)) * (X2_1 - X2_0)
const yOfRatio = (r: number) => Y2_0 + r * PLOT_H2

const CURVE_N = 181
const CURVE: [number, number, number][] = Array.from(
  { length: CURVE_N },
  (_, i): [number, number, number] => {
    const p = PCT_MIN + (i / (CURVE_N - 1)) * (PCT_MAX - PCT_MIN)
    return [xOfPct(p), yOfRatio(PCT_MIN / p), 0]
  }
)
const AXES2: [number, number, number][] = [
  [X2_0 - 0.2, Y2_0 + PLOT_H2 + 0.2, 0], [X2_0 - 0.2, Y2_0, 0], [X2_1 + 0.25, Y2_0, 0],
]
// Each marked %Z: across to the curve, then down to the axis
const MARKS = [5, 10].map((p) => ({
  pct: p,
  x: xOfPct(p),
  y: yOfRatio(PCT_MIN / p),
  guide: [
    [X2_0 - 0.2, yOfRatio(PCT_MIN / p), 0], [xOfPct(p), yOfRatio(PCT_MIN / p), 0], [xOfPct(p), Y2_0, 0],
  ] as [number, number, number][],
}))
const X_TICKS2 = [2, 5, 10, 15].map((p) => ({ p, x: xOfPct(p) }))

function PerUnit2DScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  // 0 at 2 %, 1 at 5 %, 2 at 10 %, 3 the base readout
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<unknown>(null)

  useFrame(() => {
    if (labelledFor.current === valuesRef.current || !textRefs.current[3]) return
    labelledFor.current = valuesRef.current
    const s = solve(valuesRef.current)
    const line = (p: number) => `${p} % → ${fmtA(s.Ibase / (p / 100))} · ${fmtMVA(s.Sb / (p / 100))}`
    if (textRefs.current[0]) textRefs.current[0]!.textContent = line(2)
    if (textRefs.current[1]) textRefs.current[1]!.textContent = line(5)
    if (textRefs.current[2]) textRefs.current[2]!.textContent = `${line(10)} (÷ 2)`
    textRefs.current[3]!.textContent =
      `Zbase = ${fmtOhm(s.Zbase)} · Ibase = ${fmtA(s.Ibase)} · Z = ${fmtOhm(s.Z)} = ${s.Zpu.toFixed(3)} pu`
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  return (
    <>
      <Line points={AXES2} lineWidth={1} color={c.guide} />
      {MARKS.map((m) => (
        <Line
          key={m.pct}
          points={m.guide}
          lineWidth={1}
          color={c.guide}
          dashed
          dashSize={0.08}
          gapSize={0.06}
        />
      ))}
      {/* Isc and Psc both fall as 1 ÷ %Z, so a single hyperbola carries both readings */}
      <Line points={CURVE} lineWidth={3} color={c.power} />

      <group position={[xOfPct(PCT_MIN), yOfRatio(1), 0.01]}>
        <Marker color={c.marker} size={0.06} />
      </group>
      {MARKS.map((m) => (
        <group key={m.pct} position={[m.x, m.y, 0.01]}>
          <Marker color={c.marker} size={0.065} />
        </group>
      ))}

      <Html position={[xOfPct(PCT_MIN) + 0.18, yOfRatio(1) + 0.16, 0]}>
        <span ref={(el) => { textRefs.current[0] = el }} className={axisClass} />
      </Html>
      <Html position={[MARKS[0].x + 0.18, MARKS[0].y + 0.2, 0]}>
        <span
          ref={(el) => { textRefs.current[1] = el }}
          className={accentClass}
          style={{ color: c.current }}
        />
      </Html>
      <Html position={[MARKS[1].x + 0.18, MARKS[1].y + 0.2, 0]}>
        <span
          ref={(el) => { textRefs.current[2] = el }}
          className={accentClass}
          style={{ color: c.current }}
        />
      </Html>
      <Html position={[0.1, Y2_0 + PLOT_H2 + 0.1, 0]} center>
        <div className="pointer-events-none text-center leading-4">
          <div className={labelClass}>{L10N.plotHead[locale]}</div>
          <div className={axisClass}>{L10N.halfNote[locale]}: 5 % → 10 %</div>
        </div>
      </Html>
      <Html position={[0.1, Y2_0 - 0.42, 0]} center>
        <span ref={(el) => { textRefs.current[3] = el }} className={labelClass} />
      </Html>
      {X_TICKS2.map((t) => (
        <Html key={t.p} position={[t.x, Y2_0 - 0.16, 0]} center>
          <span className={axisClass}>{t.p} %</span>
        </Html>
      ))}
    </>
  )
}

const concept: ConceptMeta = {
  id: 'per-unit-short-circuit',
  title: { en: 'Per-Unit System & Short-Circuit Current', ko: '퍼유닛(%Z) 환산과 단락전류' },
  subject: 'power-systems',
  order: 2,
  description: {
    en: `## Per-Unit System & Short-Circuit Current

**Per-unit impedance is a scale-free number: the ohms slide with the base, %Z does not.**

Two rulers describing one transformer.
- Left, ohms on a log axis: Zbase and Z both slide by decades when the base changes, yet the gap between them never changes, because that gap is the per-unit value itself.
- Right, a plain 0 to 1 pu axis: Z sits as a short segment of the 1.0 pu base and stays exactly where it is.
- Cyan liquid line: the short-circuit current at the bus, drawn with its length as the reciprocal of the per-unit segment. Shorter %Z segment, longer and faster line.
- Amber: the short-circuit capacity Psc, and the same %Z restated on a 100 MVA base.

**Bases**: Zbase = Vb² ÷ Sb, Ibase = Sb ÷ (sqrt(3)·Vb)
**Per unit**: Zpu = Z ÷ Zbase
**Fault**: Psc = Sb ÷ Zpu, Isc = Ibase ÷ Zpu = Psc ÷ (sqrt(3)·Vb)
**Base change**: Zpu_new = Zpu_old × (Sb_new ÷ Sb_old) × (Vb_old ÷ Vb_new)²

At Sb = 10 MVA, Vb = 22.9 kV, %Z = 5 %: Zbase = 52.441 Ω, Ibase = 252.12 A, Z = 2.622 Ω, Psc = 200.0 MVA, and Isc = 5042.4 A whether you divide Ibase by Zpu or divide Psc by sqrt(3)·Vb. That the two routes land on the same ampere is the payoff of working in per unit.

Move %Z: 5 % gives 5042 A, 10 % gives 2521 A, so doubling the impedance halves the fault current. Move Sb: at a fixed %Z a bigger base MVA is a stiffer source, so Zbase shrinks while Ibase and Psc grow. Move Vb alone: Zbase and Ibase swing by orders of magnitude and the per-unit bar does not budge, Psc does not change at all, and only the ampere reading converts, because amperes belong to a voltage level and per unit does not.

In 2D: Isc and Psc against %Z. Both fall as 1 ÷ %Z, so one hyperbola carries both, with the 10 % marker at literally half the height of the 5 % marker.`,
    ko: `## 퍼유닛(%Z) 환산과 단락전류

**퍼유닛 임피던스는 크기에 매이지 않는 수입니다. 기준이 바뀌면 옴 값은 통째로 움직이지만 %Z는 그대로입니다.**

같은 변압기를 두 개의 자로 재어 봅니다.
- 왼쪽, 로그 축의 옴: 기준이 바뀌면 Zbase와 Z가 몇 자릿수씩 함께 움직이지만 둘 사이 간격은 변하지 않습니다. 그 간격이 바로 퍼유닛 값입니다.
- 오른쪽, 0에서 1 pu까지의 평범한 축: Z는 1.0 pu 기준의 짧은 한 토막으로 놓이고, 기준이 바뀌어도 자리를 지킵니다.
- 시안 액체 선: 모선의 단락전류. 길이를 퍼유닛 토막의 역수로 그렸습니다. %Z 토막이 짧을수록 선은 길고 빨라집니다.
- 앰버: 단락용량 Psc, 그리고 같은 %Z를 100 MVA 기준으로 다시 쓴 값입니다.

**기준량**: Zbase = Vb² ÷ Sb, Ibase = Sb ÷ (sqrt(3)·Vb)
**퍼유닛**: Zpu = Z ÷ Zbase
**단락**: Psc = Sb ÷ Zpu, Isc = Ibase ÷ Zpu = Psc ÷ (sqrt(3)·Vb)
**기준 환산**: Zpu_new = Zpu_old × (Sb_new ÷ Sb_old) × (Vb_old ÷ Vb_new)²

Sb = 10 MVA, Vb = 22.9 kV, %Z = 5 %에서 Zbase = 52.441 Ω, Ibase = 252.12 A, Z = 2.622 Ω, Psc = 200.0 MVA이고, Ibase를 Zpu로 나누든 Psc를 sqrt(3)·Vb로 나누든 Isc = 5042.4 A가 나옵니다. 서로 다른 두 길이 같은 전류에 닿는 것이 퍼유닛으로 계산하는 보람입니다.

%Z를 움직여 보세요. 5 %에서 5042 A, 10 %에서 2521 A — 임피던스가 두 배면 단락전류는 절반입니다. Sb를 움직이면, %Z가 같을 때 기준 용량이 클수록 더 튼튼한 전원이므로 Zbase는 작아지고 Ibase와 Psc는 커집니다. Vb만 움직이면 Zbase와 Ibase는 몇 자릿수씩 흔들리는데 퍼유닛 막대는 꼼짝하지 않고 Psc도 전혀 변하지 않습니다. 암페어 값만 환산될 뿐입니다. 암페어는 전압 등급에 속한 값이고 퍼유닛은 그렇지 않기 때문입니다.

2D: %Z에 따른 Isc와 Psc입니다. 둘 다 1 ÷ %Z로 줄어들어 하나의 쌍곡선이 두 값을 함께 나타내며, 10 % 점은 5 % 점의 정확히 절반 높이에 있습니다.`,
  },
  sliderParams: [
    { key: 'pctZ',   label: { en: 'Impedance (%Z)',    ko: '퍼센트 임피던스 (%Z)' }, unit: '%',   min: 2,   max: 15,  step: 0.5, defaultValue: 5    },
    { key: 'Sb_MVA', label: { en: 'Base power (Sb)',   ko: '기준 용량 (Sb)' },       unit: 'MVA', min: 5,   max: 100, step: 5,   defaultValue: 10   },
    { key: 'Vb_kV',  label: { en: 'Base voltage (Vb)', ko: '기준 전압 (Vb)' },       unit: 'kV',  min: 6.6, max: 154, step: 0.1, defaultValue: 22.9 },
  ],
  Scene: PerUnitScene,
  Scene2D: PerUnit2DScene,
}

export default concept
