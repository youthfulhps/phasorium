import { useRef, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { useLocaleStore } from '../../lib/i18n'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import type { ConceptMeta } from '../../types/concept'

const HOURS = 24
const SAMPLES = 97 // one sample every 15 minutes, 0 h … 24 h inclusive
const SHARE_A = 6 / 15 // the three groups split the maximum demand 6 : 5 : 4,
const SHARE_B = 5 / 15 // which is 120 : 100 : 80 kW at the default 300 kW
const SHARE_C = 4 / 15
const PEAK_GAP = 8 // hours between neighbouring group peaks at full spread
const BISECT = 22

/**
 * One group's load at hour t, per unit of its own maximum: a smooth bell of half-width W
 * sitting on a base b. Peak is exactly 1 at `peak`, and the day-long mean is exactly
 * b + (1 − b)·W ÷ 24, because the smoothstep bell encloses the same area as a tent of
 * the same half-width. That is what lets the load factor set the shape exactly.
 */
function unitLoad(t: number, peak: number, b: number, W: number) {
  let d = (t - peak) % HOURS
  if (d > HOURS / 2) d -= HOURS
  if (d < -HOURS / 2) d += HOURS
  const u = 1 - Math.abs(d) / W
  return u <= 0 ? b : b + (1 - b) * u * u * (3 - 2 * u)
}

let peakHourOut = 0 // hour of the coincident peak, written by coincidentPeak (no per-frame allocation)

/** Largest value of the three groups added together, per unit of the maximum demand. */
function coincidentPeak(spread: number, b: number, W: number) {
  const pa = 12 + PEAK_GAP * spread
  const pc = 12 - PEAK_GAP * spread
  let best = 0
  for (let i = 0; i < SAMPLES - 1; i++) {
    const t = (i * HOURS) / (SAMPLES - 1)
    const sum =
      SHARE_A * unitLoad(t, pa, b, W) +
      SHARE_B * unitLoad(t, 12, b, W) +
      SHARE_C * unitLoad(t, pc, b, W)
    if (sum > best) {
      best = sum
      peakHourOut = t
    }
  }
  return best
}

// Pure physics of the capacity chain. Connected load is never all on at once, the peak is never
// held all day, and separate customers do not peak at the same minute:
//   demand factor    = maximum demand ÷ connected load
//   load factor      = average load ÷ maximum demand   (= energy ÷ (maximum demand × hours))
//   diversity factor = sum of the group maximum demands ÷ combined maximum demand   (always ≥ 1)
//   transformer      = sum of (connected load × demand factor) ÷ diversity factor, at unity pf
// Verified worked example: 500 kW connected × 0.60 → maximum demand 300 kW. 4320 kWh in 24 h →
// average 4320 ÷ 24 = 180 kW → load factor 180 ÷ 300 = 0.600. Group maxima 120 + 100 + 80 = 300 kW
// against a coincident 250 kW → diversity factor 300 ÷ 250 = 1.200. Transformer 300 ÷ 1.200 =
// 250.0 kVA. Cross-check: average = load factor × demand factor × connected load = 0.6 × 0.6 × 500
// = 180 kW. Note that load factor × diversity factor is the combined load factor, so it can never
// exceed 1: a perfectly flat load has no diversity left to win, which is why `div` is capped.
function solve({ CL = 500, DF = 0.6, LF = 0.6, DIV = 1.2 }: Record<string, number>) {
  const W = HOURS * Math.min(LF, 0.5) // ≤ 12 h, so the bell never laps the day
  const b = Math.max(0, 2 * LF - 1) // a base load appears once the bell alone cannot be this fat
  const maxDemand = CL * DF
  const average = LF * maxDemand

  // Walk the group peaks apart until the coincident peak matches the requested diversity factor
  const divMax = 1 / coincidentPeak(1, b, W)
  const target = Math.min(Math.max(DIV, 1), divMax)
  let lo = 0
  let hi = 1
  for (let k = 0; k < BISECT; k++) {
    const mid = (lo + hi) / 2
    if (1 / coincidentPeak(mid, b, W) < target) lo = mid
    else hi = mid
  }
  const spread = (lo + hi) / 2
  const peakUnit = coincidentPeak(spread, b, W)
  const coincident = maxDemand * peakUnit // what the transformer actually sees

  return {
    CL,
    DF,
    LF,
    b,
    W,
    spread,
    maxDemand,
    average,
    coincident,
    kVA: coincident, // unity power factor; divide by pf otherwise
    diversity: maxDemand / coincident,
    divMax,
    peakHour: peakHourOut,
    peakA: 12 + PEAK_GAP * spread,
    peakB: 12,
    peakC: 12 - PEAK_GAP * spread,
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

const fmtKW = (p: number) => `${p.toFixed(p < 100 ? 1 : 0)} kW`
const fmtF = (f: number) => f.toFixed(3)
const fmtClock = (h: number) => {
  const w = ((h % HOURS) + HOURS) % HOURS
  const m = Math.round(w * 60)
  return `${String(Math.floor(m / 60) % HOURS).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}`
}
// Scene labels read in the foreground color; only current (cyan) and power (amber) tint their text
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const smallClass = 'pointer-events-none whitespace-nowrap text-[10px] font-medium tabular-nums text-foreground/60'

const L = {
  en: {
    connected: 'Connected load',
    maxDemand: 'Max demand',
    average: 'Average',
    transformer: 'Transformer',
    demand: 'demand factor',
    load: 'load factor',
    diversity: 'diversity factor',
    capped: 'limited by load factor',
    groupA: 'Group A',
    groupB: 'Group B',
    groupC: 'Group C',
    h0: '00:00',
    h12: '12:00',
    h24: '24:00',
  },
  ko: {
    connected: '설비용량',
    maxDemand: '최대수요전력',
    average: '평균전력',
    transformer: '변압기',
    demand: '수용률',
    load: '부하율',
    diversity: '부등률',
    capped: '부하율 한계',
    groupA: 'A 수용가',
    groupB: 'B 수용가',
    groupC: 'C 수용가',
    h0: '00:00',
    h12: '12:00',
    h24: '24:00',
  },
} satisfies Record<string, Record<string, string>>

// Shared scratch: filled and handed straight to a geometry, so no frame ever allocates
const SEG = new Float32Array(6)
const RECT = new Float32Array(15)
const LOAD = new Float32Array(4) // group A, B, C and their sum at one instant

function setSeg(line: any, x0: number, y0: number, x1: number, y1: number, z = 0) {
  SEG[0] = x0; SEG[1] = y0; SEG[2] = z
  SEG[3] = x1; SEG[4] = y1; SEG[5] = z
  line?.geometry.setPositions(SEG)
  line?.computeLineDistances()
}

// ── 3D: one day of load, three groups behind and the transformer's own curve in front ──
const X0 = -1.6
const X1 = 1.6
const Y0 = -0.95 // zero power; high enough that nothing reaches the slider HUD at the screen's foot
const Y_SPAN = 2.0 // the connected load always fills this height, so the gaps read as ratios
const Z_FRONT = 0.5
// The three groups recede behind the combined curve, latest peak deepest: receding carries a point
// rightward on screen, so this ordering spreads the group labels apart instead of collapsing them.
const LANE_Z = [-1.0, -0.5, 0.0]
const Y_CONNECTED = Y0 + Y_SPAN
const PILE_N = 20 // one block per 100 kW of connected load, up to the slider's 2000 kW
const PILE_PITCH = 0.15
const PILE_SIZE = 0.1
const PILE_Y = Y_CONNECTED + PILE_SIZE / 2 // the blocks ride the connected-load guide they add up to
const TX_X = 1.8 // the transformer block, clear of the plot and of the floating card
const TX_W = 0.26
// Each group's label sits above its own peak, pushed right by lane so the left margin stays free
// for the max-demand and average labels, which sweep the full height of the plot.
const LABEL_SIDE = [0, 0.3, 0.6]
const LABEL_LIFT = [0.4, 0.26, 0.12]

const timeX = (t: number) => X0 + ((X1 - X0) * t) / HOURS
const CURVE_INIT: [number, number, number][] = Array.from({ length: SAMPLES }, () => [0, 0, 0])
const AXIS: [number, number, number][] = [[X0, Y0, Z_FRONT], [X1, Y0, Z_FRONT]]
const CONNECTED_GUIDE: [number, number, number][] = [
  [X0, Y_CONNECTED, Z_FRONT],
  [X1, Y_CONNECTED, Z_FRONT],
]

function DemandFactorsScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  // Group A, B, C, then the combined curve
  const bufs = useMemo(() => [0, 1, 2, 3].map(() => new Float32Array(SAMPLES * 3)), [])
  const curveRefs = useRef<any[]>([])
  const guideRefs = useRef<any[]>([]) // max demand, average, required capacity
  const tagRefs = useRef<(THREE.Group | null)[]>([]) // 0 connected, 1 MD, 2 avg, 3 kVA, 4 factors, 5-7 groups
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const txRef = useRef<THREE.Mesh>(null)
  const pileRefs = useRef<(THREE.Mesh | null)[]>([])
  const peakRef = useRef<THREE.Group>(null)
  const flow = useRef(0)
  const labelledFor = useRef<unknown>(null)

  useFrame(() => {
    const s = solve(valuesRef.current)
    const k = Y_SPAN / s.CL // scene units per kW
    const pA = s.maxDemand * SHARE_A
    const pB = s.maxDemand * SHARE_B
    const pC = s.maxDemand * SHARE_C

    for (let i = 0; i < SAMPLES; i++) {
      const t = (i * HOURS) / (SAMPLES - 1)
      const x = timeX(t)
      // The combined curve is literally the sum of the three drawn above it
      LOAD[0] = pA * unitLoad(t, s.peakA, s.b, s.W)
      LOAD[1] = pB * unitLoad(t, s.peakB, s.b, s.W)
      LOAD[2] = pC * unitLoad(t, s.peakC, s.b, s.W)
      LOAD[3] = LOAD[0] + LOAD[1] + LOAD[2]
      for (let g = 0; g < 4; g++) {
        bufs[g][i * 3] = x
        bufs[g][i * 3 + 1] = Y0 + k * LOAD[g]
        bufs[g][i * 3 + 2] = g === 3 ? Z_FRONT : LANE_Z[g]
      }
    }
    for (let g = 0; g < 4; g++) curveRefs.current[g]?.geometry.setPositions(bufs[g])

    // Sheen speed follows how hard the transformer is worked
    flow.current = 0.6 + 2.4 * (s.coincident / s.CL)

    const yMD = Y0 + k * s.maxDemand
    const yAvg = Y0 + k * s.average
    const yReq = Y0 + k * s.kVA
    setSeg(guideRefs.current[0], X0, yMD, X1, yMD, Z_FRONT)
    setSeg(guideRefs.current[1], X0, yAvg, X1, yAvg, Z_FRONT)
    setSeg(guideRefs.current[2], X0, yReq, X1, yReq, Z_FRONT)

    // The transformer is exactly as tall as the capacity it has to supply
    const h = Math.max(k * s.kVA, 1e-3)
    txRef.current?.scale.set(1, h, 1)
    txRef.current?.position.set(TX_X, Y0 + h / 2, Z_FRONT)

    // The nameplate pile: one block per 100 kW of connected load
    const blocks = Math.round(s.CL / 100)
    for (let i = 0; i < PILE_N; i++) {
      const m = pileRefs.current[i]
      if (m) m.visible = i < blocks
    }

    peakRef.current?.position.set(timeX(s.peakHour), Y0 + k * s.coincident, Z_FRONT)

    tagRefs.current[0]?.position.set(X0 + 0.1, Y_CONNECTED + 0.42, Z_FRONT)
    tagRefs.current[1]?.position.set(X0 - 0.05, yMD + 0.16, Z_FRONT)
    // Below its own line: the max-demand label is above its line and the two lines can meet
    tagRefs.current[2]?.position.set(X0 - 0.05, yAvg - 0.16, Z_FRONT)
    // A caption at the block's foot: above the block it would run into the floating card
    tagRefs.current[3]?.position.set(TX_X, Y0 - 0.32, Z_FRONT)
    tagRefs.current[4]?.position.set(0, Y0 - 0.45, Z_FRONT)
    tagRefs.current[5]?.position.set(timeX(s.peakA) + LABEL_SIDE[0], Y0 + k * pA + LABEL_LIFT[0], LANE_Z[0])
    tagRefs.current[6]?.position.set(timeX(s.peakB) + LABEL_SIDE[1], Y0 + k * pB + LABEL_LIFT[1], LANE_Z[1])
    tagRefs.current[7]?.position.set(timeX(s.peakC) + LABEL_SIDE[2], Y0 + k * pC + LABEL_LIFT[2], LANE_Z[2])

    if (labelledFor.current !== valuesRef.current && textRefs.current[7]) {
      labelledFor.current = valuesRef.current
      const t = L[locale]
      // Past the ceiling the peaks cannot spread any further, so the slider looks dead.
      // Say why in the readout instead of letting it silently disagree with the slider.
      const divNote = s.diversity > s.divMax - 1e-4 ? ` · ${t.capped}` : ''
      const texts = [
        `${t.connected} ${fmtKW(s.CL)}`,
        `${t.maxDemand} ${fmtKW(s.maxDemand)}`,
        `${t.average} ${fmtKW(s.average)}`,
        `${t.transformer} ${s.kVA.toFixed(1)} kVA`,
        `${t.demand} ${fmtF(s.DF)} · ${t.load} ${fmtF(s.LF)} · ${t.diversity} ${fmtF(s.diversity)}${divNote}`,
        `${t.groupA} ${fmtKW(pA)} · ${fmtClock(s.peakA)}`,
        `${t.groupB} ${fmtKW(pB)} · ${fmtClock(s.peakB)}`,
        `${t.groupC} ${fmtKW(pC)} · ${fmtClock(s.peakC)}`,
      ]
      texts.forEach((text, i) => { if (textRefs.current[i]) textRefs.current[i]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  const t = L[locale]
  const tagStyle = [
    undefined,
    undefined,
    { color: c.current },
    { color: c.power },
    undefined,
    undefined,
    undefined,
    undefined,
  ]

  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />

      <Line points={AXIS} lineWidth={1} color={c.guide} />
      <Line points={CONNECTED_GUIDE} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      {[c.guide, c.current, c.power].map((color, i) => (
        <Line
          key={i}
          ref={(l) => { guideRefs.current[i] = l }}
          points={AXIS}
          lineWidth={i === 2 ? 1.5 : 1}
          color={color}
          dashed
          dashSize={0.08}
          gapSize={0.06}
        />
      ))}

      {/* Three group loads, told apart by lightness; their sum is the cyan line in front */}
      {[c.voltage, c.inductance, c.resistance, c.current].map((color, g) => (
        <TrailLine
          key={g}
          ref={(l) => { curveRefs.current[g] = l }}
          points={CURVE_INIT}
          lineWidth={g === 3 ? 3 : 2}
          color={color}
          fade={false}
          flowSpeed={g === 3 ? () => flow.current : 0.4}
        />
      ))}
      <Marker ref={peakRef} color={c.marker} size={0.06} />

      {/* Connected load as a pile of nameplates, one block per 100 kW */}
      {Array.from({ length: PILE_N }, (_, i) => (
        <mesh
          key={i}
          ref={(m) => { pileRefs.current[i] = m }}
          position={[X0 + 0.05 + i * PILE_PITCH, PILE_Y, Z_FRONT]}
        >
          <boxGeometry args={[PILE_SIZE, PILE_SIZE, PILE_SIZE]} />
          <meshBasicMaterial color={c.capacitance} />
        </mesh>
      ))}

      {/* The transformer, scaled in y to the capacity it must supply */}
      <mesh ref={txRef}>
        <boxGeometry args={[TX_W, 1, TX_W]} />
        <meshBasicMaterial color={c.power} transparent opacity={0.3} depthWrite={false} />
      </mesh>

      {[0, 1, 2, 3, 4, 5, 6, 7].map((i) => (
        <group key={i} ref={(g) => { tagRefs.current[i] = g }}>
          <Html center>
            <span
              ref={(el) => { textRefs.current[i] = el }}
              className={labelClass}
              style={tagStyle[i]}
            />
          </Html>
        </group>
      ))}

      {[[0, t.h0], [12, t.h12], [24, t.h24]].map(([h, text]) => (
        <Html key={text} position={[timeX(h as number), Y0 - 0.18, Z_FRONT]} center>
          <span className={smallClass}>{text}</span>
        </Html>
      ))}
    </>
  )
}

// ── 2D: the capacity chain as three bars — connected load, × demand factor, ÷ diversity factor ──
// This reads better here than a second copy of the day profile: the 3D scene already shows *why*
// the peaks shrink, so the flat view is free to show *how far* they shrink, in one glance.
const BAR_X = [-2.7, -0.5, 1.7]
const BAR_HALF = 0.45
const BAR_Y0 = -1.5
const BAR_SPAN = 2.9
const TOPS = new Float32Array(3) // bar heights, reused every frame

function setBar(line: any, cx: number, y1: number) {
  RECT[0] = cx - BAR_HALF; RECT[1] = BAR_Y0; RECT[2] = 0
  RECT[3] = cx - BAR_HALF; RECT[4] = y1; RECT[5] = 0
  RECT[6] = cx + BAR_HALF; RECT[7] = y1; RECT[8] = 0
  RECT[9] = cx + BAR_HALF; RECT[10] = BAR_Y0; RECT[11] = 0
  RECT[12] = cx - BAR_HALF; RECT[13] = BAR_Y0; RECT[14] = 0
  line?.geometry.setPositions(RECT)
  line?.computeLineDistances()
}

const BAR_INIT: [number, number, number][] = Array.from({ length: 5 }, () => [0, 0, 0])
const SEG_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]

function DemandFactorsChain2DScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const barRefs = useRef<any[]>([]) // connected load, maximum demand, required capacity
  const stepRefs = useRef<any[]>([]) // the two drops between bars
  const avgRef = useRef<any>(null)
  const tagRefs = useRef<(THREE.Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<unknown>(null)

  useFrame(() => {
    const s = solve(valuesRef.current)
    const k = BAR_SPAN / s.CL
    TOPS[0] = BAR_Y0 + BAR_SPAN
    TOPS[1] = BAR_Y0 + k * s.maxDemand
    TOPS[2] = BAR_Y0 + k * s.kVA
    for (let i = 0; i < 3; i++) setBar(barRefs.current[i], BAR_X[i], TOPS[i])
    for (let i = 0; i < 2; i++) {
      setSeg(stepRefs.current[i], BAR_X[i] + BAR_HALF, TOPS[i], BAR_X[i + 1] - BAR_HALF, TOPS[i + 1])
    }
    const yAvg = BAR_Y0 + k * s.average
    setSeg(avgRef.current, BAR_X[0] - 0.55, yAvg, BAR_X[2] + 0.55, yAvg)

    for (let i = 0; i < 3; i++) tagRefs.current[i]?.position.set(BAR_X[i], BAR_Y0 - 0.26, 0)
    tagRefs.current[3]?.position.set((BAR_X[0] + BAR_X[1]) / 2, (TOPS[0] + TOPS[1]) / 2 + 0.22, 0)
    tagRefs.current[4]?.position.set((BAR_X[1] + BAR_X[2]) / 2, (TOPS[1] + TOPS[2]) / 2 + 0.22, 0)
    tagRefs.current[5]?.position.set(BAR_X[2] + 0.6, yAvg + 0.24, 0)

    if (labelledFor.current !== valuesRef.current && textRefs.current[5]) {
      labelledFor.current = valuesRef.current
      const t = L[locale]
      const divNote = s.diversity > s.divMax - 1e-4 ? ` · ${t.capped}` : ''
      const texts = [
        `${t.connected} ${fmtKW(s.CL)}`,
        `${t.maxDemand} ${fmtKW(s.maxDemand)}`,
        `${t.transformer} ${s.kVA.toFixed(1)} kVA`,
        `× ${t.demand} ${fmtF(s.DF)}`,
        `÷ ${t.diversity} ${fmtF(s.diversity)}${divNote}`,
        `${t.average} ${fmtKW(s.average)} · ${t.load} ${fmtF(s.LF)}`,
      ]
      texts.forEach((text, i) => { if (textRefs.current[i]) textRefs.current[i]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  const barColors = [c.capacitance, c.voltage, c.power]
  const tagStyle = [undefined, undefined, { color: c.power }, undefined, undefined, { color: c.current }]

  return (
    <>
      {barColors.map((color, i) => (
        <Line
          key={i}
          ref={(l) => { barRefs.current[i] = l }}
          points={BAR_INIT}
          lineWidth={i === 2 ? 3 : 2.5}
          color={color}
        />
      ))}
      {[0, 1].map((i) => (
        <Line
          key={i}
          ref={(l) => { stepRefs.current[i] = l }}
          points={SEG_INIT}
          lineWidth={1}
          color={c.guide}
          dashed
          dashSize={0.08}
          gapSize={0.06}
        />
      ))}
      <Line
        ref={avgRef}
        points={SEG_INIT}
        lineWidth={1.5}
        color={c.current}
        dashed
        dashSize={0.08}
        gapSize={0.06}
      />
      {[0, 1, 2, 3, 4, 5].map((i) => (
        <group key={i} ref={(g) => { tagRefs.current[i] = g }}>
          <Html center>
            <span
              ref={(el) => { textRefs.current[i] = el }}
              className={labelClass}
              style={tagStyle[i]}
            />
          </Html>
        </group>
      ))}
    </>
  )
}

const concept: ConceptMeta = {
  id: 'demand-load-factors',
  title: { en: 'Demand, Load & Diversity Factors', ko: '수용률 · 부하율 · 부등률' },
  subject: 'installations',
  order: 3,
  description: {
    en: `## Demand, Load & Diversity Factors

**Three factors shrink a pile of connected equipment down to one transformer.**

The plot is a single day, scaled so the connected load fills the frame.
- Blocks along the top: the connected load, one block per 100 kW
- Three neutral curves A, B, C: each customer group's load through the day, told apart by lightness
- Cyan liquid line: the combined load, the sum of the three, which is what the transformer carries
- Dashed guides: connected load, max demand, average in cyan, required capacity in amber
- Amber block: the transformer, exactly as tall as the capacity it must supply

**Demand factor** = maximum demand ÷ connected load
**Load factor** = average load ÷ maximum demand = energy ÷ (maximum demand × hours)
**Diversity factor** = sum of group maximum demands ÷ combined maximum demand, always ≥ 1
**Transformer** = (connected load × demand factor) ÷ diversity factor, at unity power factor

Worked example: 500 kW connected × 0.60 gives a 300 kW maximum demand. 4320 kWh in a day is a 180 kW average, so the load factor is 0.600. Group maxima 120 + 100 + 80 = 300 kW against a coincident 250 kW is a diversity factor of 1.200, and the transformer is 300 ÷ 1.200 = 250.0 kVA.

Raise the diversity factor and the three group peaks walk apart, so the cyan peak and the amber block both drop. Raise the load factor and the curves fatten toward flat. Load factor × diversity factor is the combined load factor, so it can never exceed 1: once the cyan curve is flat there is no diversity left to win, and the slider stops there.

In 2D: the same chain as three bars, connected load × demand factor ÷ diversity factor, with the average load dashed across them.`,
    ko: `## 수용률 · 부하율 · 부등률

**세 계수가 설비 더미를 변압기 한 대로 줄여 줍니다.**

그래프는 하루이고, 설비용량이 화면을 가득 채우도록 배율을 맞췄습니다.
- 위쪽 블록: 설비용량, 한 블록이 100 kW
- 중성색 곡선 A, B, C: 각 수용가 그룹의 하루 부하. 명도로 구분합니다
- 시안 액체 선: 세 곡선의 합, 즉 변압기가 실제로 메는 합성 부하
- 점선 안내선: 설비용량, 최대수요전력, 시안색 평균전력, 황색 소요 용량
- 황색 블록: 변압기. 공급해야 하는 용량만큼의 높이입니다

**수용률** = 최대수요전력 ÷ 설비용량
**부하율** = 평균전력 ÷ 최대수요전력 = 전력량 ÷ (최대수요전력 × 시간)
**부등률** = 각 그룹 최대수요전력의 합 ÷ 합성 최대수요전력, 항상 ≥ 1
**변압기** = (설비용량 × 수용률) ÷ 부등률, 역률 1 기준

계산 예: 설비용량 500 kW × 0.60이면 최대수요전력은 300 kW입니다. 하루 4320 kWh이면 평균전력이 180 kW이므로 부하율은 0.600입니다. 그룹별 최댓값 120 + 100 + 80 = 300 kW에 합성 최댓값이 250 kW이면 부등률은 1.200이고, 변압기는 300 ÷ 1.200 = 250.0 kVA입니다.

부등률을 올리면 세 그룹의 첨두가 서로 멀어져 시안 곡선의 최댓값과 황색 블록이 함께 낮아집니다. 부하율을 올리면 곡선이 통통해지다가 평탄해집니다. 부하율 × 부등률은 합성 부하율이므로 1을 넘을 수 없습니다. 시안 곡선이 완전히 평탄해지면 더 얻을 부등률이 없어서 슬라이더가 거기서 멈춥니다.

2D: 같은 흐름을 막대 세 개로 봅니다. 설비용량 × 수용률 ÷ 부등률이고, 평균전력을 점선으로 가로질러 표시했습니다.`,
  },
  sliderParams: [
    { key: 'CL',  label: { en: 'Connected load (Pc)',   ko: '설비용량 (Pc)' },  unit: 'kW', min: 100, max: 2000, step: 100,  defaultValue: 500 },
    { key: 'DF',  label: { en: 'Demand factor (fd)',    ko: '수용률 (fd)' },                min: 0.3, max: 1.0,  step: 0.01, defaultValue: 0.6 },
    { key: 'LF',  label: { en: 'Load factor (fL)',      ko: '부하율 (fL)' },                min: 0.2, max: 1.0,  step: 0.01, defaultValue: 0.6 },
    { key: 'DIV', label: { en: 'Diversity factor (fn)', ko: '부등률 (fn)' },                min: 1.0, max: 2.0,  step: 0.01, defaultValue: 1.2 },
  ],
  Scene: DemandFactorsScene,
  Scene2D: DemandFactorsChain2DScene,
}

export default concept
