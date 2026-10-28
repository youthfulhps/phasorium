import { useRef, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import type { Group } from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import { useLocaleStore } from '../../lib/i18n'
import type { ConceptMeta } from '../../types/concept'

const PTS = 401 // enough to resolve two wavelengths across the line
const MARKS = 5 // nodes land every half wavelength; 2 λ of line can hold five of them
const MARK_IDX = Array.from({ length: MARKS }, (_, i) => i)
const CURVE_INIT: [number, number, number][] = Array.from({ length: PTS }, () => [0, 0, 0])
const SEG_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]

const OMEGA = 2 * Math.PI * 0.6 // on-screen rad/s: 0.6 cycles per second, far slower than a real line
const TWO_PI = Math.PI * 2

/**
 * Envelope of the standing wave, in units of the forward amplitude V+.
 * From v(x,t) = V+[sin(ωt + βx) + Γ·sin(ωt − βx)]
 *            = V+[(1 + Γ)·sin(ωt)·cos(βx) + (1 − Γ)·cos(ωt)·sin(βx)],
 * so the peak over a cycle is sqrt((1+Γ)²cos²βx + (1−Γ)²sin²βx).
 */
function envelopeAt(gamma: number, bx: number) {
  const c = Math.cos(bx)
  const s = Math.sin(bx)
  const a = 1 + gamma
  const b = 1 - gamma
  return Math.sqrt(a * a * c * c + b * b * s * s)
}

// Reused result object: solve() runs every frame, so it must not allocate.
const OUT = {
  Z0: 50, ZL: 100, lineLen: 1,
  gamma: 0, absG: 0, vswr: 1, pRef: 0,
  maxAmp: 1, minAmp: 1, matched: true, nodeQ: 0.25,
}

// Pure physics of a lossless line driven at one frequency. x is the distance back from the LOAD,
// so the first term below is the wave heading toward the load and the second is what comes back.
//   characteristic impedance   Z0 = sqrt(L ÷ C) per unit length
//   reflection coefficient      Γ = (ZL − Z0) ÷ (ZL + Z0)
//   standing wave ratio      VSWR = (1 + |Γ|) ÷ (1 − |Γ|)
//   reflected power fraction      = |Γ|²
//   voltage on the line      v(x, t) = V+[sin(ωt + βx) + Γ·sin(ωt − βx)],  β = 2π ÷ λ
//   envelope                 |V(x)| = V+·sqrt((1+Γ)²cos²βx + (1−Γ)²sin²βx), max V+(1+|Γ|), min V+(1−|Γ|)
// Worked example, verified with node:
//   Z0 = 50 Ω, ZL = 100 Ω -> Γ = 0.3333, VSWR = 2.000, reflected power 11.1 %
//   Z0 = 50 Ω, ZL =  50 Ω -> Γ = 0,      VSWR = 1.000, reflected power 0 %   (matched)
//   Z0 = 50 Ω, ZL =  25 Ω -> Γ = −0.3333, VSWR = 2.000, reflected power 11.1 %
//   Z0 = 50 Ω, ZL = 200 Ω -> Γ = 0.6000, VSWR = 4.000, reflected power 36.0 %
//   ZL = 0 -> Γ = −1 and ZL -> ∞ -> Γ = +1, both VSWR -> ∞, all the power comes back
function solve({ ZL = 100, Z0 = 50, lineLen = 1 }: Record<string, number>) {
  const total = ZL + Z0
  const gamma = total === 0 ? -1 : (ZL - Z0) / total
  const absG = Math.abs(gamma)
  OUT.Z0 = Z0
  OUT.ZL = ZL
  OUT.lineLen = lineLen
  OUT.gamma = gamma
  OUT.absG = absG
  OUT.vswr = absG >= 1 ? Infinity : (1 + absG) / (1 - absG)
  OUT.pRef = absG * absG
  OUT.maxAmp = 1 + absG
  OUT.minAmp = 1 - absG
  OUT.matched = absG < 0.02
  // First voltage node, in wavelengths back from the load: a quarter wave for Γ > 0, at the load for Γ < 0
  OUT.nodeQ = gamma >= 0 ? 0.25 : 0
  return OUT
}

function useSliderValues() {
  const valuesRef = useRef<Record<string, number>>(useSliderStore.getState().values)
  useEffect(
    () => useSliderStore.subscribe((s) => s.values, (v) => { valuesRef.current = v }),
    []
  )
  return valuesRef
}

type Bufs = {
  fwd: Float32Array
  ref: Float32Array
  sum: Float32Array
  envU: Float32Array
  envL: Float32Array
  seg: Float32Array
}

type Lane = {
  xL: number; span: number; scale: number
  fy: number; fz: number
  ry: number; rz: number
  sy: number; sz: number
}

/** Fills the forward, reflected, summed and envelope buffers. Allocates nothing. */
function fillWaves(b: Bufs, g: Lane, t: number, gamma: number, lineLen: number) {
  const wt = OMEGA * t
  for (let i = 0; i < PTS; i++) {
    const u = i / (PTS - 1) // 0 at the generator, 1 at the load
    const bx = TWO_PI * lineLen * (1 - u) // βx, with x measured back from the load
    const x = g.xL + u * g.span
    const vi = Math.sin(wt + bx) // heading toward the load: moves right on screen
    const vr = gamma * Math.sin(wt - bx) // coming back: moves left on screen
    const env = envelopeAt(gamma, bx)
    const k = i * 3
    b.fwd[k] = x; b.fwd[k + 1] = g.fy + vi * g.scale; b.fwd[k + 2] = g.fz
    b.ref[k] = x; b.ref[k + 1] = g.ry + vr * g.scale; b.ref[k + 2] = g.rz
    b.sum[k] = x; b.sum[k + 1] = g.sy + (vi + vr) * g.scale; b.sum[k + 2] = g.sz
    b.envU[k] = x; b.envU[k + 1] = g.sy + env * g.scale; b.envU[k + 2] = g.sz
    b.envL[k] = x; b.envL[k + 1] = g.sy - env * g.scale; b.envL[k + 2] = g.sz
  }
}

/** Screen x of a point q wavelengths back from the load, or NaN when it falls off the line. */
function markX(q: number, lineLen: number, xL: number, span: number) {
  if (q > lineLen + 1e-9) return NaN
  return xL + (1 - q / lineLen) * span
}

/** Places the node dots and the antinode ticks. Allocates nothing. */
function placeMarks(
  b: Bufs, g: Lane,
  nodes: (Group | null)[], ticks: any[],
  s: typeof OUT
) {
  const antiQ = 0.25 - s.nodeQ
  const half = s.maxAmp * g.scale
  for (let k = 0; k < MARKS; k++) {
    const nx = markX(s.nodeQ + 0.5 * k, s.lineLen, g.xL, g.span)
    const dot = nodes[k]
    if (dot) {
      dot.visible = !s.matched && !Number.isNaN(nx)
      if (dot.visible) dot.position.set(nx, g.sy, g.sz + 0.01)
    }
    const ax = markX(antiQ + 0.5 * k, s.lineLen, g.xL, g.span)
    const tick = ticks[k]
    if (tick) {
      tick.visible = !s.matched && !Number.isNaN(ax)
      if (tick.visible) {
        b.seg[0] = ax; b.seg[1] = g.sy - half; b.seg[2] = g.sz
        b.seg[3] = ax; b.seg[4] = g.sy + half; b.seg[5] = g.sz
        tick.geometry.setPositions(b.seg)
        tick.computeLineDistances()
      }
    }
  }
}

function useBufs(): Bufs {
  return useMemo(() => ({
    fwd: new Float32Array(PTS * 3),
    ref: new Float32Array(PTS * 3),
    sum: new Float32Array(PTS * 3),
    envU: new Float32Array(PTS * 3),
    envL: new Float32Array(PTS * 3),
    seg: new Float32Array(6),
  }), [])
}

const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const axisClass = 'pointer-events-none whitespace-nowrap text-[11px] tabular-nums text-muted-foreground'

const vswrText = (v: number) => (Number.isFinite(v) ? v.toFixed(2) : '∞')
const L = {
  en: { matched: 'matched · envelope flat · nothing comes back', dots: 'dots = nodes · ticks = antinodes' },
  ko: { matched: '정합 · 포락선 평탄 · 반사 없음', dots: '점 = 마디 · 눈금 = 배' },
}
type Labels = (typeof L)['en']

const signed = (v: number) => (v >= 0 ? `+${v.toFixed(3)}` : v.toFixed(3))

/** Readout strings, rebuilt only when the slider values object changes identity. */
function writeLabels(texts: (HTMLSpanElement | null)[], s: typeof OUT, t: Labels) {
  if (texts[0]) texts[0].textContent = `Γ = ${signed(s.gamma)} · VSWR ${vswrText(s.vswr)} · |Γ|² ${(s.pRef * 100).toFixed(1)} %`
  if (texts[1]) texts[1].textContent = `Z0 ${s.Z0.toFixed(0)} Ω · ZL ${s.ZL.toFixed(0)} Ω · ℓ ${s.lineLen.toFixed(2)} λ`
  if (texts[2]) {
    texts[2].textContent = s.matched
      ? t.matched
      : `|V| ${s.minAmp.toFixed(2)} … ${s.maxAmp.toFixed(2)} V+ · ${t.dots}`
  }
}

// ── 3D: the two travelling waves receding into depth, their sum in front ──
const X_L3 = -2.8
const SPAN3 = 5.6
const LANE3: Lane = {
  xL: X_L3, span: SPAN3, scale: 0.55,
  fy: 0, fz: -0.25,
  ry: 0, rz: -0.95,
  sy: 0, sz: 0.45,
}
const AXIS3: [number, number, number][] = [[X_L3, 0, LANE3.sz], [X_L3 + SPAN3, 0, LANE3.sz]]
const LOAD3: [number, number, number][] = [[X_L3 + SPAN3, -1.25, LANE3.sz], [X_L3 + SPAN3, 1.25, LANE3.sz]]

function TransmissionLine3DScene() {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  const b = useBufs()
  const fwdRef = useRef<any>(null)
  const refRef = useRef<any>(null)
  const sumRef = useRef<any>(null)
  const envURef = useRef<any>(null)
  const envLRef = useRef<any>(null)
  const nodeRefs = useRef<(Group | null)[]>([])
  const tickRefs = useRef<any[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<unknown>(null)
  const labelledIn = useRef<Labels | null>(null)
  const t = L[useLocaleStore((st) => st.locale)]

  useFrame(({ clock }) => {
    const s = solve(valuesRef.current)
    fillWaves(b, LANE3, clock.getElapsedTime(), s.gamma, s.lineLen)
    fwdRef.current?.geometry.setPositions(b.fwd)
    refRef.current?.geometry.setPositions(b.ref)
    sumRef.current?.geometry.setPositions(b.sum)
    envURef.current?.geometry.setPositions(b.envU)
    envURef.current?.computeLineDistances()
    envLRef.current?.geometry.setPositions(b.envL)
    envLRef.current?.computeLineDistances()

    placeMarks(b, LANE3, nodeRefs.current, tickRefs.current, s)

    if (labelledFor.current !== valuesRef.current || labelledIn.current !== t) {
      labelledFor.current = valuesRef.current
      labelledIn.current = t
      writeLabels(textRefs.current, s, t)
    }
  })

  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />

      <Line points={AXIS3} lineWidth={1} color={c.guide} />
      <Line points={LOAD3} lineWidth={2} color={c.resistance} />

      {/* The two travelling waves, on their own lanes behind the sum */}
      <Line ref={fwdRef} points={CURVE_INIT} lineWidth={2} color={c.inductance} />
      <Line ref={refRef} points={CURVE_INIT} lineWidth={2} color={c.resistance} />

      {/* Their sum, with the envelope its peaks never leave */}
      <Line ref={envURef} points={CURVE_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.1} gapSize={0.07} />
      <Line ref={envLRef} points={CURVE_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.1} gapSize={0.07} />
      <TrailLine ref={sumRef} points={CURVE_INIT} lineWidth={3} color={c.voltage} />

      {MARK_IDX.map((k) => (
        <Line
          key={k}
          ref={(l) => { tickRefs.current[k] = l }}
          points={SEG_INIT}
          lineWidth={1}
          color={c.guide}
          dashed
          dashSize={0.07}
          gapSize={0.05}
        />
      ))}
      {MARK_IDX.map((k) => (
        <Marker key={k} ref={(gr) => { nodeRefs.current[k] = gr }} color={c.marker} size={0.05} />
      ))}

      <Html position={[X_L3, 1.85, LANE3.sz]}>
        <span ref={(el) => { textRefs.current[0] = el }} className={labelClass} />
      </Html>
      <Html position={[X_L3, 1.58, LANE3.sz]}>
        <span ref={(el) => { textRefs.current[1] = el }} className={labelClass} />
      </Html>
      <Html position={[X_L3, -1.85, LANE3.sz]}>
        <span ref={(el) => { textRefs.current[2] = el }} className={labelClass} />
      </Html>
      <Html position={[X_L3 + SPAN3 + 0.35, 1.45, LANE3.sz]} center>
        <span className={axisClass}>ZL</span>
      </Html>
      <Html position={[X_L3 - 0.4, -0.3, LANE3.sz]} center>
        <span className={axisClass}>Vg</span>
      </Html>
    </>
  )
}

// ── 2D: three stacked strips, forward over reflected over their sum ──
const X_L2 = -3.2
const SPAN2 = 5.8
const LANE2: Lane = {
  xL: X_L2, span: SPAN2, scale: 0.4,
  fy: 1.25, fz: 0,
  ry: 0.35, rz: 0,
  sy: -1.05, sz: 0,
}
const X_R2 = X_L2 + SPAN2
const AXIS_F: [number, number, number][] = [[X_L2, LANE2.fy, 0], [X_R2, LANE2.fy, 0]]
const AXIS_R: [number, number, number][] = [[X_L2, LANE2.ry, 0], [X_R2, LANE2.ry, 0]]
const AXIS_S: [number, number, number][] = [[X_L2, LANE2.sy, 0], [X_R2, LANE2.sy, 0]]
const LOAD2: [number, number, number][] = [[X_R2, -1.9, 0], [X_R2, 1.75, 0]]

function TransmissionLine2DScene() {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  const b = useBufs()
  const fwdRef = useRef<any>(null)
  const refRef = useRef<any>(null)
  const sumRef = useRef<any>(null)
  const envURef = useRef<any>(null)
  const envLRef = useRef<any>(null)
  const nodeRefs = useRef<(Group | null)[]>([])
  const tickRefs = useRef<any[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<unknown>(null)
  const labelledIn = useRef<Labels | null>(null)
  const t = L[useLocaleStore((st) => st.locale)]

  useFrame(({ clock }) => {
    const s = solve(valuesRef.current)
    fillWaves(b, LANE2, clock.getElapsedTime(), s.gamma, s.lineLen)
    fwdRef.current?.geometry.setPositions(b.fwd)
    refRef.current?.geometry.setPositions(b.ref)
    sumRef.current?.geometry.setPositions(b.sum)
    envURef.current?.geometry.setPositions(b.envU)
    envURef.current?.computeLineDistances()
    envLRef.current?.geometry.setPositions(b.envL)
    envLRef.current?.computeLineDistances()

    placeMarks(b, LANE2, nodeRefs.current, tickRefs.current, s)

    if (labelledFor.current !== valuesRef.current || labelledIn.current !== t) {
      labelledFor.current = valuesRef.current
      labelledIn.current = t
      writeLabels(textRefs.current, s, t)
    }
  })

  return (
    <>
      <Line points={AXIS_F} lineWidth={1} color={c.guide} />
      <Line points={AXIS_R} lineWidth={1} color={c.guide} />
      <Line points={AXIS_S} lineWidth={1} color={c.guide} />
      <Line points={LOAD2} lineWidth={2} color={c.resistance} />

      <Line ref={fwdRef} points={CURVE_INIT} lineWidth={2} color={c.inductance} />
      <Line ref={refRef} points={CURVE_INIT} lineWidth={2} color={c.resistance} />

      <Line ref={envURef} points={CURVE_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.1} gapSize={0.07} />
      <Line ref={envLRef} points={CURVE_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.1} gapSize={0.07} />
      <TrailLine ref={sumRef} points={CURVE_INIT} lineWidth={3} color={c.voltage} />

      {MARK_IDX.map((k) => (
        <Line
          key={k}
          ref={(l) => { tickRefs.current[k] = l }}
          points={SEG_INIT}
          lineWidth={1}
          color={c.guide}
          dashed
          dashSize={0.07}
          gapSize={0.05}
        />
      ))}
      {MARK_IDX.map((k) => (
        <Marker key={k} ref={(gr) => { nodeRefs.current[k] = gr }} color={c.marker} size={0.05} />
      ))}

      <Html position={[-3.95, 1.95, 0]}>
        <span ref={(el) => { textRefs.current[0] = el }} className={labelClass} />
      </Html>
      <Html position={[-3.95, 1.72, 0]}>
        <span ref={(el) => { textRefs.current[1] = el }} className={labelClass} />
      </Html>
      <Html position={[-3.95, -1.95, 0]}>
        <span ref={(el) => { textRefs.current[2] = el }} className={labelClass} />
      </Html>

      <Html position={[3.3, LANE2.fy, 0]} center>
        <span className={axisClass}>V+ →</span>
      </Html>
      <Html position={[3.3, LANE2.ry, 0]} center>
        <span className={axisClass}>← Γ V+</span>
      </Html>
      <Html position={[3.35, LANE2.sy, 0]} center>
        <span className={axisClass}>Σ</span>
      </Html>
      <Html position={[X_R2, 1.95, 0]} center>
        <span className={axisClass}>ZL</span>
      </Html>
      <Html position={[X_L2 - 0.25, LANE2.sy - 0.3, 0]} center>
        <span className={axisClass}>Vg</span>
      </Html>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'transmission-line',
  title: { en: 'Transmission Lines & Reflection', ko: '분포정수 회로와 반사' },
  subject: 'circuits',
  order: 24,
  cardPosition: [3.5, 1.2, -0.6],
  description: {
    en: `## Transmission Lines, Reflection and Standing Waves

**Once a line is long compared with the wavelength it stops being a wire: a wave travels along it, and whatever the load does not absorb comes straight back.**

- Light trace: the forward wave V+, travelling toward the load
- Mid trace: the reflected wave Γ·V+, travelling back toward the generator
- Bright trace: their sum, the voltage actually on the line
- Dashed outline: the envelope, the peak each point ever reaches
- Dots: nodes, where the envelope pinches to its minimum
- Dashed ticks: antinodes, where the envelope is widest

**Characteristic impedance**: Z0 = sqrt(L ÷ C) per unit length
**Reflection coefficient**: Γ = (ZL − Z0) ÷ (ZL + Z0)
**Standing wave ratio**: VSWR = (1 + |Γ|) ÷ (1 − |Γ|)
**Reflected power**: Pref ÷ Pinc = |Γ|²
**Voltage on the line**: v(x, t) = V+[sin(ωt + βx) + Γ·sin(ωt − βx)], β = 2π ÷ λ, x measured back from the load

With Z0 = 50 Ω and ZL = 100 Ω, Γ = 0.3333, VSWR = 2.000 and 11.1 % of the power turns around at the load. Set ZL = 50 Ω and Γ = 0: the envelope goes flat, the wave simply runs off the end, and every watt is delivered. A dead short, ZL = 0, gives Γ = −1 and an open end gives Γ = +1; either way VSWR runs to infinity and all of the power comes back.

Nodes and antinodes stand still because the two waves slide past each other at the same speed: they repeat every half wavelength, and the first node sits at the load for a short, a quarter wavelength back for anything above Z0.

The wave is slowed down for viewing, so one cycle on screen takes about two seconds instead of nanoseconds.

Drag ZL from 0 up through Z0 and on toward an open and watch the clean travelling wave collapse into a standing one. Then stretch the line and count how many nodes fit.`,
    ko: `## 분포정수 회로, 반사와 정재파

**선로 길이가 파장에 견줄 만큼 길어지면 그것은 더 이상 단순한 전선이 아닙니다. 파동이 선로를 따라 진행하고, 부하가 흡수하지 못한 몫은 그대로 되돌아옵니다.**

- 밝은 회색 선: 부하 쪽으로 진행하는 입사파 V+
- 중간 회색 선: 전원 쪽으로 되돌아오는 반사파 Γ·V+
- 가장 또렷한 선: 두 파의 합, 즉 선로에 실제로 걸리는 전압
- 점선 윤곽: 포락선. 각 지점이 도달하는 최대 진폭입니다
- 점: 마디(node). 포락선이 가장 좁아지는 지점입니다
- 점선 눈금: 배(antinode). 포락선이 가장 넓어지는 지점입니다

**특성 임피던스**: Z0 = sqrt(L ÷ C), 단위 길이당 값
**반사계수**: Γ = (ZL − Z0) ÷ (ZL + Z0)
**정재파비**: VSWR = (1 + |Γ|) ÷ (1 − |Γ|)
**반사 전력**: Pref ÷ Pinc = |Γ|²
**선로 전압**: v(x, t) = V+[sin(ωt + βx) + Γ·sin(ωt − βx)], β = 2π ÷ λ, x는 부하에서 거슬러 잰 거리

Z0 = 50 Ω, ZL = 100 Ω이면 Γ = 0.3333, VSWR = 2.000이고 전력의 11.1 %가 부하에서 되돌아옵니다. ZL = 50 Ω으로 맞추면 Γ = 0이 되어 포락선이 평평해지고, 파동은 끝에서 그대로 빠져나가며 전력이 한 와트도 남김없이 전달됩니다. 단락이면 ZL = 0이라 Γ = −1, 개방이면 Γ = +1이 되고, 어느 쪽이든 VSWR은 무한대로 가며 전력 전부가 되돌아옵니다.

마디와 배가 제자리에 멈춰 보이는 이유는 두 파동이 같은 속도로 서로를 스쳐 지나가기 때문입니다. 이들은 반파장마다 되풀이되며, 첫 마디는 단락일 때 부하 바로 그 자리에, Z0보다 큰 부하에서는 부하에서 1 ÷ 4 파장 떨어진 곳에 생깁니다.

파동은 눈으로 따라갈 수 있도록 느리게 재생합니다. 화면에서 한 주기가 약 2초이며, 실제 선로에서는 나노초 단위입니다.

ZL을 0에서 Z0을 지나 개방 쪽까지 끌어올리며, 깔끔하던 진행파가 정재파로 무너지는 모습을 보세요. 그다음 선로 길이를 늘려 마디가 몇 개나 들어가는지 세어 보세요.`,
  },
  sliderParams: [
    { key: 'ZL', label: { en: 'Load impedance (ZL)', ko: '부하 임피던스 (ZL)' }, unit: 'Ω', min: 0, max: 500, step: 5, defaultValue: 100 },
    { key: 'Z0', label: { en: 'Line impedance (Z0)', ko: '선로 특성 임피던스 (Z0)' }, unit: 'Ω', min: 20, max: 150, step: 5, defaultValue: 50 },
    { key: 'lineLen', label: { en: 'Line length (ℓ)', ko: '선로 길이 (ℓ)' }, unit: 'λ', min: 0.25, max: 2, step: 0.125, defaultValue: 1 },
  ],
  Scene: TransmissionLine3DScene,
  Scene2D: TransmissionLine2DScene,
}

export default concept
