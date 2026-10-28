import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import type { Group, Mesh } from 'three'
import type { LineMaterial } from 'three-stdlib'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { useLocaleStore } from '../../lib/i18n'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import type { ConceptMeta } from '../../types/concept'

const S_RATED = 100 // transformer rating [kVA]
const X_MIN = 0.05 // lowest load fraction the slider reaches
const X_MAX = 1.25 // highest load fraction the slider reaches
const P_TRUNK_REF = 140 // [kW] largest throughput the sliders can produce, for the trunk's width

// Transformer losses and efficiency.
// Iron (core) loss is fixed by the applied voltage, so it is already there at no load and never
// changes with the load. Copper loss is I²R, and I is proportional to the load, so it grows as x².
//   eta(x) = x·S·cos0 ÷ (x·S·cos0 + Pi + x²·Pcu)
// Maximum efficiency where the two losses are equal: x_opt = sqrt(Pi ÷ Pcu), and x_opt²·Pcu = Pi.
// Efficiency is symmetric about that point: eta(x) = eta(x_opt² ÷ x) exactly, so the optimum is the
// geometric mean of any two loads that give the same efficiency.
// Worked example, verified with npx tsx -e. S = 100 kVA, Pi = 1 kW, Pcu = 2 kW, cos0 = 0.8:
//   x = 1.0000 : Pout = 80.000 kW, losses = 1.000 + 2.000 = 3.000 kW, eta = 96.3855 %
//   x = 0.7071 : Pout = 56.569 kW, losses = 1.000 + 1.000 = 2.000 kW, eta = 96.5852 %  (70.71 kVA)
//   x = 0.7500 : eta = 96.5795 %
//   x = 0.5000 : eta = 96.3855 %, the same as x = 1.0000 because x_opt² ÷ 1.0 = 0.5
// The symmetry identity holds to 1.1e-16 over x = 0.1 … 1.25.
function efficiency(x: number, Pi: number, Pcu: number, pf: number) {
  const out = x * S_RATED * pf
  return out / (out + Pi + x * x * Pcu)
}

// One module-scope record, so no frame allocates
const S = {
  x: 1, Pi: 1, Pcu: 2, pf: 0.8,
  Pout: 80, Pcux: 2, Pin: 83,
  eta: 96.3855, xOpt: 0.7071, etaOpt: 96.5852, kvaOpt: 70.71,
}

function solve(v: Record<string, number>) {
  const x = v.x ?? 1
  const Pi = v.Pi ?? 1
  const Pcu = v.Pcu ?? 2
  const pf = v.pf ?? 0.8
  S.x = x
  S.Pi = Pi
  S.Pcu = Pcu
  S.pf = pf
  S.Pout = x * S_RATED * pf
  S.Pcux = x * x * Pcu
  S.Pin = S.Pout + Pi + S.Pcux
  S.eta = 100 * efficiency(x, Pi, Pcu, pf)
  S.xOpt = Math.sqrt(Pi / Pcu)
  S.etaOpt = 100 * efficiency(S.xOpt, Pi, Pcu, pf)
  S.kvaOpt = S.xOpt * S_RATED
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

const fmtKW = (p: number) => `${p.toFixed(3)} kW`
const fmtPct = (p: number) => `${p.toFixed(3)} %`
const fmtX = (x: number) => x.toFixed(4)

// Labels read in the foreground color; only the two accents (current, power) tint their text
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const accentClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums'

const L10N = {
  en: {
    ironNote: 'constant at every load',
    copperNote: 'grows as the square of the load',
    atOpt: 'losses equal: efficiency is at its peak',
    belowOpt: 'below the peak: iron loss dominates',
    aboveOpt: 'above the peak: copper loss dominates',
    rated: 'of rated',
    best: 'best',
    fullLoad: 'full load',
    etaMaxAt: 'peak',
    axisFrom: 'to',
    xAxis: 'Load fraction x',
    tieNote: (a: string, b: string, e: string) => `dashed tie: x = ${a} and x = ${b} give the same η = ${e}`,
  },
  ko: {
    ironNote: '어떤 부하에서도 일정',
    copperNote: '부하의 제곱으로 증가',
    atOpt: '두 손실이 같음: 효율이 최대입니다',
    belowOpt: '최대점 아래: 철손이 지배합니다',
    aboveOpt: '최대점 위: 동손이 지배합니다',
    rated: '정격 대비',
    best: '최대',
    fullLoad: '전부하',
    etaMaxAt: '최대',
    axisFrom: '~',
    xAxis: '부하율 x',
    tieNote: (a: string, b: string, e: string) => `점선 연결: x = ${a}와 x = ${b}의 η가 같습니다 — ${e}`,
  },
}

// ── 3D: where the power goes ──
// A trunk of real power runs across the top; two amber streams branch down out of it and land on
// their own bars. The iron stream never moves while the load slider does: same width, same sheen
// speed, same bar height. The copper stream swells as x².
const TRUNK_Y = 1.15
const X_IN = -1.85 // input end of the trunk
const X_FE = -0.75 // where the iron-loss stream leaves the trunk, and its bar
const X_CU = 0.3 // where the copper-loss stream leaves, and its bar
const X_OUT = 1.45 // output end of the trunk
const CUR_Y = 0.72 // the cyan load-current line, under the output
const X_CUR = 0.55
const BAR_BASE_Y = -1.58
const BAR_W = 0.26
const BAR_SPAN = 1.75 // scene units given to the largest loss the Pi and Pcu sliders allow

const TRUNK_N = 28
const BRANCH_N = 36
const CUR_N = 28
const SEG_INIT: [number, number, number][] = Array.from({ length: TRUNK_N }, () => [0, 0, 0])
const BRANCH_INIT: [number, number, number][] = Array.from({ length: BRANCH_N }, () => [0, 0, 0])
const CUR_INIT: [number, number, number][] = Array.from({ length: CUR_N }, () => [0, 0, 0])
const GUIDE_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]

const trunkPx = (P: number) => 1.6 + 10.5 * Math.sqrt(Math.min(P / P_TRUNK_REF, 1))
const lossPx = (P: number) => Math.min(13, Math.max(0.9, 3.2 * P))
const curPx = (x: number) => 1.2 + 4.5 * x

function fillSeg(buf: Float32Array, n: number, x0: number, y0: number, x1: number, y1: number) {
  for (let i = 0; i < n; i++) {
    const u = i / (n - 1)
    buf[i * 3] = x0 + (x1 - x0) * u
    buf[i * 3 + 1] = y0 + (y1 - y0) * u
    buf[i * 3 + 2] = 0
  }
}

/** The elbow a loss stream makes as it peels off the trunk and drops onto its bar. */
function fillBranch(buf: Float32Array, x: number, yTop: number, yBot: number) {
  const ax = x - 0.24
  for (let i = 0; i < BRANCH_N; i++) {
    const u = i / (BRANCH_N - 1)
    const w = 1 - u
    buf[i * 3] = w * w * ax + (2 * w * u + u * u) * x
    buf[i * 3 + 1] = (w * w + 2 * w * u) * yTop + u * u * yBot
    buf[i * 3 + 2] = 0
  }
}

function setBar(m: Mesh | null | undefined, x: number, h: number) {
  if (!m) return
  const hh = Math.max(h, 0.015)
  m.scale.set(BAR_W, hh, BAR_W)
  m.position.set(x, BAR_BASE_Y + hh / 2, 0)
}

function LossFlowScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  // Allocate every buffer once; each frame rewrites them in place
  const bufs = useMemo(
    () => ({
      trunk: [new Float32Array(TRUNK_N * 3), new Float32Array(TRUNK_N * 3), new Float32Array(TRUNK_N * 3)],
      branch: [new Float32Array(BRANCH_N * 3), new Float32Array(BRANCH_N * 3)],
      cur: new Float32Array(CUR_N * 3),
      guide: new Float32Array(6),
    }),
    []
  )
  const trunkRefs = useRef<any[]>([]) // 0 input, 1 after the iron branch, 2 output
  const branchRefs = useRef<any[]>([]) // 0 iron, 1 copper
  const curRef = useRef<any>(null)
  const equalRef = useRef<any>(null)
  const barRefs = useRef<(Mesh | null)[]>([]) // 0 iron, 1 copper
  const tagRefs = useRef<(Group | null)[]>([]) // 0 iron bar, 1 copper bar
  // 0 Pin, 1 Pout, 2 load current, 3 iron value, 4 iron note,
  // 5 copper value, 6 copper note, 7 eta now, 8 eta peak, 9 status
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const flows = useRef([0, 0, 0, 0]) // trunk, iron, copper, load current
  const labelledFor = useRef<unknown>(null)

  useFrame(() => {
    const s = solve(valuesRef.current)

    // Bars share one scale, fixed by the Pi and Pcu sliders alone. The load slider cannot touch it,
    // so the iron bar really does stand still while the copper bar climbs.
    const k = BAR_SPAN / Math.max(s.Pi, X_MAX * X_MAX * s.Pcu, 0.5)
    const feTop = BAR_BASE_Y + Math.max(s.Pi * k, 0.015)
    const cuTop = BAR_BASE_Y + Math.max(s.Pcux * k, 0.015)
    setBar(barRefs.current[0], X_FE, s.Pi * k)
    setBar(barRefs.current[1], X_CU, s.Pcux * k)

    // The trunk: real power in, then what is left after each loss stream leaves
    fillSeg(bufs.trunk[0], TRUNK_N, X_IN, TRUNK_Y, X_FE, TRUNK_Y)
    fillSeg(bufs.trunk[1], TRUNK_N, X_FE, TRUNK_Y, X_CU, TRUNK_Y)
    fillSeg(bufs.trunk[2], TRUNK_N, X_CU, TRUNK_Y, X_OUT, TRUNK_Y)
    const trunkP = [s.Pin, s.Pin - s.Pi, s.Pout]
    for (let t = 0; t < 3; t++) {
      const line = trunkRefs.current[t]
      if (!line) continue
      line.geometry.setPositions(bufs.trunk[t])
      ;(line.material as LineMaterial).linewidth = trunkPx(trunkP[t])
    }
    flows.current[0] = 0.5 + 2.5 * s.x

    // The two loss streams, on one shared width scale of their own
    fillBranch(bufs.branch[0], X_FE, TRUNK_Y, feTop)
    fillBranch(bufs.branch[1], X_CU, TRUNK_Y, cuTop)
    const lossP = [s.Pi, s.Pcux]
    for (let b = 0; b < 2; b++) {
      const line = branchRefs.current[b]
      if (!line) continue
      line.geometry.setPositions(bufs.branch[b])
      ;(line.material as LineMaterial).linewidth = lossPx(lossP[b])
    }
    flows.current[1] = 1.1 // the iron stream's sheen runs at a fixed rate too
    flows.current[2] = 0.4 + 3 * s.x * s.x

    // The load current leaving for the load: width and speed both follow x
    fillSeg(bufs.cur, CUR_N, X_CUR, CUR_Y, X_OUT, CUR_Y)
    const cur = curRef.current
    if (cur) {
      cur.geometry.setPositions(bufs.cur)
      ;(cur.material as LineMaterial).linewidth = curPx(s.x)
    }
    flows.current[3] = 0.5 + 3.5 * s.x

    // The equal-loss line: the iron bar's height, carried across the copper bar
    bufs.guide[0] = X_FE - 0.42
    bufs.guide[1] = feTop
    bufs.guide[2] = 0
    bufs.guide[3] = X_CU + 0.42
    bufs.guide[4] = feTop
    bufs.guide[5] = 0
    equalRef.current?.geometry.setPositions(bufs.guide)
    equalRef.current?.computeLineDistances()

    tagRefs.current[0]?.position.set(X_FE - 0.52, feTop + 0.1, 0)
    tagRefs.current[1]?.position.set(X_CU + 0.52, cuTop + 0.1, 0)

    if (labelledFor.current !== valuesRef.current && textRefs.current[9]) {
      labelledFor.current = valuesRef.current
      const t = L10N[locale]
      const status =
        Math.abs(s.x - s.xOpt) < 0.02 ? t.atOpt : s.x < s.xOpt ? t.belowOpt : t.aboveOpt
      const texts = [
        `Pin = ${fmtKW(s.Pin)}`,
        `Pout = ${fmtKW(s.Pout)}`,
        `I = ${s.x.toFixed(2)} × ${t.rated}`,
        `Pi = ${fmtKW(s.Pi)}`,
        t.ironNote,
        `Pcu·x² = ${fmtKW(s.Pcux)}`,
        t.copperNote,
        `η = ${fmtPct(s.eta)}`,
        `η ${t.etaMaxAt} = ${fmtPct(s.etaOpt)} · x_opt = ${fmtX(s.xOpt)} (${s.kvaOpt.toFixed(2)} kVA)`,
        status,
      ]
      texts.forEach((text, i) => { if (textRefs.current[i]) textRefs.current[i]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />

      {/* Background, lights, grid, and bloom come from the shared SceneStage */}
      {/* The trunk of real power: in at the left, out at the right */}
      {[0, 1, 2].map((t) => (
        <TrailLine
          key={t}
          ref={(line) => { trunkRefs.current[t] = line }}
          points={SEG_INIT}
          lineWidth={6}
          color={c.power}
          fade={false}
          flowSpeed={() => flows.current[0]}
        />
      ))}

      {/* The two loss streams branching down onto their bars */}
      {[0, 1].map((b) => (
        <TrailLine
          key={b}
          ref={(line) => { branchRefs.current[b] = line }}
          points={BRANCH_INIT}
          lineWidth={3}
          color={c.power}
          fade={false}
          flowSpeed={() => flows.current[b + 1]}
        />
      ))}

      {/* The load current, the quantity copper loss is the square of */}
      <TrailLine
        ref={curRef}
        points={CUR_INIT}
        lineWidth={2.5}
        color={c.current}
        fade={false}
        flowSpeed={() => flows.current[3]}
      />

      {/* The two losses as bars on one shared scale */}
      {[0, 1].map((b) => (
        <mesh key={b} ref={(m) => { barRefs.current[b] = m }}>
          <boxGeometry args={[1, 1, 1]} />
          <meshStandardMaterial color={c.power} roughness={0.5} />
        </mesh>
      ))}

      {/* The equal-loss line: when the copper bar reaches it, efficiency is at its peak */}
      <Line ref={equalRef} points={GUIDE_INIT} lineWidth={1.5} color={c.guide} dashed dashSize={0.1} gapSize={0.07} />

      <Html position={[-1.35, 1.5, 0]} center>
        <span ref={(el) => { textRefs.current[0] = el }} className={labelClass} />
      </Html>
      <Html position={[0.95, 1.5, 0]} center>
        <span ref={(el) => { textRefs.current[1] = el }} className={labelClass} />
      </Html>
      <Html position={[1.02, 0.95, 0]} center>
        <span ref={(el) => { textRefs.current[2] = el }} className={accentClass} style={{ color: c.current }} />
      </Html>

      {[0, 1].map((b) => (
        <group key={b} ref={(g) => { tagRefs.current[b] = g }}>
          <Html center>
            <div className="pointer-events-none text-center leading-4">
              <div>
                <span
                  ref={(el) => { textRefs.current[b === 0 ? 3 : 5] = el }}
                  className={accentClass}
                  style={{ color: c.power }}
                />
              </div>
              <div><span ref={(el) => { textRefs.current[b === 0 ? 4 : 6] = el }} className={labelClass} /></div>
            </div>
          </Html>
        </group>
      ))}

      <Html position={[-1.85, -0.05, 0]} center>
        <div className="pointer-events-none text-center leading-4">
          <div>
            <span ref={(el) => { textRefs.current[7] = el }} className={accentClass} style={{ color: c.power }} />
          </div>
          <div><span ref={(el) => { textRefs.current[8] = el }} className={labelClass} /></div>
          <div><span ref={(el) => { textRefs.current[9] = el }} className={labelClass} /></div>
        </div>
      </Html>
    </>
  )
}

// ── 2D: the efficiency curve, honestly framed ──
// eta against load fraction over the slider's whole range. The axis is fitted to the curve and its
// range is printed, so the gentle fall past the peak is never dressed up as a cliff.
const N2D = 161
const PX0 = -3.3
const PX1 = 3.3
const PY0 = -1.15
const PY1 = 1.35
const AX_Y = -1.3
const AX_X = -3.5
const CURVE_INIT: [number, number, number][] = Array.from({ length: N2D }, () => [0, 0, 0])
const X_AXIS_PTS: [number, number, number][] = [[AX_X, AX_Y, 0], [PX1 + 0.18, AX_Y, 0]]
const Y_AXIS_PTS: [number, number, number][] = [[AX_X, AX_Y, 0], [AX_X, PY1 + 0.2, 0]]

const toPx = (x: number) => PX0 + ((x - X_MIN) / (X_MAX - X_MIN)) * (PX1 - PX0)

function EfficiencyCurve2DScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const bufs = useMemo(
    () => ({ curve: new Float32Array(N2D * 3), eta: new Float32Array(N2D), seg: new Float32Array(6) }),
    []
  )
  const curveRef = useRef<any>(null)
  const guideRefs = useRef<any[]>([]) // 0 peak ordinate, 1 full load, 2 the equal-efficiency tie
  const peakRef = useRef<Group>(null)
  const liveRef = useRef<Group>(null)
  const tagRefs = useRef<(Group | null)[]>([]) // 0 peak, 1 live, 2 x_opt tick, 3 full-load tick
  // 0 peak, 1 live, 2 x_opt tick, 3 full-load tick, 4 axis range, 5 tie note
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<unknown>(null)
  const axis = useRef([0, 100])

  useFrame(() => {
    const s = solve(valuesRef.current)

    // Sweep once for the values, so the axis can be fitted to what is actually drawn
    let lo = Infinity
    let hi = -Infinity
    for (let i = 0; i < N2D; i++) {
      const x = X_MIN + ((X_MAX - X_MIN) * i) / (N2D - 1)
      const e = 100 * efficiency(x, s.Pi, s.Pcu, s.pf)
      bufs.eta[i] = e
      if (e < lo) lo = e
      if (e > hi) hi = e
    }
    const pad = Math.max((hi - lo) * 0.08, 0.05)
    lo -= pad
    hi += pad
    axis.current[0] = lo
    axis.current[1] = hi
    const toPy = (e: number) => PY0 + ((e - lo) / (hi - lo)) * (PY1 - PY0)

    for (let i = 0; i < N2D; i++) {
      bufs.curve[i * 3] = PX0 + ((PX1 - PX0) * i) / (N2D - 1)
      bufs.curve[i * 3 + 1] = toPy(bufs.eta[i])
      bufs.curve[i * 3 + 2] = 0
    }
    curveRef.current?.geometry.setPositions(bufs.curve)

    const setSeg = (g: number, x0: number, y0: number, x1: number, y1: number) => {
      bufs.seg[0] = x0; bufs.seg[1] = y0; bufs.seg[2] = 0
      bufs.seg[3] = x1; bufs.seg[4] = y1; bufs.seg[5] = 0
      guideRefs.current[g]?.geometry.setPositions(bufs.seg)
      guideRefs.current[g]?.computeLineDistances()
    }

    // The peak, clamped into the plotted range when x_opt sits beyond full load
    const xPeak = Math.min(Math.max(s.xOpt, X_MIN), X_MAX)
    const yPeak = toPy(100 * efficiency(xPeak, s.Pi, s.Pcu, s.pf))
    setSeg(0, toPx(xPeak), AX_Y, toPx(xPeak), yPeak)
    peakRef.current?.position.set(toPx(xPeak), yPeak, 0)
    tagRefs.current[0]?.position.set(toPx(xPeak), yPeak + 0.26, 0)
    tagRefs.current[2]?.position.set(toPx(xPeak), AX_Y - 0.18, 0)

    // Full load, and the lighter load that lands at exactly the same height
    const etaFull = 100 * efficiency(1, s.Pi, s.Pcu, s.pf)
    setSeg(1, toPx(1), AX_Y, toPx(1), toPy(etaFull))
    tagRefs.current[3]?.position.set(toPx(1), AX_Y - 0.36, 0)
    const xTwin = s.Pi / s.Pcu // = x_opt² ÷ 1.0
    const tied = xTwin >= X_MIN && xTwin <= X_MAX && Math.abs(xTwin - 1) > 0.03
    if (guideRefs.current[2]) guideRefs.current[2].visible = tied
    if (tied) setSeg(2, toPx(xTwin), toPy(etaFull), toPx(1), toPy(etaFull))

    liveRef.current?.position.set(toPx(s.x), toPy(s.eta), 0)
    tagRefs.current[1]?.position.set(toPx(s.x), toPy(s.eta) - 0.26, 0)

    if (labelledFor.current !== valuesRef.current && textRefs.current[5]) {
      labelledFor.current = valuesRef.current
      const t = L10N[locale]
      const texts = [
        `η ${t.etaMaxAt} ${fmtPct(s.etaOpt)}`,
        `x = ${s.x.toFixed(2)} · η = ${fmtPct(s.eta)}`,
        `x_opt = ${fmtX(s.xOpt)} (${t.best})`,
        `${t.fullLoad} x = 1.00`,
        `η: ${lo.toFixed(2)} % ${t.axisFrom} ${hi.toFixed(2)} %`,
        tied ? t.tieNote(xTwin.toFixed(2), '1.00', fmtPct(etaFull)) : '',
      ]
      texts.forEach((text, i) => { if (textRefs.current[i]) textRefs.current[i]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  return (
    <>
      <Line points={X_AXIS_PTS} lineWidth={1} color={c.guide} />
      <Line points={Y_AXIS_PTS} lineWidth={1} color={c.guide} />

      {[0, 1, 2].map((g) => (
        <Line
          key={g}
          ref={(l) => { guideRefs.current[g] = l }}
          points={GUIDE_INIT}
          lineWidth={1.5}
          color={c.guide}
          dashed
          dashSize={0.1}
          gapSize={0.07}
        />
      ))}

      <TrailLine ref={curveRef} points={CURVE_INIT} lineWidth={3} color={c.power} fade={false} flowSpeed={0.5} />

      <Marker ref={peakRef} color={c.marker} size={0.06} />
      <Marker ref={liveRef} color={c.current} size={0.06} />

      {[0, 1, 2, 3].map((g) => (
        <group key={g} ref={(el) => { tagRefs.current[g] = el }}>
          <Html center>
            <span
              ref={(el) => { textRefs.current[g] = el }}
              className={g === 0 ? accentClass : g === 1 ? accentClass : labelClass}
              style={g === 0 ? { color: c.power } : g === 1 ? { color: c.current } : undefined}
            />
          </Html>
        </group>
      ))}

      <Html position={[AX_X, PY1 + 0.52, 0]}>
        <div className="pointer-events-none leading-4">
          <div><span ref={(el) => { textRefs.current[4] = el }} className={labelClass} /></div>
          <div><span ref={(el) => { textRefs.current[5] = el }} className={labelClass} /></div>
        </div>
      </Html>
      <Html position={[0, AX_Y - 0.56, 0]} center>
        <span className={labelClass}>{L10N[locale].xAxis} (0.05 → 1.25)</span>
      </Html>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'transformer-efficiency',
  title: { en: 'Transformer Losses & Efficiency', ko: '변압기의 손실과 효율' },
  subject: 'machines',
  order: 6,
  description: {
    en: `## Transformer Losses and Efficiency

**Iron loss is already there the moment the transformer is energized and never changes, while copper loss grows as the square of the load, so efficiency peaks exactly where the two are equal.**

- The amber trunk across the top: the real power passing through, entering at the left as Pin and leaving at the right as Pout. Its width follows the square root of the power it carries, so the load slider visibly swells it
- The first amber stream branching down: iron loss Pi, the hysteresis and eddy-current loss in the core. Move the load slider and watch it refuse to respond, in width, in sheen speed and in bar height
- The second amber stream: copper loss Pcu·x², the I²R heating in the windings. It thickens as the square of the load
- The two amber bars the streams land on: the same two losses, on one shared scale, so comparing them is exact
- The dashed line across both bars sits at the iron bar's height. When the copper bar reaches it the two losses are equal, and that is the peak of the efficiency curve
- The cyan line under the output: the load current, proportional to x. Copper loss is its square

Notice how little the trunk narrows as the streams leave it. A transformer this good loses only a few percent, which is exactly why the loss streams and the bars are drawn on their own scale rather than the trunk's: at true proportion they would be hairlines.

**Efficiency**: η = x·S·cosθ ÷ (x·S·cosθ + Pi + x²·Pcu) × 100 %
**Peak efficiency**: x_opt = sqrt(Pi ÷ Pcu), where x_opt²·Pcu = Pi
**Equivalent load at the peak**: x_opt·S

Worked example with S = 100 kVA, Pi = 1 kW, Pcu = 2 kW and cosθ = 0.8. At full load the output is 80 kW against 1 + 2 = 3 kW of loss, so η = 96.386 %. The peak is at x_opt = sqrt(1 ÷ 2) = 0.7071, that is 70.71 kVA, where copper loss has fallen to exactly 1.000 kW and η = 96.585 %. At x = 0.75, just past it, η = 96.579 %.

The check worth remembering: η(0.5) = η(1.0) = 96.386 %, the same number to every decimal. Efficiency is symmetric about the optimum, because x and x_opt² ÷ x always give the same efficiency. So the optimum is the geometric mean of any two loads that tie: sqrt(0.5 × 1.0) = 0.7071. Once you see that, the peak's location stops being a formula to memorise.

Raise Pi and the peak slides toward full load, because the fixed loss now needs more copper loss to match it. Raise Pcu and it slides back toward light load. Lower cosθ and every efficiency falls, since the same current carries less real power while the losses are untouched.

In 2D: η against load fraction over x = 0.05 to 1.25. The peak is marked, full load is marked, and the dashed tie joins the two loads that land at the same height. The efficiency axis is fitted to the curve and its range is printed on screen, so a change of a tenth of a percent still looks like a tenth of a percent.`,
    ko: `## 변압기의 손실과 효율

**철손은 변압기에 전압을 걸는 순간부터 존재하며 부하와 무관하게 일정하고, 동손은 부하의 제곱으로 커집니다. 그래서 효율은 두 손실이 같아지는 바로 그 지점에서 최대가 됩니다.**

- 위를 가로지르는 호박색 줄기: 통과하는 유효전력입니다. 왼쪽에서 Pin으로 들어와 오른쪽에서 Pout으로 나갑니다. 굵기는 전력의 제곱근을 따르므로 부하 슬라이더를 움직이면 눈에 띄게 부풀어 오릅니다
- 아래로 갈라지는 첫 번째 호박색 흐름: 철손 Pi, 철심의 히스테리시스손과 와류손입니다. 부하 슬라이더를 움직여 보세요. 굵기도, 광택이 흐르는 속도도, 막대 높이도 꼼짝하지 않습니다
- 두 번째 호박색 흐름: 동손 Pcu·x², 권선의 I²R 발열입니다. 부하의 제곱으로 굵어집니다
- 흐름이 내려앉는 호박색 막대 두 개: 같은 두 손실입니다. 같은 눈금으로 재므로 서로 비교한 값이 정확합니다
- 두 막대를 가로지르는 점선: 철손 막대의 높이에 놓여 있습니다. 동손 막대가 이 선에 닿으면 두 손실이 같아지고, 그곳이 효율 곡선의 최대점입니다
- 출력 아래의 청록색 선: 부하 전류로 x에 비례합니다. 동손은 이 값의 제곱입니다

흐름이 갈라져 나가도 줄기가 거의 가늘어지지 않는 것을 보세요. 이만한 변압기는 몇 퍼센트밖에 잃지 않습니다. 그래서 손실 흐름과 막대는 줄기와 같은 눈금이 아니라 자기들만의 눈금으로 그렸습니다. 실제 비율대로라면 머리카락처럼 보일 것입니다.

**효율**: η = x·S·cosθ ÷ (x·S·cosθ + Pi + x²·Pcu) × 100 %
**최대 효율**: x_opt = sqrt(Pi ÷ Pcu), 이때 x_opt²·Pcu = Pi
**최대 효율점의 부하**: x_opt·S

계산 예: S = 100 kVA, Pi = 1 kW, Pcu = 2 kW, cosθ = 0.8. 전부하에서 출력은 80 kW, 손실은 1 + 2 = 3 kW이므로 η = 96.386 %입니다. 최대점은 x_opt = sqrt(1 ÷ 2) = 0.7071, 즉 70.71 kVA이며, 그곳에서 동손은 정확히 1.000 kW로 줄고 η = 96.585 %가 됩니다. 바로 옆 x = 0.75에서는 η = 96.579 %입니다.

기억해 둘 만한 확인: η(0.5) = η(1.0) = 96.386 %로 소수점까지 똑같습니다. x와 x_opt² ÷ x의 효율이 언제나 같기 때문에 효율은 최대점을 기준으로 대칭입니다. 그래서 효율이 같은 두 부하의 기하평균이 바로 최대점입니다. sqrt(0.5 × 1.0) = 0.7071. 이것을 한 번 보면 최대점의 위치는 더 이상 외울 공식이 아닙니다.

Pi를 올리면 최대점이 전부하 쪽으로 옮겨갑니다. 고정 손실이 커진 만큼 더 큰 동손이 필요하기 때문입니다. Pcu를 올리면 반대로 경부하 쪽으로 당겨집니다. cosθ를 낮추면 손실은 그대로인데 같은 전류가 더 적은 유효전력을 나르므로 모든 효율이 함께 떨어집니다.

2D: x = 0.05부터 1.25까지의 부하율에 대한 η 곡선입니다. 최대점과 전부하 지점을 표시하고, 같은 높이에 놓이는 두 부하를 점선으로 이었습니다. 효율 축은 곡선에 맞춰 잡고 그 범위를 화면에 적어 두었으므로, 0.1 % 변화는 화면에서도 0.1 %만큼만 보입니다.`,
  },
  sliderParams: [
    { key: 'x',   label: { en: 'Load Fraction (x)', ko: '부하율 (x)' },                             min: 0.05, max: X_MAX, step: 0.05, defaultValue: 1.0 },
    { key: 'Pi',  label: { en: 'Iron Loss (Pi)', ko: '철손 (Pi)' },                     unit: 'kW', min: 0.2,  max: 4,     step: 0.1,  defaultValue: 1.0 },
    { key: 'Pcu', label: { en: 'Full-Load Copper Loss (Pcu)', ko: '전부하 동손 (Pcu)' }, unit: 'kW', min: 0.5,  max: 6,     step: 0.1,  defaultValue: 2.0 },
    { key: 'pf',  label: { en: 'Load Power Factor (cosθ)', ko: '부하 역률 (cosθ)' },                 min: 0.6,  max: 1.0,   step: 0.01, defaultValue: 0.8 },
  ],
  Scene: LossFlowScene,
  Scene2D: EfficiencyCurve2DScene,
}

export default concept
