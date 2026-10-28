import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import { useLocaleStore, type Locale } from '../../lib/i18n'
import type { ConceptMeta } from '../../types/concept'

const V_MAX = 20 // slider ceiling [V], also the plateau that reaches full plot height

// The time axis spans WIN_N units of normalized time τ = ωn·t, so the picture keeps its shape
// when L or C move the natural frequency. One sweep takes SWEEP seconds on screen, then holds
// until CYCLE and restarts — real circuit time is therefore slowed by SWEEP ÷ (WIN_N ÷ ωn).
const WIN_N = 20
const SWEEP = 6
const CYCLE = 7.2

// ζ is read as exactly critical inside this band; the same thresholds pick the solution branch,
// so the regime named on screen is always the branch actually drawn.
const CRIT_LO = 0.999
const CRIT_HI = 1.001

const REF_ZETA = [0.3, 1, 2] // the faint reference curves drawn behind the learner's own
const LABEL_TAU = 2.5 // where the three reference curves are furthest apart, so labels don't collide
const SETTLE = 0.02 // the ±2 % settling band

const PTS = 180
const CURVE_INIT: [number, number, number][] = Array.from({ length: PTS }, () => [0, 0, 0])
const SEG_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]

/**
 * Normalized step response vC ÷ V at normalized time τ = ωn·t. Three honest branches:
 * underdamped rings, critically damped is the fastest climb without overshoot, overdamped crawls.
 */
function stepNorm(zeta: number, tau: number) {
  if (zeta < CRIT_LO) {
    const wd = Math.sqrt(1 - zeta * zeta) // ωd ÷ ωn
    return 1 - Math.exp(-zeta * tau) * (Math.cos(wd * tau) + (zeta / wd) * Math.sin(wd * tau))
  }
  if (zeta <= CRIT_HI) return 1 - (1 + tau) * Math.exp(-tau)
  const d = Math.sqrt(zeta * zeta - 1)
  const s1 = -(zeta - d) // the slow real root, in units of ωn
  const s2 = -(zeta + d) // the fast one
  return 1 - (s2 * Math.exp(s1 * tau) - s1 * Math.exp(s2 * tau)) / (s2 - s1)
}

// Reused result object so solve() allocates nothing when called inside useFrame.
const OUT = {
  R: 0, V: 0, wn: 0, zeta: 0, wd: 0, Rc: 0, regime: 0,
  tp: 0, Mp: 0, Td: 0, ts: 0, win: 0, slow: 0,
}

// Pure physics of the series RLC step response. R in Ω, L in mH, C in μF, V in V.
// ωn = 1 ÷ sqrt(LC), ζ = (R ÷ 2)·sqrt(C ÷ L), critical R = 2·sqrt(L ÷ C), ωd = ωn·sqrt(1 − ζ²).
// Overshoot Mp = e^(−ζπ ÷ sqrt(1 − ζ²)) at tp = π ÷ ωd, ringing period Td = 2π ÷ ωd,
// and the 2 % settling time ts ≈ 4 ÷ (ζ·ωn) while ζ < 1.
// Worked example, checked with node:
//   L = 100 mH, C = 10 μF -> ωn = 1000 rad ÷ s (fn = 159.15 Hz), critical R = 200 Ω
//   R =  50 Ω -> ζ = 0.25 underdamped, ωd = 968.25 rad ÷ s, Mp = 44.4 % at tp = 3.24 ms,
//                Td = 6.49 ms, ts = 16.0 ms
//   R = 200 Ω -> ζ = 1 exactly, critically damped
//   R = 400 Ω -> ζ = 2 overdamped, roots at −268 and −3732 rad ÷ s
function solve({ R = 50, L_mH = 100, C_uF = 10, V = 12 }: Record<string, number>) {
  const L = L_mH * 1e-3
  const C = C_uF * 1e-6
  const wn = 1 / Math.sqrt(L * C)
  const zeta = (R / 2) * Math.sqrt(C / L)
  const under = zeta < CRIT_LO
  const wd = under ? wn * Math.sqrt(1 - zeta * zeta) : 0
  OUT.R = R
  OUT.V = V
  OUT.wn = wn
  OUT.zeta = zeta
  OUT.wd = wd
  OUT.Rc = 2 * Math.sqrt(L / C)
  OUT.regime = under ? 0 : zeta <= CRIT_HI ? 1 : 2
  OUT.tp = under ? Math.PI / wd : 0
  OUT.Mp = under ? Math.exp((-zeta * Math.PI) / Math.sqrt(1 - zeta * zeta)) : 0
  OUT.Td = under ? (2 * Math.PI) / wd : 0
  OUT.ts = under ? 4 / (zeta * wn) : 0
  OUT.win = WIN_N / wn
  OUT.slow = SWEEP / OUT.win
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

const REGIME: Record<Locale, string[]> = {
  en: ['underdamped', 'critically damped', 'overdamped'],
  ko: ['부족감쇠', '임계감쇠', '과감쇠'],
}

const fmtT = (s: number) => (s < 1e-3 ? `${(s * 1e6).toFixed(0)} μs` : s < 1 ? `${(s * 1e3).toFixed(2)} ms` : `${s.toFixed(2)} s`)
const fmtW = (w: number) => (w < 1000 ? `${w.toFixed(1)} rad ÷ s` : `${(w / 1e3).toFixed(3)} krad ÷ s`)
const fmtR = (r: number) => `${r.toFixed(r < 100 ? 1 : 0)} Ω`

// Labels read in the foreground color; nothing here is current or power, so nothing is tinted
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const axisClass = 'pointer-events-none whitespace-nowrap text-[11px] tabular-nums text-muted-foreground'

// ── The step response plot, shared by both scenes ──
// Local coordinates: (0, 0) is the axis origin, +X is time, +Y is vC.
// zStep pushes each reference curve one step further back; 0 keeps them in the same plane.
type PlotProps = {
  position: [number, number, number]
  width: number
  height: number
  zStep: number
}

function StepPlot({ position, width, height, zStep }: PlotProps) {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  const locale = useLocaleStore((s) => s.locale)

  const liveBuf = useMemo(() => new Float32Array(PTS * 3), [])
  const refBufs = useMemo(() => REF_ZETA.map(() => new Float32Array(PTS * 3)), [])
  const seg = useMemo(() => new Float32Array(6), [])
  const axis = useMemo<[number, number, number][]>(
    () => [[0, height * 1.95, 0], [0, 0, 0], [width, 0, 0]],
    [width, height]
  )

  const liveRef = useRef<any>(null)
  const refRefs = useRef<any[]>([])
  const finalRef = useRef<any>(null)
  const bandHiRef = useRef<any>(null)
  const bandLoRef = useRef<any>(null)
  const tsRef = useRef<any>(null)
  const ringRef = useRef<any>(null)
  const nowDot = useRef<THREE.Group>(null)
  const peak1 = useRef<THREE.Group>(null)
  const peak2 = useRef<THREE.Group>(null)
  const tagRefs = useRef<(THREE.Group | null)[]>([]) // regime, overshoot, ring period, ts, 3 references
  const textRefs = useRef<(HTMLSpanElement | null)[]>([]) // the above, then readout and time note
  const labelledFor = useRef<unknown>(null)

  // Locale never changes the slider values, so nudge the label cache when it flips
  useEffect(() => { labelledFor.current = null }, [locale])

  useFrame(({ clock }) => {
    const s = solve(valuesRef.current)
    const h = height * (s.V / V_MAX)
    const xPerTau = width / WIN_N

    // Repeating switch-on: the present instant sweeps across the window, holds, then restarts
    const frac = Math.min((clock.getElapsedTime() % CYCLE) / SWEEP, 1)
    const tauNow = Math.max(frac * WIN_N, WIN_N * 1e-3)

    // The three regimes, same ωn and same final value, so only the shape differs
    for (let g = 0; g < REF_ZETA.length; g++) {
      const buf = refBufs[g]
      const zg = -(g + 1) * zStep
      for (let i = 0; i < PTS; i++) {
        const tau = (i / (PTS - 1)) * WIN_N
        buf[i * 3] = tau * xPerTau
        buf[i * 3 + 1] = h * stepNorm(REF_ZETA[g], tau)
        buf[i * 3 + 2] = zg
      }
      refRefs.current[g]?.geometry.setPositions(buf)
    }

    // The learner's own curve, growing with the sweep
    for (let i = 0; i < PTS; i++) {
      const tau = (i / (PTS - 1)) * tauNow
      liveBuf[i * 3] = tau * xPerTau
      liveBuf[i * 3 + 1] = h * stepNorm(s.zeta, tau)
      liveBuf[i * 3 + 2] = 0.01
    }
    liveRef.current?.geometry.setPositions(liveBuf)
    nowDot.current?.position.set(tauNow * xPerTau, h * stepNorm(s.zeta, tauNow), 0.02)

    // Final value V, and the ±2 % band the response has to stay inside to count as settled
    seg[0] = 0; seg[1] = h; seg[2] = 0; seg[3] = width; seg[4] = h; seg[5] = 0
    finalRef.current?.geometry.setPositions(seg)
    finalRef.current?.computeLineDistances()
    seg[1] = h * (1 + SETTLE); seg[4] = h * (1 + SETTLE)
    bandHiRef.current?.geometry.setPositions(seg)
    bandHiRef.current?.computeLineDistances()
    seg[1] = h * (1 - SETTLE); seg[4] = h * (1 - SETTLE)
    bandLoRef.current?.geometry.setPositions(seg)
    bandLoRef.current?.computeLineDistances()

    // Settling time, as a vertical: where the ringing finally fits inside the band
    const tauS = s.zeta > 0 ? 4 / s.zeta : WIN_N * 2
    const showTs = s.regime === 0 && tauS <= WIN_N
    const xTs = tauS * xPerTau
    seg[0] = xTs; seg[1] = 0; seg[2] = 0; seg[3] = xTs; seg[4] = h * (1 + SETTLE); seg[5] = 0
    tsRef.current?.geometry.setPositions(seg)
    tsRef.current?.computeLineDistances()
    if (tsRef.current) tsRef.current.visible = showTs

    // Overshoot and ringing: the first two peaks, and the bar between them is one period
    const tauP = s.regime === 0 ? Math.PI / (s.wd / s.wn) : 0
    const showPeak = s.regime === 0 && tauP <= WIN_N
    const showRing = s.regime === 0 && 3 * tauP <= WIN_N
    const yPeak = h * (1 + s.Mp)
    const x1 = tauP * xPerTau
    const x2 = 3 * tauP * xPerTau
    peak1.current?.position.set(x1, yPeak, 0.03)
    peak2.current?.position.set(x2, h * stepNorm(s.zeta, 3 * tauP), 0.03)
    seg[0] = x1; seg[1] = yPeak + 0.12; seg[2] = 0.02
    seg[3] = x2; seg[4] = yPeak + 0.12; seg[5] = 0.02
    ringRef.current?.geometry.setPositions(seg)
    if (peak1.current) peak1.current.visible = showPeak
    if (peak2.current) peak2.current.visible = showRing
    if (ringRef.current) ringRef.current.visible = showRing

    tagRefs.current[0]?.position.set(width * 0.5, height * 1.95 + 0.18, 0)
    tagRefs.current[1]?.position.set(x1, yPeak + 0.3, 0)
    tagRefs.current[2]?.position.set((x1 + x2) / 2, yPeak + 0.38, 0)
    tagRefs.current[3]?.position.set(xTs, -0.18, 0)
    for (let g = 0; g < REF_ZETA.length; g++) {
      tagRefs.current[4 + g]?.position.set(
        LABEL_TAU * xPerTau,
        h * stepNorm(REF_ZETA[g], LABEL_TAU) + 0.16,
        -(g + 1) * zStep
      )
    }
    if (textRefs.current[1]) textRefs.current[1]!.style.opacity = showPeak ? '1' : '0'
    if (textRefs.current[2]) textRefs.current[2]!.style.opacity = showRing ? '1' : '0'
    if (textRefs.current[3]) textRefs.current[3]!.style.opacity = showTs ? '1' : '0'

    // Text only when the slider values object changes identity, never per frame
    if (labelledFor.current !== valuesRef.current) {
      labelledFor.current = valuesRef.current
      const name = REGIME[locale][s.regime]
      const texts = [
        `R = ${fmtR(s.R)} · ζ = ${s.zeta.toFixed(3)} · ${name}`,
        `Mp = ${(s.Mp * 100).toFixed(1)} % · tp = ${fmtT(s.tp)}`,
        `Td = 2π ÷ ωd = ${fmtT(s.Td)}`,
        `ts (2 %) = ${fmtT(s.ts)}`,
        'ζ = 0.3',
        `ζ = 1 · R = Rc = ${fmtR(s.Rc)}`,
        'ζ = 2',
        `ωn = ${fmtW(s.wn)} (${(s.wn / (2 * Math.PI)).toFixed(1)} Hz) · ωd = ${s.regime === 0 ? fmtW(s.wd) : '—'} · Rc = ${fmtR(s.Rc)} · vC → ${s.V.toFixed(0)} V`,
        locale === 'ko'
          ? `t: 0 ~ ${fmtT(s.win)} (ωn·t = ${WIN_N}) · ${s.slow.toFixed(0)}배 느리게`
          : `t: 0 ~ ${fmtT(s.win)} (ωn·t = ${WIN_N}) · slowed ×${s.slow.toFixed(0)}`,
      ]
      for (let i = 0; i < texts.length; i++) {
        const el = textRefs.current[i]
        if (el) el.textContent = texts[i]
      }
    }
  })

  return (
    <group position={position}>
      <Line points={axis} lineWidth={1} color={c.guide} />
      <Line ref={finalRef} points={SEG_INIT} lineWidth={1.5} color={c.guide} dashed dashSize={0.1} gapSize={0.08} />
      <Line ref={bandHiRef} points={SEG_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.05} gapSize={0.05} />
      <Line ref={bandLoRef} points={SEG_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.05} gapSize={0.05} />
      <Line ref={tsRef} points={SEG_INIT} lineWidth={1.5} color={c.capacitance} dashed dashSize={0.08} gapSize={0.06} />
      <Line ref={ringRef} points={SEG_INIT} lineWidth={2.5} color={c.resistance} />

      {/* The family: ζ = 0.3, 1 and 2 at the same ωn, behind the learner's curve */}
      {REF_ZETA.map((z, g) => (
        <Line
          key={z}
          ref={(l) => { refRefs.current[g] = l }}
          points={CURVE_INIT}
          lineWidth={1}
          color={c.guide}
        />
      ))}

      <TrailLine ref={liveRef} points={CURVE_INIT} lineWidth={3} color={c.voltage} fade={false} />
      <Marker ref={peak1} color={c.marker} size={0.05} />
      <Marker ref={peak2} color={c.marker} size={0.05} />
      <Marker ref={nowDot} color={c.marker} size={0.07} />

      {[0, 1, 2, 3, 4, 5, 6].map((t) => (
        <group key={t} ref={(g) => { tagRefs.current[t] = g }}>
          <Html center>
            <span
              ref={(el) => { textRefs.current[t] = el }}
              className={t > 3 ? axisClass : labelClass}
            />
          </Html>
        </group>
      ))}

      <Html position={[0, -0.52, 0]}>
        <span ref={(el) => { textRefs.current[7] = el }} className={labelClass} />
      </Html>
      <Html position={[width, -0.18, 0]} center>
        <span ref={(el) => { textRefs.current[8] = el }} className={axisClass} />
      </Html>
      <Html position={[-0.3, height, 0]} center>
        <span className={axisClass}>vC</span>
      </Html>
    </group>
  )
}

// ── 3D: the same plot, with the three regime references receding into depth ──
function DampingScene() {
  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />
      <StepPlot position={[-1.8, -1.0, 0.9]} width={3.6} height={1.0} zStep={0.7} />
    </>
  )
}

// ── 2D: all four curves in one plane, so the regimes can be compared directly ──
function Damping2DScene() {
  return <StepPlot position={[-3.3, -1.3, 0]} width={6.2} height={1.05} zStep={0} />
}

const concept: ConceptMeta = {
  id: 'rlc-damping',
  title: { en: 'RLC Transient & Damping', ko: 'RLC 과도응답과 감쇠' },
  subject: 'circuits',
  order: 21,
  cardPosition: [3.3, 1.2, -0.6],
  description: {
    en: `## RLC Transient and Damping

**Put L and C in the same circuit and the switch-on stops being a simple exponential: one number, the damping ratio ζ, decides whether the voltage rings, arrives as fast as it can, or crawls.**

- Bright line: vC after the switch closes, for your own R, L, C and V
- Three faint curves behind it: the same circuit at ζ = 0.3, ζ = 1 and ζ = 2. Move R and your curve slides through that family
- Dashed horizontal: the final value V, with the ±2 % settling band around it
- Two small dots at the ringing peaks, and the bar between them is one period Td
- Dashed vertical: the settling time, where the ringing finally fits inside the band

**Natural frequency**: ωn = 1 ÷ sqrt(LC)
**Damping ratio**: ζ = (R ÷ 2)·sqrt(C ÷ L), so the critical resistance is Rc = 2·sqrt(L ÷ C)
**Damped frequency** (ζ < 1): ωd = ωn·sqrt(1 − ζ²)

Three regimes, three different solutions:
- ζ < 1 **underdamped**: vC = V[1 − e^(−ζωn t)(cos ωd t + (ζωn ÷ ωd) sin ωd t)] — it overshoots by Mp = e^(−ζπ ÷ sqrt(1 − ζ²)) at tp = π ÷ ωd, then rings at ωd
- ζ = 1 **critically damped**: vC = V[1 − (1 + ωn t)e^(−ωn t)] — the fastest arrival with no overshoot at all
- ζ > 1 **overdamped**: the two real roots s = −ωn(ζ ∓ sqrt(ζ² − 1)) give vC = V[1 − (s2·e^(s1 t) − s1·e^(s2 t)) ÷ (s2 − s1)] — the slow root dominates and the climb drags

With L = 100 mH and C = 10 μF: ωn = 1000 rad ÷ s (159.15 Hz) and Rc = 200 Ω. At R = 50 Ω, ζ = 0.25, ωd = 968.2 rad ÷ s, the overshoot is 44.4 % at 3.24 ms and the ringing period is 6.49 ms. At R = 200 Ω, ζ is exactly 1. At R = 400 Ω, ζ = 2 and the response never overshoots at all.

Honest about the clock: real RLC transients last milliseconds, so the time axis is slowed for viewing. It always spans 20 ÷ ωn seconds and plays in 6 seconds, and the readout gives the real duration and the slowdown factor.

Raise R and watch the ringing die out, vanish at Rc, and turn into a slow crawl. L and C move ωn, so the whole event speeds up or slows down while ζ follows them too. V only lifts the plateau; it changes no timing at all.`,
    ko: `## RLC 과도응답과 감쇠

**한 회로에 L과 C가 함께 있으면 스위치를 닫았을 때의 응답이 단순한 지수함수를 벗어납니다. 감쇠비 ζ라는 수 하나가 전압이 진동할지, 최대한 빠르게 도달할지, 느릿느릿 기어갈지를 결정합니다.**

- 밝은 선: 스위치를 닫은 뒤의 vC. 지금 설정한 R, L, C, V에 대한 응답입니다
- 뒤쪽의 흐린 곡선 셋: 같은 회로의 ζ = 0.3, ζ = 1, ζ = 2. R을 움직이면 내 곡선이 이 가족 사이를 미끄러집니다
- 가로 점선: 최종값 V와 그 둘레의 ±2 % 정정 구간
- 진동 꼭대기의 작은 점 두 개. 그 사이의 굵은 선이 한 주기 Td입니다
- 세로 점선: 진동이 마침내 구간 안에 들어오는 정정 시간

**고유 각주파수**: ωn = 1 ÷ sqrt(LC)
**감쇠비**: ζ = (R ÷ 2)·sqrt(C ÷ L), 따라서 임계 저항은 Rc = 2·sqrt(L ÷ C)
**감쇠 각주파수** (ζ < 1): ωd = ωn·sqrt(1 − ζ²)

세 영역, 세 가지 해:
- ζ < 1 **부족감쇠**: vC = V[1 − e^(−ζωn t)(cos ωd t + (ζωn ÷ ωd) sin ωd t)] — tp = π ÷ ωd에서 Mp = e^(−ζπ ÷ sqrt(1 − ζ²))만큼 넘어선 뒤 ωd로 진동합니다
- ζ = 1 **임계감쇠**: vC = V[1 − (1 + ωn t)e^(−ωn t)] — 오버슈트 없이 가장 빠르게 도달하는 경계입니다
- ζ > 1 **과감쇠**: 두 실근 s = −ωn(ζ ∓ sqrt(ζ² − 1))에서 vC = V[1 − (s2·e^(s1 t) − s1·e^(s2 t)) ÷ (s2 − s1)] — 느린 근이 지배해 상승이 늘어집니다

L = 100 mH, C = 10 μF이면 ωn = 1000 rad ÷ s (159.15 Hz), Rc = 200 Ω입니다. R = 50 Ω에서는 ζ = 0.25, ωd = 968.2 rad ÷ s이고 3.24 ms에서 44.4 % 넘어선 뒤 6.49 ms 주기로 흔들립니다. R = 200 Ω이면 ζ가 정확히 1이고, R = 400 Ω이면 ζ = 2라 한 번도 넘어서지 않습니다.

시간에 대해 솔직히 적어 둡니다. 실제 RLC 과도현상은 밀리초 단위라 시간 축을 느리게 늘여 보여 줍니다. 축은 언제나 20 ÷ ωn 초를 담고 6초에 걸쳐 재생되며, 실제 길이와 느려진 배율은 화면 아래에 표시됩니다.

R을 올리면 진동이 잦아들다가 Rc에서 사라지고 그 뒤로는 느린 포복이 됩니다. L과 C는 ωn을 바꾸므로 사건 전체가 빨라지거나 느려지고 ζ도 함께 따라옵니다. V는 평탄값만 올릴 뿐 시간에는 전혀 손대지 않습니다.`,
  },
  sliderParams: [
    { key: 'R',    label: { en: 'Resistance (R) — picks the regime', ko: '저항 (R) — 감쇠 영역을 고릅니다' }, unit: 'Ω',  min: 10, max: 600, step: 5,  defaultValue: 50 },
    { key: 'L_mH', label: { en: 'Inductance (L)',  ko: '인덕턴스 (L)' },   unit: 'mH', min: 10, max: 500, step: 10, defaultValue: 100 },
    { key: 'C_uF', label: { en: 'Capacitance (C)', ko: '커패시턴스 (C)' }, unit: 'μF', min: 1,  max: 100, step: 1,  defaultValue: 10 },
    { key: 'V',    label: { en: 'Step voltage (V)', ko: '계단 입력 전압 (V)' }, unit: 'V', min: 5, max: V_MAX, step: 1, defaultValue: 12 },
  ],
  Scene: DampingScene,
  Scene2D: Damping2DScene,
}

export default concept
