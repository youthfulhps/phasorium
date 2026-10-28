import { useRef, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { TrailLine } from '../../components/scene/TrailLine'
import type { ConceptMeta } from '../../types/concept'

const MAX_N = 25 // highest odd harmonic the slider reaches
const MAX_TERMS = (MAX_N + 1) / 2 // 13 lanes, one per odd harmonic
const LANE_IDX = Array.from({ length: MAX_TERMS }, (_, i) => i)
const WINDOW = 2 // seconds of signal across the plot

/** Peak amplitude of the n-th harmonic of a square wave of amplitude A. */
const harmonicPeak = (A: number, n: number) => (4 * A) / (Math.PI * n)

/** The partial Fourier sum at one phase angle: (4A ÷ π)·Σ sin(n·phase) ÷ n over odd n ≤ nMax. */
function sumAt(phase: number, A: number, nMax: number) {
  let acc = 0
  for (let n = 1; n <= nMax; n += 2) acc += Math.sin(n * phase) / n
  return ((4 * A) / Math.PI) * acc
}

// Reused result object: solve() runs every frame, so it must not allocate.
const OUT = { A: 0, nMax: 1, terms: 1, v1: 0, v1rms: 0, rms: 0, thd: 0, v3: 0, v5: 0 }

// Pure physics of a square wave of amplitude A as a sum of odd harmonics:
//   f(t) = (4A ÷ π) · Σ[n odd] sin(n·ω·t) ÷ n,  so the n-th peak is Vn = 4A ÷ (nπ)
//   component RMS Vn,rms = Vn ÷ sqrt(2);  total RMS = sqrt(Σ Vn,rms²)
//   THD = sqrt(Σ[n ≥ 3] Vn,rms²) ÷ V1,rms
// Worked example, A = 1 V, verified with node:
//   V1 = 4 ÷ π = 1.2732 V peak, V1,rms = 0.9003 V
//   odd harmonics to n = 9  -> total RMS 0.97959 V, THD 42.88 %
//   odd harmonics to n = 25 -> total RMS 0.99218 V, THD 46.31 %
//   odd harmonics to n = 199 -> total RMS 0.99899 V (0.999), THD 48.08 % (48.1 %)
//   limits: total RMS -> 1.000 V = A, THD -> sqrt(π² ÷ 8 − 1) = 48.34 %
function solve({ nMaxOdd = 9, amp = 1 }: Record<string, number>) {
  const A = amp
  const nMax = Math.max(1, Math.round((nMaxOdd - 1) / 2) * 2 + 1)
  const v1 = harmonicPeak(A, 1)
  const v1rms = v1 / Math.SQRT2
  let sumSq = 0
  let harmSq = 0
  for (let n = 1; n <= nMax; n += 2) {
    const r = harmonicPeak(A, n) / Math.SQRT2
    sumSq += r * r
    if (n >= 3) harmSq += r * r
  }
  OUT.A = A
  OUT.nMax = nMax
  OUT.terms = (nMax + 1) / 2
  OUT.v1 = v1
  OUT.v1rms = v1rms
  OUT.rms = Math.sqrt(sumSq)
  OUT.thd = Math.sqrt(harmSq) / v1rms
  OUT.v3 = harmonicPeak(A, 3)
  OUT.v5 = harmonicPeak(A, 5)
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

// Labels read in the foreground color; this concept has no current or power, so nothing is tinted
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const axisClass = 'pointer-events-none whitespace-nowrap text-[11px] tabular-nums text-muted-foreground'

const SEG_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]

// ── 3D: harmonics stacked into depth, their running sum in front ──
const X_L = -2.6
const X_SPAN = 5.2
const SUM_Z = 0.3 // the running sum sits nearest the camera
const LANE_GAP = 0.32 // each further harmonic recedes one gap deeper
const SPEC_X = -3.05 // the spectrum bars stand at the start of every lane
const SUM_PTS = 801 // enough samples to resolve the 25th harmonic over three cycles
const LANE_PTS = 401
const SUM_INIT: [number, number, number][] = Array.from({ length: SUM_PTS }, () => [0, 0, 0])
const LANE_INIT: [number, number, number][] = Array.from({ length: LANE_PTS }, () => [0, 0, 0])
const AXIS_3D: [number, number, number][] = [[X_L, 0, SUM_Z], [X_L + X_SPAN, 0, SUM_Z]]

const laneZ = (k: number) => SUM_Z - LANE_GAP * (k + 1)

function HarmonicStackScene() {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  const sumBuf = useMemo(() => new Float32Array(SUM_PTS * 3), [])
  const sqBuf = useMemo(() => new Float32Array(SUM_PTS * 3), [])
  const laneBufs = useMemo(() => LANE_IDX.map(() => new Float32Array(LANE_PTS * 3)), [])
  const segBuf = useMemo(() => new Float32Array(6), [])
  const sumRef = useRef<any>(null)
  const sqRef = useRef<any>(null)
  const laneRefs = useRef<any[]>([])
  const barRefs = useRef<any[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<unknown>(null)

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime()
    const s = solve(valuesRef.current)
    const freq = valuesRef.current.freq ?? 1
    const omega = 2 * Math.PI * freq

    // Running sum and the ideal square it is converging to, both on the front plane
    for (let i = 0; i < SUM_PTS; i++) {
      const frac = i / (SUM_PTS - 1)
      const phase = omega * (t - (1 - frac) * WINDOW)
      const x = X_L + frac * X_SPAN
      sumBuf[i * 3] = x
      sumBuf[i * 3 + 1] = sumAt(phase, s.A, s.nMax)
      sumBuf[i * 3 + 2] = SUM_Z
      sqBuf[i * 3] = x
      sqBuf[i * 3 + 1] = s.A * Math.sign(Math.sin(phase))
      sqBuf[i * 3 + 2] = SUM_Z - 0.03
    }
    sumRef.current?.geometry.setPositions(sumBuf)
    sqRef.current?.geometry.setPositions(sqBuf)
    sqRef.current?.computeLineDistances()

    // One lane per odd harmonic, receding along −Z, plus its bar in the spectrum
    for (let k = 0; k < MAX_TERMS; k++) {
      const shown = k < s.terms
      const lane = laneRefs.current[k]
      const bar = barRefs.current[k]
      if (lane) lane.visible = shown
      if (bar) bar.visible = shown
      if (!shown) continue
      const n = 2 * k + 1
      const peak = harmonicPeak(s.A, n)
      const z = laneZ(k)
      const buf = laneBufs[k]
      for (let i = 0; i < LANE_PTS; i++) {
        const frac = i / (LANE_PTS - 1)
        const phase = omega * (t - (1 - frac) * WINDOW)
        buf[i * 3] = X_L + frac * X_SPAN
        buf[i * 3 + 1] = peak * Math.sin(n * phase)
        buf[i * 3 + 2] = z
      }
      lane?.geometry.setPositions(buf)

      segBuf[0] = SPEC_X; segBuf[1] = 0; segBuf[2] = z
      segBuf[3] = SPEC_X; segBuf[4] = peak; segBuf[5] = z
      bar?.geometry.setPositions(segBuf)
    }

    if (labelledFor.current !== valuesRef.current && textRefs.current[2]) {
      labelledFor.current = valuesRef.current
      const a = `n ≤ ${s.nMax} · ${s.terms} odd terms`
      const b = `RMS ${s.rms.toFixed(3)} V · THD ${(s.thd * 100).toFixed(1)} %`
      const d = `V1 = 4A ÷ π = ${s.v1.toFixed(3)} V`
      if (textRefs.current[0]) textRefs.current[0]!.textContent = a
      if (textRefs.current[1]) textRefs.current[1]!.textContent = b
      if (textRefs.current[2]) textRefs.current[2]!.textContent = d
    }
  })

  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />

      <Line points={AXIS_3D} lineWidth={1} color={c.guide} />

      {/* Each harmonic on its own lane, deeper for higher n */}
      {LANE_IDX.map((k) => (
        <Line
          key={k}
          ref={(l) => { laneRefs.current[k] = l }}
          points={LANE_INIT}
          lineWidth={1.5}
          color={c.resistance}
        />
      ))}

      {/* Spectrum: one bar per harmonic, standing at the head of its lane */}
      {LANE_IDX.map((k) => (
        <Line
          key={k}
          ref={(l) => { barRefs.current[k] = l }}
          points={SEG_INIT}
          lineWidth={4}
          color={k === 0 ? c.voltage : c.resistance}
        />
      ))}

      {/* The ideal square wave, and the running sum closing in on it */}
      <Line ref={sqRef} points={SUM_INIT} lineWidth={1.5} color={c.guide} dashed dashSize={0.1} gapSize={0.07} />
      <TrailLine ref={sumRef} points={SUM_INIT} lineWidth={3} color={c.voltage} />

      <Html position={[X_L, 1.85, SUM_Z]}>
        <span ref={(el) => { textRefs.current[0] = el }} className={labelClass} />
      </Html>
      <Html position={[X_L, 1.55, SUM_Z]}>
        <span ref={(el) => { textRefs.current[1] = el }} className={labelClass} />
      </Html>
      <Html position={[X_L, -1.9, SUM_Z]}>
        <span ref={(el) => { textRefs.current[2] = el }} className={labelClass} />
      </Html>
      <Html position={[SPEC_X, 1.85, laneZ(0)]} center>
        <span className={axisClass}>Vn</span>
      </Html>
    </>
  )
}

// ── 2D: the summed wave on the left, the harmonic spectrum on the right ──
const W_X0 = -3.9
const W_SPAN = 4.0
const SPEC_X0 = 0.95
const SPEC_SPAN = 2.9
const SPEC_STEP = SPEC_SPAN / MAX_TERMS
const SPEC_Y = -1.6 // spectrum baseline
const SPEC_SCALE = 1.8 // scene units per volt of harmonic amplitude
const TICK_N = [1, 5, 9, 13, 17, 21, 25]
const WAVE_AXIS: [number, number, number][] = [[W_X0, 0, 0], [W_X0 + W_SPAN, 0, 0]]
const SPEC_AXIS: [number, number, number][] = [[SPEC_X0, SPEC_Y, 0], [SPEC_X0 + SPEC_SPAN, SPEC_Y, 0]]
const WAVE_INIT: [number, number, number][] = Array.from({ length: SUM_PTS }, () => [0, 0, 0])

const barX = (k: number) => SPEC_X0 + (k + 0.5) * SPEC_STEP

function HarmonicSpectrum2DScene() {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  const sumBuf = useMemo(() => new Float32Array(SUM_PTS * 3), [])
  const sqBuf = useMemo(() => new Float32Array(SUM_PTS * 3), [])
  const segBuf = useMemo(() => new Float32Array(6), [])
  const sumRef = useRef<any>(null)
  const sqRef = useRef<any>(null)
  const barRefs = useRef<any[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<unknown>(null)

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime()
    const s = solve(valuesRef.current)
    const freq = valuesRef.current.freq ?? 1
    const omega = 2 * Math.PI * freq

    for (let i = 0; i < SUM_PTS; i++) {
      const frac = i / (SUM_PTS - 1)
      const phase = omega * (t - (1 - frac) * WINDOW)
      const x = W_X0 + frac * W_SPAN
      sumBuf[i * 3] = x
      sumBuf[i * 3 + 1] = sumAt(phase, s.A, s.nMax)
      sumBuf[i * 3 + 2] = 0.01
      sqBuf[i * 3] = x
      sqBuf[i * 3 + 1] = s.A * Math.sign(Math.sin(phase))
      sqBuf[i * 3 + 2] = 0
    }
    sumRef.current?.geometry.setPositions(sumBuf)
    sqRef.current?.geometry.setPositions(sqBuf)
    sqRef.current?.computeLineDistances()

    for (let k = 0; k < MAX_TERMS; k++) {
      const bar = barRefs.current[k]
      const shown = k < s.terms
      if (bar) bar.visible = shown
      if (!shown) continue
      const x = barX(k)
      segBuf[0] = x; segBuf[1] = SPEC_Y; segBuf[2] = 0
      segBuf[3] = x; segBuf[4] = SPEC_Y + harmonicPeak(s.A, 2 * k + 1) * SPEC_SCALE; segBuf[5] = 0
      bar?.geometry.setPositions(segBuf)
    }

    if (labelledFor.current !== valuesRef.current && textRefs.current[2]) {
      labelledFor.current = valuesRef.current
      const a = `n ≤ ${s.nMax} · ${s.terms} odd terms · RMS ${s.rms.toFixed(3)} V · THD ${(s.thd * 100).toFixed(1)} %`
      const b = `V1 ${s.v1.toFixed(3)} V`
      const d = `V3 ${s.v3.toFixed(3)} · V5 ${s.v5.toFixed(3)} V`
      if (textRefs.current[0]) textRefs.current[0]!.textContent = a
      if (textRefs.current[1]) textRefs.current[1]!.textContent = b
      if (textRefs.current[2]) textRefs.current[2]!.textContent = d
    }
  })

  return (
    <>
      <Line points={WAVE_AXIS} lineWidth={1} color={c.guide} />
      <Line points={SPEC_AXIS} lineWidth={1} color={c.guide} />

      {/* Dashed ideal square behind the running sum: the Gibbs spike pokes above it */}
      <Line ref={sqRef} points={WAVE_INIT} lineWidth={1.5} color={c.guide} dashed dashSize={0.1} gapSize={0.07} />
      <TrailLine ref={sumRef} points={WAVE_INIT} lineWidth={3} color={c.voltage} />

      {LANE_IDX.map((k) => (
        <Line
          key={k}
          ref={(l) => { barRefs.current[k] = l }}
          points={SEG_INIT}
          lineWidth={5}
          color={k === 0 ? c.voltage : c.resistance}
        />
      ))}

      <Html position={[W_X0, -1.88, 0]}>
        <span ref={(el) => { textRefs.current[0] = el }} className={labelClass} />
      </Html>
      <Html position={[barX(0), SPEC_Y + 2.45, 0]} center>
        <span ref={(el) => { textRefs.current[1] = el }} className={labelClass} />
      </Html>
      <Html position={[SPEC_X0 + SPEC_SPAN * 0.62, SPEC_Y + 1.1, 0]} center>
        <span ref={(el) => { textRefs.current[2] = el }} className={labelClass} />
      </Html>

      {TICK_N.map((n) => (
        <Html key={n} position={[barX((n - 1) / 2), SPEC_Y - 0.2, 0]} center>
          <span className={axisClass}>{n}</span>
        </Html>
      ))}
      <Html position={[SPEC_X0 + SPEC_SPAN + 0.25, SPEC_Y, 0]} center>
        <span className={axisClass}>n</span>
      </Html>
      <Html position={[SPEC_X0 - 0.35, SPEC_Y + 2.45, 0]} center>
        <span className={axisClass}>Vn</span>
      </Html>
      <Html position={[W_X0 + W_SPAN + 0.2, 0, 0]} center>
        <span className={axisClass}>t</span>
      </Html>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'harmonics-fourier',
  title: { en: 'Harmonics & Fourier Series', ko: '비정현파·고조파' },
  subject: 'circuits',
  order: 19,
  cardPosition: [3.3, 1.2, -0.6],
  description: {
    en: `## Harmonics and Fourier Series

**Any repeating wave is a sum of sine waves at whole-number multiples of its fundamental.**

- Front wave: the running sum of every harmonic included so far
- Dashed outline: the ideal square wave it is converging to
- Waves receding into depth: the harmonics themselves, one lane per odd n
- Bars: the harmonic spectrum, one bar per harmonic, height = its amplitude

**Fourier series of a square wave**: f(t) = (4A ÷ π) × Σ sin(n·ω·t) ÷ n, odd n only
**Harmonic amplitude**: Vn = 4A ÷ (nπ)
**Component RMS**: Vn,rms = Vn ÷ sqrt(2)
**Total RMS**: Vrms = sqrt(V1,rms² + V3,rms² + V5,rms² + …)
**Distortion**: THD = sqrt(Σ Vn,rms² for n ≥ 3) ÷ V1,rms

For a 1 V square wave the fundamental alone is 4 ÷ π = 1.273 V peak, 0.900 V RMS, already taller than the square it approximates. Summing odd harmonics up to n = 199 gives a total RMS of 0.999 V, heading for exactly 1 V, and THD = 48.1 %, heading for about 48 %. By this measure an ideal square wave is roughly half distortion.

Add terms and the corners square off, but the spike at each step never shrinks, it only gets narrower. That is the Gibbs overshoot, about 9 % above the step, and it survives no matter how many terms you add.

Move the harmonic slider and watch both views at once: every new bar in the spectrum is one more ripple taken out of the time trace.`,
    ko: `## 비정현파와 푸리에 급수

**주기적인 파형은 모두 기본파의 정수배 주파수를 갖는 사인파의 합입니다.**

- 앞쪽 파형: 지금까지 포함한 고조파를 모두 더한 합성파
- 점선 윤곽: 합성파가 수렴해 가는 이상적인 구형파
- 깊이 방향으로 물러나는 파형들: 홀수 n마다 한 줄씩 놓인 각 고조파
- 막대: 고조파 스펙트럼. 고조파 하나당 막대 하나, 높이는 그 진폭입니다

**구형파의 푸리에 급수**: f(t) = (4A ÷ π) × Σ sin(n·ω·t) ÷ n, 홀수 n만
**고조파 진폭**: Vn = 4A ÷ (nπ)
**성분 실효값**: Vn,rms = Vn ÷ sqrt(2)
**전체 실효값**: Vrms = sqrt(V1,rms² + V3,rms² + V5,rms² + …)
**왜형률**: THD = sqrt(Σ Vn,rms², n ≥ 3) ÷ V1,rms

진폭 1 V 구형파에서 기본파만 보면 최댓값이 4 ÷ π = 1.273 V, 실효값은 0.900 V로 근사 대상인 구형파보다 오히려 큽니다. 홀수 고조파를 n = 199까지 더하면 전체 실효값은 0.999 V가 되어 정확히 1 V로 수렴하고, THD는 48.1 %로 약 48 %에 수렴합니다. 이 기준으로 보면 이상적인 구형파는 거의 절반이 왜곡인 셈입니다.

항을 늘리면 모서리는 점점 각이 서지만, 계단마다 생기는 뾰족한 돌출은 작아지지 않고 폭만 좁아집니다. 이것이 깁스 현상(Gibbs overshoot)으로, 계단 높이보다 약 9 % 높으며 항을 아무리 늘려도 사라지지 않습니다.

고조파 슬라이더를 움직이며 두 화면을 함께 보세요. 스펙트럼에 막대가 하나 늘 때마다 시간 파형에서 잔물결이 하나씩 빠져나갑니다.`,
  },
  sliderParams: [
    { key: 'nMaxOdd', label: { en: 'Odd harmonics up to (n)', ko: '포함 홀수 고조파 (n)' }, unit: '', min: 1, max: MAX_N, step: 2, defaultValue: 9 },
    { key: 'amp', label: { en: 'Amplitude (A)', ko: '진폭 (A)' }, unit: 'V', min: 0.3, max: 1.2, step: 0.1, defaultValue: 1 },
    { key: 'freq', label: { en: 'Fundamental frequency (f)', ko: '기본파 주파수 (f)' }, unit: 'Hz', min: 0.5, max: 1.5, step: 0.25, defaultValue: 1 },
  ],
  Scene: HarmonicStackScene,
  Scene2D: HarmonicSpectrum2DScene,
}

export default concept
