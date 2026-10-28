import { useRef, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import { Color, DoubleSide, type BufferGeometry, type Group } from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import { useLocaleStore } from '../../lib/i18n'
import type { ConceptMeta } from '../../types/concept'

const SPC = 160 // samples per input cycle
const CYCLES = 3 // input cycles across the window
const N = SPC * CYCLES + 1 // plotted samples; index N-1 repeats index 0 of the next cycle
const WARM = 12 * SPC // whole cycles simulated before recording, so the plot is steady state
const TWO_PI = Math.PI * 2

const VM_MAX = 140 // largest peak the Vm slider reaches [V]
const HALF = 0.72 // scene units for one row's full swing
const SCALE = HALF / VM_MAX // scene units per volt, the same on every row so heights compare honestly
const SWEEP = 5 // seconds for the moving dot to cross the window

const IDEAL = { hdc: 0, hrms: 0, fdc: 0, frms: 0, rippleH: 0, rippleF: 0 }

// Ideal-diode rectifier maths: diodes switch instantly and drop no voltage.
//   half-wave  Vdc = Vm / pi,     Vrms = Vm / 2
//   full-wave  Vdc = 2*Vm / pi,   Vrms = Vm / sqrt(2)
//   with a smoothing capacitor C into a load R, approximately
//   Vripple = Vm / (f*R*C) half-wave and Vm / (2*f*R*C) full-wave
// Worked example, Vm = 100 V, verified with node -e:
//   half-wave Vdc = 31.8310 V, Vrms = 50.0000 V
//   full-wave Vdc = 63.6620 V, Vrms = 70.7107 V
// The stepped simulate() below reproduces 31.827 V and 63.654 V at C = 0 (0.01 % of the
// closed form), and at Vm = 100 V, R = 1 kohm, C = 100 uF, f = 60 Hz it gives 14.09 Vpp
// half-wave and 7.01 Vpp full-wave against the approximation's 16.67 V and 8.33 V.
function solve({ Vm = 100, Cuf = 47, R = 1000, f = 60 }: Record<string, number>) {
  const C = Cuf * 1e-6
  IDEAL.hdc = Vm / Math.PI
  IDEAL.hrms = Vm / 2
  IDEAL.fdc = (2 * Vm) / Math.PI
  IDEAL.frms = Vm / Math.SQRT2
  IDEAL.rippleH = Vm / (f * R * C)
  IDEAL.rippleF = Vm / (2 * f * R * C)
  return IDEAL
}

type Traces = {
  src: Float32Array
  hw: Float32Array
  fw: Float32Array
  onHw: Uint8Array
  onFw: Uint8Array
  hdc: number
  hlo: number
  hhi: number
  fdc: number
  flo: number
  fhi: number
}

const makeTraces = (): Traces => ({
  src: new Float32Array(N),
  hw: new Float32Array(N),
  fw: new Float32Array(N),
  onHw: new Uint8Array(N),
  onFw: new Uint8Array(N),
  hdc: 0,
  hlo: 0,
  hhi: 0,
  fdc: 0,
  flo: 0,
  fhi: 0,
})

/**
 * Ideal diode + smoothing capacitor + load R, stepped at SPC samples per input cycle: the diode
 * conducts while the rectified source is above the capacitor voltage and clamps it there, otherwise
 * the capacitor discharges into R by exp(-dt / RC). C = 0 leaves the raw rectified wave.
 * Writes into the caller's buffers, so a call allocates nothing.
 */
function simulate(tr: Traces, Vm: number, Cuf: number, R: number, f: number) {
  const dt = 1 / (f * SPC)
  const decay = Math.exp(-dt / (R * Cuf * 1e-6)) // C = 0 gives exp(-Infinity) = 0: no storage at all
  const eps = 1e-6 * Vm + 1e-12
  let vh = 0
  let vf = 0
  let hSum = 0
  let fSum = 0
  let hlo = Infinity
  let hhi = -Infinity
  let flo = Infinity
  let fhi = -Infinity

  for (let k = 0; k < WARM + N; k++) {
    const s = Vm * Math.sin((TWO_PI * k) / SPC)
    const rh = s > 0 ? s : 0 // one diode passes the positive half only
    const rf = s < 0 ? -s : s // a bridge folds the negative half up
    const dh = vh * decay
    const df = vf * decay
    const onH = rh > dh + eps
    const onF = rf > df + eps
    vh = onH ? rh : dh
    vf = onF ? rf : df
    if (k < WARM) continue
    const i = k - WARM
    tr.src[i] = s
    tr.hw[i] = vh
    tr.fw[i] = vf
    tr.onHw[i] = onH ? 1 : 0
    tr.onFw[i] = onF ? 1 : 0
    if (i < N - 1) {
      hSum += vh // average over exactly CYCLES whole cycles
      fSum += vf
    }
    if (vh < hlo) hlo = vh
    if (vh > hhi) hhi = vh
    if (vf < flo) flo = vf
    if (vf > fhi) fhi = vf
  }

  tr.hdc = hSum / (N - 1)
  tr.fdc = fSum / (N - 1)
  tr.hlo = hlo
  tr.hhi = hhi
  tr.flo = flo
  tr.fhi = fhi
}

function useSliderValues() {
  const valuesRef = useRef<Record<string, number>>(useSliderStore.getState().values)
  useEffect(
    () => useSliderStore.subscribe((s) => s.values, (v) => { valuesRef.current = v }),
    []
  )
  return valuesRef
}

// Labels read in the foreground color; this concept shows only voltages, so nothing is tinted
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const noteClass = 'pointer-events-none whitespace-nowrap text-[11px] tabular-nums text-muted-foreground'

const ROW_NAMES = { en: ['half-wave', 'full-wave'], ko: ['반파', '전파'] }

type Pts = [number, number, number][]
type Row = { y: number; z: number }
type Layout = {
  xL: number
  xSpan: number
  rows: [Row, Row, Row] // source, half-wave, full-wave
  axes: [Pts, Pts, Pts]
  labelX: number
  readoutX: number
  noteY: number
}

// Small z offsets keep fill, axis, guides, average line and trace from fighting for the same depth
const Z_FILL = 0
const Z_AXIS = 0.004
const Z_GUIDE = 0.008
const Z_AVG = 0.012
const Z_TRACE = 0.02

function makeLayout(
  xL: number,
  xSpan: number,
  ys: [number, number, number],
  zs: [number, number, number]
): Layout {
  const rows = ys.map((y, i) => ({ y, z: zs[i] })) as [Row, Row, Row]
  const axes = rows.map((r): Pts => [
    [xL - 0.1, r.y, r.z + Z_AXIS],
    [xL + xSpan + 0.1, r.y, r.z + Z_AXIS],
  ]) as [Pts, Pts, Pts]
  return {
    xL,
    xSpan,
    rows,
    axes,
    labelX: xL - 0.62,
    readoutX: xL + xSpan + 0.68,
    noteY: ys[2] - 0.28,
  }
}

// Output rows grow upward from their own zero line, so only the source row needs room below it.
const L3D = makeLayout(-2.6, 5.2, [1.03, -0.61, -1.53], [-1.3, 0, 1.3])
const L2D = makeLayout(-2.85, 5.0, [1.2, -0.46, -1.4], [0, 0, 0])

const WAVE_INIT: Pts = Array.from({ length: N }, (): [number, number, number] => [0, 0, 0])
const SEG_INIT: Pts = [[0, 0, 0], [0, 0, 0]]

// Triangle strip under an output trace: vertex 2k on the curve, vertex 2k+1 on the row's zero line
const FILL_INDEX = (() => {
  const idx = new Uint16Array((N - 1) * 6)
  for (let k = 0; k < N - 1; k++) {
    const a = k * 2
    idx[k * 6] = a
    idx[k * 6 + 1] = a + 1
    idx[k * 6 + 2] = a + 2
    idx[k * 6 + 3] = a + 1
    idx[k * 6 + 4] = a + 3
    idx[k * 6 + 5] = a + 2
  }
  return idx
})()

function setLevel(buf: Float32Array, line: any, L: Layout, y: number, z: number) {
  buf[0] = L.xL - 0.1
  buf[1] = y
  buf[2] = z
  buf[3] = L.xL + L.xSpan + 0.1
  buf[4] = y
  buf[5] = z
  line?.geometry.setPositions(buf)
  line?.computeLineDistances()
}

/**
 * Source, half-wave and full-wave on one time axis. Every trace is a function of the sliders alone,
 * so the geometry is rebuilt only when the slider values (or the theme colors) change; each frame
 * just slides the markers along the buffers that are already there.
 */
function RectifierRows({ L }: { L: Layout }) {
  const c = usePhysicsColors()
  const names = ROW_NAMES[useLocaleStore((st) => st.locale)]
  const valuesRef = useSliderValues()
  const tr = useMemo(makeTraces, [])
  const srcBuf = useMemo(() => new Float32Array(N * 3), [])
  const hwBuf = useMemo(() => new Float32Array(N * 3), [])
  const fwBuf = useMemo(() => new Float32Array(N * 3), [])
  const hwFillPos = useMemo(() => new Float32Array(N * 6), [])
  const hwFillCol = useMemo(() => new Float32Array(N * 6), [])
  const fwFillPos = useMemo(() => new Float32Array(N * 6), [])
  const fwFillCol = useMemo(() => new Float32Array(N * 6), [])
  const seg = useMemo(() => new Float32Array(6), [])
  // Conducting intervals in a light neutral; the off intervals fade into the backdrop instead
  const onCol = useMemo(() => new Color(c.inductance), [c.inductance])
  const offCol = useMemo(
    () => new Color(c.inductance).lerp(new Color(c.backdrop), 0.82),
    [c.inductance, c.backdrop]
  )

  const traceRefs = useRef<any[]>([]) // 0 source, 1 half-wave, 2 full-wave
  const fillRefs = useRef<(BufferGeometry | null)[]>([]) // 0 half-wave, 1 full-wave
  const avgRefs = useRef<any[]>([]) // 0 half-wave, 1 full-wave
  const bandRefs = useRef<any[]>([]) // 0/1 half-wave low/high, 2/3 full-wave low/high
  const markerRefs = useRef<(Group | null)[]>([])
  const tagRefs = useRef<(Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const builtFor = useRef<unknown>(null)
  const builtCol = useRef<Color | null>(null)
  const builtIn = useRef<string[] | null>(null)
  // The <Html> spans may not be attached on the frame the geometry is first built, and the
  // readouts are written only inside that block, so keep retrying until they land once.
  const wroteText = useRef(false)

  useFrame(({ clock }) => {
    const vals = valuesRef.current

    if (
      builtFor.current !== vals ||
      builtCol.current !== onCol ||
      builtIn.current !== names ||
      !wroteText.current
    ) {
      builtFor.current = vals
      builtCol.current = onCol
      builtIn.current = names
      const Vm = vals.Vm ?? 100
      const ideal = solve(vals)
      simulate(tr, Vm, vals.Cuf ?? 47, vals.R ?? 1000, vals.f ?? 60)

      const ySrc = L.rows[0].y
      const yHw = L.rows[1].y
      const yFw = L.rows[2].y
      const zHwFill = L.rows[1].z + Z_FILL
      const zFwFill = L.rows[2].z + Z_FILL

      for (let i = 0; i < N; i++) {
        const x = L.xL + (i / (N - 1)) * L.xSpan
        const o = i * 3
        srcBuf[o] = x
        srcBuf[o + 1] = ySrc + tr.src[i] * SCALE
        srcBuf[o + 2] = L.rows[0].z + Z_TRACE
        hwBuf[o] = x
        hwBuf[o + 1] = yHw + tr.hw[i] * SCALE
        hwBuf[o + 2] = L.rows[1].z + Z_TRACE
        fwBuf[o] = x
        fwBuf[o + 1] = yFw + tr.fw[i] * SCALE
        fwBuf[o + 2] = L.rows[2].z + Z_TRACE

        const q = i * 6
        hwFillPos[q] = x
        hwFillPos[q + 1] = hwBuf[o + 1]
        hwFillPos[q + 2] = zHwFill
        hwFillPos[q + 3] = x
        hwFillPos[q + 4] = yHw
        hwFillPos[q + 5] = zHwFill
        fwFillPos[q] = x
        fwFillPos[q + 1] = fwBuf[o + 1]
        fwFillPos[q + 2] = zFwFill
        fwFillPos[q + 3] = x
        fwFillPos[q + 4] = yFw
        fwFillPos[q + 5] = zFwFill

        const ch = tr.onHw[i] ? onCol : offCol
        const cf = tr.onFw[i] ? onCol : offCol
        hwFillCol[q] = ch.r
        hwFillCol[q + 1] = ch.g
        hwFillCol[q + 2] = ch.b
        hwFillCol[q + 3] = ch.r
        hwFillCol[q + 4] = ch.g
        hwFillCol[q + 5] = ch.b
        fwFillCol[q] = cf.r
        fwFillCol[q + 1] = cf.g
        fwFillCol[q + 2] = cf.b
        fwFillCol[q + 3] = cf.r
        fwFillCol[q + 4] = cf.g
        fwFillCol[q + 5] = cf.b
      }

      traceRefs.current[0]?.geometry.setPositions(srcBuf)
      traceRefs.current[1]?.geometry.setPositions(hwBuf)
      traceRefs.current[2]?.geometry.setPositions(fwBuf)
      for (let g = 0; g < 2; g++) {
        const geo = fillRefs.current[g]
        if (geo) {
          geo.attributes.position.needsUpdate = true
          geo.attributes.color.needsUpdate = true
        }
      }

      // Dashed line across each output row: its average over a whole cycle, the DC the load sees
      const hAvgY = yHw + tr.hdc * SCALE
      const fAvgY = yFw + tr.fdc * SCALE
      setLevel(seg, avgRefs.current[0], L, hAvgY, L.rows[1].z + Z_AVG)
      setLevel(seg, avgRefs.current[1], L, fAvgY, L.rows[2].z + Z_AVG)
      // Faint pair above and below: the ripple band, peak to peak
      setLevel(seg, bandRefs.current[0], L, yHw + tr.hlo * SCALE, L.rows[1].z + Z_GUIDE)
      setLevel(seg, bandRefs.current[1], L, yHw + tr.hhi * SCALE, L.rows[1].z + Z_GUIDE)
      setLevel(seg, bandRefs.current[2], L, yFw + tr.flo * SCALE, L.rows[2].z + Z_GUIDE)
      setLevel(seg, bandRefs.current[3], L, yFw + tr.fhi * SCALE, L.rows[2].z + Z_GUIDE)
      tagRefs.current[0]?.position.set(L.readoutX, hAvgY, L.rows[1].z + Z_AVG)
      tagRefs.current[1]?.position.set(L.readoutX, fAvgY, L.rows[2].z + Z_AVG)

      // Readouts change only with the sliders, so the text is rebuilt only here
      if (textRefs.current[5]) {
        wroteText.current = true
        const texts = [
          `vs · Vm ${Vm.toFixed(0)} V`,
          `vo · ${names[0]}`,
          `vo · ${names[1]}`,
          `Vdc ${tr.hdc.toFixed(1)} V · Vpp ${(tr.hhi - tr.hlo).toFixed(1)} V`,
          `Vdc ${tr.fdc.toFixed(1)} V · Vpp ${(tr.fhi - tr.flo).toFixed(1)} V`,
          `C = 0 · Vm ÷ π = ${ideal.hdc.toFixed(1)} V · 2Vm ÷ π = ${ideal.fdc.toFixed(1)} V`,
        ]
        texts.forEach((text, t) => {
          if (textRefs.current[t]) textRefs.current[t]!.textContent = text
        })
      }
    }

    // One instant read off all three rows at once: when vs is negative the half-wave dot sits
    // on its zero line while the full-wave dot is up at the peak.
    const frac = (clock.getElapsedTime() % SWEEP) / SWEEP
    const i = Math.min(N - 1, Math.floor(frac * (N - 1)))
    const o = i * 3
    markerRefs.current[0]?.position.set(srcBuf[o], srcBuf[o + 1], srcBuf[o + 2])
    markerRefs.current[1]?.position.set(hwBuf[o], hwBuf[o + 1], hwBuf[o + 2])
    markerRefs.current[2]?.position.set(fwBuf[o], fwBuf[o + 1], fwBuf[o + 2])
  })

  return (
    <>
      {/* One zero line per row */}
      <Line points={L.axes[0]} lineWidth={1} color={c.guide} />
      <Line points={L.axes[1]} lineWidth={1.5} color={c.guide} />
      <Line points={L.axes[2]} lineWidth={1.5} color={c.guide} />

      {/* Area under each output trace: light neutral while a diode conducts, faded where none does */}
      {[hwFillPos, fwFillPos].map((pos, g) => (
        <mesh key={g} frustumCulled={false}>
          <bufferGeometry ref={(geo) => { fillRefs.current[g] = geo }}>
            <bufferAttribute attach="attributes-position" args={[pos, 3]} />
            <bufferAttribute attach="attributes-color" args={[g === 0 ? hwFillCol : fwFillCol, 3]} />
            <bufferAttribute attach="index" args={[FILL_INDEX, 1]} />
          </bufferGeometry>
          <meshBasicMaterial vertexColors transparent opacity={0.45} depthWrite={false} side={DoubleSide} />
        </mesh>
      ))}

      {/* Ripple band: the lowest and highest the output ever reaches */}
      {[0, 1, 2, 3].map((k) => (
        <Line
          key={k}
          ref={(l) => { bandRefs.current[k] = l }}
          points={SEG_INIT}
          lineWidth={1}
          color={c.guide}
          dashed
          dashSize={0.06}
          gapSize={0.06}
        />
      ))}

      {/* The DC average of each output */}
      {[0, 1].map((k) => (
        <Line
          key={k}
          ref={(l) => { avgRefs.current[k] = l }}
          points={SEG_INIT}
          lineWidth={2}
          color={c.voltage}
          dashed
          dashSize={0.12}
          gapSize={0.08}
        />
      ))}

      {[0, 1, 2].map((k) => (
        <TrailLine
          key={k}
          ref={(l) => { traceRefs.current[k] = l }}
          points={WAVE_INIT}
          lineWidth={k === 0 ? 2.5 : 3}
          color={c.voltage}
          fade={false}
        />
      ))}

      {[0, 1, 2].map((k) => (
        <Marker
          key={k}
          ref={(g) => { markerRefs.current[k] = g }}
          color={c.marker}
          size={0.055}
        />
      ))}

      {L.rows.map((r, t) => (
        <Html key={t} position={[L.labelX, r.y, r.z + Z_TRACE]} center>
          <span ref={(el) => { textRefs.current[t] = el }} className={labelClass} />
        </Html>
      ))}
      {[0, 1].map((k) => (
        <group key={k} ref={(g) => { tagRefs.current[k] = g }}>
          <Html center>
            <span ref={(el) => { textRefs.current[3 + k] = el }} className={labelClass} />
          </Html>
        </group>
      ))}
      <Html position={[L.xL, L.noteY, L.rows[2].z + Z_TRACE]}>
        <span ref={(el) => { textRefs.current[5] = el }} className={noteClass} />
      </Html>
    </>
  )
}

// 3D: the same three rows, each on its own plane, receding into depth
function RectifierScene() {
  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />
      {/* Background, lights, grid and bloom come from the shared SceneStage */}
      <RectifierRows L={L3D} />
    </>
  )
}

// 2D: the three rows flat and stacked, for reading the averages off against each other
function RectifierScene2D() {
  return <RectifierRows L={L2D} />
}

const concept: ConceptMeta = {
  id: 'diode-rectifier',
  title: { en: 'Diode Rectifiers', ko: '다이오드 정류회로' },
  subject: 'electronics',
  order: 1,
  description: {
    en: `## Diode Rectifiers

**A diode conducts one way only, so it cuts an AC wave down to a one-sided one, and a capacitor then fills in the gaps that are left.**

Three time rows, each on its own zero line:
- Top (strongest neutral): the source voltage vs, swinging both ways
- Middle: the half-wave output, one diode
- Bottom: the full-wave output, a four-diode bridge

On each output row the light neutral shading marks the intervals where a diode is actually conducting. Where the shading fades out, every diode is off and the smoothing capacitor alone is feeding the load, draining through R. The dashed line across each output row is that row's average, the DC the load really sees; the faint pair above and below is the ripple band, the peak-to-peak swing. The dot sweeping all three rows reads one instant off every row at once.

Half-wave throws away the negative half cycle, so with no capacitor the load gets nothing for half of every period. A full-wave bridge folds that negative half up instead: twice as many pulses, twice the average, half the ripple.

**Half-wave, ideal**: Vdc = Vm ÷ π = 0.318·Vm, Vrms = Vm ÷ 2
**Full-wave, ideal**: Vdc = 2·Vm ÷ π = 0.637·Vm, Vrms = Vm ÷ sqrt(2)
**Ripple with a smoothing capacitor, approximate**: Vripple ≈ Vm ÷ (f·R·C) half-wave, Vm ÷ (2·f·R·C) full-wave

At Vm = 100 V with C = 0 the half-wave output averages 31.83 V at 50 V RMS, and the full-wave output averages 63.66 V at 70.71 V RMS, exactly twice the DC for the same peak.

The ripple expression is only an estimate: it assumes the capacitor discharges at a constant current for the whole period, which overstates the droop. At Vm = 100 V, R = 1 kΩ, C = 100 µF, f = 60 Hz it predicts 16.7 V peak-to-peak where the stepped simulation here gives 14.1 V. The two close in on each other as C grows.

Everything drawn here treats the diodes as ideal: they switch instantly and drop no voltage. A real silicon diode drops about 0.7 V while conducting, so a real half-wave peak sits near Vm − 0.7 V, and a bridge, which always has two diodes in series, near Vm − 1.4 V. That offset is deliberately left out so the formulas stay clean.

Raise C from zero and watch the gaps fill: the shaded conducting windows shrink to short spikes near each peak, the average climbs toward Vm, and the ripple band closes. A larger R drains the capacitor more slowly, so it flattens the output too; a higher f leaves less time to droop between peaks. Vm scales every row without changing a single shape.`,
    ko: `## 다이오드 정류회로

**다이오드는 한쪽으로만 전류를 흘리므로 교류를 한쪽으로 치우친 파형으로 깎아냅니다. 그렇게 생긴 빈 구간은 커패시터가 메웁니다.**

시간축 세 줄이 각자의 영점선 위에 놓여 있습니다.
- 위 (가장 진한 중성색): 양쪽으로 모두 흔들리는 전원 전압 vs
- 가운데: 다이오드 한 개로 만든 반파 정류 출력
- 아래: 다이오드 네 개로 만든 브리지 전파 정류 출력

각 출력 줄에서 밝은 중성색으로 칠한 구간이 다이오드가 실제로 도통하는 구간입니다. 칠이 흐려지는 곳에서는 모든 다이오드가 꺼져 있고, 평활 커패시터 혼자서 R로 방전하며 부하를 먹입니다. 출력 줄을 가로지르는 점선은 그 줄의 평균, 곧 부하가 실제로 받는 직류이고, 위아래의 희미한 두 선은 맥동 폭(최대에서 최소까지)을 나타냅니다. 세 줄을 함께 지나가는 점은 같은 순간의 값을 한 번에 읽어 줍니다.

반파 정류는 음의 반주기를 버리므로, 커패시터가 없으면 매 주기의 절반 동안 부하에 아무것도 가지 않습니다. 전파 브리지는 그 음의 반주기를 위로 접어 올립니다. 맥동 횟수가 두 배, 평균도 두 배가 되고, 맥동 폭은 절반이 됩니다.

**반파, 이상적**: Vdc = Vm ÷ π = 0.318·Vm, Vrms = Vm ÷ 2
**전파, 이상적**: Vdc = 2·Vm ÷ π = 0.637·Vm, Vrms = Vm ÷ sqrt(2)
**평활 커패시터가 있을 때의 맥동, 근사식**: Vripple ≈ Vm ÷ (f·R·C) 반파, Vm ÷ (2·f·R·C) 전파

Vm = 100 V에서 C = 0이면 반파 출력의 평균은 31.83 V, 실효값은 50 V이고, 전파 출력의 평균은 63.66 V, 실효값은 70.71 V입니다. 같은 최댓값에서 직류분이 정확히 두 배입니다.

맥동 식은 어림값일 뿐입니다. 커패시터가 한 주기 내내 일정한 전류로 방전한다고 보기 때문에 전압이 떨어지는 양을 실제보다 크게 잡습니다. Vm = 100 V, R = 1 kΩ, C = 100 µF, f = 60 Hz에서 이 식은 16.7 Vpp를 주지만, 여기 쓰인 시간 단계 시뮬레이션은 14.1 Vpp를 줍니다. C가 커지면 둘은 서로 가까워집니다.

여기 그린 것은 모두 다이오드를 이상적으로 본 결과입니다. 즉시 켜지고 꺼지며 전압 강하가 없다고 봅니다. 실제 실리콘 다이오드는 도통하는 동안 약 0.7 V를 떨어뜨리므로, 실제 반파 출력의 최댓값은 Vm − 0.7 V 근처이고, 항상 두 개가 직렬로 들어가는 브리지는 Vm − 1.4 V 근처입니다. 식을 깔끔하게 두려고 이 강하는 일부러 뺐습니다.

C를 0에서 올리면 빈 구간이 메워지는 것이 보입니다. 칠한 도통 구간은 봉우리 근처의 짧은 스파이크로 줄어들고, 평균은 Vm을 향해 올라가며, 맥동 폭은 좁아집니다. R이 크면 커패시터가 더 천천히 방전하므로 출력도 평평해지고, f가 높으면 봉우리 사이에서 전압이 떨어질 시간이 줄어듭니다. Vm은 모양은 그대로 둔 채 모든 줄의 크기만 바꿉니다.`,
  },
  sliderParams: [
    { key: 'Vm',  label: { en: 'Peak input voltage (Vm)',    ko: '입력 전압 최댓값 (Vm)' },  unit: 'V',  min: 70,  max: VM_MAX, step: 5,   defaultValue: 100  },
    { key: 'Cuf', label: { en: 'Smoothing capacitance (C)',  ko: '평활 커패시턴스 (C)' },    unit: 'µF', min: 0,   max: 500,    step: 5,   defaultValue: 47   },
    { key: 'R',   label: { en: 'Load resistance (R)',        ko: '부하 저항 (R)' },          unit: 'Ω',  min: 200, max: 5000,   step: 100, defaultValue: 1000 },
    { key: 'f',   label: { en: 'Supply frequency (f)',       ko: '전원 주파수 (f)' },        unit: 'Hz', min: 30,  max: 240,    step: 10,  defaultValue: 60   },
  ],
  Scene: RectifierScene,
  Scene2D: RectifierScene2D,
}

export default concept
