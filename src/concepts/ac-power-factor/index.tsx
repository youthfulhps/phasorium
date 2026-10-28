import { useRef, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import { Color, DoubleSide, type BufferGeometry, type Group } from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { useLocaleStore } from '../../lib/i18n'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import { setLineOpacity, setOpacity, useStoryReveal } from '../../components/scene/storyReveal'
import type { ConceptMeta } from '../../types/concept'

const DISPLAY_OMEGA = 1.1 // on-screen rad/s: a real 60 Hz cycle is far too fast to follow

// Pure physics in RMS terms. v = Vm·sin(ωt), i = Im·sin(ωt − θ), p = v·i.
// Verified with node -e: Vrms = 100 V, Irms = 5 A, θ = 30° →
//   S = 500 VA, P = 433.0127 W, Q = 250.0000 var, pf = 0.8660, sqrt(P² + Q²) = 500.000000 VA.
// Same run checked the identity Vm·Im·sin(ωt)·sin(ωt − θ) = S·[cos θ − cos(2ωt − θ)]
//   (max error 6.8e-13 over a cycle) and its cycle mean, 433.0127 W, which is exactly P.
function solve({ Vrms = 100, Irms = 5, theta = 45 }: Record<string, number>) {
  const th = (theta * Math.PI) / 180
  const S = Vrms * Irms
  return {
    Vrms,
    Irms,
    deg: theta,
    th,
    S,
    P: S * Math.cos(th),
    Q: S * Math.sin(th),
    pf: Math.cos(th),
    Vm: Vrms * Math.SQRT2,
    Im: Irms * Math.SQRT2,
  }
}

// ── Narration steps drive what exists on screen. Each reveal eases 0 → 1 so nothing pops in. ──
// Step 0 is the supply voltage alone: the current, the product and the split all arrive later.
const REV_I = 0 // the current row, which is where the phase angle lives
const REV_P = 1 // the p = v·i row with its shaded area
const REV_AVG = 2 // the cycle average of p, i.e. the real power P, and the pf readout
const REV_SPQ = 3 // the S / P / Q split, and in 2D the triangle that closes it
const REV_COUNT = 4

function revealTargets(step: number, out: Float32Array) {
  out[REV_I] = step >= 1 ? 1 : 0
  out[REV_P] = step >= 2 ? 1 : 0
  out[REV_AVG] = step >= 3 ? 1 : 0
  out[REV_SPQ] = step >= 6 ? 1 : 0
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

// Shared scratch buffers: filled and handed straight to a geometry, so no frame allocates.
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

// ── 3D: three stacked time rows receding into depth — v, then i, then p with its area shaded ──
const N = 240
const X_HALF = 2.2
const CYCLES = 2 // supply cycles across the window, so p shows four of its own
const TWO_PI = Math.PI * 2

const ROW_V = 1.35
const ROW_I = 0.1
const ROW_P = -1.15
// Kept shallow: a deeper stack magnifies the nearest row until it dives under the slider HUD
const Z_V = -0.95
const Z_I = 0
const Z_P = 0.95

const V_SCALE = 0.6 / (140 * Math.SQRT2) // scene units per volt; Vrms ≤ 140 keeps the row inside ±0.6
const I_SCALE = 0.6 / (7 * Math.SQRT2) // scene units per amp, same half-height for an honest comparison
const P_HALF = 0.45 // p is drawn to its own scale: ±P_HALF is ±S, so the shape depends only on θ
const FLOW_AT_MAX = 3.5 // sheen bands per second at the largest current

const WAVE_INIT: [number, number, number][] = Array.from(
  { length: N },
  (): [number, number, number] => [0, 0, 0]
)
const rowAxis = (y: number, z: number): [number, number, number][] => [
  [-X_HALF - 0.12, y, z],
  [X_HALF + 0.12, y, z],
]
const AXIS_V = rowAxis(ROW_V, Z_V)
const AXIS_I = rowAxis(ROW_I, Z_I)
const AXIS_P = rowAxis(ROW_P, Z_P)

// Triangle strip under the p curve: vertex 2k on the curve, vertex 2k+1 on the row's zero line
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

function PowerRowsScene() {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  const vBuf = useMemo(() => new Float32Array(N * 3), [])
  const iBuf = useMemo(() => new Float32Array(N * 3), [])
  const pBuf = useMemo(() => new Float32Array(N * 3), [])
  const fillPos = useMemo(() => new Float32Array(N * 2 * 3), [])
  const fillCol = useMemo(() => new Float32Array(N * 2 * 3), [])
  // Precomputed so the per-frame loop only copies numbers
  const delivered = useMemo(() => new Color(c.power), [c.power])
  const returned = useMemo(() => new Color(c.inductance), [c.inductance])
  const vWaveRef = useRef<any>(null)
  const iWaveRef = useRef<any>(null)
  const pWaveRef = useRef<any>(null)
  const avgRef = useRef<any>(null)
  const fillRef = useRef<BufferGeometry>(null)
  const vTipRef = useRef<Group>(null)
  const iTipRef = useRef<Group>(null)
  const pTipRef = useRef<Group>(null)
  const avgTagRef = useRef<Group>(null)
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const axisRefs = useRef<any[]>([]) // v, i, p zero lines
  const fillMatRef = useRef<any>(null)
  const flow = useRef(0)
  const labelledFor = useRef<Record<string, number> | null>(null)
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(({ clock }) => {
    const s = solve(valuesRef.current)
    const wt = clock.getElapsedTime() * DISPLAY_OMEGA
    const vAmp = s.Vm * V_SCALE
    const iAmp = s.Im * I_SCALE
    const cosTh = Math.cos(s.th)

    for (let k = 0; k < N; k++) {
      const frac = k / (N - 1)
      const x = -X_HALF + frac * 2 * X_HALF
      const ph = wt - (1 - frac) * CYCLES * TWO_PI // now at the right edge, history to the left
      // p ÷ S = cos θ − cos(2ωt − θ): pulses at 2ω around the average cos θ
      const pNorm = cosTh - Math.cos(2 * ph - s.th)
      const py = ROW_P + P_HALF * pNorm

      vBuf[k * 3] = x
      vBuf[k * 3 + 1] = ROW_V + vAmp * Math.sin(ph)
      vBuf[k * 3 + 2] = Z_V
      iBuf[k * 3] = x
      iBuf[k * 3 + 1] = ROW_I + iAmp * Math.sin(ph - s.th)
      iBuf[k * 3 + 2] = Z_I
      pBuf[k * 3] = x
      pBuf[k * 3 + 1] = py
      pBuf[k * 3 + 2] = Z_P

      // Shaded area: amber where the load takes energy, light neutral where it hands it back
      const col = pNorm >= 0 ? delivered : returned
      const o = k * 6
      fillPos[o] = x; fillPos[o + 1] = py; fillPos[o + 2] = Z_P
      fillPos[o + 3] = x; fillPos[o + 4] = ROW_P; fillPos[o + 5] = Z_P
      fillCol[o] = col.r; fillCol[o + 1] = col.g; fillCol[o + 2] = col.b
      fillCol[o + 3] = col.r; fillCol[o + 4] = col.g; fillCol[o + 5] = col.b
    }
    vWaveRef.current?.geometry.setPositions(vBuf)
    iWaveRef.current?.geometry.setPositions(iBuf)
    pWaveRef.current?.geometry.setPositions(pBuf)
    const geo = fillRef.current
    if (geo) {
      geo.attributes.position.needsUpdate = true
      geo.attributes.color.needsUpdate = true
    }
    flow.current = FLOW_AT_MAX * (iAmp / 0.6)

    // The average of p over a cycle is P, and on this row it sits at P_HALF·cos θ
    const avgY = ROW_P + P_HALF * cosTh
    setSeg(avgRef.current, -X_HALF - 0.12, avgY, X_HALF + 0.12, avgY, Z_P)
    // Labelled at the left end: the right end of the row is where the narration panel starts.
    // Clamped off the row's own caption, which it would land on as pf approaches zero.
    avgTagRef.current?.position.set(-X_HALF - 0.78, Math.max(avgY, ROW_P + 0.22), Z_P)

    vTipRef.current?.position.set(X_HALF, vBuf[(N - 1) * 3 + 1], Z_V)
    iTipRef.current?.position.set(X_HALF, iBuf[(N - 1) * 3 + 1], Z_I)
    pTipRef.current?.position.set(X_HALF, pBuf[(N - 1) * 3 + 1], Z_P)

    // One row at a time: the voltage is the premise, the rest is the argument
    const wI = reveal[REV_I]
    const wP = reveal[REV_P]
    setLineOpacity(axisRefs.current[1], wI)
    setLineOpacity(iWaveRef.current, wI)
    iTipRef.current?.scale.setScalar(wI)
    setLineOpacity(axisRefs.current[2], wP)
    setLineOpacity(pWaveRef.current, wP)
    pTipRef.current?.scale.setScalar(wP)
    if (fillMatRef.current) fillMatRef.current.opacity = 0.32 * wP
    setLineOpacity(avgRef.current, reveal[REV_AVG])
    const fades = [1, wI, wP, reveal[REV_AVG], reveal[REV_SPQ]]
    fades.forEach((v, t) => setOpacity(textRefs.current[t], v))

    // Values only change with the sliders, so the text is rebuilt only then
    if (labelledFor.current !== valuesRef.current && textRefs.current[3]) {
      labelledFor.current = valuesRef.current
      const texts = [
        `v ${s.Vrms.toFixed(0)} V rms`,
        `i ${s.Irms.toFixed(1)} A rms · θ ${s.deg.toFixed(0)}°`,
        'p = v · i',
        `P ${s.P.toFixed(0)} W · pf ${s.pf.toFixed(3)}`,
        `S ${s.S.toFixed(0)} VA · P ${s.P.toFixed(0)} W · Q ${s.Q.toFixed(0)} var`,
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  const rows: { y: number; z: number; color: string }[] = [
    { y: ROW_V, z: Z_V, color: c.voltage },
    { y: ROW_I, z: Z_I, color: c.current },
    { y: ROW_P, z: Z_P, color: c.power },
  ]

  return (
    <>
      {/* Backed off and aimed right of centre, so all three rows clear the narration panel */}
      <PerspectiveCamera makeDefault position={[3, 2.2, 6]} fov={50} />
      <OrbitControls enablePan={false} target={[0.85, 0, 0]} />

      {/* One zero line per row; depth says which quantity you are looking at */}
      {[AXIS_V, AXIS_I, AXIS_P].map((pts, i) => (
        <Line
          key={i}
          ref={(l) => { axisRefs.current[i] = l }}
          points={pts}
          lineWidth={i === 2 ? 1.5 : 1}
          color={c.guide}
        />
      ))}

      {/* Area between p and its zero line: amber above, light neutral below */}
      <mesh frustumCulled={false}>
        <bufferGeometry ref={fillRef}>
          <bufferAttribute attach="attributes-position" args={[fillPos, 3]} />
          <bufferAttribute attach="attributes-color" args={[fillCol, 3]} />
          <bufferAttribute attach="index" args={[FILL_INDEX, 1]} />
        </bufferGeometry>
        <meshBasicMaterial ref={fillMatRef} vertexColors transparent opacity={0.32} depthWrite={false} side={DoubleSide} />
      </mesh>

      {/* Average of p over a cycle — the real power P */}
      <Line
        ref={avgRef}
        points={SEG_INIT}
        lineWidth={2}
        color={c.power}
        dashed
        dashSize={0.12}
        gapSize={0.08}
      />

      <TrailLine ref={vWaveRef} points={WAVE_INIT} lineWidth={2.5} color={c.voltage} />
      <TrailLine
        ref={iWaveRef}
        points={WAVE_INIT}
        lineWidth={2.5}
        color={c.current}
        flowSpeed={() => flow.current}
      />
      <TrailLine ref={pWaveRef} points={WAVE_INIT} lineWidth={3} color={c.power} />

      <Marker ref={vTipRef} color={c.marker} size={0.055} />
      <Marker ref={iTipRef} color={c.current} size={0.055} />
      <Marker ref={pTipRef} color={c.power} size={0.055} />

      {rows.map((r, t) => (
        <Html key={t} position={[-X_HALF - 0.55, r.y, r.z]} center>
          <span
            ref={(el) => { textRefs.current[t] = el }}
            className={labelClass}
            style={t === 0 ? undefined : { color: r.color }}
          />
        </Html>
      ))}
      <group ref={avgTagRef}>
        <Html center>
          <span
            ref={(el) => { textRefs.current[3] = el }}
            className={labelClass}
            style={{ color: c.power }}
          />
        </Html>
      </group>
      {/* What the source actually has to deliver, next to what of it does work */}
      <Html position={[0, ROW_P - 0.8, Z_P]} center>
        <span ref={(el) => { textRefs.current[4] = el }} className={labelClass} />
      </Html>
    </>
  )
}

// ── 2D: the power triangle at scale, with a bar showing how much of S actually works ──
const TRI_X = -2.2
const TRI_Y = 0
const VA_SCALE = 1.8 / (140 * 7) // scene units per VA; the largest S maps to 1.8
const BAR_X = 1.0
const BAR_Y = 0

function PowerTriangle2DScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  // 0 P leg, 1 Q leg, 2 S hypotenuse, 3 θ arc, 4 S bar, 5 P part of the bar
  const lineRefs = useRef<any[]>([])
  const tagRefs = useRef<(Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const cornerRef = useRef<Group>(null)
  const labelledFor = useRef<Record<string, number> | null>(null)
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(() => {
    const s = solve(valuesRef.current)
    const px = TRI_X + s.P * VA_SCALE
    const qy = TRI_Y + s.Q * VA_SCALE

    setSeg(lineRefs.current[0], TRI_X, TRI_Y, px, TRI_Y) // P across
    setSeg(lineRefs.current[1], px, TRI_Y, px, qy) // Q up when inductive, down when capacitive
    setSeg(lineRefs.current[2], TRI_X, TRI_Y, px, qy) // S closes the corner
    setArc(lineRefs.current[3], TRI_X, TRI_Y, 0.42, 0, s.th)
    cornerRef.current?.position.set(px, qy, 0)

    // Bar: the whole length is S, the amber part is P, so the amber fraction is the power factor
    const barEnd = BAR_X + s.S * VA_SCALE
    setSeg(lineRefs.current[4], BAR_X, BAR_Y, barEnd, BAR_Y)
    setSeg(lineRefs.current[5], BAR_X, BAR_Y, BAR_X + s.P * VA_SCALE, BAR_Y, 0.01)

    tagRefs.current[0]?.position.set((TRI_X + px) / 2, TRI_Y - 0.26, 0)
    tagRefs.current[1]?.position.set(px + 0.45, (TRI_Y + qy) / 2, 0)
    tagRefs.current[2]?.position.set(
      TRI_X + (px - TRI_X) * 0.45 - 0.2,
      TRI_Y + (qy - TRI_Y) * 0.45 + (s.Q >= 0 ? 0.28 : -0.28),
      0
    )
    tagRefs.current[3]?.position.set(
      TRI_X + 0.68 * Math.cos(s.th / 2),
      TRI_Y + 0.68 * Math.sin(s.th / 2),
      0
    )
    tagRefs.current[4]?.position.set((BAR_X + barEnd) / 2, BAR_Y + 0.32, 0)
    tagRefs.current[5]?.position.set((BAR_X + barEnd) / 2, BAR_Y - 0.32, 0)

    // P and the angle come with the average line; S, Q and the bar with the split that closes them
    const wP = reveal[REV_AVG]
    const wS = reveal[REV_SPQ]
    const fades = [wP, wS, wS, wP, wS, wS]
    fades.forEach((v, i) => {
      setLineOpacity(lineRefs.current[i], v)
      setOpacity(textRefs.current[i], v)
    })
    cornerRef.current?.scale.setScalar(wS)

    if (labelledFor.current !== valuesRef.current && textRefs.current[5]) {
      labelledFor.current = valuesRef.current
      const texts = [
        `P ${s.P.toFixed(0)} W`,
        `Q ${s.Q.toFixed(0)} var`,
        `S ${s.S.toFixed(0)} VA`,
        `θ ${s.deg.toFixed(0)}°`,
        `pf = cos θ = ${s.pf.toFixed(3)}`,
        locale === 'ko' ? 'S 중 앰버 부분만 일을 합니다' : 'amber part of S does the work',
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  const segs: { color: string; width: number; dashed?: boolean; arc?: boolean; accent?: boolean }[] = [
    { color: c.power, width: 3, accent: true },
    { color: c.inductance, width: 2.5 },
    { color: c.voltage, width: 2.5 },
    { color: c.guide, width: 1.5, dashed: true, arc: true },
    { color: c.inductance, width: 7 },
    { color: c.power, width: 7, accent: true },
  ]

  return (
    <>
      {segs.map((sg, i) => (
        <Line
          key={i}
          ref={(l) => { lineRefs.current[i] = l }}
          points={sg.arc ? ARC_INIT : SEG_INIT}
          lineWidth={sg.width}
          color={sg.color}
          dashed={sg.dashed}
          dashSize={0.07}
          gapSize={0.05}
        />
      ))}

      <Marker ref={cornerRef} color={c.marker} size={0.06} />

      {segs.map((sg, i) => (
        <group key={i} ref={(g) => { tagRefs.current[i] = g }}>
          <Html center>
            <span
              ref={(el) => { textRefs.current[i] = el }}
              className={labelClass}
              style={sg.accent ? { color: c.power } : undefined}
            />
          </Html>
        </group>
      ))}
    </>
  )
}

const concept: ConceptMeta = {
  id: 'ac-power-factor',
  title: { en: 'AC Power & Power Factor', ko: '교류 전력과 역률' },
  subject: 'circuits',
  order: 16,
  description: {
    en: `## AC Power and Power Factor

**Once the current is out of step with the voltage, only part of what the source delivers ever does work.**

Three stacked time rows, each on its own zero line, receding into depth:
- Top (strongest neutral): the source voltage v
- Middle (cyan liquid line): the current i, shifted by the phase angle θ
- Bottom (amber): the instantaneous power p = v · i, with the area under it shaded

Watch the bottom row. p pulses at twice the supply frequency, and as soon as θ ≠ 0 it dips below its zero line. Amber area is energy going into the load; the light neutral area below the line is energy flowing straight back to the source. The dashed amber line is the average of p over a cycle, and that average is the real power P.

**Apparent power**: S = Vrms · Irms  [VA]
**Real power**: P = S · cos θ  [W]
**Reactive power**: Q = S · sin θ  [var]
**Power triangle**: S² = P² + Q²
**Power factor**: pf = cos θ = P ÷ S

**Instantaneous power**: p(t) = Vm·Im·sin(ωt)·sin(ωt − θ) = S·[cos θ − cos(2ωt − θ)], so p swings by ±S about P at angular frequency 2ω.

Q is never consumed. It sloshes into the reactance and back out four times per cycle, yet the line and the generator still have to be sized for S. Put a capacitor in parallel with an inductive load and it supplies that same Q locally, so the line carries only P — that is power factor correction.

Slide θ away from 0 and watch the negative lobes grow, the triangle tip up, and the amber part of the bar shrink. Vrms and Irms set how big everything is without changing the shape of p.

In 2D: the power triangle at scale (P across, Q up, S on the hypotenuse, θ at the corner) and a bar whose full length is S and whose amber part is P. The p row is drawn to its own scale, ±S, and time is slowed far below 60 Hz so the pulsing can be followed.`,
    ko: `## 교류 전력과 역률

**전류가 전압과 어긋나는 순간, 전원이 보낸 것 중 일부만 실제로 일을 합니다.**

시간축 세 줄이 깊이 방향으로 겹쳐 있고, 각 줄에는 자기 영점선이 있습니다.
- 위 (가장 진한 중성색): 전원 전압 v
- 가운데 (시안 액체 선): 위상각 θ만큼 어긋난 전류 i
- 아래 (앰버): 순시 전력 p = v · i, 아래 면적을 칠했습니다

아래 줄을 보세요. p는 전원 주파수의 두 배로 맥동하고, θ ≠ 0이 되는 순간 영점선 아래로 내려갑니다. 앰버 면적은 부하로 들어가는 에너지이고, 선 아래의 밝은 중성색 면적은 전원으로 그대로 되돌아가는 에너지입니다. 점선으로 그은 앰버 선은 한 주기 동안 p의 평균이며, 그 평균이 곧 유효전력 P입니다.

**피상전력**: S = Vrms · Irms  [VA]
**유효전력**: P = S · cos θ  [W]
**무효전력**: Q = S · sin θ  [var]
**전력 삼각형**: S² = P² + Q²
**역률**: pf = cos θ = P ÷ S

**순시 전력**: p(t) = Vm·Im·sin(ωt)·sin(ωt − θ) = S·[cos θ − cos(2ωt − θ)], 즉 p는 각주파수 2ω로 P를 중심으로 ±S만큼 흔들립니다.

Q는 결코 소비되지 않습니다. 한 주기에 네 번 리액턴스로 들어갔다 나올 뿐이지만, 전선과 발전기는 여전히 S에 맞춰 용량을 잡아야 합니다. 유도성 부하에 커패시터를 병렬로 달면 그 Q를 현장에서 공급해 주므로 전선은 P만 나르면 됩니다. 이것이 역률 개선입니다.

θ를 0에서 멀리 옮기면 음의 돌출부가 커지고, 삼각형이 세워지고, 막대의 앰버 부분이 줄어듭니다. Vrms와 Irms는 p의 모양은 그대로 둔 채 전체 크기만 바꿉니다.

2D: 크기에 비례해 그린 전력 삼각형(가로 P, 세로 Q, 빗변 S, 모서리 각 θ)과, 전체 길이가 S이고 앰버 부분이 P인 막대입니다. p 줄은 ±S를 기준으로 한 자체 축척이고, 시간은 실제 60 Hz보다 훨씬 느리게 재생합니다.`,
  },
  story: [
    {
      set: { Vrms: 100, Irms: 5, theta: 0 },
      text: {
        en: 'The supply voltage. One sine wave, 100 V rms, going up and down sixty times a second. Nothing controversial yet.',
        ko: '전원 전압입니다. 사인파 하나, 실효값 100 V, 초당 60번 오르내립니다. 아직 시비 걸 데가 없습니다.',
      },
    },
    {
      set: { Vrms: 100, Irms: 5, theta: 0 },
      text: {
        en: 'Hang a pure resistor on it and the current marches in perfect step. Up together, down together, zero together.',
        ko: '순수한 저항을 달면 전류가 완벽히 발맞춰 갑니다. 같이 오르고, 같이 내리고, 같이 0이 됩니다.',
      },
    },
    {
      set: { Vrms: 100, Irms: 5, theta: 0 },
      text: {
        en: 'Multiply them at every instant and you get the power. Look at the bottom row: it pulses, but it never once dips below the line. Every joule goes into the load and stays there.',
        ko: '매 순간 둘을 곱하면 전력이 나옵니다. 맨 아래 줄을 보세요. 맥동은 하지만 영점선 아래로는 한 번도 내려가지 않습니다. 모든 줄(J)이 부하로 들어가서 거기 머뭅니다.',
      },
    },
    {
      set: { Vrms: 100, Irms: 5, theta: 0 },
      text: {
        en: 'Average that pulsing over one cycle and you get the real power — the number the meter bills you for. 500 W, and the power factor is 1.000. This is as good as it gets.',
        ko: '맥동을 한 주기 평균 내면 유효전력이 나옵니다. 계량기가 요금을 매기는 바로 그 숫자입니다. 500 W, 역률 1.000. 더 좋아질 수 없습니다.',
      },
    },
    {
      set: { Vrms: 100, Irms: 5, theta: 45 },
      text: {
        en: 'Now put a motor on it. Motors are coils, and a coil makes the current arrive late — 45° late here. And watch what happens underneath: the power curve falls through the floor.',
        ko: '이제 모터를 답니다. 모터는 코일이고, 코일은 전류를 늦게 도착하게 만듭니다. 여기선 45° 늦었습니다. 그리고 아래에서 벌어지는 일을 보세요. 전력 곡선이 바닥을 뚫고 내려갑니다.',
      },
    },
    {
      set: { Vrms: 100, Irms: 5, theta: 70 },
      text: {
        en: 'Lag it further, 70°, and the dips get deeper. That pale area below the line is energy the load borrows and hands straight back — four times every cycle. Same voltage, same current, and now only 171 W of actual work.',
        ko: '더 늦춰서 70°. 파인 곳이 더 깊어집니다. 선 아래 밝은 면적은 부하가 빌렸다가 그대로 돌려주는 에너지입니다. 한 주기에 네 번씩. 전압도 그대로, 전류도 그대로인데 실제로 한 일은 171 W뿐입니다.',
      },
    },
    {
      set: { Vrms: 100, Irms: 5, theta: 45 },
      text: {
        en: 'So the one number splits into three. The source still has to deliver 500 VA. Of that, 354 W does work and 354 var just sloshes back and forth. `S² = P² + Q²`',
        ko: '그래서 숫자 하나가 셋으로 갈라집니다. 전원은 여전히 500 VA를 내보내야 합니다. 그중 354 W가 일을 하고, 354 var는 그냥 왔다 갔다 할 뿐입니다. `S² = P² + Q²`',
      },
    },
    {
      set: { Vrms: 100, Irms: 5, theta: -45 },
      text: {
        en: 'Here is the escape hatch. Push θ negative — a capacitor, where the current arrives early instead. P is unchanged at 354 W, but Q flips sign to −354 var. Coils and capacitors cancel.',
        ko: '여기 빠져나갈 구멍이 있습니다. θ를 음수로 밀어 보세요. 커패시터입니다. 전류가 늦는 대신 먼저 도착합니다. P는 354 W 그대로인데 Q가 −354 var로 부호를 뒤집습니다. 코일과 커패시터는 서로 상쇄됩니다.',
      },
    },
    {
      set: { Vrms: 100, Irms: 3.5, theta: 0 },
      text: {
        en: 'So park a capacitor next to the motor and its Q eats the coil\'s Q. The work barely moves — 350 W — but the source now sends 350 VA instead of 500, and the line current falls from 5.0 A to 3.5 A. `pf = cos θ = P ÷ S`',
        ko: '그래서 모터 옆에 커패시터를 세워 두면 그 Q가 코일의 Q를 먹어 치웁니다. 한 일은 거의 그대로 350 W인데, 전원이 내보내는 건 500 VA에서 350 VA로 줄고 전선에 흐르는 전류는 5.0 A에서 3.5 A로 떨어집니다. `pf = cos θ = P ÷ S`',
      },
    },
    {
      set: { Vrms: 100, Irms: 5, theta: 45 },
      text: {
        en: '`S = V × I` `P = S cos θ` `Q = S sin θ` You pay for P, but the wires, the breakers and the transformer are all sized for S. A bad power factor does not cost you more work — it costs you more copper. The sliders are yours.',
        ko: '`S = V × I` `P = S cos θ` `Q = S sin θ` 요금은 P로 내지만 전선도, 차단기도, 변압기도 전부 S에 맞춰 고릅니다. 역률이 나쁘면 일이 늘어나는 게 아니라 구리가 늘어납니다. 이제 슬라이더는 당신 것입니다.',
      },
    },
  ],
  sliderParams: [
    { key: 'Vrms',  label: { en: 'RMS voltage (Vrms)', ko: '실효 전압 (Vrms)' }, unit: 'V', min: 60,  max: 140, step: 5,   defaultValue: 100 },
    { key: 'Irms',  label: { en: 'RMS current (Irms)', ko: '실효 전류 (Irms)' }, unit: 'A', min: 2,   max: 7,   step: 0.5, defaultValue: 5   },
    { key: 'theta', label: { en: 'Phase angle (θ)',    ko: '위상각 (θ)' },       unit: '°', min: -90, max: 90,  step: 5,   defaultValue: 45  },
  ],
  Scene: PowerRowsScene,
  Scene2D: PowerTriangle2DScene,
}

export default concept
