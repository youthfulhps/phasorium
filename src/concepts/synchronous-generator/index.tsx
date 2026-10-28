import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import type { Group } from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { useLocaleStore } from '../../lib/i18n'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import type { ConceptMeta } from '../../types/concept'

const V_PU = 1 // terminal voltage, taken as the phase reference [pu]
const RA = 0 // armature resistance, neglected here: in a real machine it is a few percent of Xs
const DISPLAY_OMEGA = 0.9 // on-screen rad/s: a real 60 Hz phasor set is far too fast to follow

/*
 * Synchronous generator, per unit, V as the reference and the current at −φ for a lagging load:
 *   E = V + I·(Ra + j·Xs)
 *   voltage regulation = (|E| − V) ÷ V × 100 %
 *   P = |E|·V·sin δ ÷ Xs = V·I·cos φ
 * Verified with npx tsx for V = 1.0 pu, I = 1.0 pu, Ra = 0, Xs = 1.0 pu:
 *   pf 0.8 lagging  →  |E| = 1.7889 pu,  δ = 26.565°,  regulation = +78.89 %
 *   pf 1.0          →  |E| = 1.4142 pu,  δ = 45.000°,  regulation = +41.42 %
 *   pf 0.8 leading  →  |E| = 0.8944 pu,  δ = 63.435°,  regulation = −10.56 %
 *   power identity at pf 0.8 lagging: 1.7889 × 1 × sin 26.565° ÷ 1 = 1 × 1 × 0.8 = 0.8000 pu
 * With Ra = 0 the identity is exact, since |E|·sin δ is just the imaginary part I·Xs·cos φ.
 */
const S = {
  I: 1,
  pf: 0.8,
  Xs: 1,
  leading: false,
  phi: 0.6435011087932844,
  Emag: 1.7888543819998317,
  delta: 0.4636476090008061,
  reg: 78.88543819998317,
  Pangle: 0.8,
  Pload: 0.8,
  Pmax: 1.7888543819998317,
}

function solve(v: Record<string, number>) {
  const I = Math.max(v.I_pu ?? 1, 0)
  const pf = Math.min(Math.max(v.pf ?? 0.8, 0), 1)
  const Xs = Math.max(v.Xs ?? 1, 1e-6)
  const leading = (v.dir ?? 1) >= 2
  const phi = Math.acos(pf)
  const sgn = leading ? -1 : 1 // the current sits at −φ for a lagging load, at +φ for a leading one
  const s = sgn * Math.sin(phi)
  const re = V_PU + I * (pf * RA + s * Xs) // real part of E = V + I·(Ra + j·Xs)
  const im = I * (pf * Xs - s * RA)
  const Emag = Math.hypot(re, im)
  const delta = Math.atan2(im, re)
  S.I = I
  S.pf = pf
  S.Xs = Xs
  S.leading = leading
  S.phi = phi
  S.Emag = Emag
  S.delta = delta
  S.reg = ((Emag - V_PU) / V_PU) * 100
  S.Pangle = (Emag * V_PU * Math.sin(delta)) / Xs
  S.Pload = V_PU * I * pf
  S.Pmax = (Emag * V_PU) / Xs
  return S
}
/* ── end of physics ── */

function useSliderValues() {
  const valuesRef = useRef<Record<string, number>>(useSliderStore.getState().values)
  useEffect(
    () => useSliderStore.subscribe((s) => s.values, (v) => { valuesRef.current = v }),
    []
  )
  return valuesRef
}

const pu = (x: number) => `${x.toFixed(4)} pu`
const deg = (rad: number) => `${((rad * 180) / Math.PI).toFixed(3)}°`
const signedPct = (x: number) => `${x >= 0 ? '+' : '−'}${Math.abs(x).toFixed(2)} %`
// Scene text reads in the foreground color; only current (cyan) and power (amber) tint their labels
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const accentClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums'
const axisClass = 'pointer-events-none whitespace-nowrap text-[10px] tabular-nums text-muted-foreground'

const L10N = {
  en: {
    lag: 'lagging',
    lead: 'leading',
    reg: 'regulation',
    pullOut: 'pull-out at δ 90° · stability limit',
    circle: 'circle at |V|',
  },
  ko: {
    lag: '지상',
    lead: '진상',
    reg: '전압변동률',
    pullOut: 'δ 90° 탈출점 · 안정 한계',
    circle: '|V| 원',
  },
} satisfies Record<string, Record<string, string>>

// Shared scratch, filled then handed straight to a geometry, so no frame ever allocates
const SEG = new Float32Array(6)
const ARC_N = 28
const ARC = new Float32Array(ARC_N * 3)
const CIRCLE_N = 97
const CIRCLE = new Float32Array(CIRCLE_N * 3)

function setSeg(line: any, x0: number, y0: number, x1: number, y1: number) {
  SEG[0] = x0; SEG[1] = y0; SEG[2] = 0
  SEG[3] = x1; SEG[4] = y1; SEG[5] = 0
  line?.geometry.setPositions(SEG)
  line?.computeLineDistances()
}

/** Arc of radius r from a0 to a1, centred on the diagram origin. */
function setArc(line: any, r: number, a0: number, a1: number) {
  for (let i = 0; i < ARC_N; i++) {
    const a = a0 + ((a1 - a0) * i) / (ARC_N - 1)
    ARC[i * 3] = r * Math.cos(a)
    ARC[i * 3 + 1] = r * Math.sin(a)
    ARC[i * 3 + 2] = 0
  }
  line?.geometry.setPositions(ARC)
  line?.computeLineDistances()
}

function setCircle(line: any, r: number) {
  for (let i = 0; i < CIRCLE_N; i++) {
    const a = (i / (CIRCLE_N - 1)) * Math.PI * 2
    CIRCLE[i * 3] = r * Math.cos(a)
    CIRCLE[i * 3 + 1] = r * Math.sin(a)
    CIRCLE[i * 3 + 2] = 0
  }
  line?.geometry.setPositions(CIRCLE)
  line?.computeLineDistances()
}

const SEG_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]
const ARC_INIT: [number, number, number][] = Array.from(
  { length: ARC_N },
  (): [number, number, number] => [0, 0, 0]
)
const CIRCLE_INIT: [number, number, number][] = Array.from(
  { length: CIRCLE_N },
  (): [number, number, number] => [0, 0, 0]
)

// ── 3D: the phasor triangle, built explicitly and turning as one rigid set ──
// The longest voltage phasor is drawn this long, so V and E always share one honest scale:
// E outside the dashed circle at |V| is positive regulation, inside is negative.
const LMAX = 1.3
const I_SCALE = 0.62 // scene units per pu of current, its own scale (1.5 pu → 0.93 units)
const FLOW_PER_PU = 2.6 // sheen bands per second at 1 pu of current
const READOUT_X = -1.45 // the two power readouts sit low and left, clear of the card and the slider HUD
const HALF_PI = Math.PI / 2

function SynchronousGeneratorScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const t = L10N[locale]
  const valuesRef = useSliderValues()
  // 0 V, 1 I·Xs, 2 E, 3 regulation dimension line, 4 δ arc, 5 φ arc, 6 circle at |V|
  const lineRefs = useRef<any[]>([])
  const iRef = useRef<any>(null)
  const tagRefs = useRef<(Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const vTipRef = useRef<Group>(null)
  const eTipRef = useRef<Group>(null)
  const iTipRef = useRef<Group>(null)
  const iBuf = useMemo(() => new Float32Array(6), [])
  const flow = useRef(0)
  const labelledFor = useRef<unknown>(null)
  const labelledLocale = useRef<string>('')

  useFrame(({ clock }) => {
    const s = solve(valuesRef.current)
    const wt = clock.getElapsedTime() * DISPLAY_OMEGA
    const sgn = s.leading ? -1 : 1
    const k = LMAX / Math.max(s.Emag, V_PU) // one scale for both voltages, so lengths compare honestly
    const vLen = V_PU * k
    const eLen = s.Emag * k
    const iLen = s.I * I_SCALE
    const aV = wt // V is the reference; the whole set turns counter-clockwise with it
    const aI = wt - sgn * s.phi
    const aE = wt + s.delta
    const vx = vLen * Math.cos(aV)
    const vy = vLen * Math.sin(aV)
    const ex = eLen * Math.cos(aE)
    const ey = eLen * Math.sin(aE)
    const ix = iLen * Math.cos(aI)
    const iy = iLen * Math.sin(aI)
    const live = s.I > 1e-3

    // The triangle: V along the reference, I·Xs from its tip (perpendicular to I), E closing it
    setSeg(lineRefs.current[0], 0, 0, vx, vy)
    setSeg(lineRefs.current[1], vx, vy, ex, ey)
    setSeg(lineRefs.current[2], 0, 0, ex, ey)
    setCircle(lineRefs.current[6], vLen)

    // Regulation as a dimension line: the circle at |V| out to (or back from) the tip of E
    const nx = -Math.sin(aE) * 0.14 // offset sideways so it does not hide under the E phasor
    const ny = Math.cos(aE) * 0.14
    setSeg(
      lineRefs.current[3],
      vLen * Math.cos(aE) + nx, vLen * Math.sin(aE) + ny,
      ex + nx, ey + ny
    )

    const m = Math.min(vLen, eLen)
    setArc(lineRefs.current[4], m * 0.62, aV, aE) // δ, between V and E
    setArc(lineRefs.current[5], m * 0.33, aV, aI) // φ, between V and I
    if (lineRefs.current[1]) lineRefs.current[1].visible = live
    if (lineRefs.current[3]) lineRefs.current[3].visible = live
    if (lineRefs.current[4]) lineRefs.current[4].visible = s.delta > 5e-3
    if (lineRefs.current[5]) lineRefs.current[5].visible = live && s.phi > 5e-3

    iBuf[0] = 0; iBuf[1] = 0; iBuf[2] = 0
    iBuf[3] = ix; iBuf[4] = iy; iBuf[5] = 0
    iRef.current?.geometry.setPositions(iBuf)
    if (iRef.current) iRef.current.visible = live
    flow.current = FLOW_PER_PU * s.I

    vTipRef.current?.position.set(vx, vy, 0)
    eTipRef.current?.position.set(ex, ey, 0)
    iTipRef.current?.position.set(ix, iy, 0)
    if (iTipRef.current) iTipRef.current.visible = live

    // Every tag is anchored to the piece of geometry it names, so it travels with the rotation
    tagRefs.current[0]?.position.set(
      vx * 0.55 + 0.24 * Math.cos(aV - HALF_PI),
      vy * 0.55 + 0.24 * Math.sin(aV - HALF_PI),
      0
    )
    const mx = (vx + ex) / 2
    const my = (vy + ey) / 2
    const mr = Math.max(Math.hypot(mx, my), 1e-3)
    tagRefs.current[1]?.position.set(mx + (mx / mr) * 0.3, my + (my / mr) * 0.3, 0)
    tagRefs.current[2]?.position.set(ex + 0.28 * Math.cos(aE), ey + 0.28 * Math.sin(aE), 0)
    const aIs = aI - sgn * HALF_PI // the side of I that faces away from V stays clear
    tagRefs.current[3]?.position.set(ix / 2 + 0.28 * Math.cos(aIs), iy / 2 + 0.28 * Math.sin(aIs), 0)
    const rMid = (vLen + eLen) / 2
    tagRefs.current[4]?.position.set(
      rMid * Math.cos(aE) + 0.4 * Math.cos(aE + HALF_PI),
      rMid * Math.sin(aE) + 0.4 * Math.sin(aE + HALF_PI),
      0
    )
    tagRefs.current[5]?.position.set(
      (m * 0.62 + 0.18) * Math.cos(aV + s.delta / 2),
      (m * 0.62 + 0.18) * Math.sin(aV + s.delta / 2),
      0
    )
    tagRefs.current[6]?.position.set(
      (m * 0.33 + 0.16) * Math.cos(aV - (sgn * s.phi) / 2),
      (m * 0.33 + 0.16) * Math.sin(aV - (sgn * s.phi) / 2),
      0
    )

    // Values only move with the sliders, so the text is rebuilt only then
    if ((labelledFor.current !== valuesRef.current || labelledLocale.current !== locale) && textRefs.current[8]) {
      labelledFor.current = valuesRef.current
      labelledLocale.current = locale
      const texts = [
        `V ${pu(V_PU)}`,
        `I·Xs ${pu(s.I * s.Xs)} ⟂ I`,
        `E ${pu(s.Emag)}`,
        `I ${s.I.toFixed(2)} pu · pf ${s.pf.toFixed(2)} ${s.leading ? t.lead : t.lag}`,
        `${t.reg} ${signedPct(s.reg)}`,
        `δ ${deg(s.delta)}`,
        s.phi > 5e-3 ? `φ ${deg(s.phi)}` : '',
        `P = E·V·sin δ ÷ Xs = ${pu(s.Pangle)}`,
        `P = V·I·cos φ = ${pu(s.Pload)}`,
      ]
      texts.forEach((text, i) => { if (textRefs.current[i]) textRefs.current[i]!.textContent = text })
    }
  })

  const segs: { color: string; width: number; dashed?: boolean; shape?: 'arc' | 'circle' }[] = [
    { color: c.resistance, width: 3 },
    { color: c.inductance, width: 2.5 },
    { color: c.voltage, width: 3 },
    { color: c.marker, width: 2 },
    { color: c.guide, width: 1.5, dashed: true, shape: 'arc' },
    { color: c.guide, width: 1.5, dashed: true, shape: 'arc' },
    { color: c.guide, width: 1, dashed: true, shape: 'circle' },
  ]
  // 0 V, 1 I·Xs, 2 E, 3 I, 4 regulation, 5 δ, 6 φ, then the two power readouts
  const tagStyle = [undefined, undefined, undefined, c.current, undefined, undefined, undefined]

  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />

      {segs.map((sg, i) => (
        <Line
          key={i}
          ref={(l) => { lineRefs.current[i] = l }}
          points={sg.shape === 'circle' ? CIRCLE_INIT : sg.shape === 'arc' ? ARC_INIT : SEG_INIT}
          lineWidth={sg.width}
          color={sg.color}
          dashed={sg.dashed}
          dashSize={0.08}
          gapSize={0.06}
        />
      ))}

      {/* The load current, as a liquid line whose sheen runs faster the harder it flows */}
      <TrailLine
        ref={iRef}
        points={SEG_INIT}
        lineWidth={2.5}
        color={c.current}
        fade={false}
        flowSpeed={() => flow.current}
      />

      <Marker ref={vTipRef} color={c.marker} size={0.05} />
      <Marker ref={eTipRef} color={c.marker} size={0.065} />
      <Marker ref={iTipRef} color={c.current} size={0.055} />

      {[0, 1, 2, 3, 4, 5, 6].map((i) => (
        <group key={i} ref={(g) => { tagRefs.current[i] = g }}>
          <Html center>
            <span
              ref={(el) => { textRefs.current[i] = el }}
              className={tagStyle[i] ? accentClass : labelClass}
              style={tagStyle[i] ? { color: tagStyle[i] } : undefined}
            />
          </Html>
        </group>
      ))}

      {/* The same real power twice over, from the angle and from the load: they must agree */}
      {[0, 1].map((i) => (
        <Html key={i} position={[READOUT_X, -1.5 - i * 0.28, 0]} center>
          <span
            ref={(el) => { textRefs.current[7 + i] = el }}
            className={accentClass}
            style={{ color: c.power }}
          />
        </Html>
      ))}
      <Html position={[0, -1.02, 0]} center>
        <span className={axisClass}>{t.circle}</span>
      </Html>
    </>
  )
}

// ── 2D: the power-angle curve, with the operating point and the pull-out point on it ──
const CURVE_N = 181
const X_HALF = 3.4
const BASE_Y = -1.6
// One fixed vertical scale, so the curve's height really is |E|·V ÷ Xs and not a rescaling.
// Its largest value over the whole slider space is 3.8734 pu (I 1.5 pu, pf 0.5 lagging, Xs 0.4 pu).
const P_FULL_SCALE = 3.88
const KP = 3.1 / P_FULL_SCALE
const xOfDeg = (d: number) => -X_HALF + (d / 180) * 2 * X_HALF
const DELTA_AXIS: [number, number, number][] = [[-X_HALF - 0.15, BASE_Y, 0], [X_HALF + 0.15, BASE_Y, 0]]
const P_AXIS: [number, number, number][] = [[-X_HALF, BASE_Y, 0], [-X_HALF, BASE_Y + 3.2, 0]]
const ONE_PU: [number, number, number][] = [
  [-X_HALF, BASE_Y + KP, 0],
  [X_HALF, BASE_Y + KP, 0],
]
const CURVE_INIT: [number, number, number][] = Array.from(
  { length: CURVE_N },
  (): [number, number, number] => [0, 0, 0]
)

function PowerAngleCurve2DScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const t = L10N[locale]
  const valuesRef = useSliderValues()
  // 0 pull-out line at δ 90°, 1 dropline to the δ axis, 2 level line to the P axis
  const lineRefs = useRef<any[]>([])
  const curveRef = useRef<any>(null)
  const tagRefs = useRef<(Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const dotRef = useRef<Group>(null)
  const curveBuf = useMemo(() => new Float32Array(CURVE_N * 3), [])
  const labelledFor = useRef<unknown>(null)
  const labelledLocale = useRef<string>('')

  useFrame(() => {
    const s = solve(valuesRef.current)

    for (let i = 0; i < CURVE_N; i++) {
      const d = i // one sample per degree
      curveBuf[i * 3] = xOfDeg(d)
      curveBuf[i * 3 + 1] = BASE_Y + KP * s.Pmax * Math.sin((d * Math.PI) / 180)
      curveBuf[i * 3 + 2] = 0
    }
    curveRef.current?.geometry.setPositions(curveBuf)

    const topY = BASE_Y + KP * s.Pmax
    setSeg(lineRefs.current[0], xOfDeg(90), BASE_Y, xOfDeg(90), topY)
    const dx = xOfDeg((s.delta * 180) / Math.PI)
    const dy = BASE_Y + KP * s.Pangle
    setSeg(lineRefs.current[1], dx, BASE_Y, dx, dy)
    setSeg(lineRefs.current[2], -X_HALF, dy, dx, dy)
    dotRef.current?.position.set(dx, dy, 0)

    tagRefs.current[0]?.position.set(dx, dy + 0.26, 0)
    tagRefs.current[1]?.position.set(xOfDeg(90), topY + 0.22, 0)

    if ((labelledFor.current !== valuesRef.current || labelledLocale.current !== locale) && textRefs.current[1]) {
      labelledFor.current = valuesRef.current
      labelledLocale.current = locale
      const texts = [
        `δ ${deg(s.delta)} · P ${pu(s.Pangle)} = V·I·cos φ ${pu(s.Pload)}`,
        `${t.pullOut} · Pmax = E·V ÷ Xs ${pu(s.Pmax)}`,
      ]
      texts.forEach((text, i) => { if (textRefs.current[i]) textRefs.current[i]!.textContent = text })
    }
  })

  const segs: { color: string; width: number }[] = [
    { color: c.guide, width: 1.5 },
    { color: c.guide, width: 1 },
    { color: c.guide, width: 1 },
  ]

  return (
    <>
      <Line points={DELTA_AXIS} lineWidth={1} color={c.guide} />
      <Line points={P_AXIS} lineWidth={1} color={c.guide} />
      <Line points={ONE_PU} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.08} />
      <Line ref={curveRef} points={CURVE_INIT} lineWidth={2.5} color={c.power} />

      {segs.map((sg, i) => (
        <Line
          key={i}
          ref={(l) => { lineRefs.current[i] = l }}
          points={SEG_INIT}
          lineWidth={sg.width}
          color={sg.color}
          dashed
          dashSize={0.07}
          gapSize={0.05}
        />
      ))}

      <Marker ref={dotRef} color={c.power} size={0.06} />

      {[0, 1].map((i) => (
        <group key={i} ref={(g) => { tagRefs.current[i] = g }}>
          <Html center>
            <span
              ref={(el) => { textRefs.current[i] = el }}
              className={accentClass}
              style={{ color: c.power }}
            />
          </Html>
        </group>
      ))}

      <Html position={[xOfDeg(0), BASE_Y - 0.24, 0]} center>
        <span className={axisClass}>δ 0°</span>
      </Html>
      <Html position={[xOfDeg(90), BASE_Y - 0.24, 0]} center>
        <span className={axisClass}>90°</span>
      </Html>
      <Html position={[xOfDeg(180), BASE_Y - 0.24, 0]} center>
        <span className={axisClass}>180°</span>
      </Html>
      <Html position={[-X_HALF - 0.3, BASE_Y + KP, 0]} center>
        <span className={axisClass}>P 1.0 pu</span>
      </Html>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'synchronous-generator',
  title: {
    en: 'Synchronous Generator: Regulation & Power Angle',
    ko: '동기발전기의 전압변동률과 부하각',
  },
  subject: 'machines',
  order: 4,
  description: {
    en: `## Synchronous Generator: Regulation and Power Angle

**The excitation EMF E and the terminal voltage V differ only by the drop across the synchronous reactance, and the power factor of the load decides whether the terminal voltage sags or rises.**

- Mid neutral arrow, its tip riding the dashed circle: the terminal voltage V, taken as the phase reference
- Strongest neutral arrow: the excitation EMF E, what the field winding induces behind Xs
- Light neutral arrow closing the triangle: the reactance drop I·Xs, always perpendicular to the current
- Cyan liquid arrow: the load current I, behind V by φ for a lagging load and ahead of it for a leading one
- Faint arcs at the centre: φ between V and I, and the power angle δ between V and E
- Short dimension line beside the tip of E: the regulation, the plain length difference between |E| and the dashed circle at |V|
- Amber readouts: the same real power worked out twice, once from δ and once from the load

The triangle turns counter-clockwise as one rigid set, because that is what it is inside the machine: E turns with the rotor, and δ is how far the rotor runs ahead of the terminal voltage.

**EMF**: E = V + I·(Ra + j·Xs), with Ra neglected, so E = V + j·I·Xs
**Voltage regulation**: (|E| − V) ÷ V × 100 %
**Real power**: P = |E|·V·sin δ ÷ Xs = V·I·cos φ

At the defaults (V = 1.0 pu, I = 1.0 pu, Xs = 1.0 pu, pf 0.8 lagging) E reaches 1.7889 pu at δ = 26.565°, a regulation of +78.89 %, and both power forms read 0.8000 pu.

Walk pf up to 1.00 and E shrinks to 1.4142 pu while δ opens to 45.000°. Switch the load type to leading and E falls inside the circle: at pf 0.8 leading |E| = 0.8944 pu, the regulation is −10.56 %, and the terminal voltage rises under load. Raise I or Xs and the drop stretches, pushing δ further round.

In 2D: the power-angle curve P = |E|·V·sin δ ÷ Xs over δ from 0° to 180°, the operating point marked on it, and the pull-out point at δ = 90° as the stability limit. The vertical scale is fixed, so the curve genuinely grows with |E|: excitation is what buys stability margin. A deeply leading load with a large Xs pushes the operating point past 90°, where steady operation is no longer possible.`,
    ko: `## 동기발전기의 전압변동률과 부하각

**유도기전력 E와 단자전압 V의 차이는 동기 리액턴스에 걸리는 강하뿐이고, 단자전압이 떨어질지 오히려 올라갈지는 부하의 역률이 결정합니다.**

- 끝점이 점선 원을 따라 도는 중간 밝기 중성색 화살표: 위상 기준으로 잡은 단자전압 V
- 가장 진한 중성색 화살표: Xs 뒤에서 계자 권선이 유도하는 기전력 E
- 삼각형을 닫는 밝은 중성색 화살표: 전류에 항상 수직인 리액턴스 강하 I·Xs
- 시안 액체 화살표: 부하전류 I. 지상 부하면 V보다 φ만큼 뒤지고, 진상 부하면 그만큼 앞섭니다
- 중심의 옅은 호: V와 I 사이의 φ, 그리고 V와 E 사이의 부하각 δ
- E 끝 옆의 짧은 치수선: 전압변동률, 곧 |V| 점선 원과 |E|의 길이 차이 그 자체
- 앰버 수치: 같은 유효전력을 δ에서 한 번, 부하에서 한 번 계산한 값

삼각형은 한 덩어리로 반시계 방향으로 돕니다. 기계 안에서 실제로 그렇기 때문입니다. E는 회전자와 함께 돌고, δ는 회전자가 단자전압보다 앞서 나간 각입니다.

**기전력**: E = V + I·(Ra + j·Xs), 여기서는 Ra를 무시하므로 E = V + j·I·Xs
**전압변동률**: (|E| − V) ÷ V × 100 %
**유효전력**: P = |E|·V·sin δ ÷ Xs = V·I·cos φ

기본값(V = 1.0 pu, I = 1.0 pu, Xs = 1.0 pu, 역률 0.8 지상)에서 E는 1.7889 pu에 이르고 δ = 26.565°, 전압변동률은 +78.89 %이며, 두 전력식이 모두 0.8000 pu를 가리킵니다.

역률을 1.00까지 올리면 E는 1.4142 pu로 줄고 δ는 45.000°로 벌어집니다. 부하 종류를 진상으로 바꾸면 E가 원 안으로 들어갑니다. 역률 0.8 진상에서 |E| = 0.8944 pu, 전압변동률은 −10.56 %가 되어 부하를 걸었는데도 단자전압이 올라갑니다. I나 Xs를 올리면 강하가 길어지며 δ가 더 벌어집니다.

2D: δ가 0°에서 180°까지 갈 때의 전력-부하각 곡선 P = |E|·V·sin δ ÷ Xs입니다. 현재 운전점을 찍고, δ = 90°의 탈출점을 안정 한계로 표시했습니다. 세로 축척이 고정이므로 곡선 높이는 그대로 |E|를 따라갑니다. 안정 여유를 사는 것은 결국 계자 전류입니다. Xs가 큰 상태에서 진상 부하를 깊게 걸면 운전점이 90°를 넘어가 정상 운전이 불가능한 영역으로 들어갑니다.`,
  },
  sliderParams: [
    { key: 'I_pu', label: { en: 'Load current (I)',              ko: '부하전류 (I)' },                 unit: 'pu', min: 0,   max: 1.5, step: 0.05, defaultValue: 1   },
    { key: 'pf',   label: { en: 'Power factor (cos φ)',           ko: '역률 (cos φ)' },                             min: 0.5, max: 1,   step: 0.01, defaultValue: 0.8 },
    { key: 'dir',  label: { en: 'Load type — 1 lagging, 2 leading', ko: '부하 종류 — 1 지상, 2 진상' },             min: 1,   max: 2,   step: 1,    defaultValue: 1   },
    { key: 'Xs',   label: { en: 'Synchronous reactance (Xs)',     ko: '동기 리액턴스 (Xs)' },           unit: 'pu', min: 0.4, max: 2,   step: 0.05, defaultValue: 1   },
  ],
  Scene: SynchronousGeneratorScene,
  Scene2D: PowerAngleCurve2DScene,
}

export default concept
