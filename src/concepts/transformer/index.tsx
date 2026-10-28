import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import type { Group, Mesh } from 'three'
import type { LineMaterial } from 'three-stdlib'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { useLocaleStore } from '../../lib/i18n'
import { TrailLine } from '../../components/scene/TrailLine'
import { Resistor, setResistorHeat } from '../../components/scene/Resistor'
import { setLineOpacity, setOpacity, useStoryReveal } from '../../components/scene/storyReveal'
import type { ConceptMeta } from '../../types/concept'

const TAU = Math.PI * 2
const V1_MAX = 240 // primary-voltage slider maximum [V]
const V2_MAX = 480 // largest secondary voltage reachable (a = 0.5 doubles V1)
const P_REF = 1000 // [W] display reference for the load resistor's filament
const TURNS_MAX = 12 // drawn turns on the larger of the two windings

// Ideal transformer. One shared core ties the windings together, so the turns ratio sets all of it:
//   V1 ÷ V2 = N1 ÷ N2 = a,  I1 ÷ I2 = N2 ÷ N1 = 1 ÷ a,  S1 = S2,  Zin = a²·ZL
// Worked example, verified with node -e:
//   V1 = 220 V, a = 10, ZL = 10 Ω
//     V2  = 220 ÷ 10   =   22.000 V
//     I2  =  22 ÷ 10   =    2.200 A
//     I1  = 2.2 ÷ 10   =    0.220 A
//     Zin = 10² × 10   = 1000.000 Ω   (and V1 ÷ I1 = 220 ÷ 0.22 = 1000, the same number)
//     S1  = 220 × 0.22 =   48.400 W
//     S2  =  22 × 2.2  =   48.400 W   (equal, as an ideal transformer must be)
// Results live in one module-scope record, so no frame allocates.
const S = {
  V1: 220, a: 10, ZL: 10,
  V2: 22, I1: 0.22, I2: 2.2,
  Zin: 1000, S1: 48.4, S2: 48.4,
  N1: 12, N2: 1.2,
}

function solve(v: Record<string, number>) {
  const V1 = v.V1 ?? 220
  const a = v.a ?? 10
  const ZL = v.ZL ?? 10
  const V2 = V1 / a
  const I2 = V2 / ZL
  const I1 = I2 / a
  S.V1 = V1
  S.a = a
  S.ZL = ZL
  S.V2 = V2
  S.I2 = I2
  S.I1 = I1
  S.Zin = a * a * ZL
  S.S1 = V1 * I1
  S.S2 = V2 * I2
  // Drawn turns stay in the true ratio a, with the larger winding pinned at TURNS_MAX
  S.N1 = a >= 1 ? TURNS_MAX : TURNS_MAX * a
  S.N2 = a >= 1 ? TURNS_MAX / a : TURNS_MAX
  return S
}

// ── Narration steps drive what exists on screen. Each reveal eases 0 → 1 so nothing pops in. ──
// Step 0 is one coil on an iron core: the second winding, the load and the trade all come later.
const REV_FLUX = 0 // the alternating flux the primary drives round the core
const REV_SEC = 1 // the secondary winding, sharing nothing but that flux
const REV_VOLT = 2 // the two voltage bars and their readouts, plus the turns ratio
const REV_LOAD = 3 // the load, and with it the current that actually flows
const REV_POWER = 4 // the S1 = S2 readout
const REV_ZIN = 5 // the impedance seen back through the ratio
const REV_COUNT = 6

function revealTargets(step: number, out: Float32Array) {
  out[REV_FLUX] = step >= 1 ? 1 : 0
  out[REV_SEC] = step >= 2 ? 1 : 0
  out[REV_VOLT] = step >= 3 ? 1 : 0
  out[REV_LOAD] = step >= 5 ? 1 : 0
  out[REV_POWER] = step >= 6 ? 1 : 0
  out[REV_ZIN] = step >= 8 ? 1 : 0
}

function useSliderValues() {
  const valuesRef = useRef<Record<string, number>>(useSliderStore.getState().values)
  useEffect(
    () => useSliderStore.subscribe((s) => s.values, (v) => { valuesRef.current = v }),
    []
  )
  return valuesRef
}

const fmt = (x: number, unit: string) => `${x.toFixed(x >= 1000 ? 0 : x < 10 ? 2 : 1)} ${unit}`
const fmtOhm = (x: number) => (x >= 100 ? `${Math.round(x)} Ω` : fmt(x, 'Ω'))
const fmtA = (x: number) => (x % 1 === 0 ? x.toFixed(0) : x.toFixed(1))

// Labels read in the foreground color; only the two accents tint their text, so the current and
// power readouts take their color from the tokens and everything else stays foreground.
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const accentClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums'

const widthFor = (I: number) => Math.min(1.2 + 1.6 * Math.sqrt(I), 8)
const flowFor = (I: number) => Math.min(0.5 + 0.6 * I, 8)
const radiusFor = (ZL: number) => 0.11 * Math.sqrt(1 / Math.max(ZL, 0.5)) + 0.03

// ── 3D: one iron core, two windings, and the flux that circulates between them ──
// The primary carries many turns and a thin current line; the secondary carries few turns and a
// thick one. Same core, same flux, same power — only the ratio moves.
const X_SRC = -2.6 // source wire
const X_P = -1.4 // primary leg center
const X_S = 0.4 // secondary leg center
const X_LOAD = 1.6 // load wire
const XM = (X_P + X_S) / 2
const CORE_W = X_S - X_P
const CORE_HALF = 0.18 // half the iron's thickness
const CORE_Y = 0.87 // yoke center height
const COIL_R = 0.34 // winding radius, wider than the leg it wraps
const Y_TOP = 0.6
const Y_BOT = -0.6
const Y_BASE = -1.35 // the 0 V foot of the two voltage bars
const HEIGHT_PER_VOLT = 1.75 / V2_MAX // one scale for both bars, so the shrink is honest
const BAR_3D = 0.14

const LEG_ARGS: [number, number, number] = [2 * CORE_HALF, 2 * (CORE_Y + CORE_HALF), 2 * CORE_HALF]
const YOKE_ARGS: [number, number, number] = [CORE_W, 2 * CORE_HALF, 2 * CORE_HALF]
const CORE_PARTS: [number, number, number][] = [
  [X_P, 0, 0], [X_S, 0, 0], [XM, CORE_Y, 0], [XM, -CORE_Y, 0],
]

// The flux: one closed loop running the length of the core, squared off to follow the iron
const FLUX_N = 161
const FLUX_PTS: [number, number, number][] = Array.from({ length: FLUX_N }, (_, i) => {
  const t = (i / (FLUX_N - 1)) * TAU
  const cs = Math.cos(t)
  const sn = Math.sin(t)
  return [
    XM + (CORE_W / 2) * Math.sign(cs) * Math.sqrt(Math.abs(cs)),
    CORE_Y * Math.sign(sn) * Math.sqrt(Math.abs(sn)),
    0,
  ]
})
const FLUX_FLOW = 2.4 // sheen bands per second at the peak of the alternating flux
const FLUX_OMEGA = 1.1 // on-screen rad/s: a real mains cycle is far too fast to follow

// One winding loop: the helix up its leg, a lead to the terminal, the terminal wire, a lead back
const COIL_N = 193
const LEAD_N = 12
const SIDE_N = 14
const WIND_N = COIL_N + LEAD_N + SIDE_N + LEAD_N
const WIND_INIT: [number, number, number][] = Array.from(
  { length: WIND_N },
  (): [number, number, number] => [0, 0, 0]
)

function fillWinding(buf: Float32Array, xLeg: number, phase0: number, turns: number, xTerm: number) {
  let n = 0
  const span = Y_TOP - Y_BOT
  for (let i = 0; i < COIL_N; i++) {
    const u = i / (COIL_N - 1)
    const th = phase0 + u * turns * TAU
    buf[n++] = xLeg + COIL_R * Math.cos(th)
    buf[n++] = Y_BOT + u * span
    buf[n++] = COIL_R * Math.sin(th)
  }
  const thTop = phase0 + turns * TAU
  const xTopC = xLeg + COIL_R * Math.cos(thTop)
  const zTopC = COIL_R * Math.sin(thTop)
  const xBotC = xLeg + COIL_R * Math.cos(phase0)
  const zBotC = COIL_R * Math.sin(phase0)
  for (let i = 1; i <= LEAD_N; i++) {
    const u = i / LEAD_N
    buf[n++] = xTopC + (xTerm - xTopC) * u
    buf[n++] = Y_TOP
    buf[n++] = zTopC - zTopC * u
  }
  for (let i = 1; i <= SIDE_N; i++) {
    const u = i / SIDE_N
    buf[n++] = xTerm
    buf[n++] = Y_TOP + (Y_BOT - Y_TOP) * u
    buf[n++] = 0
  }
  for (let i = 1; i <= LEAD_N; i++) {
    const u = i / LEAD_N
    buf[n++] = xTerm + (xBotC - xTerm) * u
    buf[n++] = Y_BOT
    buf[n++] = zBotC * u
  }
}

function setBar3D(m: Mesh | null | undefined, x: number, h: number) {
  if (!m) return
  const hh = Math.max(h, 1e-3)
  m.scale.set(BAR_3D, hh, BAR_3D)
  m.position.set(x, Y_BASE + hh / 2, 0)
}

function TransformerCoreScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  // Allocate the winding buffers once; every frame rewrites them in place
  const bufs = useMemo(() => [new Float32Array(WIND_N * 3), new Float32Array(WIND_N * 3)], [])
  const windRefs = useRef<any[]>([]) // 0 primary, 1 secondary
  const fluxRef = useRef<any>(null)
  const barRefs = useRef<(Mesh | null)[]>([]) // 0 V1, 1 V2
  const loadRef = useRef<Group>(null)
  // 0 primary V, 1 primary I, 2 secondary V, 3 secondary I, 4 ratio, 5 Zin, 6 power
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const flows = useRef([0, 0, 0]) // primary current, secondary current, flux
  const labelledFor = useRef<unknown>(null)
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(({ clock }) => {
    const s = solve(valuesRef.current)

    // Each winding is drawn as the current itself: turn count from the ratio, width from the amps
    fillWinding(bufs[0], X_P, Math.PI, s.N1, X_SRC)
    fillWinding(bufs[1], X_S, 0, s.N2, X_LOAD)
    // Before the narration hangs a load on the secondary, that winding is open: no secondary
    // current at all, and only the magnetizing trickle in the primary, which is flowFor's floor.
    const wLoad = reveal[REV_LOAD]
    for (let w = 0; w < 2; w++) {
      const line = windRefs.current[w]
      if (!line) continue
      line.geometry.setPositions(bufs[w])
      const I = (w === 0 ? s.I1 : s.I2) * wLoad
      ;(line.material as LineMaterial).linewidth = widthFor(I)
      flows.current[w] = w === 0 ? flowFor(I) : flowFor(s.I2) * wLoad
    }
    setLineOpacity(windRefs.current[1], reveal[REV_SEC])

    // The flux alternates, so its sheen runs one way and then the other; V1 sets how much of it
    const fluxMat = fluxRef.current?.material as LineMaterial | undefined
    if (fluxMat) fluxMat.linewidth = 1.5 + 3 * Math.sqrt(s.V1 / V1_MAX)
    flows.current[2] = FLUX_FLOW * Math.sin(clock.getElapsedTime() * FLUX_OMEGA)
    setLineOpacity(fluxRef.current, reveal[REV_FLUX])

    const wVolt = reveal[REV_VOLT]
    setBar3D(barRefs.current[0], X_SRC - 0.38, s.V1 * HEIGHT_PER_VOLT * wVolt)
    setBar3D(barRefs.current[1], X_LOAD + 0.38, s.V2 * HEIGHT_PER_VOLT * wVolt * reveal[REV_SEC])

    const load = loadRef.current
    if (load) {
      const r = radiusFor(s.ZL) * wLoad
      load.rotation.z = Math.PI / 2 // the Resistor is unit-sized along local X, so stand it upright
      load.position.set(X_LOAD, 0, 0)
      load.scale.set((Y_TOP - Y_BOT) * wLoad, r, r)
    }
    setResistorHeat(load, Math.sqrt(s.S2 / P_REF) * wLoad)

    const fades = [wVolt, wLoad, wVolt * reveal[REV_SEC], wLoad, wVolt, reveal[REV_ZIN], reveal[REV_POWER]]
    fades.forEach((v, t) => setOpacity(textRefs.current[t], v))

    if (labelledFor.current !== valuesRef.current && textRefs.current[6]) {
      labelledFor.current = valuesRef.current
      const ko = locale === 'ko'
      if (textRefs.current[0]) {
        textRefs.current[0]!.textContent = `${ko ? '1차' : 'Primary'} · V1 ${fmt(s.V1, 'V')}`
      }
      if (textRefs.current[1]) textRefs.current[1]!.textContent = `I1 ${fmt(s.I1, 'A')}`
      if (textRefs.current[2]) {
        textRefs.current[2]!.textContent = `${ko ? '2차' : 'Secondary'} · V2 ${fmt(s.V2, 'V')}`
      }
      if (textRefs.current[3]) textRefs.current[3]!.textContent = `I2 ${fmt(s.I2, 'A')}`
      if (textRefs.current[4]) {
        textRefs.current[4]!.textContent = `a = N1 ÷ N2 = ${fmtA(s.a)}`
      }
      if (textRefs.current[5]) {
        textRefs.current[5]!.textContent = `Zin = a²·ZL = ${fmtOhm(s.Zin)}`
      }
      textRefs.current[6]!.textContent = `S1 = S2 = ${fmt(s.S2, 'W')}`
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  return (
    <>
      {/* Aimed right of centre, so the secondary's load and readouts clear the narration panel */}
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} target={[1.0, 0, 0]} />

      {/* Background, lights, grid, and bloom come from the shared SceneStage */}
      {/* The iron core, drawn as glass so the flux inside it stays visible */}
      {CORE_PARTS.map((p, i) => (
        <mesh key={i} position={p}>
          <boxGeometry args={i < 2 ? LEG_ARGS : YOKE_ARGS} />
          <meshStandardMaterial
            color={c.inductance}
            roughness={0.6}
            metalness={0.25}
            transparent
            opacity={0.4}
            depthWrite={false}
          />
        </mesh>
      ))}

      {/* The shared flux: one closed loop threading both windings */}
      <TrailLine
        ref={fluxRef}
        points={FLUX_PTS}
        lineWidth={4}
        color={c['b-field']}
        fade={false}
        flowSpeed={() => flows.current[2]}
      />

      {/* Each winding is the current path: many thin turns on one side, few thick ones on the other */}
      {[0, 1].map((w) => (
        <TrailLine
          key={w}
          ref={(line) => { windRefs.current[w] = line }}
          points={WIND_INIT}
          lineWidth={2.5}
          color={c.current}
          fade={false}
          flowSpeed={() => flows.current[w]}
        />
      ))}

      {/* The load on the secondary, a glass tube whose filament thickens with the power it takes */}
      <Resistor ref={loadRef} />

      {/* V1 and V2 as bars on one shared scale, standing on the same 0 V foot */}
      {[0, 1].map((b) => (
        <mesh key={b} ref={(m) => { barRefs.current[b] = m }}>
          <boxGeometry args={[1, 1, 1]} />
          <meshStandardMaterial color={c.voltage} roughness={0.5} />
        </mesh>
      ))}

      <Html position={[X_SRC, -1.62, 0]} center>
        <div className="pointer-events-none text-center leading-5">
          <div><span ref={(el) => { textRefs.current[0] = el }} className={labelClass} /></div>
          <div>
            <span ref={(el) => { textRefs.current[1] = el }} className={accentClass} style={{ color: c.current }} />
          </div>
        </div>
      </Html>
      <Html position={[X_LOAD, -1.62, 0]} center>
        <div className="pointer-events-none text-center leading-5">
          <div><span ref={(el) => { textRefs.current[2] = el }} className={labelClass} /></div>
          <div>
            <span ref={(el) => { textRefs.current[3] = el }} className={accentClass} style={{ color: c.current }} />
          </div>
        </div>
      </Html>
      <Html position={[XM, 1.5, 0]} center>
        <div className="pointer-events-none text-center leading-5">
          <div><span ref={(el) => { textRefs.current[4] = el }} className={labelClass} /></div>
          <div><span ref={(el) => { textRefs.current[5] = el }} className={labelClass} /></div>
        </div>
      </Html>
      <Html position={[XM, -1.98, 0]} center>
        <span ref={(el) => { textRefs.current[6] = el }} className={accentClass} style={{ color: c.power }} />
      </Html>
    </>
  )
}

// ── 2D: the trade as six bars ──
// Voltage and current are each measured against the larger member of their own pair, so the two
// staircases mirror each other: V2 is 1 ÷ a of V1 exactly as I2 is a times I1. The two power bars
// are drawn to the same mark because S1 = S2 always, whatever the sliders say.
const BAR_X0 = -3.3
const SPAN = 4.6
const BAR_H = 0.3
const ROW_Y = [1.55, 1.08, 0.3, -0.17, -0.95, -1.42] // V1, V2, I1, I2, S1, S2
const POWER_FRAC = 0.72
const X_EQUAL = BAR_X0 + SPAN * POWER_FRAC
const LABEL_X = BAR_X0 + SPAN + 0.16
const EQUAL_PTS: [number, number, number][] = [
  [X_EQUAL, ROW_Y[5] - BAR_H, 0],
  [X_EQUAL, ROW_Y[4] + BAR_H, 0],
]
const lengths = new Float32Array(6)

function TransformerBars2DScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const barRefs = useRef<(Mesh | null)[]>([])
  // 0..5 bar readouts, 6 the ratio and reflected-impedance line
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<unknown>(null)
  const equalRef = useRef<any>(null)
  const headRef = useRef<HTMLSpanElement>(null)
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(() => {
    const s = solve(valuesRef.current)
    const vMax = Math.max(s.V1, s.V2)
    const iMax = Math.max(s.I1, s.I2)
    lengths[0] = (SPAN * s.V1) / vMax
    lengths[1] = (SPAN * s.V2) / vMax
    lengths[2] = (SPAN * s.I1) / iMax
    lengths[3] = (SPAN * s.I2) / iMax
    lengths[4] = SPAN * POWER_FRAC
    lengths[5] = SPAN * POWER_FRAC
    // Each pair of bars arrives with the sentence that introduces that quantity
    const wVolt = reveal[REV_VOLT]
    const wSec = reveal[REV_SEC]
    const fades = [wVolt, wVolt * wSec, reveal[REV_LOAD], reveal[REV_LOAD], reveal[REV_POWER], reveal[REV_POWER]]
    for (let b = 0; b < 6; b++) {
      const w = Math.max(lengths[b], 1e-3)
      barRefs.current[b]?.scale.set(w, BAR_H * fades[b], 1)
      barRefs.current[b]?.position.set(BAR_X0 + w / 2, ROW_Y[b], 0)
      setOpacity(textRefs.current[b], fades[b])
    }
    setLineOpacity(equalRef.current, reveal[REV_POWER])
    setOpacity(headRef.current, reveal[REV_POWER])
    setOpacity(textRefs.current[6], reveal[REV_ZIN])

    if (labelledFor.current !== valuesRef.current && textRefs.current[6]) {
      labelledFor.current = valuesRef.current
      const ko = locale === 'ko'
      if (textRefs.current[0]) textRefs.current[0]!.textContent = `V1 ${fmt(s.V1, 'V')}`
      if (textRefs.current[1]) textRefs.current[1]!.textContent = `V2 ${fmt(s.V2, 'V')}`
      if (textRefs.current[2]) textRefs.current[2]!.textContent = `I1 ${fmt(s.I1, 'A')}`
      if (textRefs.current[3]) textRefs.current[3]!.textContent = `I2 ${fmt(s.I2, 'A')}`
      if (textRefs.current[4]) textRefs.current[4]!.textContent = `S1 ${fmt(s.S1, 'W')}`
      if (textRefs.current[5]) textRefs.current[5]!.textContent = `S2 ${fmt(s.S2, 'W')}`
      textRefs.current[6]!.textContent =
        `a = N1 ÷ N2 = ${fmtA(s.a)} · ZL = ${fmtOhm(s.ZL)} · Zin = a²·ZL = ${fmtOhm(s.Zin)}` +
        ` ${ko ? '(1차에서 본 값)' : '(seen from the primary)'}`
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  const barColors = [c.voltage, c.voltage, c.current, c.current, c.power, c.power]
  const barOpacity = [0.9, 0.55, 0.9, 0.55, 0.9, 0.55]

  return (
    <>
      {barColors.map((color, b) => (
        <mesh key={b} ref={(m) => { barRefs.current[b] = m }}>
          <planeGeometry args={[1, 1]} />
          <meshBasicMaterial color={color} transparent opacity={barOpacity[b]} depthWrite={false} />
        </mesh>
      ))}

      {/* The equal-power line: both power bars always reach it */}
      <Line ref={equalRef} points={EQUAL_PTS} lineWidth={1.5} color={c.power} dashed dashSize={0.1} gapSize={0.07} />

      {ROW_Y.map((y, b) => (
        <Html key={b} position={[LABEL_X, y, 0]}>
          <span
            ref={(el) => { textRefs.current[b] = el }}
            className={b < 2 ? labelClass : accentClass}
            style={b < 2 ? undefined : { color: b < 4 ? c.current : c.power }}
          />
        </Html>
      ))}

      <Html position={[BAR_X0, 1.95, 0]}>
        <span ref={headRef} className={labelClass}>V2 ÷ V1 = 1 ÷ a · I2 ÷ I1 = a · S1 = S2</span>
      </Html>
      <Html position={[BAR_X0, -1.8, 0]}>
        <span ref={(el) => { textRefs.current[6] = el }} className={labelClass} />
      </Html>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'transformer',
  title: { en: 'Transformer', ko: '변압기' },
  subject: 'machines',
  order: 1,
  description: {
    en: `## The Transformer

**A shared core lets the turns ratio trade voltage for current: multiply one by a and divide the other by a, and the power that crosses is unchanged.**

- The glassy ring: the iron core, the magnetic path both windings sit on
- The deep cool loop running through it: the flux. It thickens as V1 rises, and its sheen reverses because the flux alternates
- The two cyan helices: the windings, drawn as the current itself. The primary has many turns and a thin line; the secondary has few turns and a thick one. Drawn turn counts keep the true ratio a, with the larger winding pinned at 12 turns
- The two upright bars: V1 and V2 on one shared scale, standing on the same 0 V foot
- The tube on the right: the load ZL, its filament thickening with the power it absorbs
- The amber readout: S1 and S2, which stay equal

**Turns ratio**: a = N1 ÷ N2
**Voltage**: V1 ÷ V2 = a, so V2 = V1 ÷ a
**Current**: I1 ÷ I2 = 1 ÷ a, so I1 = I2 ÷ a
**Power**: S1 = V1·I1 = V2·I2 = S2
**Reflected impedance**: Zin = a²·ZL

Worked example: V1 = 220 V, a = 10 and ZL = 10 Ω give V2 = 22 V, I2 = 2.2 A and I1 = 0.22 A. Both sides carry 48.4 W, and the primary sees Zin = 10² × 10 = 1000 Ω, which is also 220 V ÷ 0.22 A. That last line is the one learners forget: the turns ratio scales impedance by a², not by a.

Raise a and the secondary bar collapses while its current line swells, with the power untouched and Zin climbing as the square. Push a below 1 and the transformer steps up instead. Raise V1 to thicken the flux and lift every quantity. Lower ZL to draw more current and heat the load.

This is the ideal case. A real transformer also has winding resistance, leakage reactance and core loss (hysteresis and eddy currents), so its secondary voltage sags under load and a little power is lost on the way through.

In 2D: the six bars side by side. Each pair is measured against its own larger member, so the voltage and current staircases mirror each other, and the two power bars reach the same dashed mark no matter where the sliders go.`,
    ko: `## 변압기

**하나의 철심이 권수비로 전압과 전류를 맞바꿉니다. 한쪽을 a배 하면 다른 쪽이 a분의 1이 되고, 넘어가는 전력은 그대로입니다.**

- 유리처럼 비치는 고리: 철심. 두 권선이 함께 올라앉은 자기 회로입니다
- 철심을 따라 도는 짙은 한색 고리: 자속. V1이 커지면 굵어지고, 교류이므로 광택이 방향을 바꿔 흐릅니다
- 청록색 나선 두 개: 권선. 전류 그 자체로 그렸습니다. 1차는 권수가 많고 선이 가늘며, 2차는 권수가 적고 선이 굵습니다. 그린 권수는 실제 비 a를 지키고, 많은 쪽을 12턴으로 고정했습니다
- 서 있는 막대 두 개: V1과 V2. 같은 눈금, 같은 0 V 바닥에서 재습니다
- 오른쪽 관: 부하 ZL. 흡수하는 전력만큼 내부 필라멘트가 굵어집니다
- 호박색 수치: S1과 S2. 언제나 같습니다

**권수비**: a = N1 ÷ N2
**전압**: V1 ÷ V2 = a, 즉 V2 = V1 ÷ a
**전류**: I1 ÷ I2 = 1 ÷ a, 즉 I1 = I2 ÷ a
**전력**: S1 = V1·I1 = V2·I2 = S2
**임피던스 환산**: Zin = a²·ZL

계산 예: V1 = 220 V, a = 10, ZL = 10 Ω이면 V2 = 22 V, I2 = 2.2 A, I1 = 0.22 A입니다. 양쪽 모두 48.4 W를 보내고, 1차에서 본 임피던스는 Zin = 10² × 10 = 1000 Ω으로 220 V ÷ 0.22 A와 같습니다. 바로 이 마지막 줄을 자주 놓칩니다. 권수비는 임피던스를 a배가 아니라 a²배로 바꿉니다.

a를 올리면 2차 전압 막대가 주저앉고 전류 선이 부풀지만 전력은 그대로이며, Zin은 제곱으로 치솟습니다. a를 1 아래로 내리면 승압이 됩니다. V1을 올리면 자속이 굵어지고 모든 값이 함께 커집니다. ZL을 낮추면 전류가 더 흐르고 부하가 달아오릅니다.

여기까지는 이상 변압기입니다. 실제 변압기에는 권선 저항, 누설 리액턴스, 철손(히스테리시스손과 와류손)이 있어서 부하가 걸리면 2차 전압이 조금 내려가고 전력도 일부 잃습니다.

2D: 여섯 개의 막대를 나란히 놓았습니다. 각 쌍은 그 쌍에서 큰 쪽을 기준으로 재므로 전압과 전류의 계단이 서로 거울처럼 뒤바뀌고, 전력 막대 두 개는 슬라이더를 어떻게 움직여도 같은 점선 눈금에 닿습니다.`,
  },
  story: [
    {
      set: { V1: 220, a: 10, ZL: 10 },
      text: {
        en: 'Wind a coil around a lump of iron and push alternating current through it. That is the whole apparatus so far.',
        ko: '쇳덩이에 코일을 감고 교류를 흘립니다. 지금까지의 장치는 이게 전부입니다.',
      },
    },
    {
      set: { V1: 220, a: 10, ZL: 10 },
      text: {
        en: 'A changing current makes a changing magnetic flux, and the iron gives it a path to run round. Watch the sheen reverse — alternating current, alternating flux.',
        ko: '전류가 변하면 자속이 변하고, 철심이 그 자속에 길을 내줍니다. 광택이 방향을 뒤집는 걸 보세요. 교류니까 자속도 왔다 갔다 합니다.',
      },
    },
    {
      set: { V1: 220, a: 10, ZL: 10 },
      text: {
        en: 'Now wind a second coil on the far side of the same iron. The two coils never touch. No wire runs between them. They share one thing: the flux.',
        ko: '이제 같은 철심 반대편에 두 번째 코일을 감습니다. 두 코일은 서로 닿지 않습니다. 전선 한 가닥도 이어져 있지 않습니다. 공유하는 건 단 하나, 자속입니다.',
      },
    },
    {
      set: { V1: 220, a: 10, ZL: 10 },
      text: {
        en: 'The same flux threads every turn, so every turn gets the same induced volts. Count turns and you have counted volts. Ten times fewer turns, ten times less voltage: 220 V in, 22.0 V out.',
        ko: '같은 자속이 모든 감은 선을 지나니, 한 바퀴마다 유도되는 전압이 똑같습니다. 감은 수를 센 것이 곧 전압을 센 것입니다. 감은 수가 10분의 1이면 전압도 10분의 1. 220 V 들어가서 22.0 V 나옵니다.',
      },
    },
    {
      set: { V1: 220, a: 0.5, ZL: 10 },
      text: {
        en: 'Flip the ratio and it climbs instead. a = 0.5 turns 220 V into 440.0 V. The same device steps down or steps up — the only difference is which side you call the input.',
        ko: '비율을 뒤집으면 반대로 올라갑니다. a = 0.5면 220 V가 440.0 V가 됩니다. 같은 장치가 강압기도 되고 승압기도 됩니다. 어느 쪽을 입력이라 부르느냐의 차이뿐입니다.',
      },
    },
    {
      set: { V1: 220, a: 10, ZL: 10 },
      text: {
        en: 'But it is not free. Hang a 10 Ω load on the secondary and current starts to flow: 2.20 A.',
        ko: '하지만 공짜는 아닙니다. 2차에 10 Ω 부하를 답니다. 전류가 흐르기 시작합니다. 2.20 A.',
      },
    },
    {
      set: { V1: 220, a: 10, ZL: 10 },
      text: {
        en: 'And the primary only draws 0.22 A to cover it. Voltage went down ten times, so current went up ten times. `S1 = S2 = 48.4 W` Whatever goes in comes out. A transformer is a trade, not a gift.',
        ko: '그런데 1차는 그걸 감당하는 데 0.22 A만 끌어갑니다. 전압이 10배 내려갔으니 전류가 10배 올라간 겁니다. `S1 = S2 = 48.4 W` 들어간 만큼 나옵니다. 변압기는 거래이지 선물이 아닙니다.',
      },
    },
    {
      set: { V1: 220, a: 10, ZL: 50 },
      text: {
        en: 'Swap in a 50 Ω load and the secondary current falls to 0.44 A, the power to 9.68 W. The transformer did not decide that. The load did. It only moves power; it never makes any.',
        ko: '부하를 50 Ω으로 바꾸면 2차 전류는 0.44 A, 전력은 9.68 W로 떨어집니다. 변압기가 정한 게 아닙니다. 부하가 정했습니다. 변압기는 전력을 옮길 뿐, 만들어내지는 않습니다.',
      },
    },
    {
      set: { V1: 220, a: 10, ZL: 10 },
      text: {
        en: 'One last gift from the ratio. Looking in from the primary, that 10 Ω load does not look like 10 Ω at all. `Zin = a² × ZL = 10² × 10 = 1000 Ω` Voltage scales by a, current by 1 ÷ a, so impedance scales by a².',
        ko: '권수비가 주는 마지막 선물입니다. 1차에서 들여다보면 저 10 Ω 부하가 10 Ω으로 보이지 않습니다. `Zin = a² × ZL = 10² × 10 = 1000 Ω` 전압은 a배, 전류는 1/a배이니 임피던스는 a²배가 됩니다.',
      },
    },
    {
      set: { V1: 220, a: 10, ZL: 10 },
      text: {
        en: '`V1 ÷ V2 = a` `I1 ÷ I2 = 1 ÷ a` `S1 = S2` Trade voltage for current, leave the power alone. That is why the pylons outside your window run at hundreds of kilovolts — high voltage means low current, and low current means the wires stop burning themselves up. The sliders are yours.',
        ko: '`V1 ÷ V2 = a` `I1 ÷ I2 = 1 ÷ a` `S1 = S2` 전압과 전류를 맞바꾸고 전력은 건드리지 않습니다. 창밖 송전탑이 수십만 볼트로 달리는 이유가 이겁니다. 전압이 높으면 전류가 작고, 전류가 작으면 전선이 자기 자신을 태우지 않습니다. 이제 슬라이더는 당신 것입니다.',
      },
    },
  ],
  sliderParams: [
    { key: 'V1', label: { en: 'Primary Voltage (V1)', ko: '1차 전압 (V1)' },        unit: 'V', min: 20,  max: V1_MAX, step: 10,  defaultValue: 220 },
    { key: 'a',  label: { en: 'Turns Ratio (a = N1 ÷ N2)', ko: '권수비 (a = N1 ÷ N2)' },        min: 0.5, max: 12,     step: 0.5, defaultValue: 10 },
    { key: 'ZL', label: { en: 'Load Impedance (ZL)', ko: '부하 임피던스 (ZL)' },     unit: 'Ω', min: 1,   max: 50,     step: 1,   defaultValue: 10 },
  ],
  Scene: TransformerCoreScene,
  Scene2D: TransformerBars2DScene,
}

export default concept
