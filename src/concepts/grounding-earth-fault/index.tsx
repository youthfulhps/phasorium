import { useRef, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { useLocaleStore } from '../../lib/i18n'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import { FloorCaption, PotentialRuler, useRiseIn } from '../../components/scene/PotentialLandscape'
import { Resistor, setResistorHeat } from '../../components/scene/Resistor'
import type { ConceptMeta } from '../../types/concept'

const VP = 220 // phase voltage, line to neutral [V]
const ROD_D = 0.014 // driven rod diameter [m]
const SAFE_V = 50 // conventional touch-voltage limit [V]
const RCD_MA = 30 // residual-current device rating [mA]

// Slider extremes, used to normalize the visuals
const RHO_MIN = 30, RHO_MAX = 500, L_MIN = 1, L_MAX = 6, RN_MIN = 1

// ── Physics ──────────────────────────────────────────────────────────────────
// Driven-rod ground resistance (Dwight's rod formula, as in IEEE Std 80), rho in ohm-metres,
// L and d in metres:
//     Rg = rho / (2*pi*L) * ln(4L / d)
// When a live conductor touches an earthed enclosure, the equipment electrode Rg and the system
// neutral electrode Rn sit in SERIES across the phase voltage, so the loop is a voltage divider:
//     If = Vp / (Rg + Rn),  Vtouch = If * Rg,  Vn = If * Rn,  Vtouch + Vn = Vp
//     Vtouch / Vp = Rg / (Rg + Rn)
// Verified rows (Vp = 220 V, d = 14 mm), checked against the divider identity:
//   rho = 100, L = 3 -> Rg = 35.83 ohm; Rn = 10 -> If = 4.800 A, Vtouch = 172.0 V, Vn = 48.0 V
//       (172.0 / 220 = 0.7818 = 35.83 / 45.83)
//   Rg = 10 ohm,       Rn = 10 -> If = 11.000 A, Vtouch = 110.0 V, Vn = 110.0 V (0.5000 = 10 / 20)
//   Rg = 5 ohm,        Rn = 10 -> If = 14.667 A, Vtouch =  73.3 V, Vn = 146.7 V (0.3333 = 5 / 15)
//   rho = 100, L = 1 -> Rg = 90.00 ohm;  Rn = 10 -> If = 2.200 A, Vtouch = 198.0 V (0.9000)
//   rho = 300, L = 3 -> Rg = 107.49 ohm; Rn = 10 -> If = 1.873 A, Vtouch = 201.3 V (0.9149)
// Note the counter-intuitive pairing: dropping Rg from 35.83 to 5 ohm TRIPLES the fault current
// yet more than halves the touch voltage. A 30 mA RCD measures the line-minus-neutral imbalance,
// which equals If, and If is amperes in every row above, so it trips whatever Rg happens to be.
function solve({ L_m = 3, rho = 100, Rn = 10 }: Record<string, number>) {
  const Rg = (rho / (2 * Math.PI * L_m)) * Math.log((4 * L_m) / ROD_D)
  const If = VP / (Rg + Rn)
  return { Rg, Rn, If, Vtouch: If * Rg, Vn: If * Rn }
}

const RG_MIN = solve({ L_m: L_MAX, rho: RHO_MIN }).Rg // 5.93 ohm, the best electrode in range
const P_REF = (VP * VP) / (RG_MIN + RN_MIN) // largest power the loop can burn, for the filaments

function useSliderValues() {
  const valuesRef = useRef<Record<string, number>>(useSliderStore.getState().values)
  useEffect(
    () => useSliderStore.subscribe((s) => s.values, (v) => { valuesRef.current = v }),
    []
  )
  return valuesRef
}

const fmtV = (v: number) => `${v.toFixed(1)} V`
const fmtR = (r: number) => `${r.toFixed(r < 100 ? 1 : 0)} Ω`
const fmtI = (a: number) => `${a.toFixed(2)} A`

// Scene labels: identical symbols in both locales, only the words differ
const L10N = {
  en: { rcd: 'RCD', trips: 'trips', limit: `${SAFE_V} V limit`, over: 'above the limit', under: 'within the limit' },
  ko: { rcd: '누전차단기', trips: '차단', limit: `${SAFE_V} V 한계`, over: '한계 초과', under: '한계 이내' },
}

const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const axisClass = 'pointer-events-none whitespace-nowrap text-[11px] text-muted-foreground'

// ── 3D: the earth-fault loop as a potential landscape ────────────────────────
// Floor = the earth mass at 0 V. Walking the loop in the direction of the fault current:
// A (source neutral, -Vn) -> B (phase terminal, +Vtouch) through the source, B -> C along the
// feeder past the RCD to the fault on the enclosure, C -> D down the rod (drops Vtouch across Rg),
// D -> A back through the earth and up the neutral electrode (drops Vn across Rn).
const BASE_Y = -0.2
const HEIGHT_PER_VOLT = 1.8 / VP
const RULER_VOLTS = [-VP, -VP / 2, 0, VP / 2, VP]
const A = new THREE.Vector3(-1.8, 0, 1.1)
const B = new THREE.Vector3(-1.8, 0, -1.1)
const C = new THREE.Vector3(0.8, 0, -1.1)
const D = new THREE.Vector3(0.8, 0, 1.1)
const CORNERS = [A, B, C, D, A]
const SIDES = [2.2, 2.6, 2.2, 2.6] // |AB|, |BC|, |CD|, |DA|
const LOOP_LEN = 9.6
const SRC: [number, number] = [0, 2.2] // the source lifts the loop by Vp
const EL_RG: [number, number] = [5.1, 6.7] // the rod, on side C->D
const EL_RN: [number, number] = [7.9, 8.9] // the neutral electrode, on side D->A
const D_RCD = 3.5 // where the breaker sits on the feeder

const PATH_POINTS = 241
const FLOW_SPEED = 0.25 // scene units per second per amp of sheen travel (on-screen, not drift)
const SHEEN_BANDS = 14

// The person stands directly below the enclosure, so the arm is a vertical line and the touch
// voltage reads as a plain height. Kept left of x = 1.1 to stay clear of the floating card.
const PERSON_X = C.x
const GAP_DX = 0.3 // offset of the dashed measuring plumb from the person
const BAND_X0 = -1.6
const BAND_X1 = 1.3

/** Horizontal position at distance d along the loop. */
function loopXZ(d: number, out: THREE.Vector3) {
  let rest = ((d % LOOP_LEN) + LOOP_LEN) % LOOP_LEN
  for (let k = 0; k < 4; k++) {
    if (rest <= SIDES[k] || k === 3) return out.lerpVectors(CORNERS[k], CORNERS[k + 1], Math.min(rest / SIDES[k], 1))
    rest -= SIDES[k]
  }
  return out
}

const ramp = (d: number, [a, b]: [number, number]) => Math.min(Math.max((d - a) / (b - a), 0), 1)

/** Potential at distance d, taking the earth mass as 0 V. */
function potential(d: number, vTouch: number, vN: number) {
  return -vN + VP * ramp(d, SRC) - vTouch * ramp(d, EL_RG) - vN * ramp(d, EL_RN)
}

const PATH_INIT: [number, number, number][] = Array.from({ length: PATH_POINTS }, () => [0, 0, 0])
const SEG_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]
const FOOTPRINT: [number, number, number][] = CORNERS.map((p) => [p.x, BASE_Y, p.z])
const X_AXIS = new THREE.Vector3(1, 0, 0)
// Scratch objects reused every frame (no allocations in useFrame)
const p0 = new THREE.Vector3()
const p1 = new THREE.Vector3()
const dir = new THREE.Vector3()
const unit = new THREE.Vector3()

/** Seat a resistor tube on its stretch of the landscape and set its filament. */
function placeResistor(
  g: THREE.Group | null,
  tag: THREE.Group | null,
  span: [number, number],
  vTouch: number,
  vN: number,
  k: number,
  R: number,
  rMin: number,
  heat: number,
  tagDx: number,
  tagDy: number
) {
  loopXZ(span[0], p0).setY(BASE_Y + k * potential(span[0], vTouch, vN))
  loopXZ(span[1], p1).setY(BASE_Y + k * potential(span[1], vTouch, vN))
  dir.subVectors(p1, p0)
  if (g) {
    g.position.addVectors(p0, p1).multiplyScalar(0.5)
    g.quaternion.setFromUnitVectors(X_AXIS, unit.copy(dir).normalize())
    const r = 0.14 * Math.sqrt(rMin / Math.max(R, rMin)) + 0.02
    g.scale.set(dir.length() * 0.92, r, r)
  }
  setResistorHeat(g, heat)
  tag?.position.set((p0.x + p1.x) / 2 + tagDx, (p0.y + p1.y) / 2 + tagDy, (p0.z + p1.z) / 2)
}

function GroundingScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const rise = useRiseIn()
  const pathBuf = useMemo(() => new Float32Array(PATH_POINTS * 3), [])
  const segBuf = useMemo(() => new Float32Array(6), [])
  const pathRef = useRef<any>(null)
  const armRef = useRef<any>(null)
  const gapRef = useRef<any>(null)
  const bandRef = useRef<any>(null)
  const resRefs = useRef<(THREE.Group | null)[]>([]) // Rg, Rn
  const encRef = useRef<THREE.Mesh>(null)
  const rcdRef = useRef<THREE.Mesh>(null)
  const faultRef = useRef<THREE.Group>(null)
  const tagRefs = useRef<(THREE.Group | null)[]>([]) // Vp, Rg, Rn, Vtouch, RCD, limit
  const textRefs = useRef<(HTMLSpanElement | null)[]>([]) // Vp, Rg, Rn, Vtouch, RCD, limit, If, sum
  const flow = useRef(0)
  const labelledFor = useRef<unknown>(null)

  useFrame(() => {
    const s = solve(valuesRef.current)
    const k = HEIGHT_PER_VOLT * rise.current
    const yTouch = BASE_Y + k * s.Vtouch

    // The whole loop profile: height is the potential at that point
    for (let i = 0; i < PATH_POINTS; i++) {
      const d = (i / (PATH_POINTS - 1)) * LOOP_LEN
      loopXZ(d, p0)
      pathBuf[i * 3] = p0.x
      pathBuf[i * 3 + 1] = BASE_Y + k * potential(d, s.Vtouch, s.Vn)
      pathBuf[i * 3 + 2] = p0.z
    }
    pathRef.current?.geometry.setPositions(pathBuf)
    flow.current = (s.If * FLOW_SPEED * SHEEN_BANDS) / LOOP_LEN

    // Both filaments share one reference, so equal glow means equal power; their ratio is Vtouch : Vn
    placeResistor(resRefs.current[0], tagRefs.current[1], EL_RG, s.Vtouch, s.Vn, k, s.Rg, RG_MIN,
      Math.sqrt((s.Vtouch * s.If) / P_REF), -0.45, 0)
    placeResistor(resRefs.current[1], tagRefs.current[2], EL_RN, s.Vtouch, s.Vn, k, s.Rn, RN_MIN,
      Math.sqrt((s.Vn * s.If) / P_REF), 0, -0.3)

    // Enclosure and the fault on it both sit at exactly Vtouch above the earth
    encRef.current?.position.set(C.x, yTouch, C.z)
    faultRef.current?.position.set(C.x, yTouch, C.z)
    loopXZ(D_RCD, p0)
    rcdRef.current?.position.set(p0.x, yTouch, p0.z)
    tagRefs.current[4]?.position.set(p0.x, yTouch + 0.28, p0.z)

    // The person: feet on the earth, hand on the enclosure right above. The arm is vertical, so
    // the gap it bridges reads as a height rather than a diagonal.
    segBuf[0] = PERSON_X; segBuf[1] = BASE_Y + 0.44; segBuf[2] = C.z
    segBuf[3] = PERSON_X; segBuf[4] = yTouch; segBuf[5] = C.z
    armRef.current?.geometry.setPositions(segBuf)
    // Dashed plumb beside it, measuring the full Vtouch from the earth up
    segBuf[0] = PERSON_X + GAP_DX; segBuf[1] = BASE_Y; segBuf[2] = C.z
    segBuf[3] = PERSON_X + GAP_DX; segBuf[4] = yTouch; segBuf[5] = C.z
    gapRef.current?.geometry.setPositions(segBuf)
    gapRef.current?.computeLineDistances()
    tagRefs.current[3]?.position.set(PERSON_X, BASE_Y - 0.34, C.z)

    // The safety band: crossing it is the whole point of lowering Rg
    const yBand = BASE_Y + k * SAFE_V
    segBuf[0] = BAND_X0; segBuf[1] = yBand; segBuf[2] = C.z
    segBuf[3] = BAND_X1; segBuf[4] = yBand; segBuf[5] = C.z
    bandRef.current?.geometry.setPositions(segBuf)
    bandRef.current?.computeLineDistances()
    tagRefs.current[5]?.position.set(BAND_X0 - 0.05, yBand + 0.14, C.z)

    loopXZ((SRC[0] + SRC[1]) / 2, p0)
    tagRefs.current[0]?.position.set(p0.x - 0.35, BASE_Y + (k * (s.Vtouch - s.Vn)) / 2, p0.z)

    // Readouts only change with the sliders or the locale
    if (labelledFor.current !== valuesRef.current && textRefs.current[7]) {
      labelledFor.current = valuesRef.current
      const t = L10N[locale]
      const texts = [
        `Vp ${VP} V`,
        `Rg ${fmtR(s.Rg)}`,
        `Rn ${fmtR(s.Rn)}`,
        `Vtouch ${fmtV(s.Vtouch)} · ${s.Vtouch > SAFE_V ? t.over : t.under}`,
        `${t.rcd} ${RCD_MA} mA · ${t.trips} · If = ${Math.round(s.If / (RCD_MA * 1e-3))}× IΔn`,
        t.limit,
        `If ${fmtI(s.If)}`,
        `Vtouch + Vn = ${fmtV(s.Vtouch)} + ${fmtV(s.Vn)} = ${VP} V`,
      ]
      texts.forEach((text, i) => { if (textRefs.current[i]) textRefs.current[i]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />

      <Line points={FOOTPRINT} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      <FloorCaption position={[D.x + 0.25, BASE_Y, D.z + 0.45]} />
      <PotentialRuler
        x={B.x - 0.4}
        z={B.z - 0.3}
        baseY={BASE_Y}
        volts={RULER_VOLTS}
        unitsPerVolt={() => HEIGHT_PER_VOLT * rise.current}
      />

      {/* The fault current itself: one liquid loop whose sheen flows at a speed proportional to If */}
      <TrailLine
        ref={pathRef}
        points={PATH_INIT}
        lineWidth={4}
        color={c.current}
        fade={false}
        flowSpeed={() => flow.current}
      />

      {/* Rg on the rod, Rn on the neutral electrode: thinner tube means larger resistance */}
      <Resistor ref={(g) => { resRefs.current[0] = g }} />
      <Resistor ref={(g) => { resRefs.current[1] = g }} />

      {/* The earthed enclosure, floating at Vtouch above the earth */}
      <mesh ref={encRef}>
        <boxGeometry args={[0.48, 0.3, 0.38]} />
        <meshBasicMaterial color={c.inductance} transparent opacity={0.35} depthWrite={false} />
      </mesh>
      <Marker ref={faultRef} color={c.marker} size={0.055} />

      {/* The breaker on the feeder */}
      <mesh ref={rcdRef}>
        <boxGeometry args={[0.3, 0.22, 0.26]} />
        <meshBasicMaterial color={c.resistance} transparent opacity={0.5} depthWrite={false} />
      </mesh>

      {/* The person: body on the earth, arm bridging the touch voltage */}
      <mesh position={[PERSON_X, BASE_Y + 0.18, C.z]}>
        <boxGeometry args={[0.12, 0.36, 0.12]} />
        <meshBasicMaterial color={c.resistance} />
      </mesh>
      <mesh position={[PERSON_X, BASE_Y + 0.46, C.z]}>
        <sphereGeometry args={[0.085, 20, 20]} />
        <meshBasicMaterial color={c.resistance} />
      </mesh>
      <Line ref={armRef} points={SEG_INIT} lineWidth={2.5} color={c.voltage} />
      <Line ref={gapRef} points={SEG_INIT} lineWidth={1.5} color={c.guide} dashed dashSize={0.07} gapSize={0.05} />
      <Line ref={bandRef} points={SEG_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.1} gapSize={0.08} />

      {[0, 1, 2, 3, 4, 5].map((i) => (
        <group key={i} ref={(g) => { tagRefs.current[i] = g }}>
          <Html center>
            <span ref={(el) => { textRefs.current[i] = el }} className={labelClass} />
          </Html>
        </group>
      ))}
      <Html position={[-0.5, BASE_Y - 0.38, D.z + 0.35]} center>
        <span ref={(el) => { textRefs.current[6] = el }} className={labelClass} style={{ color: c.current }} />
      </Html>
      <Html position={[-0.4, BASE_Y + 1.95, 0]} center>
        <span ref={(el) => { textRefs.current[7] = el }} className={labelClass} />
      </Html>
    </>
  )
}

// ── 2D: Vtouch and If against Rg. One rises, one falls, and they cross. ──────
const RG_LO = 1
const RG_HI = 60
const CURVE_PTS = 121
const X0 = -3.3
const Y0 = -1.6
const PLOT_W = 6.4
const PLOT_H = 3.0
const X_PER_OHM = PLOT_W / (RG_HI - RG_LO)
const Y_PER_VOLT = PLOT_H / VP
const REF_RG = [5, 10, 35.829] // the three verified rows
const AXES: [number, number, number][] = [
  [X0, Y0 + PLOT_H + 0.3, 0], [X0, Y0, 0], [X0 + PLOT_W + 0.3, Y0, 0],
]
const BAND_TOP = Y0 + SAFE_V * Y_PER_VOLT
const BAND_LINE: [number, number, number][] = [[X0, BAND_TOP, 0], [X0 + PLOT_W, BAND_TOP, 0]]
const CURVE_INIT: [number, number, number][] = Array.from({ length: CURVE_PTS }, () => [X0, Y0, 0])
const xOf = (Rg: number) => X0 + (Rg - RG_LO) * X_PER_OHM

function Grounding2DScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const vBuf = useMemo(() => new Float32Array(CURVE_PTS * 3), [])
  const iBuf = useMemo(() => new Float32Array(CURVE_PTS * 3), [])
  const vRef = useRef<any>(null)
  const iRef = useRef<any>(null)
  const opRef = useRef<THREE.Group>(null)
  const refMarks = useRef<(THREE.Group | null)[]>([])
  const refTags = useRef<(THREE.Group | null)[]>([])
  const opTag = useRef<THREE.Group>(null)
  const textRefs = useRef<(HTMLSpanElement | null)[]>([]) // ref0..2, operating point, If axis, limit
  const labelledFor = useRef<unknown>(null)

  useFrame(() => {
    const s = solve(valuesRef.current)
    const ifMax = VP / (RG_LO + s.Rn) // full scale of the If axis, so both curves fill the plot

    for (let i = 0; i < CURVE_PTS; i++) {
      const Rg = RG_LO + (i / (CURVE_PTS - 1)) * (RG_HI - RG_LO)
      const x = xOf(Rg)
      vBuf[i * 3] = x
      vBuf[i * 3 + 1] = Y0 + ((VP * Rg) / (Rg + s.Rn)) * Y_PER_VOLT
      vBuf[i * 3 + 2] = 0
      iBuf[i * 3] = x
      iBuf[i * 3 + 1] = Y0 + (VP / (Rg + s.Rn) / ifMax) * PLOT_H
      iBuf[i * 3 + 2] = 0
    }
    vRef.current?.geometry.setPositions(vBuf)
    iRef.current?.geometry.setPositions(iBuf)

    for (let m = 0; m < REF_RG.length; m++) {
      const Rg = REF_RG[m]
      const y = Y0 + ((VP * Rg) / (Rg + s.Rn)) * Y_PER_VOLT
      refMarks.current[m]?.position.set(xOf(Rg), y, 0.01)
      refTags.current[m]?.position.set(xOf(Rg) + 0.1, y - 0.22, 0)
    }

    const opY = Y0 + s.Vtouch * Y_PER_VOLT
    const opX = xOf(Math.min(Math.max(s.Rg, RG_LO), RG_HI))
    opRef.current?.position.set(opX, opY, 0.02)
    opTag.current?.position.set(opX, opY + 0.26, 0)

    if (labelledFor.current !== valuesRef.current && textRefs.current[5]) {
      labelledFor.current = valuesRef.current
      const t = L10N[locale]
      const texts = [
        ...REF_RG.map((Rg) => `${fmtV((VP * Rg) / (Rg + s.Rn))}`),
        `Rg ${fmtR(s.Rg)} · Vtouch ${fmtV(s.Vtouch)} · If ${fmtI(s.If)}`,
        `If (A) 0–${ifMax.toFixed(1)}`,
        t.limit,
      ]
      texts.forEach((text, i) => { if (textRefs.current[i]) textRefs.current[i]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  return (
    <>
      <Line points={AXES} lineWidth={1} color={c.guide} />
      {/* The band below the touch-voltage limit: operating points inside it are the safe ones */}
      <mesh position={[X0 + PLOT_W / 2, (Y0 + BAND_TOP) / 2, -0.02]}>
        <planeGeometry args={[PLOT_W, BAND_TOP - Y0]} />
        <meshBasicMaterial color={c.guide} transparent opacity={0.25} depthWrite={false} />
      </mesh>
      <Line points={BAND_LINE} lineWidth={1} color={c.guide} dashed dashSize={0.1} gapSize={0.08} />

      <Line ref={vRef} points={CURVE_INIT} lineWidth={3} color={c.voltage} />
      <Line ref={iRef} points={CURVE_INIT} lineWidth={2.5} color={c.current} />

      {REF_RG.map((Rg, m) => (
        <group key={Rg}>
          <Marker ref={(g) => { refMarks.current[m] = g }} color={c.marker} size={0.05} />
          <group ref={(g) => { refTags.current[m] = g }}>
            <Html center>
              <span ref={(el) => { textRefs.current[m] = el }} className={labelClass} />
            </Html>
          </group>
        </group>
      ))}
      <Marker ref={opRef} color={c.marker} size={0.075} />
      <group ref={opTag}>
        <Html center>
          <span ref={(el) => { textRefs.current[3] = el }} className={labelClass} />
        </Html>
      </group>

      <Html position={[X0 + PLOT_W + 0.3, Y0 - 0.22, 0]} center>
        <span className={axisClass}>Rg (Ω)</span>
      </Html>
      <Html position={[X0 + 0.55, Y0 + PLOT_H + 0.3, 0]} center>
        <span className={axisClass}>Vtouch (V) 0–{VP}</span>
      </Html>
      <Html position={[X0 + PLOT_W - 0.5, Y0 + PLOT_H + 0.3, 0]} center>
        <span ref={(el) => { textRefs.current[4] = el }} className={axisClass} style={{ color: c.current }} />
      </Html>
      <Html position={[X0 + PLOT_W - 0.4, BAND_TOP + 0.16, 0]} center>
        <span ref={(el) => { textRefs.current[5] = el }} className={axisClass} />
      </Html>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'grounding-earth-fault',
  title: { en: 'Grounding & Earth-Fault Protection', ko: '접지와 지락보호 (접촉전압과 누전차단기)' },
  subject: 'installations',
  order: 2,
  description: {
    en: `## Grounding and Earth-Fault Protection

**A fault makes the two earthing resistances a voltage divider, and the share that lands on the enclosure is exactly what a person feels.**

Height = potential, with the floor being the earth's own 0 V.
- Cyan liquid line: the fault current If — out of the source, past the breaker, into the fault, down the rod, through the earth, up the neutral electrode; its sheen flows faster when If is larger
- Glass tube on the rod: Rg, the equipment's own ground resistance (thinner tube = larger Rg)
- Glass tube in the earth return: Rn, the system neutral grounding resistance
- Amber filaments: the power each resistance burns, on one shared scale, so their glow splits like Vtouch to Vn
- The figure standing on the earth: the height its arm bridges is Vtouch
- Dashed horizontal: the 50 V touch-voltage limit
- Dashed outline on the floor: the same installation drawn flat, all at 0 V

**Rod resistance**: Rg = ρ ÷ (2π L) × ln(4L ÷ d)

**Fault loop**: If = Vp ÷ (Rg + Rn), then Vtouch = If × Rg and Vn = If × Rn

The two heights always add back up to the source, Vtouch + Vn = Vp, which is the divider rule Vtouch ÷ Vp = Rg ÷ (Rg + Rn).

Lengthen the rod or wet the soil: Rg falls, the enclosure sinks toward the earth, and the flow speeds up. At Rg = 35.8 Ω the enclosure sits at 172 V; at Rg = 5 Ω it sits at 73 V even though If is three times larger. More current, far less danger — current is what the breaker reads, touch voltage is what the body feels. That is the entire reason for grounding.

The RCD compares the line and neutral currents and opens on a ${RCD_MA} mA difference. That difference is If, which is amperes in every setting here, so the breaker trips however bad Rg is; Rg only decides how high the enclosure rises during the fraction of a second before it does.

In 2D: Vtouch rising and If falling against Rg, crossing, with markers at 5, 10, and 35.8 Ω and the 50 V band shaded. If has its own full scale, printed on its axis.`,
    ko: `## 접지와 지락보호

**지락이 일어나면 두 접지저항이 전압 분배기가 되고, 그중 외함에 걸리는 몫이 사람이 실제로 느끼는 전압입니다.**

높이 = 전위이고, 바닥이 대지 자신의 0 V입니다.
- 시안 액체 선: 지락전류 If — 전원에서 차단기를 지나 지락점으로, 접지극을 타고 내려가 대지를 건너 중성점 접지극으로 돌아옵니다. If가 클수록 광택이 빠르게 흐릅니다
- 접지극의 유리관: Rg, 기기 자체의 접지저항 (관이 가늘수록 Rg가 큼)
- 대지 귀로의 유리관: Rn, 계통 중성점 접지저항
- 앰버 필라멘트: 각 저항이 소비하는 전력. 같은 기준으로 그렸으므로 두 빛의 비가 Vtouch : Vn입니다
- 대지에 서 있는 사람: 팔이 건너는 높이가 Vtouch입니다
- 점선 수평선: 접촉전압 한계 50 V
- 바닥의 점선 윤곽: 같은 설비를 평면으로 그린 모습, 전체가 0 V입니다

**접지극 저항**: Rg = ρ ÷ (2π L) × ln(4L ÷ d)

**지락 루프**: If = Vp ÷ (Rg + Rn), 그리고 Vtouch = If × Rg, Vn = If × Rn

두 높이는 항상 전원으로 되돌아와 합이 맞습니다. Vtouch + Vn = Vp이고, 이는 분배 법칙 Vtouch ÷ Vp = Rg ÷ (Rg + Rn)과 같습니다.

접지극을 길게 박거나 땅이 젖으면 Rg가 작아지고, 외함이 대지 쪽으로 내려앉으며 흐름은 빨라집니다. Rg = 35.8 Ω에서 외함은 172 V에 있지만, Rg = 5 Ω에서는 If가 세 배나 큰데도 73 V로 내려옵니다. 전류는 늘고 위험은 크게 줄어드는 것입니다. 전류는 차단기가 읽는 값이고 접촉전압은 몸이 받는 값이니까요. 접지를 하는 이유가 바로 이것입니다.

누전차단기는 전선과 중성선의 전류를 비교해 그 차이가 ${RCD_MA} mA가 되면 차단합니다. 그 차이가 곧 If이고 여기서는 어느 설정에서도 암페어 단위이므로, Rg가 아무리 나빠도 차단기는 동작합니다. Rg는 차단되기까지의 짧은 순간에 외함이 얼마나 높이 올라가는지만 결정합니다.

2D: Rg에 대해 Vtouch는 올라가고 If는 내려가며 서로 교차합니다. 5, 10, 35.8 Ω에 마커를 찍고 50 V 띠를 음영으로 표시했습니다. If는 자기 전체 범위를 축에 적어 두었습니다.`,
  },
  sliderParams: [
    { key: 'L_m', label: { en: 'Rod length (L)', ko: '접지극 길이 (L)' }, unit: 'm', min: L_MIN, max: L_MAX, step: 0.1, defaultValue: 3 },
    { key: 'rho', label: { en: 'Soil resistivity (ρ)', ko: '대지 고유저항 (ρ)' }, unit: 'Ω·m', min: RHO_MIN, max: RHO_MAX, step: 10, defaultValue: 100 },
    { key: 'Rn', label: { en: 'Neutral ground resistance (Rn)', ko: '중성점 접지저항 (Rn)' }, unit: 'Ω', min: RN_MIN, max: 50, step: 1, defaultValue: 10 },
  ],
  Scene: GroundingScene,
  Scene2D: Grounding2DScene,
}

export default concept
