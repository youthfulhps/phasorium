import { useRef, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { useLocaleStore } from '../../lib/i18n'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import { FloorCaption, PotentialRuler, useRiseIn } from '../../components/scene/PotentialLandscape'
import { Resistor, setResistorHeat } from '../../components/scene/Resistor'
import type { ConceptMeta } from '../../types/concept'

// Slider ranges (also used to normalize the visuals)
const V_MAX = 24 // [V]
const RTH_MIN = 2 // [Ω]
const RL_MIN = 0.5 // [Ω]
const RL_MAX = 30 // [Ω]
const MATCH_TOL = 0.26 // |RL − Rth| below this reads as matched [Ω]

// Pure physics: a Thevenin source (Vth behind Rth) feeding a load RL.
// I = Vth / (Rth + RL), PL = I²·RL, P_int = I²·Rth, η = PL / (Vth·I) = RL / (Rth + RL).
// Worked example (verified with node): Vth = 12 V, Rth = 6 Ω
//   RL = 6 Ω  → I = 1.0000 A, PL = 6.0000 W, P_int = 6.0000 W, P_total = 12 W, η = 50.00 %
//   RL = 18 Ω → I = 0.5000 A, PL = 4.5000 W, P_int = 1.5000 W, P_total =  6 W, η = 75.00 %
//   RL = 2 Ω  → I = 1.5000 A, PL = 4.5000 W, P_int = 13.500 W, P_total = 18 W, η = 25.00 %
// Peak check: Vth² / (4·Rth) = 144 / 24 = 6.0000 W, reached exactly at RL = Rth = 6 Ω.

/** Load power alone, without building an object — the 2D sweep calls this once per sample. */
function loadPower(Vth: number, Rth: number, RL: number) {
  const i = Vth / (Rth + RL)
  return i * i * RL
}

/** The best any load can ever get from this source: PL at RL = Rth. */
const peakPower = (Vth: number, Rth: number) => (Vth * Vth) / (4 * Rth)

function useSliderValues() {
  const valuesRef = useRef<Record<string, number>>(useSliderStore.getState().values)
  useEffect(
    () => useSliderStore.subscribe((s) => s.values, (v) => { valuesRef.current = v }),
    []
  )
  return valuesRef
}

const fmtV = (x: number) => `${x.toFixed(2)} V`
const fmtW = (x: number) => `${x.toFixed(2)} W`
const fmtR = (x: number) => `${x.toFixed(1)} Ω`
const fmtI = (x: number) => `${x.toFixed(2)} A`
const fmtPct = (x: number) => `${(x * 100).toFixed(0)} %`
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const axisClass = 'pointer-events-none whitespace-nowrap text-[11px] text-muted-foreground'

// ── 3D: the source and its load as one potential landscape ──
// Walked in the direction of conventional current: source ramp lifts charge to Vth, Rth drops it to
// VL, the load drops the rest to 0. The height left over for the load IS the load voltage.
const BASE_Y = -0.8
const HEIGHT_PER_VOLT = 1.8 / V_MAX
const RULER_VOLTS = [0, 12, 24]
const SEG_PTS = 24

type XZ = [number, number]
const LOOP_XZ: XZ[] = [
  [-2.0, 1.2], // 0  source terminal on the floor, 0 V
  [-2.0, -1.2], // 1  top of the source ramp, Vth
  [-1.4, -1.2], // 2  wire at Vth
  [-0.2, -1.2], // 3  bottom of Rth, VL
  [1.2, -1.2], // 4  wire at VL
  [1.2, 1.2], // 5  bottom of RL, 0 V
  [-2.0, 1.2], // 6  return wire, closes the loop
]
const loopV = new Float32Array(LOOP_XZ.length)
const LOOP_PTS = (LOOP_XZ.length - 1) * SEG_PTS + 1
const LOOP_INIT: [number, number, number][] = Array.from({ length: LOOP_PTS }, () => [0, 0, 0])
const FOOTPRINT: [number, number, number][] = LOOP_XZ.map(([x, z]) => [x, BASE_Y, z])
const GUIDE_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]

const RTH_A = LOOP_XZ[2]
const RTH_B = LOOP_XZ[3]
const RL_A = LOOP_XZ[4]
const RL_B = LOOP_XZ[5]

let loopLen = 0
for (let s = 0; s < LOOP_XZ.length - 1; s++) {
  loopLen += Math.hypot(LOOP_XZ[s + 1][0] - LOOP_XZ[s][0], LOOP_XZ[s + 1][1] - LOOP_XZ[s][1])
}
const FLOW_SPEED = 0.35 // sheen bands per second per amp
const SHEEN_BANDS = 14 // bands along the loop in TrailLine's shader

/** Fill a flat XZ path, raising every sample to its potential. */
function fillPath(buf: Float32Array, xz: XZ[], volts: Float32Array, k: number) {
  let n = 0
  for (let s = 0; s < xz.length - 1; s++) {
    const last = s === xz.length - 2
    for (let j = 0; j < SEG_PTS + (last ? 1 : 0); j++) {
      const u = j / SEG_PTS
      buf[n++] = xz[s][0] + (xz[s + 1][0] - xz[s][0]) * u
      buf[n++] = BASE_Y + k * (volts[s] + (volts[s + 1] - volts[s]) * u)
      buf[n++] = xz[s][1] + (xz[s + 1][1] - xz[s][1]) * u
    }
  }
}

const X_AXIS = new THREE.Vector3(1, 0, 0)
const p0 = new THREE.Vector3()
const p1 = new THREE.Vector3()
const dir = new THREE.Vector3()
const unit = new THREE.Vector3()

function placeResistor(
  g: THREE.Group | null, tag: THREE.Group | null,
  a: XZ, va: number, b: XZ, vb: number, k: number, R: number, heat: number, tagDx: number
) {
  p0.set(a[0], BASE_Y + k * va, a[1])
  p1.set(b[0], BASE_Y + k * vb, b[1])
  dir.subVectors(p1, p0)
  if (g) {
    g.position.addVectors(p0, p1).multiplyScalar(0.5)
    g.quaternion.setFromUnitVectors(X_AXIS, unit.copy(dir).normalize())
    const r = 0.15 * Math.sqrt(RTH_MIN / Math.max(R, 1)) + 0.02
    g.scale.set(dir.length() * 0.9, r, r)
  }
  setResistorHeat(g, heat)
  tag?.position.set((p0.x + p1.x) / 2 + tagDx, (p0.y + p1.y) / 2 + 0.3, (p0.z + p1.z) / 2)
}

function MaxPowerScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const rise = useRiseIn()
  const loopBuf = useMemo(() => new Float32Array(LOOP_PTS * 3), [])
  const guideBuf = useMemo(() => new Float32Array(6), [])
  const loopRef = useRef<any>(null)
  const dropSrc = useRef<any>(null)
  const dropLoad = useRef<any>(null)
  const levelRef = useRef<any>(null)
  const resRefs = useRef<(THREE.Group | null)[]>([]) // Rth, RL
  const tagRefs = useRef<(THREE.Group | null)[]>([]) // Vth, Rth, RL, VL
  const textRefs = useRef<(HTMLSpanElement | null)[]>([]) // Vth, Rth, RL, VL, I, summary
  const flow = useRef(0)
  const labelledFor = useRef<unknown>(null)

  useFrame(() => {
    const { Vth = 12, Rth = 6, RL = 18 } = valuesRef.current
    const I = Vth / (Rth + RL)
    const VL = I * RL
    const PL = I * I * RL
    const Pint = I * I * Rth
    const eta = RL / (Rth + RL)
    const k = HEIGHT_PER_VOLT * rise.current

    loopV[0] = 0
    loopV[1] = loopV[2] = Vth
    loopV[3] = loopV[4] = VL
    loopV[5] = loopV[6] = 0

    fillPath(loopBuf, LOOP_XZ, loopV, k)
    loopRef.current?.geometry.setPositions(loopBuf)
    flow.current = (I * FLOW_SPEED * SHEEN_BANDS) / loopLen

    // Vertical guides so height reads as height: one at the top of the source ramp, one at VL
    guideBuf[0] = LOOP_XZ[1][0]; guideBuf[1] = BASE_Y; guideBuf[2] = LOOP_XZ[1][1]
    guideBuf[3] = LOOP_XZ[1][0]; guideBuf[4] = BASE_Y + k * Vth; guideBuf[5] = LOOP_XZ[1][1]
    dropSrc.current?.geometry.setPositions(guideBuf)
    dropSrc.current?.computeLineDistances()
    guideBuf[0] = RL_A[0]; guideBuf[1] = BASE_Y; guideBuf[2] = RL_A[1]
    guideBuf[3] = RL_A[0]; guideBuf[4] = BASE_Y + k * VL; guideBuf[5] = RL_A[1]
    dropLoad.current?.geometry.setPositions(guideBuf)
    dropLoad.current?.computeLineDistances()

    // Level line at VL, run back under the Vth rail: the gap above it is what Rth eats
    guideBuf[0] = LOOP_XZ[1][0]; guideBuf[1] = BASE_Y + k * VL; guideBuf[2] = LOOP_XZ[1][1]
    guideBuf[3] = RL_A[0]; guideBuf[4] = BASE_Y + k * VL; guideBuf[5] = RL_A[1]
    levelRef.current?.geometry.setPositions(guideBuf)
    levelRef.current?.computeLineDistances()

    // Both filaments are scaled against the same reference, so equal glow means equal power
    const ref = peakPower(Vth, Rth)
    placeResistor(resRefs.current[0], tagRefs.current[1], RTH_A, Vth, RTH_B, VL, k, Rth, Math.sqrt(Pint / ref) * 0.8, 0)
    placeResistor(resRefs.current[1], tagRefs.current[2], RL_A, VL, RL_B, 0, k, RL, Math.sqrt(PL / ref) * 0.8, 0.55)

    tagRefs.current[0]?.position.set(LOOP_XZ[1][0] - 0.3, BASE_Y + (k * Vth) / 2, 0)
    tagRefs.current[3]?.position.set(RL_A[0] - 0.45, BASE_Y + k * VL + 0.2, RL_A[1])

    if (labelledFor.current !== valuesRef.current && textRefs.current[5]) {
      labelledFor.current = valuesRef.current
      const ko = locale === 'ko'
      const matched = Math.abs(RL - Rth) < MATCH_TOL
      const texts = [
        `Vth ${fmtV(Vth)}`,
        `Rth ${fmtR(Rth)} · P_int ${fmtW(Pint)}`,
        `RL ${fmtR(RL)} · PL ${fmtW(PL)}`,
        `VL ${fmtV(VL)}`,
        `I ${fmtI(I)}`,
        matched
          ? `${ko ? '정합' : 'Matched'} · RL = Rth · PL,max ${fmtW(ref)} · η ${fmtPct(eta)}`
          : `η ${fmtPct(eta)} · ${ko ? '최대 전력은' : 'peak PL at'} RL = ${fmtR(Rth)} (${fmtW(ref)})`,
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />

      <Line points={FOOTPRINT} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      <FloorCaption position={[-0.3, BASE_Y, 1.55]} />
      <PotentialRuler
        x={-2.0}
        z={-1.75}
        baseY={BASE_Y}
        volts={RULER_VOLTS}
        unitsPerVolt={() => HEIGHT_PER_VOLT * rise.current}
      />

      {/* The current itself: one liquid loop whose height is the potential, flowing at a speed ∝ I */}
      <TrailLine
        ref={loopRef}
        points={LOOP_INIT}
        lineWidth={4}
        color={c.current}
        fade={false}
        flowSpeed={() => flow.current}
      />
      <Line ref={dropSrc} points={GUIDE_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.06} gapSize={0.05} />
      <Line ref={dropLoad} points={GUIDE_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.06} gapSize={0.05} />
      <Line ref={levelRef} points={GUIDE_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />

      {[0, 1].map((r) => (
        <Resistor key={r} ref={(g) => { resRefs.current[r] = g }} />
      ))}

      {[0, 1, 2, 3].map((t) => (
        <group key={t} ref={(g) => { tagRefs.current[t] = g }}>
          <Html center>
            <span
              ref={(el) => { textRefs.current[t] = el }}
              className={labelClass}
              style={t === 1 || t === 2 ? { color: c.power } : undefined}
            />
          </Html>
        </group>
      ))}
      <Html position={[-0.3, BASE_Y - 0.22, 1.2]} center>
        <span
          ref={(el) => { textRefs.current[4] = el }}
          className="pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums"
          style={{ color: c.current }}
        />
      </Html>
      <Html position={[-0.3, BASE_Y - 0.5, 1.2]} center>
        <span ref={(el) => { textRefs.current[5] = el }} className={labelClass} />
      </Html>
    </>
  )
}

// ── 2D: sweep RL. PL peaks exactly at RL = Rth, where efficiency is still only half ──
const X0 = -3.2
const X_SPAN = 6.4
const Y0 = -1.7
const Y_SPAN = 3.2 // 1.0 normalized → y = 1.5
const CURVE_PTS = 161
const CURVE_INIT: [number, number, number][] = Array.from({ length: CURVE_PTS }, () => [0, 0, 0])
const AXIS_INIT: [number, number, number][] = [[X0, Y0, 0], [X0 + X_SPAN, Y0, 0]]
const TOP_INIT: [number, number, number][] = [[X0, Y0 + Y_SPAN, 0], [X0 + X_SPAN, Y0 + Y_SPAN, 0]]
const HALF_INIT: [number, number, number][] = [[X0, Y0 + Y_SPAN / 2, 0], [X0 + X_SPAN, Y0 + Y_SPAN / 2, 0]]
const MARK_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]
const xOf = (RL: number) => X0 + (RL / RL_MAX) * X_SPAN

function MaxPower2DScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const powerBuf = useMemo(() => new Float32Array(CURVE_PTS * 3), [])
  const etaBuf = useMemo(() => new Float32Array(CURVE_PTS * 3), [])
  const markBuf = useMemo(() => new Float32Array(6), [])
  const powerRef = useRef<any>(null)
  const etaRef = useRef<any>(null)
  const matchRef = useRef<any>(null)
  const powerDot = useRef<THREE.Group>(null)
  const etaDot = useRef<THREE.Group>(null)
  const tagRefs = useRef<(THREE.Group | null)[]>([]) // readout, matched note, PL, η
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<unknown>(null)

  useFrame(() => {
    const { Vth = 12, Rth = 6, RL = 18 } = valuesRef.current
    const ref = peakPower(Vth, Rth)

    for (let i = 0; i < CURVE_PTS; i++) {
      const rl = (i / (CURVE_PTS - 1)) * RL_MAX
      const x = X0 + (i / (CURVE_PTS - 1)) * X_SPAN
      powerBuf[i * 3] = x
      powerBuf[i * 3 + 1] = Y0 + (loadPower(Vth, Rth, rl) / ref) * Y_SPAN
      powerBuf[i * 3 + 2] = 0
      etaBuf[i * 3] = x
      etaBuf[i * 3 + 1] = Y0 + (rl / (Rth + rl)) * Y_SPAN
      etaBuf[i * 3 + 2] = 0
    }
    powerRef.current?.geometry.setPositions(powerBuf)
    etaRef.current?.geometry.setPositions(etaBuf)

    // Vertical line at the match, where PL tops out and η is exactly one half
    const xm = xOf(Rth)
    markBuf[0] = xm; markBuf[1] = Y0; markBuf[2] = 0
    markBuf[3] = xm; markBuf[4] = Y0 + Y_SPAN; markBuf[5] = 0
    matchRef.current?.geometry.setPositions(markBuf)
    matchRef.current?.computeLineDistances()

    const PL = loadPower(Vth, Rth, RL)
    const eta = RL / (Rth + RL)
    const x = xOf(RL)
    const yP = Y0 + (PL / ref) * Y_SPAN
    const yE = Y0 + eta * Y_SPAN
    powerDot.current?.position.set(x, yP, 0.01)
    etaDot.current?.position.set(x, yE, 0.01)
    tagRefs.current[0]?.position.set(x, yP + 0.3, 0)
    tagRefs.current[1]?.position.set(xm, Y0 - 0.55, 0)
    tagRefs.current[2]?.position.set(xm, Y0 + Y_SPAN + 0.22, 0)
    tagRefs.current[3]?.position.set(X0 + X_SPAN - 0.2, Y0 + (RL_MAX / (Rth + RL_MAX)) * Y_SPAN + 0.24, 0)

    if (labelledFor.current !== valuesRef.current && textRefs.current[1]) {
      labelledFor.current = valuesRef.current
      const ko = locale === 'ko'
      const texts = [
        `RL ${fmtR(RL)} · PL ${fmtW(PL)} · η ${fmtPct(eta)}`,
        `${ko ? '정합' : 'matched'} RL = Rth = ${fmtR(Rth)} · PL,max = Vth² ÷ 4Rth = ${fmtW(ref)} · η 50 %`,
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  return (
    <>
      <Line points={AXIS_INIT} lineWidth={1} color={c.guide} />
      <Line points={TOP_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      <Line points={HALF_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      <Line ref={matchRef} points={MARK_INIT} lineWidth={1.5} color={c.guide} dashed dashSize={0.1} gapSize={0.07} />
      <Line ref={etaRef} points={CURVE_INIT} lineWidth={2} color={c.voltage} />
      <Line ref={powerRef} points={CURVE_INIT} lineWidth={3} color={c.power} />
      <Marker ref={powerDot} color={c.marker} size={0.07} />
      <Marker ref={etaDot} color={c.marker} size={0.055} />

      {[0, 1].map((t) => (
        <group key={t} ref={(g) => { tagRefs.current[t] = g }}>
          <Html center>
            <span ref={(el) => { textRefs.current[t] = el }} className={labelClass} />
          </Html>
        </group>
      ))}
      <group ref={(g) => { tagRefs.current[2] = g }}>
        <Html center>
          <span className={labelClass} style={{ color: c.power }}>PL</span>
        </Html>
      </group>
      <group ref={(g) => { tagRefs.current[3] = g }}>
        <Html center>
          <span className={labelClass}>η</span>
        </Html>
      </group>

      <Html position={[X0 + X_SPAN + 0.3, Y0, 0]} center>
        <span className={axisClass}>RL (Ω)</span>
      </Html>
      <Html position={[X0 - 0.45, Y0 + Y_SPAN, 0]} center>
        <span className={axisClass}>PL,max</span>
      </Html>
      <Html position={[X0 - 0.45, Y0 + Y_SPAN / 2, 0]} center>
        <span className={axisClass}>η 50 %</span>
      </Html>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'max-power-transfer',
  title: { en: 'Maximum Power Transfer', ko: '최대 전력 전달' },
  subject: 'circuits',
  order: 9,
  description: {
    en: `## Maximum Power Transfer

**A source hands its load the most power when the load matches the source's own internal resistance — and at that very point it burns just as much inside itself.**

The circuit is drawn as a potential landscape: height = electric potential, so the source voltage splits between the two resistors and the height left for the load is VL.
- Left ramp: the source lifts charge by Vth; the dashed level line marks VL, so the gap above it is what Rth takes
- Cyan liquid line: the current I; the faster its sheen flows, the larger I
- Two glass tubes: Rth inside the source, RL the load; thinner means larger R
- Amber filaments: the power each one turns into heat. At RL = Rth the two glow alike, because they burn exactly the same amount

**Current**: I = Vth ÷ (Rth + RL)

**Load power**: PL = I² × RL, **inside the source**: P_int = I² × Rth

**Efficiency**: η = PL ÷ (Vth × I) = RL ÷ (Rth + RL) = VL ÷ Vth

**Matched**: RL = Rth gives the largest PL, PL,max = Vth² ÷ (4 × Rth), with η = 50 %

Leave Vth at 12 V and Rth at 6 Ω, then slide RL down to 6 Ω: PL peaks at 6.00 W and η is 50 %. Push RL up to 18 Ω and efficiency climbs to 75 %, but the load now gets only 4.50 W. Maximum power and maximum efficiency are two different goals, which is why matching belongs to antennas and audio, not to a power grid.

In 2D: PL against RL peaks exactly where the dashed line stands, at RL = Rth, while η rises forever toward 1 without reaching it. Where one is at its top, the other is only half.`,
    ko: `## 최대 전력 전달

**전원은 부하 저항이 자기 내부 저항과 같아질 때 부하에 가장 큰 전력을 넘겨줍니다. 그리고 바로 그 지점에서 똑같은 양을 자기 안에서 태웁니다.**

회로를 전위 지형으로 그렸습니다. 높이 = 전위이므로 전원 전압이 두 저항에 나뉘고, 부하에게 남은 높이가 곧 VL입니다.
- 왼쪽 오르막: 전원이 전하를 Vth만큼 끌어올립니다. 점선 수평선이 VL 높이이고, 그 위로 벌어진 만큼을 Rth가 가져갑니다
- 시안 액체 선: 전류 I. 광택이 빨리 흐를수록 I가 큽니다
- 유리관 두 개: 전원 안의 Rth와 부하 RL. 가늘수록 R이 큽니다
- 앰버 필라멘트: 각자가 열로 바꾸는 전력. RL = Rth에서 둘이 똑같이 빛납니다. 태우는 양이 정확히 같기 때문입니다

**전류**: I = Vth ÷ (Rth + RL)

**부하 전력**: PL = I² × RL, **전원 내부 손실**: P_int = I² × Rth

**효율**: η = PL ÷ (Vth × I) = RL ÷ (Rth + RL) = VL ÷ Vth

**정합 조건**: RL = Rth일 때 PL이 최대가 되고, PL,max = Vth² ÷ (4 × Rth), 이때 η = 50 %

Vth를 12 V, Rth를 6 Ω으로 두고 RL을 6 Ω까지 내려 보세요. PL이 6.00 W로 최대가 되고 η는 50 %입니다. RL을 18 Ω으로 올리면 효율은 75 %까지 오르지만 부하가 받는 전력은 4.50 W로 줄어듭니다. 최대 전력과 최대 효율은 서로 다른 목표입니다. 정합이 안테나와 오디오의 이야기이고 전력 계통의 이야기가 아닌 이유입니다.

2D: PL 곡선은 점선이 선 자리, 즉 RL = Rth에서 정확히 최대가 됩니다. 반면 η는 1을 향해 끝없이 오르기만 하고 닿지는 않습니다. 한쪽이 정점일 때 다른 쪽은 겨우 절반입니다.`,
  },
  sliderParams: [
    { key: 'Vth', label: { en: 'Source EMF (Vth)', ko: '전원 기전력 (Vth)' }, unit: 'V', min: 2, max: V_MAX, step: 0.5, defaultValue: 12 },
    { key: 'Rth', label: { en: 'Internal Resistance (Rth)', ko: '내부 저항 (Rth)' }, unit: 'Ω', min: RTH_MIN, max: 20, step: 0.5, defaultValue: 6 },
    { key: 'RL', label: { en: 'Load Resistance (RL)', ko: '부하 저항 (RL)' }, unit: 'Ω', min: RL_MIN, max: RL_MAX, step: 0.5, defaultValue: 18 },
  ],
  Scene: MaxPowerScene,
  Scene2D: MaxPower2DScene,
}

export default concept
