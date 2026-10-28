import { useRef, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import type { Group } from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { useLocaleStore } from '../../lib/i18n'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import type { ConceptMeta } from '../../types/concept'

const PTS = 241 // points along the line; enough for a smooth bow at 600 km
const PROFILE_INIT: [number, number, number][] = Array.from({ length: PTS }, () => [0, 0, 0])
const SEG_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]

const V_PROP = 3e5 // propagation velocity on an overhead line [km ÷ s]
const ZC = 400 // surge (characteristic) impedance sqrt(L ÷ C) [Ω]
const VS_LINE_KV = 154 // sending-end line voltage [kV]
const VS_PHASE = (VS_LINE_KV * 1e3) / Math.sqrt(3) // 88911.9 V per phase
const VS_OVER_ZC = VS_PHASE / ZC // 222.28 A, the current unit of the charging-current profile

/**
 * Ratio of receiving-end to sending-end voltage magnitude for a lossless distributed line
 * closed on a resistive load Zl = Zc ÷ k, where k is the load in units of the surge
 * impedance loading. k = 0 is the open receiving end.
 *   Vs ÷ Vr = cos(βl) + j·k·sin(βl)   so   Vr ÷ Vs = 1 ÷ sqrt(cos²βl + k²sin²βl)
 */
function ratioAt(km: number, beta: number, load: number) {
  const bl = beta * km
  return 1 / Math.hypot(Math.cos(bl), load * Math.sin(bl))
}

// Reused result object: solve() runs every frame, so it must not allocate.
const OUT = {
  lineKm: 300, freq: 60, load: 0,
  beta: 0, bl: 0, ratio: 1, vsOverVr: 1, rise: 0, ic: 0, vrKV: 0,
}

// Lossless distributed line (series L and shunt C only, R and G dropped). Open receiving end:
//   β    = ω·sqrt(L·C) = 2π·f ÷ v          [rad ÷ km],  v ≈ 3×10^5 km ÷ s
//   Zc   = sqrt(L ÷ C) = 400 Ω
//   Vr ÷ Vs = 1 ÷ cos(βl)                 larger than 1, so the open end sits ABOVE the source
//   V(x) = Vs·cos(β(l − x)) ÷ cos(βl)      x measured from the SENDING end: Vs at x = 0, Vr at x = l
//   Is   = Vs·tan(βl) ÷ Zc                 leading charging current drawn by the line's own C
// Both length and frequency enter only through βl, so the rise grows with either one.
// With a resistive load Zl = Zc ÷ k (k in units of surge impedance loading, k = 0 open):
//   V(d) = Vr·sqrt(cos²βd + k²sin²βd),  I(d) = (Vr ÷ Zc)·sqrt(sin²βd + k²cos²βd),  d = l − x
// Worked example, verified with npx tsx:
//   f = 60 Hz -> β = 1.2566×10⁻³ rad ÷ km
//     l = 100 km -> βl =  7.200°, Vr ÷ Vs = 1.00795 (rise  0.795 %)
//     l = 300 km -> βl = 21.600°, Vr ÷ Vs = 1.07553 (rise  7.553 %)
//     l = 500 km -> βl = 36.000°, Vr ÷ Vs = 1.23607 (rise 23.607 %)
//   f = 50 Hz, l = 300 km -> βl = 18.000°, Vr ÷ Vs = 1.05146 (rise 5.146 %)
//   profile at l = 300 km, 60 Hz, open end: V(0) ÷ Vs = 1.00000, V(150) ÷ Vs = 1.05648,
//     V(300) ÷ Vs = 1.07553 — it meets Vs at the source and Vr at the open end
//   charging current at l = 300 km, 60 Hz, Vs = 154 kV ÷ sqrt(3) = 88911.9 V per phase, Zc = 400 Ω:
//     Is = 88911.9 × tan(21.600°) ÷ 400 = 88911.9 × 0.395928 ÷ 400 = 88.007 A per phase
//   loading at l = 300 km, 60 Hz: k = 0 -> 1.07553, k = 0.5 -> 1.05505, k = 1 -> 1.00000 (flat),
//     k = 1.5 -> 0.92474, k = 2 -> 0.84318 (the bow has turned into a drop)
function solve({ lineKm = 300, freq = 60, load = 0 }: Record<string, number>) {
  const beta = (2 * Math.PI * freq) / V_PROP
  const bl = beta * lineKm
  const ratio = ratioAt(lineKm, beta, load)
  OUT.lineKm = lineKm
  OUT.freq = freq
  OUT.load = load
  OUT.beta = beta
  OUT.bl = bl
  OUT.ratio = ratio
  OUT.vsOverVr = 1 / ratio
  OUT.rise = (ratio - 1) * 100
  // Sending-end current: (Vr ÷ Zc)·sqrt(sin²βl + k²cos²βl), which is Vs·tan(βl) ÷ Zc when k = 0
  OUT.ic = VS_OVER_ZC * ratio * Math.hypot(Math.sin(bl), load * Math.cos(bl))
  OUT.vrKV = VS_LINE_KV * ratio
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

const deg = (rad: number) => `${((rad * 180) / Math.PI).toFixed(3)}°`
const signedPct = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(3)} %`
// Scene text reads in the foreground color; only current (cyan) and power (amber) tint their labels
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const accentClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums'
const axisClass = 'pointer-events-none whitespace-nowrap text-[10px] tabular-nums text-muted-foreground'

const L10N = {
  en: {
    sending: 'sending end',
    openEnd: 'open end',
    ref: 'Vs level',
    rise: 'rise',
    drop: 'drop',
    taper: 'Ic: largest at the source, zero at the open end',
    charging: 'charging current Ic',
    lengthAxis: 'line length l',
    openCurve: 'dashed: open end · solid: at this load',
    resonance: 'βl reaches 90° near 1250 km at 60 Hz, where the lossless ratio runs away',
  },
  ko: {
    sending: '송전단',
    openEnd: '개방된 수전단',
    ref: 'Vs 기준선',
    rise: '상승',
    drop: '강하',
    taper: 'Ic: 송전단에서 최대, 개방단에서 0',
    charging: '충전전류 Ic',
    lengthAxis: '선로 길이 l',
    openCurve: '점선: 무부하 · 실선: 현재 부하',
    resonance: '60 Hz에서 약 1250 km이면 βl이 90°에 이르고, 무손실 비는 발산합니다',
  },
} satisfies Record<string, Record<string, string>>

// Shared scratch, written then handed straight to a geometry, so no frame ever allocates
const SEG = new Float32Array(6)
function setSeg(line: any, x0: number, y0: number, z0: number, x1: number, y1: number, z1: number) {
  SEG[0] = x0; SEG[1] = y0; SEG[2] = z0
  SEG[3] = x1; SEG[4] = y1; SEG[5] = z1
  line?.geometry.setPositions(SEG)
  line?.computeLineDistances()
}

// ── 3D: the line along X, voltage magnitude as height ──
// The dashed line at REF_Y is the sending-end level, so the gap above it at the open end IS the rise.
// Both lanes run at constant z and the camera sits on the z axis (see CAM below), so a lane with no
// rise projects to an exactly horizontal screen line and a rising Vr can only read as rising.
const X_S = -2.45
const X_R = 0.8
const SPAN = X_R - X_S
const V_Z = 0.55
const REF_Y = 0.4
const V_GAIN = 2.5 // scene units per unit of Vr ÷ Vs: 1.372 -> +0.93, 0.645 -> −0.89
const I_Z = -0.85
const I_BASE_Y = -1.35
const I_GAIN = 0.65 // scene units per unit of Vs ÷ Zc; the profile peaks near 1.04 of that
const FLOW_PER_AMP = 0.012 // sheen bands per second per amp of charging current

// Head-on, on the z axis: an off-axis camera makes the far end of a level line project LOWER than the
// near end (−233 px over this span from [3, 2, 5]), which would read as a voltage drop and teach the
// opposite of the concept. From the z axis, height is the only vertical signal. OrbitControls still
// lets the learner swing round for the depth between the two lanes.
const CAM: [number, number, number] = [0, 0.9, 5.9]
// Everything stays left of the floating description card, which sits at its default [2.6, 0.6, -1.5]
const REF_LINE: [number, number, number][] = [[X_S, REF_Y, V_Z], [X_R + 0.15, REF_Y, V_Z]]
const I_BASE: [number, number, number][] = [[X_S, I_BASE_Y, I_Z], [X_R, I_BASE_Y, I_Z]]

/** Fills the voltage and charging-current profiles along the line. Allocates nothing. */
function fillProfiles(vBuf: Float32Array, iBuf: Float32Array, s: typeof OUT) {
  for (let i = 0; i < PTS; i++) {
    const u = i / (PTS - 1) // 0 at the sending end, 1 at the open end
    const bd = s.bl * (1 - u) // β(l − x)
    const cos = Math.cos(bd)
    const sin = Math.sin(bd)
    const vRel = Math.hypot(cos, s.load * sin) / s.vsOverVr // |V(x)| ÷ Vs
    const iRel = Math.hypot(sin, s.load * cos) / s.vsOverVr // |I(x)| ÷ (Vs ÷ Zc)
    const x = X_S + u * SPAN
    const k = i * 3
    vBuf[k] = x; vBuf[k + 1] = REF_Y + (vRel - 1) * V_GAIN; vBuf[k + 2] = V_Z
    iBuf[k] = x; iBuf[k + 1] = I_BASE_Y + iRel * I_GAIN; iBuf[k + 2] = I_Z
  }
}

function FerrantiProfile3DScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const t = L10N[locale]
  const valuesRef = useSliderValues()
  const vBuf = useMemo(() => new Float32Array(PTS * 3), [])
  const iBuf = useMemo(() => new Float32Array(PTS * 3), [])
  const vRef = useRef<any>(null)
  const iRef = useRef<any>(null)
  const gapRef = useRef<any>(null)
  const dotRefs = useRef<(Group | null)[]>([]) // sending end, open end
  const riseTagRef = useRef<Group>(null)
  const textRefs = useRef<(HTMLSpanElement | null)[]>([]) // state, params, rise, Ic
  const flow = useRef(0)
  const labelledFor = useRef<unknown>(null)
  const labelledLocale = useRef<string>('')

  useFrame(() => {
    const s = solve(valuesRef.current)
    fillProfiles(vBuf, iBuf, s)
    vRef.current?.geometry.setPositions(vBuf)
    iRef.current?.geometry.setPositions(iBuf)

    // The receiving end of the voltage curve is the last sample in the buffer
    const yR = vBuf[(PTS - 1) * 3 + 1]
    setSeg(gapRef.current, X_R, REF_Y, V_Z, X_R, yR, V_Z)
    dotRefs.current[0]?.position.set(X_S, REF_Y, V_Z)
    dotRefs.current[1]?.position.set(X_R, yR, V_Z)
    // Inside the gap and left of the tick: the profile is tangentially flat at the open end, so the
    // gap is at its widest here, and the tag stays well clear of the description card on the right.
    riseTagRef.current?.position.set(X_R - 0.5, (REF_Y + yR) / 2, V_Z)

    flow.current = FLOW_PER_AMP * s.ic

    if (labelledFor.current !== valuesRef.current || labelledLocale.current !== locale) {
      labelledFor.current = valuesRef.current
      labelledLocale.current = locale
      const texts = [
        `βl ${deg(s.bl)} · Vr ÷ Vs ${s.ratio.toFixed(5)} · ${s.rise >= 0 ? t.rise : t.drop} ${signedPct(s.rise)}`,
        `l ${s.lineKm.toFixed(0)} km · f ${s.freq.toFixed(0)} Hz · Zc ${ZC} Ω · Vs ${VS_LINE_KV} kV`,
        signedPct(s.rise), // bare number in the gap; the word for it is on the first row
        `${t.charging} ${s.ic.toFixed(1)} A`,
      ]
      texts.forEach((text, i) => { if (textRefs.current[i]) textRefs.current[i]!.textContent = text })
    }
  })

  return (
    <>
      <PerspectiveCamera makeDefault position={CAM} fov={50} />
      <OrbitControls enablePan={false} />

      {/* The sending-end level: everything above it is voltage rise */}
      <Line points={REF_LINE} lineWidth={1} color={c.guide} dashed dashSize={0.11} gapSize={0.08} />
      <Line ref={gapRef} points={SEG_INIT} lineWidth={1.5} color={c.guide} dashed dashSize={0.07} gapSize={0.05} />
      <Line points={I_BASE} lineWidth={1} color={c.guide} />

      {/* Voltage magnitude along the line: the Ferranti bow */}
      <Line ref={vRef} points={PROFILE_INIT} lineWidth={3} color={c.voltage} />
      {/* Leading charging current, drawn from the source (index 0) to the open end */}
      <TrailLine
        ref={iRef}
        points={PROFILE_INIT}
        lineWidth={2.5}
        color={c.current}
        fade={false}
        flowSpeed={() => flow.current}
      />

      <Marker ref={(g) => { dotRefs.current[0] = g }} color={c.marker} size={0.06} />
      <Marker ref={(g) => { dotRefs.current[1] = g }} color={c.marker} size={0.07} />

      <Html position={[X_S, 1.8, V_Z]}>
        <span ref={(el) => { textRefs.current[0] = el }} className={labelClass} />
      </Html>
      <Html position={[X_S, 1.57, V_Z]}>
        <span ref={(el) => { textRefs.current[1] = el }} className={labelClass} />
      </Html>
      <group ref={riseTagRef}>
        <Html center>
          <span ref={(el) => { textRefs.current[2] = el }} className={labelClass} />
        </Html>
      </group>
      {/* Both current-lane captions sit below the baseline, where the TrailLine never reaches */}
      <Html position={[X_S, -1.6, I_Z]}>
        <span ref={(el) => { textRefs.current[3] = el }} className={accentClass} style={{ color: c.current }} />
      </Html>
      <Html position={[X_S, -1.86, I_Z]}>
        <span className={axisClass}>{t.taper}</span>
      </Html>

      {/* Above the dashed reference, near the sending end, where the profile is pinned to Vs */}
      <Html position={[X_S + 0.45, REF_Y + 0.22, V_Z]} center>
        <span className={axisClass}>{t.ref}</span>
      </Html>
      {/* Below the deepest the profile can sag, and above the charging-current lane behind it */}
      <Html position={[X_S, -0.65, V_Z]} center>
        <span className={axisClass}>{t.sending}</span>
      </Html>
      <Html position={[X_R, -0.65, V_Z]} center>
        <span className={axisClass}>{t.openEnd}</span>
      </Html>
    </>
  )
}

// ── 2D: Vr ÷ Vs against line length, 50–600 km ──
const KM_MIN = 50
const KM_MAX = 600
const PX_L = -3.4
const PX_SPAN = 6.8
const Y1 = -0.2 // where Vr ÷ Vs = 1 sits
const GAIN2 = 4.5 // 1.372 -> +1.47, 0.645 -> −1.80
const Y_TOP = 1.85 // clamp: the lossless ratio runs away as βl approaches 90°
const Y_BOTTOM = -1.9
const PLOT_PTS = 221
const PLOT_INIT: [number, number, number][] = Array.from({ length: PLOT_PTS }, () => [0, 0, 0])
const MARK_KM = [100, 300, 500]
const ONE_LINE: [number, number, number][] = [[PX_L, Y1, 0], [PX_L + PX_SPAN, Y1, 0]]

const plotX = (km: number) => PX_L + ((km - KM_MIN) / (KM_MAX - KM_MIN)) * PX_SPAN
const plotY = (ratio: number) => Math.min(Math.max(Y1 + (ratio - 1) * GAIN2, Y_BOTTOM), Y_TOP)

/** Fills the open-end and loaded curves over the length range. Allocates nothing. */
function fillCurves(openBuf: Float32Array, loadBuf: Float32Array, s: typeof OUT) {
  for (let i = 0; i < PLOT_PTS; i++) {
    const km = KM_MIN + (i / (PLOT_PTS - 1)) * (KM_MAX - KM_MIN)
    const x = plotX(km)
    const k = i * 3
    openBuf[k] = x; openBuf[k + 1] = plotY(ratioAt(km, s.beta, 0)); openBuf[k + 2] = 0
    loadBuf[k] = x; loadBuf[k + 1] = plotY(ratioAt(km, s.beta, s.load)); loadBuf[k + 2] = 0
  }
}

function FerrantiRatio2DScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const t = L10N[locale]
  const valuesRef = useSliderValues()
  const openBuf = useMemo(() => new Float32Array(PLOT_PTS * 3), [])
  const loadBuf = useMemo(() => new Float32Array(PLOT_PTS * 3), [])
  const openRef = useRef<any>(null)
  const loadRef = useRef<any>(null)
  const dropRef = useRef<any>(null)
  const dotRefs = useRef<(Group | null)[]>([]) // 100 km, 300 km, 500 km, current length
  const tagRefs = useRef<(Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([]) // three marks, then the readout
  const labelledFor = useRef<unknown>(null)
  const labelledLocale = useRef<string>('')

  useFrame(() => {
    const s = solve(valuesRef.current)
    fillCurves(openBuf, loadBuf, s)
    openRef.current?.geometry.setPositions(openBuf)
    openRef.current?.computeLineDistances()
    loadRef.current?.geometry.setPositions(loadBuf)

    for (let m = 0; m < MARK_KM.length; m++) {
      const x = plotX(MARK_KM[m])
      const y = plotY(ratioAt(MARK_KM[m], s.beta, 0))
      dotRefs.current[m]?.position.set(x, y, 0)
      tagRefs.current[m]?.position.set(x, y + 0.26, 0)
    }
    const cx = plotX(s.lineKm)
    const cy = plotY(s.ratio)
    dotRefs.current[3]?.position.set(cx, cy, 0)
    setSeg(dropRef.current, cx, Y1, 0, cx, cy, 0)

    if (labelledFor.current !== valuesRef.current || labelledLocale.current !== locale) {
      labelledFor.current = valuesRef.current
      labelledLocale.current = locale
      const texts = [
        ...MARK_KM.map((km) => `${km} km · ${ratioAt(km, s.beta, 0).toFixed(3)}`),
        `l ${s.lineKm.toFixed(0)} km · f ${s.freq.toFixed(0)} Hz · Vr ÷ Vs ${s.ratio.toFixed(5)} · ${s.rise >= 0 ? t.rise : t.drop} ${signedPct(s.rise)}`,
      ]
      texts.forEach((text, i) => { if (textRefs.current[i]) textRefs.current[i]!.textContent = text })
    }
  })

  return (
    <>
      <Line points={ONE_LINE} lineWidth={1} color={c.guide} />
      <Line ref={dropRef} points={SEG_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />

      {/* Open receiving end: 1 ÷ cos(βl), climbing toward the βl = 90° resonance */}
      <Line ref={openRef} points={PLOT_INIT} lineWidth={1.5} color={c.guide} dashed dashSize={0.1} gapSize={0.07} />
      {/* The same line at the load set by the slider */}
      <Line ref={loadRef} points={PLOT_INIT} lineWidth={3} color={c.voltage} />

      {MARK_KM.map((km, m) => (
        <group key={km}>
          <Marker ref={(g) => { dotRefs.current[m] = g }} color={c.marker} size={0.055} />
          <group ref={(g) => { tagRefs.current[m] = g }}>
            <Html center>
              <span ref={(el) => { textRefs.current[m] = el }} className={axisClass} />
            </Html>
          </group>
        </group>
      ))}
      <Marker ref={(g) => { dotRefs.current[3] = g }} color={c.marker} size={0.075} />

      <Html position={[-3.9, 1.85, 0]}>
        <span ref={(el) => { textRefs.current[3] = el }} className={labelClass} />
      </Html>
      <Html position={[-3.9, 1.6, 0]}>
        <span className={axisClass}>{t.openCurve}</span>
      </Html>
      <Html position={[-3.9, -1.75, 0]}>
        <span className={axisClass}>{t.resonance}</span>
      </Html>
      <Html position={[PX_L + PX_SPAN + 0.35, Y1, 0]} center>
        <span className={axisClass}>1.00</span>
      </Html>
      <Html position={[PX_L, Y1 - 0.28, 0]} center>
        <span className={axisClass}>{KM_MIN} km</span>
      </Html>
      <Html position={[PX_L + PX_SPAN, Y1 - 0.28, 0]} center>
        <span className={axisClass}>{KM_MAX} km</span>
      </Html>
      <Html position={[PX_L + PX_SPAN / 2, Y1 - 0.28, 0]} center>
        <span className={axisClass}>{t.lengthAxis}</span>
      </Html>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'ferranti-effect',
  title: { en: 'Ferranti Effect', ko: '페란티 현상 (무부하 선로의 전압 상승)' },
  subject: 'power-systems',
  order: 4,
  description: {
    en: `## Ferranti Effect

**On a long line with no load, the line's own capacitance pulls a leading charging current through the series inductance, and the open receiving end ends up at a HIGHER voltage than the source.**

- Bright curve: the voltage magnitude along the line, bowing up toward the open end
- Dashed horizontal line: the sending-end level Vs, so the gap above it at the far end is the rise itself
- Dashed vertical tick: that gap, labelled as a percentage
- Dots: the sending end and the open receiving end
- Cyan liquid line: the charging current, largest at the source and tapering to zero at the open end — every amp of it has been absorbed by the line capacitance along the way

**Phase constant**: β = ω·sqrt(L·C) = 2π·f ÷ v, with v ≈ 3×10^5 km ÷ s
**Open-end voltage**: Vr ÷ Vs = 1 ÷ cos(βl)
**Voltage profile**: V(x) = Vs·cos(β(l − x)) ÷ cos(βl), x measured from the sending end
**Charging current**: Is = Vs·tan(βl) ÷ Zc, with Zc = sqrt(L ÷ C)

These are the lossless forms: series resistance and leakage are dropped, which is a good approximation for a transmission line and makes the effect exact and easy to read.

At f = 60 Hz, β = 1.2566×10⁻³ rad ÷ km, so at l = 100 km βl = 7.200° and Vr ÷ Vs = 1.00795 (a 0.795 % rise); at 300 km βl = 21.600° and the ratio is 1.07553 (7.553 %); at 500 km βl = 36.000° and it reaches 1.23607 (23.607 %). Drop to 50 Hz at 300 km and βl falls to 18.000° with the ratio at 1.05146 (5.146 %). Length and frequency both enter only through βl, which is why the effect grows with either one and runs away as βl approaches 90°.

With Zc = 400 Ω and Vs = 154 kV, that is 88911.9 V per phase, the 300 km line draws Is = 88911.9 × tan(21.600°) ÷ 400 = 88.007 A per phase with nothing connected at the far end.

Line voltage drop showed the voltage sagging under load; this is the opposite case, and the contrast is the point. Add load and watch the bow flatten: at one unit of surge impedance loading the profile is perfectly flat, and beyond it the line sags again like any loaded feeder.`,
    ko: `## 페란티 현상

**무부하 장거리 선로에서는 선로 자체의 정전용량이 직렬 인덕턴스를 통해 진상 충전전류를 끌어당기고, 그 결과 개방된 수전단 전압이 송전단보다 더 높아집니다.**

- 가장 또렷한 곡선: 선로를 따라가는 전압 크기. 개방단 쪽으로 활처럼 휘어 올라갑니다
- 수평 점선: 송전단 전압 Vs의 높이. 끝에서 이 선 위로 벌어진 간격이 곧 전압 상승분입니다
- 수직 점선: 그 간격이며, 백분율로 표시됩니다
- 점: 송전단과 개방된 수전단
- 시안 액체 선: 충전전류. 송전단에서 가장 크고 개방단으로 갈수록 0으로 줄어듭니다 — 그만큼이 가는 길에 선로 정전용량으로 모두 흘러들었기 때문입니다

**위상정수**: β = ω·sqrt(L·C) = 2π·f ÷ v, v ≈ 3×10^5 km ÷ s
**무부하 수전단 전압**: Vr ÷ Vs = 1 ÷ cos(βl)
**전압 분포**: V(x) = Vs·cos(β(l − x)) ÷ cos(βl), x는 송전단에서 잰 거리
**충전전류**: Is = Vs·tan(βl) ÷ Zc, Zc = sqrt(L ÷ C)

여기서는 무손실 근사를 씁니다. 직렬 저항과 누설을 무시한 식이며, 송전선에서는 충분히 좋은 근사이고 현상을 정확하고 읽기 쉽게 보여 줍니다.

f = 60 Hz에서 β = 1.2566×10⁻³ rad ÷ km입니다. l = 100 km이면 βl = 7.200°, Vr ÷ Vs = 1.00795으로 0.795 % 상승이고, 300 km이면 βl = 21.600°, 비는 1.07553으로 7.553 %, 500 km이면 βl = 36.000°, 비는 1.23607으로 23.607 %에 이릅니다. 300 km에서 50 Hz로 내리면 βl이 18.000°로 줄고 비는 1.05146, 즉 5.146 %가 됩니다. 길이와 주파수는 모두 βl을 통해서만 들어오므로, 어느 쪽을 키워도 현상이 커지고 βl이 90°에 가까워지면 발산합니다.

Zc = 400 Ω, Vs = 154 kV, 즉 상전압 88911.9 V일 때 300 km 선로는 끝에 아무것도 달지 않아도 Is = 88911.9 × tan(21.600°) ÷ 400 = 88.007 A의 충전전류를 상당 끌어갑니다.

전압강하 개념에서는 부하가 걸리면 전압이 처졌습니다. 여기는 정반대 경우이고, 그 대비가 핵심입니다. 부하를 올리며 활이 펴지는 모습을 보세요. 고유부하(서지 임피던스 부하) 1배에서 분포가 완전히 평평해지고, 그보다 커지면 선로는 다시 보통의 부하 선로처럼 처집니다.`,
  },
  sliderParams: [
    { key: 'lineKm', label: { en: 'Line length (l)', ko: '선로 길이 (l)' }, unit: 'km', min: 50, max: 600, step: 10, defaultValue: 300 },
    { key: 'freq', label: { en: 'Frequency (f)', ko: '주파수 (f)' }, unit: 'Hz', min: 50, max: 60, step: 1, defaultValue: 60 },
    { key: 'load', label: { en: 'Load (× surge impedance loading)', ko: '부하 (고유부하 배수)' }, unit: '×', min: 0, max: 2, step: 0.05, defaultValue: 0 },
  ],
  Scene: FerrantiProfile3DScene,
  Scene2D: FerrantiRatio2DScene,
}

export default concept
