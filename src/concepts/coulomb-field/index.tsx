import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { setLineOpacity, setOpacity, useStoryReveal } from '../../components/scene/storyReveal'
import { useLocaleStore } from '../../lib/i18n'
import { Marker } from '../../components/scene/Marker'
import type { ConceptMeta } from '../../types/concept'

const K = 8.99e9 // Coulomb constant [N·m² ÷ C²]
const Q_UNIT = 1e-6 // sliders are in μC
const R_MIN_CM = 4
const R_MAX_CM = 40
const SCENE_PER_METER = 6 // scene units per metre, so the drawn gap is proportional to r

/*
 * Worked example (re-verified with node -e):
 *   q1 = +1 μC, q2 = +2 μC, r = 10 cm
 *   F  = k q1 q2 ÷ r² = 8.99e9 × 1e-6 × 2e-6 ÷ 0.01 = 1.798 N, sign positive = repulsive
 *   Doubling r to 20 cm gives F = 0.4495 N, exactly one quarter.
 * The probes show the same 1 ÷ r² on the field of one charge, E = k q ÷ r²:
 *   at 12.5 cm E = 5.75e5 V ÷ m, at 25 cm E = 1.44e5 V ÷ m.
 */
function solve({ q1 = 1, q2 = 2, r = 10 }: Record<string, number>) {
  const Q1 = q1 * Q_UNIT
  const Q2 = q2 * Q_UNIT
  const R = r / 100
  return {
    q1, q2, r, Q1, Q2, R,
    F: (K * Q1 * Q2) / (R * R), // + repulsive, − attractive
  }
}

function useSliderValues() {
  const valuesRef = useRef<Record<string, number>>(useSliderStore.getState().values)
  useEffect(
    () => useSliderStore.subscribe((s) => s.values, (v) => { valuesRef.current = v }),
    []
  )
  return valuesRef
}

const fmtQ = (q: number) => `${q < 0 ? '−' : '+'}${Math.abs(q).toFixed(1)} μC`
const fmtN = (f: number) => {
  const a = Math.abs(f)
  return a >= 10 ? `${a.toFixed(2)} N` : a >= 0.01 ? `${a.toFixed(3)} N` : `${(a * 1e3).toFixed(1)} mN`
}
const fmtEfield = (e: number) => {
  const a = Math.abs(e)
  return a >= 1e6 ? `${(a / 1e6).toFixed(2)} MV ÷ m` : `${(a / 1e3).toFixed(0)} kV ÷ m`
}
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const axisClass = 'pointer-events-none whitespace-nowrap text-[11px] text-muted-foreground'

// ── Field-line tracing, done once per seed in the xy-plane and then rotated into planes about X ──
const TRACE_PTS = 44
const TRACE_STEP = 0.13
const SEED_R = 0.17 // where a line starts, just outside the charge's dot
const STOP_R = 0.14 // a line ends when it reaches the other charge
const VIEW_R = 2.2 // and also when it leaves the framed area (DESIGN.md: content within about ±2)
const VIEW_X = 1.15 // camera and orbit target share this x, framing the scene left of the narration

const SEED_DEG = [30, 90, 150, 210, 270, 330]
const SEED_COS = new Float32Array(SEED_DEG.map((d) => Math.cos((d * Math.PI) / 180)))
const SEED_SIN = new Float32Array(SEED_DEG.map((d) => Math.sin((d * Math.PI) / 180)))
const PLANE_DEG = [0, 60]
const PLANE_COS = new Float32Array(PLANE_DEG.map((d) => Math.cos((d * Math.PI) / 180)))
const PLANE_SIN = new Float32Array(PLANE_DEG.map((d) => Math.sin((d * Math.PI) / 180)))
const LINE_COUNT = 2 * SEED_DEG.length * PLANE_DEG.length
const TRACE_INIT: [number, number, number][] = Array.from({ length: TRACE_PTS }, () => [0, 0, 0])

const field = new Float32Array(2) // scratch for the summed field at a point
const trace = new Float32Array(TRACE_PTS * 2) // scratch for one traced line in the xy-plane

/** Field of the pair at (px, py), in scene coordinates. Only its direction is used. */
function fieldAt(px: number, py: number, x1: number, Q1: number, x2: number, Q2: number) {
  let ex = 0
  let ey = 0
  let dx = px - x1
  let d2 = dx * dx + py * py
  if (d2 > 1e-12) {
    const m = Q1 / (d2 * Math.sqrt(d2))
    ex += m * dx
    ey += m * py
  }
  dx = px - x2
  d2 = dx * dx + py * py
  if (d2 > 1e-12) {
    const m = Q2 / (d2 * Math.sqrt(d2))
    ex += m * dx
    ey += m * py
  }
  field[0] = ex
  field[1] = ey
}

/** Walks the field direction from a seed point; once it stops, the remaining samples pile up there. */
function traceFieldLine(
  sx: number, sy: number, dir: number,
  x1: number, Q1: number, x2: number, Q2: number, otherX: number
) {
  let px = sx
  let py = sy
  let live = true
  for (let i = 0; i < TRACE_PTS; i++) {
    trace[i * 2] = px
    trace[i * 2 + 1] = py
    if (!live) continue
    fieldAt(px, py, x1, Q1, x2, Q2)
    const m = Math.sqrt(field[0] * field[0] + field[1] * field[1])
    if (m < 1e-18) {
      live = false
      continue
    }
    px += (dir * TRACE_STEP * field[0]) / m
    py += (dir * TRACE_STEP * field[1]) / m
    const ox = px - otherX
    if (ox * ox + py * py < STOP_R * STOP_R || px * px + py * py > VIEW_R * VIEW_R) live = false
  }
}

// ── Force arrows: one 5-point polyline each (shaft, then two barbs folded back from the tip) ──
const ARROW_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
const ARROW_SCALE = 0.56 // scene units per newton, linear so the 1 ÷ r² collapse stays honest
const ARROW_MIN = 0.07
const ARROW_MAX = 1.5

function fillArrow(buf: Float32Array, x0: number, dirX: number, len: number) {
  const tipX = x0 + dirX * len
  const backX = tipX - dirX * Math.min(0.15, len * 0.45)
  const barb = Math.min(0.085, len * 0.3)
  buf[0] = x0; buf[1] = 0; buf[2] = 0
  buf[3] = tipX; buf[4] = 0; buf[5] = 0
  buf[6] = backX; buf[7] = barb; buf[8] = 0
  buf[9] = tipX; buf[10] = 0; buf[11] = 0
  buf[12] = backX; buf[13] = -barb; buf[14] = 0
}

const GAP_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]
const RAY_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]

// ── Narration steps drive what exists on screen. Each reveal eases 0 → 1 so nothing pops in. ──
const REV_LINES = 0 // field lines of the pair
const REV_PROBE = 1 // the two probes that show E falling as 1 ÷ r²
const REV_PAIR = 2 // q2, the force arrow it feels, and the force readout
const REV_ARROW1 = 3 // the equal and opposite arrow back on q1
const REV_COUNT = 4

/** Which reveals the sentence at `step` asks for. Index 0 is the lone charge, so nothing yet. */
function revealTargets(step: number, out: Float32Array) {
  out[REV_LINES] = step >= 1 ? 1 : 0
  out[REV_PROBE] = step === 2 ? 1 : 0
  out[REV_PAIR] = step >= 3 ? 1 : 0
  out[REV_ARROW1] = step >= 4 ? 1 : 0
}

// Probe travel along +Y from q1, in scene units. The far probe sits at exactly twice the near one,
// so 2× the distance landing on ¼ the field is visible rather than asserted.
const PROBE_MIN = 0.75
const PROBE_MAX = 1.05
const PROBE_TAG_X = 0.95 // readouts sit clear to the right of the ray
const PROBE_PERIOD = 7 // seconds for one out-and-back

// Readout slots. 0/1 ride along with the charges, 2 is centred under them, 3/4 ride the probes.
const TEXT_Q1 = 0
const TEXT_Q2 = 1
const TEXT_FORCE = 2
const TEXT_NEAR = 3
const TEXT_FAR = 4
/** Which text slot each positioned tag group carries, in tagRefs order. */
const TAG_TEXT_SLOT = [TEXT_Q1, TEXT_Q2, TEXT_NEAR, TEXT_FAR]

function CoulombFieldScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  const lineBufs = useMemo(
    () => Array.from({ length: LINE_COUNT }, () => new Float32Array(TRACE_PTS * 3)),
    []
  )
  const arrowBufs = useMemo(() => [new Float32Array(15), new Float32Array(15)], [])
  const gapBuf = useMemo(() => new Float32Array(6), [])
  const rayBuf = useMemo(() => new Float32Array(6), [])

  const lineRefs = useRef<any[]>([])
  const arrowRefs = useRef<any[]>([])
  const gapRef = useRef<any>(null)
  const rayRef = useRef<any>(null)
  const chargeRefs = useRef<(THREE.Group | null)[]>([])
  const probeRefs = useRef<(THREE.Group | null)[]>([]) // near probe, far probe
  const tagRefs = useRef<(THREE.Group | null)[]>([]) // q1 tag, q2 tag, near probe tag, far probe tag
  const textRefs = useRef<(HTMLSpanElement | null)[]>([]) // see TEXT_*; index 2 is the centred readout
  const labelledFor = useRef<unknown>(null)

  useFrame(({ clock }) => {
    const s = solve(valuesRef.current)

    const revLines = reveal[REV_LINES]
    const revProbe = reveal[REV_PROBE]
    const revPair = reveal[REV_PAIR]
    const revArrow1 = reveal[REV_ARROW1]

    // Before q2 exists, q1 sits at the origin and slides left to make room for it
    const half = (s.R * SCENE_PER_METER) / 2
    const x1 = -half * revPair
    const x2 = half
    const Q2eff = s.Q2 * revPair

    let li = 0
    for (let ch = 0; ch < 2; ch++) {
      const seedX = ch === 0 ? x1 : x2
      const otherX = ch === 0 ? x2 : x1
      const dir = (ch === 0 ? s.Q1 : Q2eff) < 0 ? -1 : 1
      for (let a = 0; a < SEED_COS.length; a++) {
        traceFieldLine(
          seedX + SEED_R * SEED_COS[a], SEED_R * SEED_SIN[a], dir,
          x1, s.Q1, x2, Q2eff, otherX
        )
        for (let p = 0; p < PLANE_COS.length; p++) {
          const buf = lineBufs[li]
          const cp = PLANE_COS[p]
          const sp = PLANE_SIN[p]
          for (let i = 0; i < TRACE_PTS; i++) {
            const ty = trace[i * 2 + 1]
            buf[i * 3] = trace[i * 2]
            buf[i * 3 + 1] = ty * cp
            buf[i * 3 + 2] = ty * sp
          }
          const line = lineRefs.current[li]
          line?.geometry.setPositions(buf)
          // Charge 2's own lines belong to the pair, so they wait for it
          setLineOpacity(line, ch === 0 ? revLines : revLines * revPair)
          li++
        }
      }
    }

    // Equal and opposite: same length, mirrored directions. Repulsion points outward
    const len = Math.min(ARROW_MAX, Math.max(ARROW_MIN, Math.abs(s.F) * ARROW_SCALE))
    const outward = s.F >= 0 ? 1 : -1
    fillArrow(arrowBufs[0], x1, -outward, len * revArrow1)
    fillArrow(arrowBufs[1], x2, outward, len * revPair)
    arrowRefs.current[0]?.geometry.setPositions(arrowBufs[0])
    arrowRefs.current[1]?.geometry.setPositions(arrowBufs[1])
    setLineOpacity(arrowRefs.current[0], revArrow1)
    setLineOpacity(arrowRefs.current[1], revPair)

    gapBuf[0] = x1; gapBuf[1] = -0.42; gapBuf[2] = 0
    gapBuf[3] = x2; gapBuf[4] = -0.42; gapBuf[5] = 0
    gapRef.current?.geometry.setPositions(gapBuf)
    gapRef.current?.computeLineDistances()
    setLineOpacity(gapRef.current, revPair)

    chargeRefs.current[0]?.position.set(x1, 0, 0)
    chargeRefs.current[1]?.position.set(x2, 0, 0)
    chargeRefs.current[0]?.scale.setScalar(0.55 + 0.25 * Math.abs(s.q1))
    chargeRefs.current[1]?.scale.setScalar((0.55 + 0.25 * Math.abs(s.q2)) * revPair)
    tagRefs.current[0]?.position.set(x1, 0.42, 0)
    tagRefs.current[1]?.position.set(x2, 0.42, 0)

    // The two probes: same field, one at r and one at 2r
    const dNear = PROBE_MIN + (PROBE_MAX - PROBE_MIN) * (0.5 - 0.5 * Math.cos((clock.getElapsedTime() * 2 * Math.PI) / PROBE_PERIOD))
    const dFar = dNear * 2
    const rNear = dNear / SCENE_PER_METER
    probeRefs.current[0]?.position.set(x1, dNear, 0)
    probeRefs.current[1]?.position.set(x1, dFar, 0)
    probeRefs.current[0]?.scale.setScalar(revProbe)
    probeRefs.current[1]?.scale.setScalar(revProbe)
    tagRefs.current[2]?.position.set(x1 + PROBE_TAG_X, dNear, 0)
    tagRefs.current[3]?.position.set(x1 + PROBE_TAG_X, dFar, 0)
    rayBuf[0] = x1; rayBuf[1] = 0; rayBuf[2] = 0
    rayBuf[3] = x1; rayBuf[4] = dFar; rayBuf[5] = 0
    rayRef.current?.geometry.setPositions(rayBuf)
    rayRef.current?.computeLineDistances()
    setLineOpacity(rayRef.current, revProbe * 0.8)

    const ko = locale === 'ko'
    if (revProbe > 0.01 && textRefs.current[TEXT_FAR]) {
      const eNear = (K * Math.abs(s.Q1)) / (rNear * rNear)
      textRefs.current[TEXT_NEAR]!.textContent = `r ${(rNear * 100).toFixed(1)} cm · E ${fmtEfield(eNear)}`
      textRefs.current[TEXT_FAR]!.textContent = `2r ${(rNear * 200).toFixed(1)} cm · E ${fmtEfield(eNear / 4)} · ${ko ? '4분의 1' : 'a quarter'}`
    }
    setOpacity(textRefs.current[TEXT_NEAR], revProbe)
    setOpacity(textRefs.current[TEXT_FAR], revProbe)
    setOpacity(textRefs.current[TEXT_Q2], revPair)
    setOpacity(textRefs.current[TEXT_FORCE], revPair)

    if (labelledFor.current !== valuesRef.current && textRefs.current[TEXT_FORCE]) {
      labelledFor.current = valuesRef.current
      const push = s.F >= 0 ? (ko ? '반발' : 'repulsive') : (ko ? '흡인' : 'attractive')
      textRefs.current[TEXT_Q1]!.textContent = `q1 ${fmtQ(s.q1)}`
      textRefs.current[TEXT_Q2]!.textContent = `q2 ${fmtQ(s.q2)}`
      textRefs.current[TEXT_FORCE]!.textContent = `r ${s.r.toFixed(0)} cm · F ${fmtN(s.F)} · ${push}`
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  return (
    <>
      {/* Camera and target share an x, so the view direction has no x-component and the
          two mirrored force arrows project to exactly the same screen length. Off-axis
          (the usual [3, 2, 5]) foreshortens the far one by 22%, which flatly contradicts
          "equal and opposite". The shared x is offset right, which slides the scene left
          out from under the narration text. */}
      <PerspectiveCamera makeDefault position={[VIEW_X, 1.7, 5.7]} fov={50} />
      <OrbitControls enablePan={false} target={[VIEW_X, 0, 0]} />

      {lineBufs.map((_, l) => (
        <Line
          key={l}
          ref={(line) => { lineRefs.current[l] = line }}
          points={TRACE_INIT}
          lineWidth={1.5}
          color={c['e-field']}
        />
      ))}

      <Line ref={gapRef} points={GAP_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      <Line ref={rayRef} points={RAY_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.07} gapSize={0.05} />
      <Marker ref={(g) => { chargeRefs.current[0] = g }} color={c.voltage} size={0.1} />
      <Marker ref={(g) => { chargeRefs.current[1] = g }} color={c.marker} size={0.1} />
      <Marker ref={(g) => { probeRefs.current[0] = g }} color={c['e-field']} size={0.05} />
      <Marker ref={(g) => { probeRefs.current[1] = g }} color={c['e-field']} size={0.05} />
      {[0, 1].map((a) => (
        <Line
          key={a}
          ref={(line) => { arrowRefs.current[a] = line }}
          points={ARROW_INIT}
          lineWidth={2.5}
          color={c.marker}
        />
      ))}

      {TAG_TEXT_SLOT.map((slot, t) => (
        <group key={t} ref={(g) => { tagRefs.current[t] = g }}>
          <Html center>
            <span ref={(el) => { textRefs.current[slot] = el }} className={labelClass} />
          </Html>
        </group>
      ))}
      <Html position={[0, -0.78, 0]} center>
        <span ref={(el) => { textRefs.current[TEXT_FORCE] = el }} className={labelClass} />
      </Html>
    </>
  )
}

// ── 2D: force against distance. The dot at 2r sits at exactly a quarter of the dot at r ──
const X0 = -3.4
const X_SPAN = 6.8
const Y0 = -1.6
const Y_PER_N = 0.4125 // 8 N fills the plot height
const Y_MAX = 1.78
const CURVE_PTS = 161
const CURVE_INIT: [number, number, number][] = Array.from({ length: CURVE_PTS }, () => [0, 0, 0])
const AXIS_X_INIT: [number, number, number][] = [[X0, Y0, 0], [X0 + X_SPAN, Y0, 0]]
const AXIS_Y_INIT: [number, number, number][] = [[X0, Y0, 0], [X0, Y_MAX, 0]]

const plotX = (rcm: number) => X0 + ((rcm - R_MIN_CM) / (R_MAX_CM - R_MIN_CM)) * X_SPAN
const plotY = (f: number) => Math.min(Y_MAX, Y0 + Math.abs(f) * Y_PER_N)

function CoulombFalloff2DScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const curveBuf = useMemo(() => new Float32Array(CURVE_PTS * 3), [])
  const curveRef = useRef<any>(null)
  const dotRefs = useRef<(THREE.Group | null)[]>([]) // dot at r, dot at 2r
  const tagRefs = useRef<(THREE.Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<unknown>(null)

  useFrame(() => {
    const s = solve(valuesRef.current)
    const kq = K * s.Q1 * s.Q2

    for (let i = 0; i < CURVE_PTS; i++) {
      const rcm = R_MIN_CM + (i / (CURVE_PTS - 1)) * (R_MAX_CM - R_MIN_CM)
      const rm = rcm / 100
      curveBuf[i * 3] = plotX(rcm)
      curveBuf[i * 3 + 1] = plotY(kq / (rm * rm))
      curveBuf[i * 3 + 2] = 0
    }
    curveRef.current?.geometry.setPositions(curveBuf)

    const x = plotX(s.r)
    const y = plotY(s.F)
    dotRefs.current[0]?.position.set(x, y, 0.01)
    tagRefs.current[0]?.position.set(x, y + 0.26, 0)

    const r2 = s.r * 2
    const inRange = r2 <= R_MAX_CM
    const x2 = plotX(Math.min(r2, R_MAX_CM))
    const y2 = plotY(s.F / 4)
    if (dotRefs.current[1]) {
      dotRefs.current[1].visible = inRange
      dotRefs.current[1].position.set(x2, y2, 0.01)
    }
    if (tagRefs.current[1]) {
      tagRefs.current[1].visible = inRange
      tagRefs.current[1].position.set(x2, y2 + 0.26, 0)
    }

    if (labelledFor.current !== valuesRef.current && textRefs.current[1]) {
      labelledFor.current = valuesRef.current
      const ko = locale === 'ko'
      const texts = [
        `r ${s.r.toFixed(0)} cm · F ${fmtN(s.F)}`,
        `2r ${r2.toFixed(0)} cm · F ${fmtN(s.F / 4)} · ${ko ? '거리 2배, 힘 4분의 1' : 'double r, quarter F'}`,
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  return (
    <>
      <Line points={AXIS_X_INIT} lineWidth={1} color={c.guide} />
      <Line points={AXIS_Y_INIT} lineWidth={1} color={c.guide} />
      <Line ref={curveRef} points={CURVE_INIT} lineWidth={2.5} color={c['e-field']} />
      <Marker ref={(g) => { dotRefs.current[0] = g }} color={c.marker} size={0.07} />
      <Marker ref={(g) => { dotRefs.current[1] = g }} color={c.voltage} size={0.06} />
      {[0, 1].map((t) => (
        <group key={t} ref={(g) => { tagRefs.current[t] = g }}>
          <Html center>
            <span ref={(el) => { textRefs.current[t] = el }} className={labelClass} />
          </Html>
        </group>
      ))}
      <Html position={[X0 + X_SPAN + 0.3, Y0, 0]} center>
        <span className={axisClass}>r</span>
      </Html>
      <Html position={[X0 - 0.3, Y_MAX, 0]} center>
        <span className={axisClass}>F</span>
      </Html>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'coulomb-field',
  title: { en: "Coulomb's Law & Electric Field", ko: '쿨롱의 법칙과 전기장' },
  subject: 'electromagnetics',
  order: 1,
  story: [
    {
      set: { q1: 1, q2: 2, r: 10 },
      text: {
        en: 'One charge. Nothing around it, nothing touching it. And the space right next to it is already not empty space anymore.',
        ko: '전하 하나. 주변엔 아무것도 없습니다. 닿아 있는 것도 없습니다. 그런데 이 전하 바로 옆의 공간은, 이미 빈 공간이 아닙니다.',
      },
    },
    {
      text: {
        en: 'That difference is the electric field. Invisible threads pouring out in every direction — the charge reaching out, before there is anything out there to touch.',
        ko: '그 차이가 전기장입니다. 사방으로 쏟아져 나가는 보이지 않는 실. 닿을 상대도 아직 없는데, 전하는 벌써 손을 뻗고 있습니다.',
      },
    },
    {
      text: {
        en: 'Follow them out. They spread. Go twice as far and the same bundle has to cover four times the area, so the field is four times weaker. Nobody invented the 1 ÷ r². It is just a sphere getting bigger.',
        ko: '실을 따라 나가 보세요. 벌어집니다. 두 배 멀어지면 같은 다발이 네 배 넓은 면을 덮어야 하니, 전기장은 네 배 약해집니다. 1 ÷ r²은 누가 정한 게 아닙니다. 구가 커지는 것뿐입니다.',
      },
    },
    {
      text: {
        en: 'Now bring in a second charge. It has no idea where the first one is. It only feels the field at its own feet. Field times charge is force. `F = q × E`',
        ko: '이제 두 번째 전하를 데려옵니다. 이 전하는 첫 번째가 어디 있는지 모릅니다. 제 발밑의 전기장만 느낄 뿐입니다. 전기장 × 전하 = 힘. `F = q × E`',
      },
    },
    {
      text: {
        en: 'q2 is twice the size of q1. So it gets pushed twice as hard, right? Look at the arrows. Dead equal. The big charge sits in the weak field, the small one in the strong field, and the two numbers land right on top of each other.',
        ko: 'q2는 q1의 두 배입니다. 그러면 두 배로 세게 밀리겠죠? 화살표를 보세요. 똑같습니다. 큰 전하는 약한 전기장에, 작은 전하는 강한 전기장에 있어서 두 값이 정확히 포개집니다.',
      },
    },
    {
      set: { q1: 1, q2: 2, r: 20 },
      text: {
        en: 'Now pull them apart. 10 cm to 20 cm — just double. The force goes 1.798 N → 0.449 N. Not half. A quarter. Distance is the most brutal term in the formula.',
        ko: '이제 떼어 놓습니다. 10 cm에서 20 cm. 딱 두 배입니다. 힘은 1.798 N에서 0.449 N. 절반이 아닙니다. 4분의 1입니다. 공식에서 거리가 제일 잔인합니다.',
      },
    },
    {
      set: { q1: 1, q2: -2, r: 10 },
      text: {
        en: 'Flip q2 negative. The whole picture turns inside out. The threads stop flaring apart and thread straight from one charge into the other, and both arrows swing inward. Same formula. One minus sign did all of that.',
        ko: 'q2를 음수로 뒤집습니다. 그림이 통째로 뒤집힙니다. 벌어지던 실들이 한쪽에서 나와 다른 쪽으로 곧장 꿰이고, 화살표는 둘 다 안으로 돌아섭니다. 공식은 그대로입니다. 마이너스 하나가 이걸 다 했습니다.',
      },
    },
    {
      set: { q1: 1, q2: 2, r: 10 },
      text: {
        en: '`F = k × q1 × q2 ÷ r², k = 8.99 × 10⁹ N·m² ÷ C²` One line, and everything you just watched is folded inside it. The sliders are open now — make them unequal, drag them apart, flip a sign.',
        ko: '`F = k × q1 × q2 ÷ r², k = 8.99 × 10⁹ N·m² ÷ C²` 한 줄입니다. 방금 본 전부가 이 안에 접혀 있습니다. 이제 슬라이더를 직접 만져 보세요. 크기를 다르게, 멀리 떼어 놓고, 부호를 뒤집어 보세요.',
      },
    },
  ],
  description: {
    en: `## Coulomb's Law and the Electric Field

**Two charges push or pull with a force that falls off as 1 ÷ r², and it is cleaner to say each charge fills space with a field E that any other charge then feels.**

- Bright dots: the two charges, q1 on the left and q2 on the right. Each dot grows with |q|
- Faint lines: the electric field of the pair, the vector sum of the two point-charge fields. They leave a positive charge and enter a negative one
- Arrows on the charges: the force each one feels. Equal in size and opposite in direction, always, whatever q1 and q2 are

**Coulomb's law**: F = k × q1 × q2 ÷ r², with k = 8.99 × 10⁹ N·m² ÷ C²
**Field of a point charge**: E = k × q ÷ r², and its potential V = k × q ÷ r
**Force from a field**: F = q × E

With q1 = +1 μC, q2 = +2 μC and r = 10 cm the force is 1.798 N, repulsive. The field q1 alone makes at that spot is 8.99 × 10⁵ V ÷ m, and its potential there is 8.99 × 10⁴ V.

Pull r apart and the arrows collapse far faster than the gap grows: double the distance, quarter the force. Flip q2 negative and the field lines switch from flaring apart to threading straight from one charge into the other. The arrow length is proportional to F, so what you see is the real 1 ÷ r² falloff.

In 2D: force against distance. A dot marks the current r, and a second dot at 2r sits at exactly a quarter of its height.`,
    ko: `## 쿨롱의 법칙과 전기장

**두 전하는 1 ÷ r²로 약해지는 힘으로 서로 밀거나 당깁니다. 각 전하가 공간을 전기장 E로 채우고 다른 전하가 그 장을 느낀다고 보는 편이 더 깔끔합니다.**

- 밝은 점: 두 전하입니다. 왼쪽이 q1, 오른쪽이 q2이고 점은 |q|가 클수록 커집니다
- 흐린 선: 두 전하가 만드는 전기장이며, 각 점전하 전기장의 벡터 합입니다. 양전하에서 나와 음전하로 들어갑니다
- 전하에 붙은 화살표: 각 전하가 받는 힘입니다. q1과 q2가 무엇이든 크기는 같고 방향은 반대입니다

**쿨롱의 법칙**: F = k × q1 × q2 ÷ r², k = 8.99 × 10⁹ N·m² ÷ C²
**점전하의 전기장**: E = k × q ÷ r², 전위는 V = k × q ÷ r
**전기장이 주는 힘**: F = q × E

q1 = +1 μC, q2 = +2 μC, r = 10 cm이면 힘은 1.798 N이고 서로 밀어냅니다. 그 지점에서 q1만이 만드는 전기장은 8.99 × 10⁵ V ÷ m, 전위는 8.99 × 10⁴ V입니다.

r을 벌리면 간격이 늘어나는 속도보다 화살표가 훨씬 빨리 줄어듭니다. 거리가 2배가 되면 힘은 4분의 1입니다. q2를 음수로 바꾸면 전기장선이 서로 벌어지는 모양에서 한 전하를 떠나 다른 전하로 곧장 꿰는 모양으로 바뀝니다. 화살표 길이는 F에 비례하므로, 보이는 그대로가 실제 1 ÷ r² 감소입니다.

2D: 거리에 따른 힘입니다. 현재 r에 점이 하나 찍히고, 2r에 찍히는 점은 정확히 그 4분의 1 높이에 놓입니다.`,
  },
  sliderParams: [
    { key: 'q1', label: { en: 'Charge 1 (q1)',  ko: '전하 1 (q1)' }, unit: 'μC', min: -3,        max: 3,        step: 0.1, defaultValue: 1  },
    { key: 'q2', label: { en: 'Charge 2 (q2)',  ko: '전하 2 (q2)' }, unit: 'μC', min: -3,        max: 3,        step: 0.1, defaultValue: 2  },
    { key: 'r',  label: { en: 'Separation (r)', ko: '거리 (r)'    }, unit: 'cm', min: R_MIN_CM, max: R_MAX_CM, step: 1,   defaultValue: 10 },
  ],
  Scene: CoulombFieldScene,
  Scene2D: CoulombFalloff2DScene,
}

export default concept
