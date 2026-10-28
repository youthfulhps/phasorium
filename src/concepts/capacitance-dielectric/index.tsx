import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Edges, Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { useLocaleStore } from '../../lib/i18n'
import { Marker } from '../../components/scene/Marker'
import {
  setLineOpacity, setOpacity, useStoryReveal,
} from '../../components/scene/storyReveal'
import type { ConceptMeta } from '../../types/concept'

const EPS0 = 8.854e-12 // vacuum permittivity [F ÷ m]

// Slider ranges, also used by the 2D plot
const A_MIN = 20 // [cm²]
const A_MAX = 200
const D_MIN = 0.5 // [mm]
const D_MAX = 3

// Drawing scale. Plate side and gap are both proportional to the real geometry,
// so a thinner gap really looks thinner and a wider plate really looks wider.
const SIDE_PER_SQRT_CM2 = 0.2 // scene units per sqrt(cm²); 100 cm² -> side 2.0
const GAP_PER_MM = 0.55 // scene units per mm; 1 mm -> 0.55
const PLATE_T = 0.045 // drawn plate thickness [scene units]

// Reused result object so solve() allocates nothing when called inside useFrame.
const OUT = { A: 0, d: 0, er: 0, V: 0, C: 0, Q: 0, E: 0, W: 0, u: 0 }

/*
 * Pure physics. A in cm², d in mm, er dimensionless, V in volts.
 *   C = eps0 × er × A ÷ d      Q = C × V      E = V ÷ d
 *   W = ½ × C × V² = ½ × Q × V       u = ½ × eps0 × er × E²
 *
 * Worked example, re-verified with node -e:
 *   A = 100 cm² = 0.01 m², d = 1 mm, er = 1
 *     C = 8.854e-12 × 1 × 0.01 ÷ 0.001 = 8.854e-11 F = 88.54 pF
 *     at V = 100 V: Q = 8.854 nC, E = 100 V ÷ 0.001 m = 100 kV ÷ m,
 *                   W = ½ × 8.854e-11 × 100² = 4.427e-7 J = 0.443 μJ
 *   Same geometry with er = 4: C = 354.16 pF, Q = 35.416 nC, W = 1.771 μJ,
 *                   u = ½ × 8.854e-12 × 4 × (1e5)² = 0.177 J ÷ m³
 */
function solve({ A = 100, d = 1, er = 4, V = 100 }: Record<string, number>) {
  const Am = A * 1e-4 // cm² -> m²
  const dm = d * 1e-3 // mm -> m
  const C = (EPS0 * er * Am) / dm
  const E = V / dm
  OUT.A = A
  OUT.d = d
  OUT.er = er
  OUT.V = V
  OUT.C = C
  OUT.Q = C * V
  OUT.E = E
  OUT.W = 0.5 * C * V * V
  OUT.u = 0.5 * EPS0 * er * E * E
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

const fmtC = (f: number) => {
  const pf = f * 1e12
  return pf >= 1000 ? `${(pf / 1000).toFixed(2)} nF` : `${pf.toFixed(2)} pF`
}
const fmtQ = (q: number) => {
  const nc = q * 1e9
  return nc >= 1000 ? `${(nc / 1000).toFixed(2)} μC` : `${nc.toFixed(2)} nC`
}
const fmtW = (w: number) => {
  const uj = w * 1e6
  return uj >= 1000 ? `${(uj / 1000).toFixed(2)} mJ` : `${uj.toFixed(3)} μJ`
}
const fmtE = (e: number) =>
  e >= 1e6 ? `${(e / 1e6).toFixed(2)} MV ÷ m` : `${(e / 1e3).toFixed(0)} kV ÷ m`

const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
// Scene text that sits in front of the plates gets a backdrop chip: the plates are bright in dark
// mode (and bloom spreads them) and dark in light mode, so bare text loses contrast either way.
const chipClass = 'pointer-events-none whitespace-nowrap rounded-sm bg-background/80 px-1.5 py-0.5 text-[11px] font-semibold tabular-nums text-foreground'
const noteClass = 'pointer-events-none whitespace-nowrap rounded-sm bg-background/80 px-1.5 py-0.5 text-[11px] text-muted-foreground'
const energyClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-power'
const axisClass = 'pointer-events-none whitespace-nowrap text-[11px] text-muted-foreground'

// ── Charge symbols drawn on the plates: one symbol per CHARGE_PER_SYMBOL of stored charge ──
const CHARGE_PER_SYMBOL = 4e-9 // [C]
const SYM_MIN = 2
const SYM_MAX = 14
const symbolCount = (q: number) =>
  Math.max(SYM_MIN, Math.min(SYM_MAX, Math.round(q / CHARGE_PER_SYMBOL)))
const symbolRow = (sign: string, n: number) => `${sign} `.repeat(n).trimEnd()

// ── Field arrows: a 4 × 4 grid over the plate footprint, drawn so their density tracks E ──
// Flux through the plates is E × A, so the number of arrows shown is proportional to that
// product and the arrows per unit plate area come out proportional to E itself.
const GRID = 4
const ARROW_MAX = GRID * GRID
const FLUX_PER_ARROW = 1250 // (kV ÷ m) × cm² per arrow; 100 kV ÷ m over 100 cm² -> 8 arrows
// Bayer-style rank per grid cell, so a partial set of arrows stays evenly spread out
const ARROW_RANK = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5]
const ARROW_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]

// ── Polarisation arrows inside the slab, pointing against E ──
const POL_X = [-1, 1, -1, 1]
const POL_Z = [-1, -1, 1, 1]

// ── Narration: what each step wants on screen ──
// The dielectric itself is not a slot: the slab and its P arrows already follow εr, so a step
// simply sets εr = 1 to take the material back out of the gap.
const REV_Q = 0 // the +Q and −Q symbol rows on the plates
const REV_READ = 1 // the C · Q · E readout, and the 2D read-off line
const REV_E = 2 // the field arrows crossing the gap
const REV_W = 3 // the stored-energy line, in both views
const REV_COUNT = 4

function revealTargets(step: number, out: Float32Array) {
  out[REV_Q] = step >= 1 ? 1 : 0
  out[REV_READ] = step >= 2 ? 1 : 0
  out[REV_E] = step >= 3 ? 1 : 0
  out[REV_W] = step >= 8 ? 1 : 0
}

/** Vertical arrow as a 5-point polyline: shaft from y0 to y1, then two barbs folded back from the tip. */
function fillVArrow(buf: Float32Array, x: number, z: number, y0: number, y1: number) {
  const dir = y1 >= y0 ? 1 : -1
  const len = Math.abs(y1 - y0)
  const head = Math.min(0.1, len * 0.3)
  const barb = Math.min(0.055, len * 0.18)
  buf[0] = x; buf[1] = y0; buf[2] = z
  buf[3] = x; buf[4] = y1; buf[5] = z
  buf[6] = x - barb; buf[7] = y1 - dir * head; buf[8] = z
  buf[9] = x; buf[10] = y1; buf[11] = z
  buf[12] = x + barb; buf[13] = y1 - dir * head; buf[14] = z
}

function CapacitorScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()

  const arrowBufs = useMemo(
    () => Array.from({ length: ARROW_MAX }, () => new Float32Array(15)),
    []
  )
  const polBufs = useMemo(() => Array.from({ length: POL_X.length }, () => new Float32Array(15)), [])

  const arrowRefs = useRef<any[]>([])
  const polRefs = useRef<any[]>([])
  const plateRefs = useRef<(THREE.Mesh | null)[]>([])
  const slabRef = useRef<THREE.Mesh>(null)
  const tagRefs = useRef<(THREE.Group | null)[]>([]) // plus row, minus row, polarisation label
  const textRefs = useRef<(HTMLSpanElement | null)[]>([]) // plus, minus, polarisation, readout, energy
  const labelledFor = useRef<unknown>(null)
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(() => {
    const s = solve(valuesRef.current)
    const side = SIDE_PER_SQRT_CM2 * Math.sqrt(s.A)
    const gap = GAP_PER_MM * s.d
    const half = gap / 2
    const spacing = side / GRID

    plateRefs.current[0]?.scale.set(side, PLATE_T, side)
    plateRefs.current[0]?.position.set(0, half + PLATE_T / 2, 0)
    plateRefs.current[1]?.scale.set(side, PLATE_T, side)
    plateRefs.current[1]?.position.set(0, -half - PLATE_T / 2, 0)

    // E points from the positive plate down to the negative one, and E = V ÷ d whether or not
    // the slab is in, so the arrow length always spans the gap exactly.
    const shown = Math.max(1, Math.min(ARROW_MAX, Math.round((s.E / 1e3) * s.A / FLUX_PER_ARROW)))
    for (let i = 0; i < ARROW_MAX; i++) {
      const line = arrowRefs.current[i]
      const visible = ARROW_RANK[i] < shown && reveal[REV_E] > 0.01
      if (line) line.visible = visible
      if (!visible) continue
      setLineOpacity(line, reveal[REV_E])
      const col = i % GRID
      const row = (i - col) / GRID
      const x = (col - (GRID - 1) / 2) * spacing
      const z = (row - (GRID - 1) / 2) * spacing
      fillVArrow(arrowBufs[i], x, z, half, -half)
      line?.geometry.setPositions(arrowBufs[i])
    }

    // The slab fills the gap; its polarisation opposes E, which is why the plates must hold
    // er times more charge to keep the same E across the same gap.
    const polFrac = 1 - 1 / s.er
    const hasSlab = polFrac > 0.02
    if (slabRef.current) {
      slabRef.current.visible = hasSlab
      slabRef.current.scale.set(side * 0.92, gap, side * 0.92)
      const mat = slabRef.current.material as THREE.MeshBasicMaterial
      mat.opacity = 0.12 + 0.26 * polFrac
    }
    const polLen = gap * 0.7 * polFrac
    for (let p = 0; p < POL_X.length; p++) {
      const line = polRefs.current[p]
      if (line) line.visible = hasSlab
      if (!hasSlab) continue
      fillVArrow(polBufs[p], POL_X[p] * spacing, POL_Z[p] * spacing, -polLen / 2, polLen / 2)
      line?.geometry.setPositions(polBufs[p])
    }

    tagRefs.current[0]?.position.set(0, half + PLATE_T + 0.24, 0)
    tagRefs.current[1]?.position.set(0, -half - PLATE_T - 0.24, 0)
    // Mid-gap and in front of the stack, but kept well inside the plate's own footprint: pushed
    // out past the corner it projected over the sidebar wheel at the widest plate setting.
    tagRefs.current[2]?.position.set(-side * 0.25 - 0.2, 0, side / 2 + 0.5)

    setOpacity(textRefs.current[0], reveal[REV_Q])
    setOpacity(textRefs.current[1], reveal[REV_Q])
    setOpacity(textRefs.current[3], reveal[REV_READ])
    setOpacity(textRefs.current[4], reveal[REV_W])

    if (labelledFor.current !== valuesRef.current && textRefs.current[4]) {
      labelledFor.current = valuesRef.current
      const ko = locale === 'ko'
      // drei's Html ignores a hidden parent group, so the P label hides itself
      if (textRefs.current[2]) textRefs.current[2]!.style.opacity = hasSlab ? '1' : '0'
      const n = symbolCount(s.Q)
      const texts = [
        `${symbolRow('+', n)}   +Q`,
        `${symbolRow('−', n)}   −Q`,
        ko ? 'P, E의 반대 방향' : 'P, opposite to E',
        `C ${fmtC(s.C)} · Q ${fmtQ(s.Q)} · E ${fmtE(s.E)}`,
        `W ${fmtW(s.W)} · u ${s.u.toFixed(3)} J ÷ m³`,
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  return (
    <>
      {/* Aimed right of centre so the plates clear the narration panel on the right */}
      <PerspectiveCamera makeDefault position={[3, 2, 5.6]} fov={50} />
      <OrbitControls enablePan={false} target={[0.8, 0, 0]} />

      {[0, 1].map((p) => (
        <mesh key={p} ref={(m) => { plateRefs.current[p] = m }}>
          <boxGeometry args={[1, 1, 1]} />
          {/* Back faces only, like Resistor's glass tube: the plate reads as a thin pane whose
              outline carries the shape, so it never clips to black in light mode nor turns into
              one big bloom-blown white surface in dark mode, and the charge rows, the dielectric
              and the field arrows stay visible through it. */}
          <meshBasicMaterial
            color={c.voltage}
            transparent
            opacity={0.22}
            depthWrite={false}
            side={THREE.BackSide}
          />
          <Edges color={c.voltage} lineWidth={2} />
        </mesh>
      ))}

      <mesh ref={slabRef}>
        <boxGeometry args={[1, 1, 1]} />
        <meshBasicMaterial
          color={c.capacitance}
          transparent
          opacity={0.2}
          depthWrite={false}
          side={THREE.BackSide}
        />
        <Edges color={c.capacitance} lineWidth={1} />
      </mesh>

      {arrowBufs.map((_, a) => (
        <Line
          key={a}
          ref={(line) => { arrowRefs.current[a] = line }}
          points={ARROW_INIT}
          lineWidth={2}
          color={c['e-field']}
        />
      ))}

      {polBufs.map((_, p) => (
        <Line
          key={p}
          ref={(line) => { polRefs.current[p] = line }}
          points={ARROW_INIT}
          lineWidth={2}
          color={c.capacitance}
        />
      ))}

      {[0, 1, 2].map((t) => (
        <group key={t} ref={(g) => { tagRefs.current[t] = g }}>
          <Html center>
            <span
              ref={(el) => { textRefs.current[t] = el }}
              className={t === 2 ? noteClass : chipClass}
            />
          </Html>
        </group>
      ))}
      <Html position={[0, -1.3, 0]} center>
        <span ref={(el) => { textRefs.current[3] = el }} className={labelClass} />
      </Html>
      <Html position={[0, -1.6, 0]} center>
        <span ref={(el) => { textRefs.current[4] = el }} className={energyClass} />
      </Html>
    </>
  )
}

// ── 2D: capacitance against plate separation, the 1 ÷ d hyperbola ──
const X0 = -3.4
const X_SPAN = 6.8
const Y0 = -1.6
const Y_MAX = 1.8
const PF_FULL = 800 // pF at the top of the plot
const Y_PER_PF = (Y_MAX - Y0) / PF_FULL
const CURVE_PTS = 161
const CURVE_INIT: [number, number, number][] = Array.from({ length: CURVE_PTS }, () => [0, 0, 0])
const AXIS_X_INIT: [number, number, number][] = [[X0, Y0, 0], [X0 + X_SPAN, Y0, 0]]
const AXIS_Y_INIT: [number, number, number][] = [[X0, Y0, 0], [X0, Y_MAX, 0]]
const GUIDE_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]

const plotX = (dmm: number) => X0 + ((dmm - D_MIN) / (D_MAX - D_MIN)) * X_SPAN
const plotY = (farads: number) => Math.min(Y_MAX, Y0 + farads * 1e12 * Y_PER_PF)
/**
 * The gap at which C reaches the top of the plot. A 1 ÷ d curve shoots off the ceiling at small
 * gaps whenever the plates are wide or the dielectric is strong, and capping C there would draw a
 * flat lid — a horizontal line saying the hyperbola levels off, which it never does. So the curve
 * is trimmed to the stretch that fits instead: it starts at the left edge only while it can, and
 * otherwise begins partway in, running off the top exactly where it really does.
 */
const gapAtCeiling = (numerator: number) => (numerator / (PF_FULL * 1e-12)) * 1e3

function CapacitanceVsGap2DScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()

  const curveBuf = useMemo(() => new Float32Array(CURVE_PTS * 3), [])
  const guideBuf = useMemo(() => new Float32Array(6), [])
  const curveRef = useRef<any>(null)
  const guideRef = useRef<any>(null)
  const dotRef = useRef<THREE.Group>(null)
  const tagRef = useRef<THREE.Group>(null)
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<unknown>(null)
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(() => {
    const s = solve(valuesRef.current)
    const numerator = EPS0 * s.er * s.A * 1e-4 // C × d, held fixed along the curve

    const dStart = Math.max(D_MIN, gapAtCeiling(numerator))
    for (let i = 0; i < CURVE_PTS; i++) {
      const dmm = dStart + (i / (CURVE_PTS - 1)) * (D_MAX - dStart)
      curveBuf[i * 3] = plotX(dmm)
      curveBuf[i * 3 + 1] = plotY(numerator / (dmm * 1e-3))
      curveBuf[i * 3 + 2] = 0
    }
    curveRef.current?.geometry.setPositions(curveBuf)

    // Park the marker at the curve's own start when the setting is off the top of the plot, so it
    // stays on the line it is reading rather than floating above its end.
    const x = plotX(Math.max(s.d, dStart))
    const y = plotY(s.C)
    dotRef.current?.position.set(x, y, 0.01)
    tagRef.current?.position.set(x, y + 0.26, 0)
    guideBuf[0] = X0; guideBuf[1] = y; guideBuf[2] = 0
    guideBuf[3] = x; guideBuf[4] = y; guideBuf[5] = 0
    guideRef.current?.geometry.setPositions(guideBuf)
    guideRef.current?.computeLineDistances()

    // The curve is the material fact and stays; what the story reveals is the reading off it.
    setLineOpacity(guideRef.current, reveal[REV_READ])
    dotRef.current?.scale.setScalar(reveal[REV_READ])
    setOpacity(textRefs.current[0], reveal[REV_READ])
    setOpacity(textRefs.current[1], reveal[REV_W])

    if (labelledFor.current !== valuesRef.current && textRefs.current[1]) {
      labelledFor.current = valuesRef.current
      const ko = locale === 'ko'
      const texts = [
        `d ${s.d.toFixed(1)} mm · C ${fmtC(s.C)}`,
        `${ko ? '저장 에너지' : 'stored energy'} W ${fmtW(s.W)}`,
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  return (
    <>
      <Line points={AXIS_X_INIT} lineWidth={1} color={c.guide} />
      <Line points={AXIS_Y_INIT} lineWidth={1} color={c.guide} />
      <Line ref={guideRef} points={GUIDE_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      <Line ref={curveRef} points={CURVE_INIT} lineWidth={2.5} color={c.capacitance} />
      <Marker ref={dotRef} color={c.marker} size={0.07} />
      <group ref={tagRef}>
        <Html center>
          <span ref={(el) => { textRefs.current[0] = el }} className={labelClass} />
        </Html>
      </group>
      <Html position={[X0 + 1.1, Y_MAX - 0.05, 0]} center>
        <span ref={(el) => { textRefs.current[1] = el }} className={energyClass} />
      </Html>
      <Html position={[X0 + X_SPAN + 0.3, Y0, 0]} center>
        <span className={axisClass}>d</span>
      </Html>
      <Html position={[X0 - 0.3, Y_MAX, 0]} center>
        <span className={axisClass}>C</span>
      </Html>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'capacitance-dielectric',
  title: { en: 'Capacitance & Dielectrics', ko: '정전용량과 유전체' },
  subject: 'electromagnetics',
  order: 4,
  description: {
    en: `## Capacitance, Dielectrics and Stored Energy

**Two plates hold charge in proportion to the voltage across them, C = Q ÷ V, and the geometry plus the material in between is what sets that ratio.**

- Solid slabs: the two plates. The symbol rows read +Q and −Q, one symbol per 4 nC, so the row grows as the plates take on more charge
- Arrows across the gap: the uniform field E, pointing from the positive plate to the negative one. Their density tracks E, since the number of arrows follows the flux E × A
- Translucent block: the dielectric, with the four faint P arrows inside it pointing against E. That is the polarisation of the material
- Amber readout: the stored energy W and the energy density u

**Capacitance**: C = ε0 × εr × A ÷ d, with ε0 = 8.854 × 10⁻¹² F ÷ m
**Charge and field**: Q = C × V, E = V ÷ d
**Stored energy**: W = ½ × C × V² = ½ × Q × V, and per unit volume u = ½ × ε0 × εr × E²

With A = 100 cm², d = 1 mm and εr = 1 the capacitance is 88.54 pF. At V = 100 V that holds 8.854 nC, the field is 100 kV ÷ m and the stored energy is 0.443 μJ. Raise εr to 4 and the same box becomes 354.2 pF, holding 35.42 nC and 1.771 μJ.

Widen the plates or squeeze the gap and C climbs, exactly as A ÷ d says. Then watch εr: the field arrows do not change, because E = V ÷ d is fixed by the source, but the P arrows appear and the plate symbol rows get longer. The polarisation cancels part of what the plate charge would otherwise produce, so the source has to push εr times more charge in to hold the same field, and that is the capacitance rising.

In 2D: C against the gap d, a 1 ÷ d curve with a marker and a dashed read-off at the current setting. The plot tops out at 800 pF.`,
    ko: `## 정전용량과 유전체, 그리고 저장 에너지

**두 극판은 양단 전압에 비례하는 전하를 담으며, C = Q ÷ V입니다. 그 비율을 정하는 것은 극판의 형상과 사이에 든 물질입니다.**

- 꽉 찬 판: 두 극판입니다. 기호 줄은 각각 +Q와 −Q이고 4 nC마다 기호 하나이므로, 극판이 전하를 더 담으면 줄이 길어집니다
- 간격을 지나는 화살표: 균일한 전기장 E이며 양극판에서 음극판으로 향합니다. 화살표 개수는 플럭스 E × A를 따르므로 화살표 밀도가 E를 나타냅니다
- 반투명한 덩어리: 유전체입니다. 그 안의 흐린 P 화살표 4개는 E의 반대 방향을 가리키며, 물질의 분극을 뜻합니다
- 황색 수치: 저장 에너지 W와 에너지 밀도 u입니다

**정전용량**: C = ε0 × εr × A ÷ d, ε0 = 8.854 × 10⁻¹² F ÷ m
**전하와 전기장**: Q = C × V, E = V ÷ d
**저장 에너지**: W = ½ × C × V² = ½ × Q × V, 단위 체적당 u = ½ × ε0 × εr × E²

A = 100 cm², d = 1 mm, εr = 1이면 정전용량은 88.54 pF입니다. V = 100 V에서 전하는 8.854 nC, 전기장은 100 kV ÷ m, 저장 에너지는 0.443 μJ입니다. εr을 4로 올리면 같은 구조가 354.2 pF이 되어 35.42 nC과 1.771 μJ을 담습니다.

극판을 넓히거나 간격을 좁히면 A ÷ d 그대로 C가 커집니다. 이제 εr을 움직여 보세요. E = V ÷ d는 전원이 고정하므로 전기장 화살표는 그대로지만, P 화살표가 나타나고 극판의 기호 줄이 길어집니다. 분극이 극판 전하가 만들려던 전기장의 일부를 상쇄하므로, 같은 전기장을 유지하려면 전원이 εr배의 전하를 더 밀어넣어야 합니다. 그것이 바로 정전용량의 증가입니다.

2D: 간격 d에 대한 C이며, 1 ÷ d 곡선 위에 현재 설정을 가리키는 점과 눈금을 읽는 점선이 놓입니다. 그래프 상단은 800 pF입니다.`,
  },
  story: [
    {
      set: { A: 100, d: 1, er: 1, V: 100 },
      text: {
        en: 'Two metal plates facing each other across a gap. Nothing joins them, nothing touches. That is already a capacitor.',
        ko: '금속판 두 장이 간격을 두고 마주 보고 있습니다. 연결된 것도, 닿은 것도 없습니다. 이게 벌써 커패시터입니다.',
      },
    },
    {
      set: { A: 100, d: 1, er: 1, V: 100 },
      text: {
        en: 'Put 100 V across them and the source pumps charge off one plate and onto the other, so one ends up +Q and the other exactly −Q.',
        ko: '여기에 100 V를 걸면 전원이 한쪽 판의 전하를 퍼서 반대쪽에 쌓습니다. 한쪽은 +Q, 반대쪽은 정확히 같은 양의 −Q가 됩니다.',
      },
    },
    {
      set: { A: 100, d: 1, er: 1, V: 100 },
      text: {
        en: 'That stack of charge comes to 8.85 nC. Divide it by the 100 V holding it there and you get 88.54 pF, the capacitance of this one pair of plates. `C = Q ÷ V`',
        ko: '쌓인 전하는 8.85 nC. 이걸 붙잡고 있는 100 V로 나누면 88.54 pF, 이 판 한 쌍의 정전용량입니다. `C = Q ÷ V`',
      },
    },
    {
      set: { A: 100, d: 1, er: 1, V: 100 },
      text: {
        en: 'Between the plates the field is the same strength everywhere. 100 V spread across 1 mm of gap is 100 kV ÷ m. `E = V ÷ d`',
        ko: '두 판 사이는 어디서나 세기가 같은 전기장입니다. 100 V를 1 mm에 나눠 걸었으니 100 kV ÷ m. `E = V ÷ d`',
      },
    },
    {
      set: { A: 100, d: 2, er: 1, V: 100 },
      text: {
        en: 'Pull the gap out to 2 mm and both halve: 44.27 pF of capacitance, 50 kV ÷ m of field. Same voltage, twice the distance to cross.',
        ko: '간격을 2 mm로 벌리면 둘 다 절반이 됩니다. 정전용량 44.27 pF, 전기장 50 kV ÷ m. 전압은 같은데 건너야 할 거리가 두 배가 되었으니까요.',
      },
    },
    {
      set: { A: 200, d: 1, er: 1, V: 100 },
      text: {
        en: 'Close the gap again and double the plate area instead: 177.08 pF. Area multiplies, gap divides, and that is the whole geometry of a capacitor. `C = ε0 × εr × A ÷ d`',
        ko: '간격을 되돌리고 대신 판 면적을 두 배로 넓히면 177.08 pF. 면적은 곱하고 간격은 나눕니다. 커패시터의 형상은 이것으로 끝입니다. `C = ε0 × εr × A ÷ d`',
      },
    },
    {
      set: { A: 200, d: 1, er: 4, V: 100 },
      text: {
        en: 'Now slide a dielectric into the gap. The field arrows do not budge, still 100 kV ÷ m, but the charge on the plates jumps four times over to 70.83 nC.',
        ko: '이제 사이에 유전체를 끼웁니다. 전기장 화살표는 100 kV ÷ m에서 꼼짝도 하지 않는데, 판에 쌓인 전하만 네 배로 뛰어 70.83 nC이 됩니다.',
      },
    },
    {
      set: { A: 200, d: 1, er: 4, V: 100 },
      text: {
        en: 'The faint arrows inside the slab are the reason. The material polarises against the field and cancels part of it, so the source has to push εr times more charge in just to hold the same 100 kV ÷ m. That is the capacitance climbing to 708.32 pF.',
        ko: '덩어리 안의 흐린 화살표가 그 이유입니다. 유전체가 전기장을 깎는 쪽으로 분극되니, 같은 100 kV ÷ m를 유지하려면 전원이 εr배의 전하를 더 밀어넣어야 합니다. 그게 708.32 pF로 커진 정전용량입니다.',
      },
    },
    {
      set: { A: 200, d: 1, er: 4, V: 100 },
      text: {
        en: 'Charge sitting under voltage is stored energy. Right now the gap is holding 3.542 μJ, which is 0.177 J in every cubic metre of it.',
        ko: '전압 아래 쌓인 전하는 저장된 에너지이기도 합니다. 지금 이 간격이 3.542 μJ을 담고 있고, 1 m³마다 0.177 J씩입니다.',
      },
    },
    {
      set: { A: 200, d: 1, er: 4, V: 200 },
      text: {
        en: 'Double the voltage and the energy goes up four times, to 14.166 μJ, because the voltage enters squared. Charge scales, energy squares. `W = ½ × C × V²`',
        ko: '전압을 두 배로 올리면 에너지는 네 배인 14.166 μJ이 됩니다. 전압이 제곱으로 들어가기 때문입니다. 전하는 비례, 에너지는 제곱. `W = ½ × C × V²`',
      },
    },
  ],
  sliderParams: [
    { key: 'A',  label: { en: 'Plate area (A)',            ko: '극판 면적 (A)'     }, unit: 'cm²', min: A_MIN, max: A_MAX, step: 5,   defaultValue: 100 },
    { key: 'd',  label: { en: 'Separation (d)',            ko: '극판 간격 (d)'     }, unit: 'mm',  min: D_MIN, max: D_MAX, step: 0.1, defaultValue: 1   },
    { key: 'er', label: { en: 'Relative permittivity (εr)', ko: '비유전율 (εr)'    },              min: 1,     max: 10,    step: 0.1, defaultValue: 4   },
    { key: 'V',  label: { en: 'Applied voltage (V)',       ko: '인가 전압 (V)'     }, unit: 'V',   min: 20,    max: 200,   step: 5,   defaultValue: 100 },
  ],
  Scene: CapacitorScene,
  Scene2D: CapacitanceVsGap2DScene,
}

export default concept
