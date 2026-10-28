import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { useLocaleStore } from '../../lib/i18n'
import { Marker } from '../../components/scene/Marker'
import { setLineOpacity, setOpacity, useStoryReveal } from '../../components/scene/storyReveal'
import type { ConceptMeta } from '../../types/concept'

const EPS0 = 8.854e-12 // permittivity of vacuum [F ÷ m]
const Q_UNIT = 1e-6 // the charge slider is in μC
const R_MIN_CM = 5
const R_MAX_CM = 16
const D_MAX_CM = 22 // above R_MAX_CM, so the charge can be pushed outside at every r
// One honest linear scale for both r and d: doubling r doubles the drawn radius, and the drawn
// centre is inside the drawn surface exactly when d < r. The slider ranges above are what make
// it fit: the farthest drawn point is (d + r) × SCENE_PER_METER = 1.90 scene units.
const SCENE_PER_METER = 5 // scene units per metre, so the drawn surface grows with r

/*
 * Worked example (re-verified with node -e):
 *   Q = 1 μC, Gaussian sphere of any radius
 *   total flux Ψ = ∮D·dS = Q_enclosed = 1.000 μC, independent of r and of the shape
 *   at r = 10 cm: D = Q ÷ (4π r²) = 1e-6 ÷ (4π × 0.01) = 7.9577e-6 C ÷ m²
 *                 E = D ÷ ε0      = 7.9577e-6 ÷ 8.854e-12 = 8.9877e5 V ÷ m
 *   Coulomb's law gives k Q ÷ r² = 8.99e9 × 1e-6 ÷ 0.01 = 8.990e5 V ÷ m, the same number
 *   doubling r to 20 cm: D = 1.9894e-6 C ÷ m², one quarter, while Ψ is still 1.000 μC
 *   infinite line, λ = 1 μC ÷ m at 5 cm: E = λ ÷ (2π ε0 r) = 3.5951e5 V ÷ m
 *   infinite sheet, σ = 1 μC ÷ m²:       E = σ ÷ (2 ε0)    = 5.6472e4 V ÷ m
 *   charge pushed outside the surface: Q_enclosed = 0, so Ψ = 0 however strong the local field is
 */
function solve({ Q = 1, r = 10, d = 0 }: Record<string, number>) {
  const Qc = Q * Q_UNIT
  const Rm = r / 100
  const dm = d / 100
  const enclosed = dm < Rm
  const Dfield = Qc / (4 * Math.PI * Rm * Rm) // D at distance r from the charge
  return {
    Q, r, d, Qc, Rm, dm, enclosed,
    Qenc: enclosed ? Qc : 0,
    flux: enclosed ? Qc : 0, // ∮D·dS = Q_enclosed
    Dfield,
    E: Dfield / EPS0,
  }
}

// ── Narration steps drive what exists on screen. Each reveal eases 0 → 1 so nothing pops in. ──
// Step 0 is the bare charge with its field lines: the Gaussian surface is a later idea.
const REV_SURFACE = 0 // the imaginary closed surface, its r/d readout and the offset line
const REV_SEG = 1 // the thickened stretch of each line inside the surface, i.e. what gets counted
const REV_FLUX = 2 // the Q_enc and Ψ readout
const REV_DE = 3 // the D and E readout, which unlike Ψ does care about r
const REV_COUNT = 4

function revealTargets(step: number, out: Float32Array) {
  out[REV_SURFACE] = step >= 1 ? 1 : 0
  out[REV_SEG] = step >= 2 ? 1 : 0
  out[REV_FLUX] = step >= 3 ? 1 : 0
  out[REV_DE] = step >= 5 ? 1 : 0
}

function useSliderValues() {
  const valuesRef = useRef<Record<string, number>>(useSliderStore.getState().values)
  useEffect(
    () => useSliderStore.subscribe((s) => s.values, (v) => { valuesRef.current = v }),
    []
  )
  return valuesRef
}

const fmtQ = (q: number) => `+${q.toFixed(2)} μC`
const fmtFlux = (coulombs: number) => `${(coulombs / Q_UNIT).toFixed(3)} μC`
const fmtD = (d: number) => `${(d * 1e6).toFixed(3)} μC ÷ m²`
const fmtE = (e: number) =>
  e >= 1e6 ? `${(e / 1e6).toFixed(2)} MV ÷ m` : `${(e / 1e3).toFixed(1)} kV ÷ m`
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const axisClass = 'pointer-events-none whitespace-nowrap text-[11px] text-muted-foreground'

// ── Field lines: straight rays leaving the charge. Their NUMBER stands for Q, never for r ──
const RAY_MAX = 24
const RAYS_PER_UC = 8
const RAY_R0 = 0.17 // a ray starts just outside the charge's dot, even at Q = 3
const RAY_LEN = 1.9 // drawn radius of the field: every ray stops inside the frame
const STRIDE = 7 // coprime with RAY_MAX, so any prefix of the rays is still spread over the sphere
const GOLDEN = 2.399963229728653

const RAY_DIR = new Float32Array(RAY_MAX * 3)
for (let i = 0; i < RAY_MAX; i++) {
  const j = (i * STRIDE) % RAY_MAX
  const y = 1 - (2 * (j + 0.5)) / RAY_MAX
  const ring = Math.sqrt(Math.max(0, 1 - y * y))
  const phi = j * GOLDEN
  RAY_DIR[i * 3] = ring * Math.cos(phi)
  RAY_DIR[i * 3 + 1] = y
  RAY_DIR[i * 3 + 2] = ring * Math.sin(phi)
}

const RAY_POINTS: [number, number, number][][] = Array.from({ length: RAY_MAX }, (_, i) => {
  const ux = RAY_DIR[i * 3]
  const uy = RAY_DIR[i * 3 + 1]
  const uz = RAY_DIR[i * 3 + 2]
  return [
    [ux * RAY_R0, uy * RAY_R0, uz * RAY_R0],
    [ux * RAY_LEN, uy * RAY_LEN, uz * RAY_LEN],
  ]
})

const SEG_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]

function GaussLawScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()

  // One 2-point buffer per ray for the piece of it that lies inside the Gaussian surface
  const segBufs = useMemo(() => Array.from({ length: RAY_MAX }, () => new Float32Array(6)), [])
  const offsetBuf = useMemo(() => new Float32Array(6), [])

  const rayRefs = useRef<any[]>([])
  const segRefs = useRef<any[]>([])
  const offsetRef = useRef<any>(null)
  const surfaceRef = useRef<THREE.Group>(null)
  const wireMatRef = useRef<THREE.MeshBasicMaterial>(null)
  const shellMatRef = useRef<THREE.MeshBasicMaterial>(null)
  const chargeRef = useRef<THREE.Group>(null)
  const surfTagRef = useRef<THREE.Group>(null)
  const textRefs = useRef<(HTMLSpanElement | null)[]>([]) // charge, surface, flux, density
  const labelledFor = useRef<unknown>(null)
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(() => {
    const s = solve(valuesRef.current)
    const cx = s.dm * SCENE_PER_METER // Gaussian surface centre, on the x-axis
    const Rs = s.Rm * SCENE_PER_METER
    const outside = cx * cx - Rs * Rs // negative while the charge sits inside the surface

    const shown = Math.min(RAY_MAX, Math.max(2, Math.round(s.Q * RAYS_PER_UC)))
    for (let i = 0; i < RAY_MAX; i++) {
      const ray = rayRefs.current[i]
      const seg = segRefs.current[i]
      if (ray) ray.visible = i < shown
      if (!seg) continue
      if (i >= shown) {
        seg.visible = false
        continue
      }
      const ux = RAY_DIR[i * 3]
      const b = ux * cx
      const disc = b * b - outside
      if (disc <= 0) {
        seg.visible = false
        continue
      }
      const root = Math.sqrt(disc)
      let t0 = b - root // where the ray enters the surface, negative when the charge is inside
      let t1 = b + root // where it leaves
      if (t1 <= RAY_R0) {
        seg.visible = false
        continue
      }
      if (t0 < RAY_R0) t0 = RAY_R0
      if (t1 > RAY_LEN) t1 = RAY_LEN
      if (t1 - t0 < 1e-3) {
        seg.visible = false
        continue
      }
      const uy = RAY_DIR[i * 3 + 1]
      const uz = RAY_DIR[i * 3 + 2]
      const buf = segBufs[i]
      buf[0] = ux * t0; buf[1] = uy * t0; buf[2] = uz * t0
      buf[3] = ux * t1; buf[4] = uy * t1; buf[5] = uz * t1
      seg.visible = true
      seg.geometry.setPositions(buf)
      setLineOpacity(seg, reveal[REV_SEG])
    }

    surfaceRef.current?.position.set(cx, 0, 0)
    surfaceRef.current?.scale.setScalar(Rs)
    if (wireMatRef.current) wireMatRef.current.opacity = (s.enclosed ? 0.32 : 0.13) * reveal[REV_SURFACE]
    if (shellMatRef.current) shellMatRef.current.opacity = (s.enclosed ? 0.06 : 0.02) * reveal[REV_SURFACE]
    chargeRef.current?.scale.setScalar(0.8 + 0.3 * s.Q)
    surfTagRef.current?.position.set(cx, Rs + 0.45, 0) // clear above the surface, not on it

    offsetBuf[0] = 0; offsetBuf[1] = -0.3; offsetBuf[2] = 0
    offsetBuf[3] = cx; offsetBuf[4] = -0.3; offsetBuf[5] = 0
    if (offsetRef.current) {
      offsetRef.current.visible = cx > 0.02
      offsetRef.current.geometry.setPositions(offsetBuf)
      offsetRef.current.computeLineDistances()
      setLineOpacity(offsetRef.current, reveal[REV_SURFACE])
    }

    // Every readout fades in with the idea it belongs to
    setOpacity(textRefs.current[1], reveal[REV_SURFACE])
    setOpacity(textRefs.current[2], reveal[REV_FLUX])
    setOpacity(textRefs.current[3], reveal[REV_DE])

    if (labelledFor.current !== valuesRef.current && textRefs.current[3]) {
      labelledFor.current = valuesRef.current
      const ko = locale === 'ko'
      const where = s.enclosed
        ? (ko ? '전하가 안에 있음' : 'charge inside')
        : (ko ? '전하가 밖에 있음' : 'charge outside')
      const texts = [
        `Q ${fmtQ(s.Q)}`,
        `r ${s.r.toFixed(0)} cm · d ${s.d.toFixed(0)} cm · ${where}`,
        `Q_enc ${fmtFlux(s.Qenc)} · Ψ ${fmtFlux(s.flux)}`,
        `D ${fmtD(s.Dfield)} · E ${fmtE(s.E)}`,
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />

      {RAY_POINTS.map((pts, i) => (
        <Line
          key={i}
          ref={(line) => { rayRefs.current[i] = line }}
          points={pts}
          lineWidth={1.5}
          color={c['e-field']}
        />
      ))}
      {RAY_POINTS.map((_, i) => (
        <Line
          key={i}
          ref={(line) => { segRefs.current[i] = line }}
          points={SEG_INIT}
          lineWidth={3}
          color={c['e-field']}
        />
      ))}

      <group ref={surfaceRef}>
        <mesh>
          <sphereGeometry args={[1, 20, 12]} />
          <meshBasicMaterial ref={wireMatRef} color={c.guide} wireframe transparent opacity={0.32} depthWrite={false} />
        </mesh>
        <mesh>
          <sphereGeometry args={[1, 20, 12]} />
          <meshBasicMaterial color={c.guide} transparent opacity={0.06} depthWrite={false} side={THREE.DoubleSide} ref={shellMatRef} />
        </mesh>
      </group>

      <Line ref={offsetRef} points={SEG_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      <Marker ref={chargeRef} color={c.voltage} size={0.1} />

      <Html position={[0, -1.1, 0]} center>
        <span ref={(el) => { textRefs.current[0] = el }} className={labelClass} />
      </Html>
      <group ref={surfTagRef}>
        <Html center>
          <span ref={(el) => { textRefs.current[1] = el }} className={labelClass} />
        </Html>
      </group>
      <Html position={[0, -1.4, 0]} center>
        <span ref={(el) => { textRefs.current[2] = el }} className={labelClass} />
      </Html>
      <Html position={[0, -1.68, 0]} center>
        <span ref={(el) => { textRefs.current[3] = el }} className={labelClass} />
      </Html>
    </>
  )
}

// ── 2D: D plunges as 1 ÷ r² while the flux line stays perfectly flat ──
const X0 = -3.4
const X_SPAN = 6.8
const Y0 = -1.6
const Y_MAX = 1.8
const D_REF = Q_UNIT / (4 * Math.PI * 0.01) // D of 1 μC at 10 cm, the plot's unit height
const Y_PER_REF = 0.8
const Q_MAX_UC = 3
const CURVE_PTS = 161
const CURVE_INIT: [number, number, number][] = Array.from({ length: CURVE_PTS }, () => [0, 0, 0])
const AXIS_X_INIT: [number, number, number][] = [[X0, Y0, 0], [X0 + X_SPAN, Y0, 0]]
const AXIS_Y_INIT: [number, number, number][] = [[X0, Y0, 0], [X0, Y_MAX, 0]]

const plotX = (rcm: number) => X0 + ((rcm - R_MIN_CM) / (R_MAX_CM - R_MIN_CM)) * X_SPAN
const plotD = (d: number) => Math.min(Y_MAX, Y0 + (d / D_REF) * Y_PER_REF)
const plotFlux = (coulombs: number) => Y0 + (coulombs / Q_UNIT / Q_MAX_UC) * 3

function GaussFlux2DScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const ko = locale === 'ko'

  const curveBuf = useMemo(() => new Float32Array(CURVE_PTS * 3), [])
  const fluxBuf = useMemo(() => new Float32Array(6), [])
  const curveRef = useRef<any>(null)
  const fluxRef = useRef<any>(null)
  const dropRef = useRef<any>(null)
  const dropBuf = useMemo(() => new Float32Array(6), [])
  const dotRefs = useRef<(THREE.Group | null)[]>([]) // on the D curve, on the flux line
  const tagRefs = useRef<(THREE.Group | null)[]>([])
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<unknown>(null)
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(() => {
    const s = solve(valuesRef.current)

    for (let i = 0; i < CURVE_PTS; i++) {
      const rcm = R_MIN_CM + (i / (CURVE_PTS - 1)) * (R_MAX_CM - R_MIN_CM)
      const rm = rcm / 100
      curveBuf[i * 3] = plotX(rcm)
      curveBuf[i * 3 + 1] = plotD(s.Qc / (4 * Math.PI * rm * rm))
      curveBuf[i * 3 + 2] = 0
    }
    curveRef.current?.geometry.setPositions(curveBuf)

    const yFlux = plotFlux(s.flux)
    fluxBuf[0] = X0; fluxBuf[1] = yFlux; fluxBuf[2] = 0
    fluxBuf[3] = X0 + X_SPAN; fluxBuf[4] = yFlux; fluxBuf[5] = 0
    fluxRef.current?.geometry.setPositions(fluxBuf)

    const x = plotX(s.r)
    const yD = plotD(s.Dfield)
    dropBuf[0] = x; dropBuf[1] = Y0; dropBuf[2] = 0
    dropBuf[3] = x; dropBuf[4] = yD; dropBuf[5] = 0
    if (dropRef.current) {
      dropRef.current.geometry.setPositions(dropBuf)
      dropRef.current.computeLineDistances()
    }

    dotRefs.current[0]?.position.set(x, yD, 0.01)
    tagRefs.current[0]?.position.set(x, yD + 0.26, 0)
    dotRefs.current[1]?.position.set(x, yFlux, 0.01)
    tagRefs.current[1]?.position.set(x, yFlux + 0.26, 0)

    const wFlux = reveal[REV_FLUX]
    setLineOpacity(fluxRef.current, wFlux)
    dotRefs.current[1]?.scale.setScalar(wFlux)
    setOpacity(textRefs.current[1], wFlux)

    if (labelledFor.current !== valuesRef.current && textRefs.current[1]) {
      labelledFor.current = valuesRef.current
      const flat = s.enclosed
        ? (ko ? 'r이 변해도 일정' : 'flat as r changes')
        : (ko ? '전하가 밖이라 0' : 'zero, charge outside')
      const texts = [
        `D ${fmtD(s.Dfield)} · r ${s.r.toFixed(0)} cm`,
        `Ψ ${fmtFlux(s.flux)} · ${flat}`,
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  return (
    <>
      <Line points={AXIS_X_INIT} lineWidth={1} color={c.guide} />
      <Line points={AXIS_Y_INIT} lineWidth={1} color={c.guide} />
      <Line ref={dropRef} points={SEG_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.07} gapSize={0.06} />
      <Line ref={curveRef} points={CURVE_INIT} lineWidth={2.5} color={c['e-field']} />
      <Line ref={fluxRef} points={SEG_INIT} lineWidth={2.5} color={c.voltage} />
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
      <Html position={[X0 - 0.35, Y_MAX, 0]} center>
        <span className={axisClass}>D, Ψ</span>
      </Html>
      <Html position={[X0 + X_SPAN - 0.9, Y_MAX - 0.1, 0]} center>
        <span className={axisClass}>{ko ? '아래로 휘는 곡선: D' : 'falling curve: D'}</span>
      </Html>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'gauss-law',
  title: { en: "Gauss's Law", ko: '가우스 법칙' },
  subject: 'electromagnetics',
  order: 3,
  cardPosition: [2.8, 1.0, -2.0],
  description: {
    en: `## Gauss's Law and Electric Flux Density

**The total electric flux leaving any closed surface equals the charge inside it, so the surface's size and shape never matter, only what it encloses.**

- Bright dot: the point charge Q. It grows with Q
- Faint straight lines: the field of the charge. Their number stands for Q, so growing the surface never adds or removes one
- Thick stretches of those lines: the part of each line that is inside the Gaussian surface. With the charge inside, every line starts there and leaves once. With the charge outside, a line that reaches the surface enters on one side and leaves on the other, so it contributes nothing
- Faint sphere: the Gaussian surface. It fades when it encloses no charge
- Faint dashed line: the offset d between the charge and the centre of the surface

**Gauss's law**: Ψ = ∮D·dS = Q_enclosed, with D = ε0 × E and ε0 = 8.854 × 10⁻¹² F ÷ m
**Point charge**: D = Q ÷ (4π × r²), so E = Q ÷ (4π × ε0 × r²)
**Infinite line charge**: E = λ ÷ (2π × ε0 × r)
**Infinite sheet of charge**: E = σ ÷ (2 × ε0), the same everywhere

With Q = 1 μC the total flux is 1 μC through any closed surface. At r = 10 cm, D = 7.958 × 10⁻⁶ C ÷ m² and E = 8.988 × 10⁵ V ÷ m, which is exactly what Coulomb's law gives. A line charge λ = 1 μC ÷ m gives E = 3.595 × 10⁵ V ÷ m at 5 cm, and a sheet σ = 1 μC ÷ m² gives E = 5.647 × 10⁴ V ÷ m at any distance.

Grow r: D and E collapse as 1 ÷ r², yet Ψ does not budge. That is why a symmetric problem is one line of algebra instead of an integral. Then push d past r so the charge sits outside: the lines still thread through the surface, but each one enters and leaves, Q_enc goes to zero, and so does Ψ.

In 2D: the falling curve is D against r, the flat line is Ψ. One plunges, the other never moves, until the charge leaves the surface and Ψ drops to the axis.`,
    ko: `## 가우스 법칙과 전속밀도

**닫힌 면을 빠져나가는 전기 선속의 총량은 그 안에 든 전하와 같습니다. 면의 크기와 모양은 아무 상관이 없고, 무엇을 감싸는지만 중요합니다.**

- 밝은 점: 점전하 Q이며 Q가 클수록 커집니다
- 흐린 직선: 전하가 만드는 전기장입니다. 선의 개수가 Q를 나타내므로 면을 키워도 선이 늘거나 줄지 않습니다
- 그 선의 굵어진 구간: 각 선이 가우스 면 안쪽을 지나는 부분입니다. 전하가 안에 있으면 모든 선이 거기서 출발해 한 번 나갑니다. 전하가 밖에 있으면 면에 닿은 선이 한쪽으로 들어와 다른 쪽으로 나가므로 기여가 0입니다
- 흐린 구: 가우스 면입니다. 감싼 전하가 없으면 더 흐려집니다
- 흐린 점선: 전하와 면 중심 사이의 거리 d입니다

**가우스 법칙**: Ψ = ∮D·dS = Q_enclosed, D = ε0 × E, ε0 = 8.854 × 10⁻¹² F ÷ m
**점전하**: D = Q ÷ (4π × r²), 따라서 E = Q ÷ (4π × ε0 × r²)
**무한 선전하**: E = λ ÷ (2π × ε0 × r)
**무한 평면전하**: E = σ ÷ (2 × ε0), 거리와 무관하게 일정합니다

Q = 1 μC이면 어떤 닫힌 면을 지나도 총 선속은 1 μC입니다. r = 10 cm에서 D = 7.958 × 10⁻⁶ C ÷ m², E = 8.988 × 10⁵ V ÷ m이고, 이는 쿨롱의 법칙으로 얻는 값과 정확히 같습니다. 선전하 λ = 1 μC ÷ m는 5 cm에서 E = 3.595 × 10⁵ V ÷ m, 평면전하 σ = 1 μC ÷ m²는 거리와 무관하게 E = 5.647 × 10⁴ V ÷ m입니다.

r을 키우면 D와 E는 1 ÷ r²로 급격히 작아지는데 Ψ는 꿈쩍도 하지 않습니다. 대칭적인 문제가 적분이 아니라 한 줄의 계산으로 풀리는 이유입니다. 이제 d를 r보다 크게 밀어 전하를 면 밖으로 내보내 보세요. 선은 여전히 면을 꿰뚫지만 하나하나가 들어왔다 나가므로 Q_enc가 0이 되고, Ψ도 0이 됩니다.

2D: 아래로 휘는 곡선은 r에 따른 D이고, 평평한 선은 Ψ입니다. 하나는 곤두박질치는데 다른 하나는 전혀 움직이지 않다가, 전하가 면을 벗어나는 순간 Ψ만 축으로 떨어집니다.`,
  },
  story: [
    {
      set: { Q: 1, r: 10, d: 0 },
      text: {
        en: 'One charge, sitting there. Field lines spraying out in every direction. Count them — eight. Remember that number.',
        ko: '전하 하나가 놓여 있습니다. 전기력선이 사방으로 뻗어 나갑니다. 세어 보세요. 여덟 개. 이 숫자를 기억하세요.',
      },
    },
    {
      set: { Q: 1, r: 10, d: 0 },
      text: {
        en: 'Now wrap an imaginary bag around it. It is not a thing. There is no bag. You drew it, and that is the only reason it exists.',
        ko: '이제 상상의 봉투를 씌웁니다. 물건이 아닙니다. 봉투 같은 건 없습니다. 당신이 그렸기 때문에 존재할 뿐입니다.',
      },
    },
    {
      set: { Q: 1, r: 10, d: 0 },
      text: {
        en: 'Watch where the lines cross it. Every line born inside punches out exactly once, and never comes back. Eight lines in, eight lines out.',
        ko: '선이 봉투를 뚫는 자리를 보세요. 안에서 태어난 선은 정확히 한 번 뚫고 나가고 다시는 돌아오지 않습니다. 여덟 개가 태어나 여덟 개가 나갑니다.',
      },
    },
    {
      set: { Q: 1, r: 10, d: 0 },
      text: {
        en: 'So counting the lines leaving the bag is the same as counting the charge inside it. That sentence is the whole law. `Ψ = ∮D·dS = Q_enc`',
        ko: '그러니까 봉투를 나가는 선을 세는 것은 안에 든 전하를 세는 것과 같습니다. 이 문장이 법칙의 전부입니다. `Ψ = ∮D·dS = Q_enc`',
      },
    },
    {
      set: { Q: 1, r: 16, d: 0 },
      text: {
        en: 'Blow the bag up. The surface area is now 2.56 times bigger — and the flux reads 1.000 μC, exactly what it read before. The bag\'s size has no vote.',
        ko: '봉투를 부풀립니다. 표면적이 2.56배가 됐습니다. 그런데 선속은 1.000 μC, 아까와 똑같습니다. 봉투 크기에는 투표권이 없습니다.',
      },
    },
    {
      set: { Q: 1, r: 5, d: 0 },
      text: {
        en: 'But the strength at one spot is a different question. Pull in to 5 cm and the field is fierce: D = 31.831 μC ÷ m².',
        ko: '하지만 한 점에서의 세기는 다른 이야기입니다. 5 cm까지 다가가면 전기장은 사납습니다. D = 31.831 μC ÷ m².',
      },
    },
    {
      set: { Q: 1, r: 10, d: 0 },
      text: {
        en: 'Double the distance and it collapses to a quarter — 7.958. Same lines, spread over four times the area. `D = Q ÷ (4π r²)` Total flux: unchanged.',
        ko: '거리를 두 배로 하면 4분의 1인 7.958로 주저앉습니다. 같은 선이 4배 넓은 면에 퍼진 겁니다. `D = Q ÷ (4π r²)` 총 선속은 그대로입니다.',
      },
    },
    {
      set: { Q: 3, r: 10, d: 0 },
      text: {
        en: 'Triple the charge instead. Three times the lines, three times the flux: 3.000 μC. Flux only ever looks at one thing.',
        ko: '이번엔 전하를 세 배로 합니다. 선도 세 배, 선속도 세 배인 3.000 μC. 선속이 보는 건 딱 하나뿐입니다.',
      },
    },
    {
      set: { Q: 1, r: 10, d: 20 },
      text: {
        en: 'Now the trick that makes it click. Slide the bag off the charge entirely. The lines still stab right through it — but each one enters one side and leaves the other. In, out, cancelled. Q_enc = 0, so Ψ = 0.',
        ko: '이제 감이 오게 만드는 묘기입니다. 봉투를 전하 바깥으로 완전히 밀어냅니다. 선은 여전히 봉투를 관통합니다. 그런데 하나하나가 한쪽으로 들어와 반대쪽으로 나갑니다. 들어오고, 나가고, 상계. Q_enc = 0이므로 Ψ = 0입니다.',
      },
    },
    {
      set: { Q: 1, r: 10, d: 0 },
      text: {
        en: '`Ψ = Q_enc` It never asks the bag\'s size, shape or position. Only what is inside. That is why a symmetric problem collapses from an integral into one line of arithmetic. The sliders are yours.',
        ko: '`Ψ = Q_enc` 봉투의 크기도, 모양도, 위치도 묻지 않습니다. 안에 뭐가 있는지만 묻습니다. 대칭적인 문제가 적분에서 산수 한 줄로 쪼그라드는 이유입니다. 이제 슬라이더는 당신 것입니다.',
      },
    },
  ],
  sliderParams: [
    { key: 'Q', label: { en: 'Enclosed charge (Q)',    ko: '전하량 (Q)'        }, unit: 'μC', min: 0.2,      max: Q_MAX_UC, step: 0.1, defaultValue: 1  },
    { key: 'r', label: { en: 'Surface radius (r)',     ko: '가우스 면 반지름 (r)' }, unit: 'cm', min: R_MIN_CM, max: R_MAX_CM, step: 1,   defaultValue: 10 },
    { key: 'd', label: { en: 'Surface offset (d)',     ko: '면 중심 이동 (d)'    }, unit: 'cm', min: 0,        max: D_MAX_CM, step: 1,   defaultValue: 0  },
  ],
  Scene: GaussLawScene,
  Scene2D: GaussFlux2DScene,
}

export default concept
