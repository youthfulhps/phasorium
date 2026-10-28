import { useRef, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import type { Group, Mesh } from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import type { ConceptMeta } from '../../types/concept'

// Pure physics. Shared flux couples the two coils: M = k·sqrt(L1·L2) with 0 ≤ k ≤ 1.
// In series the windings either help or fight each other: L = L1 + L2 ± 2M.
// Worked example, verified with node -e:
//   L1 = 100 mH, L2 = 400 mH, k = 0.5 -> M = 0.5·sqrt(0.1 × 0.4) H = 100.000 mH
//     series aiding   = 100 + 400 + 2×100 = 700.000 mH
//     series opposing = 100 + 400 − 2×100 = 300.000 mH
//   k = 1   -> M = 200.000 mH, aiding = 900.000 mH, opposing = 100.000 mH
//   k = 0   -> M =   0.000 mH, aiding = opposing = 500.000 mH
// Result lives in one module-scope record so no frame allocates.
const S = { L1: 100, L2: 400, k: 0.5, M: 100, aiding: 700, opposing: 300 }

function solve(v: Record<string, number>) {
  const L1 = v.L1_mH ?? 100
  const L2 = v.L2_mH ?? 400
  const k = v.k ?? 0.5
  const M = k * Math.sqrt(L1 * L2)
  S.L1 = L1
  S.L2 = L2
  S.k = k
  S.M = M
  S.aiding = L1 + L2 + 2 * M
  S.opposing = L1 + L2 - 2 * M
  return S
}

function useSliderValues() {
  const valuesRef = useRef<Record<string, number>>(useSliderStore.getState().values)
  useEffect(
    () => useSliderStore.subscribe((s) => s.values, (v) => { valuesRef.current = v }),
    []
  )
  return valuesRef
}

const fmtL = (mH: number) => (mH >= 1000 ? `${(mH / 1000).toFixed(2)} H` : `${mH.toFixed(0)} mH`)
// Labels read in the foreground color; only the accents (current, power) tint their text, and
// neither appears here, so every label in this concept is plain foreground.
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'

const L_FULL = 500 // slider maximum, used to map inductance to a drawn coil radius
const rOf = (mH: number) => 0.18 + 0.42 * Math.sqrt(Math.min(mH, L_FULL) / L_FULL)

// ── 3D: two helices side by side, with the flux that threads them ──
// Each coil carries its own closed flux loops. Raising k slides every loop toward the midpoint and
// stretches it until, at k = 1, all six span both coils: that is what "fully coupled" looks like.
const X1 = -1.9
const X2 = 0.3
const XM = (X1 + X2) / 2
const HALF = 0.45 // half the drawn length of one coil
const PAD = 0.42 // how far a flux loop reaches past the coil it belongs to
const A_OWN = HALF + PAD
const A_SHARED = (X2 - X1) / 2 + HALF + PAD

const COIL_N = 161
const COIL_TURNS = 8
// Unit helix along local x ∈ [−0.5, 0.5] at radius 1; the group is scaled to length and radius
const COIL: [number, number, number][] = Array.from({ length: COIL_N }, (_, i) => {
  const u = i / (COIL_N - 1)
  const a = u * COIL_TURNS * 2 * Math.PI
  return [u - 0.5, Math.sin(a), Math.cos(a)]
})

const LOOP_N = 72
const LOOP_INIT: [number, number, number][] = Array.from(
  { length: LOOP_N },
  (): [number, number, number] => [0, 0, 0]
)
// A solenoid field line is a closed loop in the half plane that holds the axis: out through the
// bore at a small radius, back along the outside at a large one. Shape is fixed, so it is
// tabulated once as a rounded rectangle and only stretched each frame.
const UX = new Float32Array(LOOP_N)
const UY = new Float32Array(LOOP_N)
for (let i = 0; i < LOOP_N; i++) {
  const t = (i / (LOOP_N - 1)) * 2 * Math.PI
  const cs = Math.cos(t)
  const sn = Math.sin(t)
  UX[i] = Math.sign(cs) * Math.sqrt(Math.abs(cs))
  UY[i] = Math.sign(sn) * Math.sqrt(Math.abs(sn))
}
const LOOP_COIL = [0, 0, 0, 1, 1, 1]
const LOOP_PHI = [0, Math.PI / 3, (2 * Math.PI) / 3, Math.PI / 6, Math.PI / 2, (5 * Math.PI) / 6]
const LOOP_IN = [0.25, 0.5, 0.75, 0.25, 0.5, 0.75] // bore radius, in coil radii
const LOOP_OUT = [1.3, 1.75, 2.25, 1.3, 1.75, 2.25] // return radius, in coil radii
const FLUX_FLOW = 2.2 // sheen bands per second at the peak of the alternating flux
const FLUX_OMEGA = 1.1 // on-screen rad/s: a real mains cycle is far too fast to follow

function MutualInductanceCoilScene() {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  const loopBufs = useMemo(
    () => Array.from({ length: 6 }, () => new Float32Array(LOOP_N * 3)),
    []
  )
  const loopRefs = useRef<any[]>([])
  const coilRefs = useRef<(Group | null)[]>([])
  const dotRefs = useRef<(Group | null)[]>([])
  // 0 L1, 1 L2, 2 M, 3 aiding, 4 opposing
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const flow = useRef(0)
  const labelledFor = useRef<Record<string, number> | null>(null)

  useFrame(({ clock }) => {
    const s = solve(valuesRef.current)
    const r1 = rOf(s.L1)
    const r2 = rOf(s.L2)
    coilRefs.current[0]?.scale.set(2 * HALF, r1, r1)
    coilRefs.current[1]?.scale.set(2 * HALF, r2, r2)
    // Dots mark the two terminals wound the same way, both at the near end here
    dotRefs.current[0]?.position.set(X1 - HALF - 0.12, r1 + 0.1, 0)
    dotRefs.current[1]?.position.set(X2 - HALF - 0.12, r2 + 0.1, 0)

    const rk = Math.max(r1, r2)
    for (let j = 0; j < 6; j++) {
      const own = LOOP_COIL[j] === 0
      const xcOwn = own ? X1 : X2
      const rOwn = own ? r1 : r2
      // k slides each loop from its own coil to the pair and widens it to span both
      const xc = xcOwn + (XM - xcOwn) * s.k
      const a = A_OWN + (A_SHARED - A_OWN) * s.k
      const rb = rOwn + (rk - rOwn) * s.k
      const rin = rb * LOOP_IN[j]
      const rout = rb * LOOP_OUT[j]
      const rm = (rin + rout) / 2
      const rh = (rout - rin) / 2
      const cos = Math.cos(LOOP_PHI[j])
      const sin = Math.sin(LOOP_PHI[j])
      const buf = loopBufs[j]
      for (let i = 0; i < LOOP_N; i++) {
        const rho = rm + rh * UY[i]
        buf[i * 3] = xc + a * UX[i]
        buf[i * 3 + 1] = rho * cos
        buf[i * 3 + 2] = rho * sin
      }
      loopRefs.current[j]?.geometry.setPositions(buf)
    }

    // The flux alternates, so the sheen runs one way then the other
    flow.current = FLUX_FLOW * Math.sin(clock.getElapsedTime() * FLUX_OMEGA)

    if (labelledFor.current !== valuesRef.current && textRefs.current[4]) {
      labelledFor.current = valuesRef.current
      if (textRefs.current[0]) textRefs.current[0]!.textContent = `L1 ${fmtL(s.L1)}`
      if (textRefs.current[1]) textRefs.current[1]!.textContent = `L2 ${fmtL(s.L2)}`
      if (textRefs.current[2]) {
        textRefs.current[2]!.textContent = `M = k·sqrt(L1·L2) = ${fmtL(s.M)}`
      }
      if (textRefs.current[3]) {
        textRefs.current[3]!.textContent = `L1 + L2 + 2M = ${fmtL(s.aiding)}`
      }
      textRefs.current[4]!.textContent = `L1 + L2 − 2M = ${fmtL(s.opposing)}`
    }
  })

  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />

      {/* Background, lights, grid, and bloom come from the shared SceneStage */}
      {/* The shared flux: six closed field lines that stretch across as k rises */}
      {LOOP_COIL.map((_, j) => (
        <TrailLine
          key={j}
          ref={(l) => { loopRefs.current[j] = l }}
          points={LOOP_INIT}
          lineWidth={2}
          color={c['b-field']}
          fade={false}
          flowSpeed={() => flow.current}
        />
      ))}

      {/* The two coils, each a helix whose radius grows with its inductance */}
      <group position={[X1, 0, 0]} ref={(g) => { coilRefs.current[0] = g }}>
        <Line points={COIL} lineWidth={2} color={c.inductance} />
      </group>
      <group position={[X2, 0, 0]} ref={(g) => { coilRefs.current[1] = g }}>
        <Line points={COIL} lineWidth={2} color={c.resistance} />
      </group>

      {/* Dot convention: the terminal of each coil that the winding starts from */}
      <Marker ref={(g) => { dotRefs.current[0] = g }} color={c.marker} size={0.055} />
      <Marker ref={(g) => { dotRefs.current[1] = g }} color={c.marker} size={0.055} />

      <Html position={[X1, -1.6, 0]} center>
        <span ref={(el) => { textRefs.current[0] = el }} className={labelClass} />
      </Html>
      <Html position={[X2, -1.6, 0]} center>
        <span ref={(el) => { textRefs.current[1] = el }} className={labelClass} />
      </Html>
      <Html position={[XM, 1.72, 0]} center>
        <div className="pointer-events-none text-center leading-5">
          <div><span ref={(el) => { textRefs.current[2] = el }} className={labelClass} /></div>
          <div><span ref={(el) => { textRefs.current[3] = el }} className={labelClass} /></div>
          <div><span ref={(el) => { textRefs.current[4] = el }} className={labelClass} /></div>
        </div>
      </Html>
    </>
  )
}

// ── 2D: the two series connections as bars against the same L1 + L2 mark ──
// Both rows start from L1 and L2 laid end to end. Aiding adds a 2M piece past that mark, opposing
// takes the same piece back off it, so the gap between the two totals is 4M and k opens it.
const X0 = -3.5
const SPAN = 6.5
const Y_A = 0.75
const Y_B = -0.75
const BAR_H = 0.34
const SEG = new Float32Array(6)
const SEG_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]

function setBar(m: Mesh | null | undefined, xa: number, xb: number, y: number, z: number) {
  if (!m) return
  m.scale.set(Math.max(xb - xa, 1e-4), 1, 1)
  m.position.set((xa + xb) / 2, y, z)
}
const put = (g: Group | null | undefined, x: number, y: number) => { g?.position.set(x, y, 0) }

function MutualInductanceBars2DScene() {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  // 0 aiding L1, 1 aiding L2, 2 aiding +2M, 3 opposing L1, 4 opposing L2, 5 opposing −2M
  const barRefs = useRef<(Mesh | null)[]>([])
  // 0 aiding L1, 1 aiding L2, 2 +2M, 3 opposing L1, 4 opposing L2, 5 −2M
  const tagRefs = useRef<(Group | null)[]>([])
  // 0 aiding total, 1 opposing total, 2 L1 + L2 mark, 3 M
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const totalRefs = useRef<(Group | null)[]>([])
  // 0 aiding end, 1 opposing end, 2 3 aiding dots, 4 5 opposing dots
  const dotRefs = useRef<(Group | null)[]>([])
  const guideRef = useRef<any>(null)
  const markRef = useRef<Group>(null)
  const labelledFor = useRef<Record<string, number> | null>(null)

  useFrame(() => {
    const s = solve(valuesRef.current)
    // One scale for both rows, pinned to the largest total these two coils can reach
    // (that is k = 1 aiding, which equals (sqrt(L1) + sqrt(L2))²), so the rows compare honestly
    const kx = SPAN / (s.L1 + s.L2 + 2 * Math.sqrt(s.L1 * s.L2))
    const x1 = X0 + s.L1 * kx
    const xRef = X0 + (s.L1 + s.L2) * kx
    const xAid = X0 + s.aiding * kx
    const xOpp = X0 + s.opposing * kx

    setBar(barRefs.current[0], X0, x1, Y_A, 0)
    setBar(barRefs.current[1], x1, xRef, Y_A, 0)
    setBar(barRefs.current[2], xRef, xAid, Y_A, 0)
    setBar(barRefs.current[3], X0, x1, Y_B, 0)
    setBar(barRefs.current[4], x1, xRef, Y_B, 0)
    // The piece 2M takes back off L1 + L2, drawn over the bar it eats into
    setBar(barRefs.current[5], xOpp, xRef, Y_B, 0.01)

    put(tagRefs.current[0], (X0 + x1) / 2, Y_A - 0.33)
    put(tagRefs.current[1], (x1 + xRef) / 2, Y_A - 0.33)
    put(tagRefs.current[2], (xRef + xAid) / 2, Y_A - 0.33)
    put(tagRefs.current[3], (X0 + x1) / 2, Y_B - 0.33)
    put(tagRefs.current[4], (x1 + xRef) / 2, Y_B - 0.33)
    put(tagRefs.current[5], (xOpp + xRef) / 2, Y_B - 0.33)
    put(totalRefs.current[0], (X0 + xAid) / 2, Y_A + 0.62)
    put(totalRefs.current[1], (X0 + xOpp) / 2, Y_B + 0.62)

    dotRefs.current[0]?.position.set(xAid, Y_A, 0.02)
    dotRefs.current[1]?.position.set(xOpp, Y_B, 0.02)
    // Aiding: both dots at the same end of their coil. Opposing: at opposite ends.
    dotRefs.current[2]?.position.set(X0 + 0.14, Y_A + 0.27, 0)
    dotRefs.current[3]?.position.set(x1 + 0.14, Y_A + 0.27, 0)
    dotRefs.current[4]?.position.set(X0 + 0.14, Y_B + 0.27, 0)
    dotRefs.current[5]?.position.set(xRef - 0.14, Y_B + 0.27, 0)

    // The common L1 + L2 mark: aiding runs past it, opposing stops short of it
    SEG[0] = xRef; SEG[1] = Y_B - 0.5; SEG[2] = 0
    SEG[3] = xRef; SEG[4] = Y_A + 0.5; SEG[5] = 0
    guideRef.current?.geometry.setPositions(SEG)
    guideRef.current?.computeLineDistances()
    markRef.current?.position.set(xRef, Y_A + 0.95, 0)

    if (labelledFor.current !== valuesRef.current && textRefs.current[3]) {
      labelledFor.current = valuesRef.current
      if (textRefs.current[0]) {
        textRefs.current[0]!.textContent = `L1 + L2 + 2M = ${fmtL(s.aiding)}`
      }
      if (textRefs.current[1]) {
        textRefs.current[1]!.textContent = `L1 + L2 − 2M = ${fmtL(s.opposing)}`
      }
      if (textRefs.current[2]) textRefs.current[2]!.textContent = `L1 + L2 = ${fmtL(s.L1 + s.L2)}`
      textRefs.current[3]!.textContent = `M = k·sqrt(L1·L2) = ${fmtL(s.M)}`
    }
  })

  const barColors = [c.inductance, c.resistance, c['b-field'], c.inductance, c.resistance, c['b-field']]
  const tagTexts = ['L1', 'L2', '+2M', 'L1', 'L2', '−2M']

  return (
    <>
      <Line
        ref={guideRef}
        points={SEG_INIT}
        lineWidth={1}
        color={c.guide}
        dashed
        dashSize={0.08}
        gapSize={0.06}
      />

      {barColors.map((color, i) => (
        <mesh key={i} ref={(m) => { barRefs.current[i] = m }}>
          <planeGeometry args={[1, BAR_H]} />
          <meshBasicMaterial color={color} />
        </mesh>
      ))}

      {tagTexts.map((text, i) => (
        <group key={i} ref={(g) => { tagRefs.current[i] = g }}>
          <Html center>
            <span className={labelClass}>{text}</span>
          </Html>
        </group>
      ))}

      {[0, 1].map((i) => (
        <group key={i} ref={(g) => { totalRefs.current[i] = g }}>
          <Html center>
            <span ref={(el) => { textRefs.current[i] = el }} className={labelClass} />
          </Html>
        </group>
      ))}

      <group ref={markRef}>
        <Html center>
          <span ref={(el) => { textRefs.current[2] = el }} className={labelClass} />
        </Html>
      </group>
      <Html position={[0, -1.7, 0]} center>
        <span ref={(el) => { textRefs.current[3] = el }} className={labelClass} />
      </Html>

      <Marker ref={(g) => { dotRefs.current[0] = g }} color={c.marker} size={0.07} />
      <Marker ref={(g) => { dotRefs.current[1] = g }} color={c.marker} size={0.07} />
      {/* Dot convention: same end in the top row, opposite ends in the bottom one */}
      {[2, 3, 4, 5].map((i) => (
        <Marker key={i} ref={(g) => { dotRefs.current[i] = g }} color={c.marker} size={0.045} />
      ))}
    </>
  )
}

const concept: ConceptMeta = {
  id: 'mutual-inductance',
  title: { en: 'Mutual Inductance & Coupling', ko: '상호 인덕턴스와 결합계수' },
  subject: 'circuits',
  order: 22,
  description: {
    en: `## Mutual Inductance and Coupling

**Two coils that share flux change each other, and once they are in series the total inductance depends on which way they are wound.**

- L1 and L2: the two coils, drawn as helices whose radius grows with their inductance
- The deep cool loops threading them: the magnetic flux. At k = 0 each loop stays with its own coil; as k rises the loops stretch across until at k = 1 every line threads both
- The dots on the coils: the dot convention, marking the terminal each winding starts from
- The three readouts: M, and the total inductance of the same pair in series wound to help or to fight each other

**Mutual inductance**: M = k·sqrt(L1·L2), with 0 ≤ k ≤ 1
**Series aiding** (dots at the same end): L = L1 + L2 + 2M
**Series opposing** (dots at opposite ends): L = L1 + L2 − 2M
**Induced voltage**: v2 = M·(di1 ÷ dt)

Worked example: L1 = 100 mH, L2 = 400 mH and k = 0.5 give M = 100 mH, so series aiding is 700 mH and series opposing is 300 mH. Push k to 1 and M doubles to 200 mH, splitting the pair to 900 mH against 100 mH.

This is the transformer. Wind both coils on one iron core so k approaches 1, and nearly all the flux the first coil makes threads the second; the turns ratio then sets the voltages, V2 ÷ V1 = N2 ÷ N1.

Raise k and watch the flux loops reach across while the two totals pull apart by 4M. Raise L1 or L2 to grow a coil, which lifts M with it.

In 2D: the two connections as bars built from L1, L2 and the 2M piece, both measured against the same L1 + L2 mark, so aiding runs past it by exactly as much as opposing falls short.`,
    ko: `## 상호 인덕턴스와 결합계수

**자속을 나눠 갖는 두 코일은 서로에게 영향을 줍니다. 직렬로 이으면 감은 방향에 따라 합성 인덕턴스가 달라집니다.**

- L1, L2: 두 코일. 나선으로 그렸고 반지름이 인덕턴스에 따라 커집니다
- 코일을 꿰는 짙은 한색 고리: 자속. k = 0에서는 고리가 각자의 코일에만 머물고, k가 커질수록 늘어나 k = 1에서는 모든 선이 두 코일을 함께 꿰뚫습니다
- 코일 위의 점: 도트 규약. 각 권선이 시작되는 단자를 표시합니다
- 세 개의 수치: M과, 같은 두 코일을 직렬로 이었을 때 서로 돕도록 감은 경우와 거스르도록 감은 경우의 합성 인덕턴스

**상호 인덕턴스**: M = k·sqrt(L1·L2), 0 ≤ k ≤ 1
**가동 접속** (점이 같은 쪽 끝): L = L1 + L2 + 2M
**차동 접속** (점이 반대쪽 끝): L = L1 + L2 − 2M
**유도 전압**: v2 = M·(di1 ÷ dt)

계산 예: L1 = 100 mH, L2 = 400 mH, k = 0.5이면 M = 100 mH이므로 가동 접속은 700 mH, 차동 접속은 300 mH입니다. k를 1까지 올리면 M이 200 mH로 두 배가 되어 900 mH와 100 mH로 더 벌어집니다.

이것이 바로 변압기의 원리입니다. 두 코일을 하나의 철심에 감아 k를 1에 가깝게 만들면 1차 코일이 만든 자속이 거의 전부 2차 코일을 꿰뚫고, 그때부터는 권수비가 전압을 결정합니다. V2 ÷ V1 = N2 ÷ N1.

k를 올리면 자속 고리가 건너편까지 뻗어 나가고 두 합성값이 4M만큼 벌어집니다. L1이나 L2를 올리면 해당 코일이 커지면서 M도 함께 커집니다.

2D: 두 접속을 L1, L2와 2M 조각으로 쌓은 막대로 그렸습니다. 둘 다 같은 L1 + L2 눈금을 기준으로 재므로, 가동 접속이 눈금을 넘어선 만큼 차동 접속이 눈금에 못 미칩니다.`,
  },
  sliderParams: [
    { key: 'L1_mH', label: { en: 'Coil 1 Inductance (L1)', ko: '코일 1 인덕턴스 (L1)' }, unit: 'mH', min: 10, max: 500, step: 10,   defaultValue: 100 },
    { key: 'L2_mH', label: { en: 'Coil 2 Inductance (L2)', ko: '코일 2 인덕턴스 (L2)' }, unit: 'mH', min: 10, max: 500, step: 10,   defaultValue: 400 },
    { key: 'k',     label: { en: 'Coupling Coefficient (k)', ko: '결합계수 (k)' },                    min: 0,  max: 1,   step: 0.01, defaultValue: 0.5 },
  ],
  Scene: MutualInductanceCoilScene,
  Scene2D: MutualInductanceBars2DScene,
}

export default concept
