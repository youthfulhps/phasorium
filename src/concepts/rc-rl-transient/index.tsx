import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import { setLineOpacity, setOpacity, useStoryReveal } from '../../components/scene/storyReveal'
import type { ConceptMeta } from '../../types/concept'

// Slider ranges (also used to normalize the plateau heights)
const V_MAX = 24 // [V]
const RL_R = 10 // [Ω] the RL circuit keeps a fixed resistor so its τ stays watchable

// Time axes: how much circuit time each row's horizontal axis spans
const WINDOW_RC = 6 // [s]
const WINDOW_RL = 0.3 // [s]
// One charge sweep takes SWEEP seconds on screen, then holds until CYCLE and restarts.
// So the RC row plays at real speed (6 s in 6 s) and the RL row is slowed 20× (0.3 s in 6 s).
const SWEEP = 6
const CYCLE = 7.2

const F63 = 1 - Math.exp(-1) // 0.632121
const F99 = 1 - Math.exp(-5) // 0.993262
const P63 = (F63 * 100).toFixed(1)
const P99 = (F99 * 100).toFixed(1)

const C_PTS = 160
const CURVE_INIT: [number, number, number][] = Array.from({ length: C_PTS }, () => [0, 0, 0])
const SEG_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]

// ── Narration steps drive what exists on screen. Each reveal eases 0 → 1 so nothing pops in. ──
// Step 0 is the bare rising curve: no guides, no markers, nothing to read off yet.
const REV_FINAL = 0 // the plateau the curve is heading for, and its readout
const REV_63 = 1 // the 63.2 % level
const REV_TAU = 2 // the 1τ vertical, i.e. where τ can be read off the picture
const REV_TAU5 = 3 // the 5τ vertical, i.e. "settled"
const REV_RL = 4 // the whole RL row, brought in as the mirror of the RC one
const REV_COUNT = 5

function revealTargets(step: number, out: Float32Array) {
  out[REV_FINAL] = step >= 1 ? 1 : 0
  out[REV_63] = step >= 2 ? 1 : 0
  out[REV_TAU] = step >= 3 ? 1 : 0
  out[REV_TAU5] = step >= 6 ? 1 : 0
  out[REV_RL] = step >= 7 ? 1 : 0
}

// Reused result object so solve() allocates nothing when called inside useFrame.
const OUT = { tauRC: 0, tauRL: 0, vFinal: 0, iFinal: 0 }

// Pure physics. R in kΩ, C in μF, L in H, V in V.
// RC charging: vC = V × (1 − e^(−t ÷ τ)), τ = R × C.  RL rise: i = (V ÷ R) × (1 − e^(−t ÷ τ)), τ = L ÷ R.
// Worked example, checked with node:
//   RC  V = 12 V, R = 10 kΩ, C = 100 μF → τ = 1.00 s, vC(τ) = 7.59 V (63.2 %), vC(5τ) = 11.92 V (99.3 %)
//   RL  V = 12 V, R = 10 Ω (fixed), L = 0.5 H → τ = 50 ms, final current 1.20 A, i(τ) = 0.759 A
function solve({ V = 12, R = 10, C = 100, L = 0.5 }: Record<string, number>) {
  OUT.tauRC = R * C * 1e-3 // kΩ × μF = 1e3 Ω × 1e-6 F = 1e-3 s
  OUT.tauRL = L / RL_R
  OUT.vFinal = V
  OUT.iFinal = V / RL_R
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

const fmtTime = (s: number) => (s < 1 ? `${(s * 1000).toFixed(0)} ms` : `${s.toFixed(2)} s`)
const tagClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const axisClass = 'pointer-events-none whitespace-nowrap text-[10px] tabular-nums text-muted-foreground'

// ── One transient: a time axis with the rising curve, the 63.2 % level and the τ markers ──
// Local coordinates: (0, 0) is the axis origin, +X is time, +Y is the quantity.
type RowProps = {
  kind: 'rc' | 'rl'
  position: [number, number, number]
  width: number
  height: number
}

function TransientRow({ kind, position, width, height }: RowProps) {
  const c = usePhysicsColors()
  const valuesRef = useSliderValues()
  const rc = kind === 'rc'
  const win = rc ? WINDOW_RC : WINDOW_RL
  const maxFinal = rc ? V_MAX : V_MAX / RL_R

  const curveBuf = useMemo(() => new Float32Array(C_PTS * 3), [])
  const seg = useMemo(() => new Float32Array(6), [])
  const axis = useMemo<[number, number, number][]>(
    () => [[0, height * 1.12, 0], [0, 0, 0], [width, 0, 0]],
    [width, height]
  )

  const curveRef = useRef<any>(null)
  const finalRef = useRef<any>(null)
  const p63Ref = useRef<any>(null)
  const tauRef = useRef<any>(null)
  const tau5Ref = useRef<any>(null)
  const dotRef = useRef<THREE.Group>(null)
  const finalPos = useRef<THREE.Group>(null)
  const p63Pos = useRef<THREE.Group>(null)
  const tauPos = useRef<THREE.Group>(null)
  const tau5Pos = useRef<THREE.Group>(null)
  const finalText = useRef<HTMLSpanElement>(null)
  const tauText = useRef<HTMLSpanElement>(null)
  const tau5Text = useRef<HTMLSpanElement>(null)
  const headText = useRef<HTMLSpanElement>(null)
  const axisText = useRef<HTMLSpanElement>(null)
  const p63Text = useRef<HTMLSpanElement>(null)
  const axisRef = useRef<any>(null)
  const labelledFor = useRef<unknown>(null)
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(({ clock }) => {
    const s = solve(valuesRef.current)
    const tau = rc ? s.tauRC : s.tauRL
    const final = rc ? s.vFinal : s.iFinal
    const h = height * (final / maxFinal)
    const xPerSec = width / win

    // Repeating switch-on: sweep the present instant across the window, hold, restart
    const frac = Math.min((clock.getElapsedTime() % CYCLE) / SWEEP, 1)
    const tNow = Math.max(frac * win, win * 1e-3)

    for (let i = 0; i < C_PTS; i++) {
      const t = (i / (C_PTS - 1)) * tNow
      curveBuf[i * 3] = t * xPerSec
      curveBuf[i * 3 + 1] = h * (1 - Math.exp(-t / tau))
      curveBuf[i * 3 + 2] = 0
    }
    curveRef.current?.geometry.setPositions(curveBuf)

    // Final value and the 63.2 % level, as horizontal guides across the whole window
    const y63 = h * F63
    seg[0] = 0; seg[1] = h; seg[2] = 0; seg[3] = width; seg[4] = h; seg[5] = 0
    finalRef.current?.geometry.setPositions(seg)
    finalRef.current?.computeLineDistances()
    seg[1] = y63; seg[4] = y63
    p63Ref.current?.geometry.setPositions(seg)
    p63Ref.current?.computeLineDistances()

    // τ and 5τ, as verticals: this is where the learner reads the time constant off the picture
    const xTau = Math.min(tau, win) * xPerSec
    seg[0] = xTau; seg[1] = 0; seg[2] = 0; seg[3] = xTau; seg[4] = h; seg[5] = 0
    tauRef.current?.geometry.setPositions(seg)
    tauRef.current?.computeLineDistances()
    const xTau5 = Math.min(5 * tau, win) * xPerSec
    seg[0] = xTau5; seg[3] = xTau5
    tau5Ref.current?.geometry.setPositions(seg)
    tau5Ref.current?.computeLineDistances()

    // The RL row only exists once the narration reaches it; everything in it fades with the row.
    const row = rc ? 1 : reveal[REV_RL]
    const show5 = 5 * tau <= win ? 1 : 0
    setLineOpacity(axisRef.current, row)
    setLineOpacity(curveRef.current, row)
    dotRef.current?.scale.setScalar(row)
    setLineOpacity(finalRef.current, reveal[REV_FINAL] * row)
    setLineOpacity(p63Ref.current, reveal[REV_63] * row)
    setLineOpacity(tauRef.current, reveal[REV_TAU] * row)
    setLineOpacity(tau5Ref.current, reveal[REV_TAU5] * row * show5)
    setOpacity(headText.current, row)
    setOpacity(axisText.current, row)
    setOpacity(finalText.current, reveal[REV_FINAL] * row)
    setOpacity(p63Text.current, reveal[REV_63] * row)
    setOpacity(tauText.current, reveal[REV_TAU] * row)
    setOpacity(tau5Text.current, reveal[REV_TAU5] * row * show5)

    dotRef.current?.position.set(tNow * xPerSec, h * (1 - Math.exp(-tNow / tau)), 0.01)
    finalPos.current?.position.set(width + 0.1, h, 0)
    p63Pos.current?.position.set(-0.12, y63, 0)
    tauPos.current?.position.set(xTau, -0.17, 0)
    tau5Pos.current?.position.set(xTau5, -0.17, 0)

    // Text only when the slider values object changes identity, never per frame
    if (labelledFor.current !== valuesRef.current) {
      labelledFor.current = valuesRef.current
      if (finalText.current) {
        finalText.current.textContent = rc ? `vC → ${final.toFixed(1)} V` : `i → ${final.toFixed(2)} A`
      }
      if (tauText.current) tauText.current.textContent = `1τ = ${fmtTime(tau)} → ${P63}%`
      if (tau5Text.current) tau5Text.current.textContent = `5τ = ${fmtTime(5 * tau)} → ${P99}%`
    }
  })

  const guideColor = rc ? c.capacitance : c.inductance

  return (
    <group position={position}>
      <Line ref={axisRef} points={axis} lineWidth={1} color={c.guide} />
      <Line ref={finalRef} points={SEG_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.1} gapSize={0.08} />
      <Line ref={p63Ref} points={SEG_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.06} gapSize={0.05} />
      <Line ref={tauRef} points={SEG_INIT} lineWidth={1.5} color={guideColor} dashed dashSize={0.08} gapSize={0.06} />
      <Line ref={tau5Ref} points={SEG_INIT} lineWidth={1.5} color={guideColor} dashed dashSize={0.08} gapSize={0.06} />

      <TrailLine ref={curveRef} points={CURVE_INIT} lineWidth={3} color={rc ? c.voltage : c.current} />
      <Marker ref={dotRef} color={c.marker} size={0.06} />

      <Html position={[0.55, height + 0.2, 0]} center>
        <span ref={headText} className={tagClass}>{rc ? 'RC · τ = R × C' : 'RL · τ = L ÷ R'}</span>
      </Html>
      <Html position={[width, -0.4, 0]} center>
        <span ref={axisText} className={axisClass}>{rc ? 't (s) · 0 ~ 6 s · ×1' : 't (ms) · 0 ~ 300 ms · ×0.05'}</span>
      </Html>

      <group ref={finalPos}>
        <Html center>
          <span
            ref={finalText}
            className={rc ? tagClass : 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums'}
            style={rc ? undefined : { color: c.current }}
          />
        </Html>
      </group>
      <group ref={p63Pos}>
        <Html center>
          <span ref={p63Text} className={axisClass} style={{ transform: 'translateX(-50%)' }}>{P63}%</span>
        </Html>
      </group>
      <group ref={tauPos}>
        <Html center>
          <span ref={tauText} className={tagClass} />
        </Html>
      </group>
      <group ref={tau5Pos}>
        <Html center>
          <span ref={tau5Text} className={tagClass} />
        </Html>
      </group>
    </group>
  )
}

// ── 3D: the two time axes side by side in depth, so you can look down them ──
const W3 = 3.6
const H3 = 1.25

function TransientScene() {
  return (
    <>
      {/* Backed off and shifted left, so the far end of both axes clears the narration panel */}
      <PerspectiveCamera makeDefault position={[3, 2, 5.8]} fov={50} />
      <OrbitControls enablePan={false} />
      <TransientRow kind="rc" position={[-3.0, 0.05, 1.0]} width={W3} height={H3} />
      <TransientRow kind="rl" position={[-3.0, -2.0, -1.0]} width={W3} height={H3} />
    </>
  )
}

// ── 2D: both transients stacked on one flat plot, the time axis running full width ──
const W2 = 6.2
const H2 = 1.25

function Transient2DScene() {
  return (
    <>
      <TransientRow kind="rc" position={[-3.3, 0.45, 0]} width={W2} height={H2} />
      <TransientRow kind="rl" position={[-3.3, -1.45, 0]} width={W2} height={H2} />
    </>
  )
}

const concept: ConceptMeta = {
  id: 'rc-rl-transient',
  title: { en: 'RC & RL Transients', ko: 'RC·RL 과도현상' },
  subject: 'circuits',
  order: 17,
  description: {
    en: `## RC and RL Transients

**Switch a DC source on and nothing jumps: the circuit slides toward its final state, closing 63.2 % of the gap in one time constant τ and settling after about 5τ.**

- Upper trace: vC, the voltage on the capacitor of the RC circuit
- Lower cyan trace: i, the current in the inductor of the RL circuit
- Faint dashed horizontals: the final value and the 63.2 % level
- Stronger dashed verticals: 1τ and 5τ, so you can read τ straight off the picture
- Bright dot: the present instant; each sweep charges, pauses, then starts again

**RC**: vC = V × (1 − e^(−t ÷ τ)) with τ = R × C, and the charging current i = (V ÷ R) × e^(−t ÷ τ)

**RL**: i = (V ÷ R) × (1 − e^(−t ÷ τ)) with τ = L ÷ R, and the inductor voltage vL = V × e^(−t ÷ τ)

At t = τ both curves stand at 63.2 % of their final value, at 5τ at 99.3 %. The capacitor resists a change of voltage, the inductor resists a change of current: the same curve, mirrored.

Honest about the clock: the RL circuit keeps a fixed R of 10 Ω so that its τ stays watchable, and the two rows run on their own time scales. The RC axis spans 6 s and plays at real speed; the RL axis spans 300 ms and is slowed 20 times.

Raise R or C and the RC curve leans over as its 1τ marker slides right. Raise L and the RL curve does the same. Raise V and both plateaus rise while the markers stay where they are, because τ does not depend on the source.`,
    ko: `## RC·RL 과도현상

**직류 전원을 켜도 회로는 뛰어오르지 않습니다. 최종 상태를 향해 지수적으로 다가가며, 시상수 τ 하나 만에 남은 거리의 63.2 %를 좁히고 약 5τ 뒤에 자리를 잡습니다.**

- 위쪽 곡선: RC 회로 커패시터의 전압 vC
- 아래쪽 시안 곡선: RL 회로 인덕터의 전류 i
- 흐린 점선 가로선: 최종값과 63.2 % 높이
- 진한 점선 세로선: 1τ와 5τ. 그림에서 τ를 바로 읽을 수 있습니다
- 밝은 점: 현재 시각. 한 번 충전한 뒤 잠시 멈췄다가 다시 시작합니다

**RC**: vC = V × (1 − e^(−t ÷ τ)), τ = R × C, 충전 전류 i = (V ÷ R) × e^(−t ÷ τ)

**RL**: i = (V ÷ R) × (1 − e^(−t ÷ τ)), τ = L ÷ R, 인덕터 전압 vL = V × e^(−t ÷ τ)

t = τ에서 두 곡선 모두 최종값의 63.2 %, 5τ에서 99.3 %에 있습니다. 커패시터는 전압 변화를, 인덕터는 전류 변화를 거부합니다. 같은 곡선을 거울에 비춘 셈입니다.

시간에 대해 솔직히 적어 둡니다. RL 회로는 τ가 보기 좋은 값이 되도록 R을 10 Ω으로 고정했고, 두 줄은 각자의 시간 축을 씁니다. RC 축은 6 s를 담고 실제 속도로 재생되며, RL 축은 300 ms를 담아 20배 느리게 재생됩니다.

R이나 C를 올리면 RC 곡선이 완만해지고 1τ 표시가 오른쪽으로 밀립니다. L을 올리면 RL 곡선이 똑같이 반응합니다. V를 올리면 두 평탄값은 높아지지만 표시는 그대로입니다. τ는 전원과 무관하기 때문입니다.`,
  },
  story: [
    {
      set: { V: 12, R: 10, C: 100, L: 0.5 },
      text: {
        en: 'Flip the switch. The capacitor does not snap to 12 V — it leans toward it. Nothing in a real circuit jumps.',
        ko: '스위치를 올립니다. 커패시터는 12 V로 튀어오르지 않습니다. 기대듯이 다가갑니다. 실제 회로에서 순간이동하는 건 없습니다.',
      },
    },
    {
      set: { V: 12, R: 10, C: 100, L: 0.5 },
      text: {
        en: 'Where is it headed? 12 V, the source. It will get there — eventually. Watch how it crawls the last stretch.',
        ko: '어디로 가는 걸까요. 전원 전압 12 V입니다. 언젠가는 도착합니다. 마지막 구간을 얼마나 기어가는지 보세요.',
      },
    },
    {
      set: { V: 12, R: 10, C: 100, L: 0.5 },
      text: {
        en: 'So engineers stopped asking when it arrives and asked a better question: how long to close 63.2 % of the gap? Not half. 63.2 %. That number falls out of the exponential and it never changes.',
        ko: '그래서 공학자들은 언제 도착하냐고 묻기를 그만두고 더 나은 질문을 던졌습니다. 남은 거리의 63.2 %를 좁히는 데 얼마나 걸리나. 절반이 아닙니다. 63.2 %. 지수함수에서 떨어지는 값이고 절대 바뀌지 않습니다.',
      },
    },
    {
      set: { V: 12, R: 10, C: 100, L: 0.5 },
      text: {
        en: 'That time is the time constant τ, and you can read it straight off the picture. `τ = R × C = 10 kΩ × 100 μF = 1 s`',
        ko: '그 시간이 시상수 τ입니다. 그림에서 바로 읽을 수 있습니다. `τ = R × C = 10 kΩ × 100 μF = 1 s`',
      },
    },
    {
      set: { V: 12, R: 10, C: 200, L: 0.5 },
      text: {
        en: 'Double the capacitor. τ doubles to 2 s and the whole curve leans over. Bigger bucket, same tap, longer fill.',
        ko: '커패시터를 두 배로 키웁니다. τ가 2초로 두 배가 되고 곡선 전체가 눕습니다. 양동이가 커졌으니 같은 수도로 더 오래 받아야죠.',
      },
    },
    {
      set: { V: 24, R: 10, C: 200, L: 0.5 },
      text: {
        en: 'Now raise the source to 24 V. The plateau climbs — and the τ marker does not budge. τ is built from R and C only. The source has no say in it.',
        ko: '이번엔 전원을 24 V로 올립니다. 평탄값은 올라가는데 τ 표시는 꿈쩍도 하지 않습니다. τ는 R과 C로만 만들어집니다. 전원은 발언권이 없습니다.',
      },
    },
    {
      set: { V: 12, R: 10, C: 100, L: 0.5 },
      text: {
        en: 'Five time constants gets you to 99.3 %, and every engineer on earth calls that done. Five seconds and this circuit has stopped being interesting.',
        ko: '시상수 다섯 개면 99.3 %입니다. 지구상의 모든 공학자가 이걸 끝났다고 부릅니다. 5초면 이 회로는 더 볼 게 없어집니다.',
      },
    },
    {
      set: { V: 12, R: 10, C: 100, L: 0.5 },
      text: {
        en: 'Second row, and it is the same story wearing a different hat. A capacitor refuses to change its voltage. A coil refuses to change its current. Watch the shape.',
        ko: '두 번째 줄입니다. 같은 이야기가 모자만 바꿔 쓴 겁니다. 커패시터는 전압이 바뀌는 걸 거부합니다. 코일은 전류가 바뀌는 걸 거부합니다. 모양을 보세요.',
      },
    },
    {
      set: { V: 12, R: 10, C: 100, L: 0.5 },
      text: {
        en: 'Identical curve, identical 63.2 %, identical 5τ. Only the recipe flips: one multiplies, the other divides. `τ = L ÷ R = 0.5 H ÷ 10 Ω = 50 ms`',
        ko: '똑같은 곡선, 똑같은 63.2 %, 똑같은 5τ. 레시피만 뒤집힙니다. 하나는 곱하고 하나는 나눕니다. `τ = L ÷ R = 0.5 H ÷ 10 Ω = 50 ms`',
      },
    },
    {
      set: { V: 12, R: 10, C: 100, L: 0.2 },
      text: {
        en: 'Shrink the coil and τ shrinks with it — 20 ms. `RC: τ = R × C` `RL: τ = L ÷ R` Two different circuits, one curve, one number to remember. The sliders are yours.',
        ko: '코일을 작게 하면 τ도 같이 작아집니다. 20 ms. `RC: τ = R × C` `RL: τ = L ÷ R` 서로 다른 두 회로, 하나의 곡선, 외울 숫자는 하나. 이제 슬라이더는 당신 것입니다.',
      },
    },
  ],
  sliderParams: [
    { key: 'V', label: { en: 'Source voltage (V)', ko: '전원 전압 (V)' }, unit: 'V', min: 1, max: V_MAX, step: 0.5, defaultValue: 12 },
    { key: 'R', label: { en: 'Resistance, RC side (R)', ko: '저항, RC 쪽 (R)' }, unit: 'kΩ', min: 1, max: 20, step: 0.5, defaultValue: 10 },
    { key: 'C', label: { en: 'Capacitance (C)', ko: '정전용량 (C)' }, unit: 'μF', min: 10, max: 300, step: 5, defaultValue: 100 },
    { key: 'L', label: { en: 'Inductance (L)', ko: '인덕턴스 (L)' }, unit: 'H', min: 0.05, max: 2, step: 0.05, defaultValue: 0.5 },
  ],
  Scene: TransientScene,
  Scene2D: Transient2DScene,
}

export default concept
