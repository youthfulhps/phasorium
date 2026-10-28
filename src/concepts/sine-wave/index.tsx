import { useRef, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { useLocaleStore } from '../../lib/i18n'
import { setLineOpacity, setOpacity, useStoryReveal } from '../../components/scene/storyReveal'
import type { ConceptMeta } from '../../types/concept'

const HELIX_POINTS = 300
const CIRCLE_POINTS = 65
const VISIBLE_DURATION = 4
const FLAT_POINTS = 200

const HELIX_INIT: [number, number, number][] = Array.from(
  { length: HELIX_POINTS },
  (_, i) => [0, 0, -(i / (HELIX_POINTS - 1)) * 4]
)
const CIRCLE_INIT: [number, number, number][] = Array.from(
  { length: CIRCLE_POINTS },
  (_, i) => {
    const theta = (i / (CIRCLE_POINTS - 1)) * Math.PI * 2
    return [Math.cos(theta), Math.sin(theta), 0.01]
  }
)
const PROJ_INIT: [number, number, number][] = [[0, 0, 0.02], [1, 0, 0.02]]
const RMS_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, -4]]
const RMS_K = 1 / Math.SQRT2 // a sine's heating value is its peak over √2

// ── Narration steps drive what exists on screen. Each reveal eases 0 → 1 so nothing pops in. ──
const REV_PROJ = 0 // the drop to the Y axis: the sine value itself
const REV_HELIX = 1 // the height dragged out along time, i.e. the wave
const REV_SPEC = 2 // the Vm / f / ω / T / φ readout
const REV_RMS = 3 // the RMS level and its readout
const REV_COUNT = 4

/** Which reveals the sentence at `step` asks for. Step 0 is a bare dot going round. */
function revealTargets(step: number, out: Float32Array) {
  out[REV_PROJ] = step >= 1 ? 1 : 0
  out[REV_HELIX] = step >= 2 ? 1 : 0
  out[REV_SPEC] = step >= 3 ? 1 : 0
  out[REV_RMS] = step >= 7 ? 1 : 0
}

const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'

function useSliderValues() {
  const valuesRef = useRef<Record<string, number>>(useSliderStore.getState().values)
  useEffect(
    () => useSliderStore.subscribe((s) => s.values, (v) => { valuesRef.current = v }),
    []
  )
  return valuesRef
}

/** The one-line spec of the wave on screen, in the order the narration introduces it. */
function specLine({ amplitude = 1, frequency = 1, phase = 0 }: Record<string, number>) {
  return `Vm ${amplitude.toFixed(2)} V · f ${frequency.toFixed(1)} Hz · ω ${(2 * Math.PI * frequency).toFixed(2)} rad/s · T ${(1 / frequency).toFixed(2)} s · φ ${phase.toFixed(0)}°`
}
const FLAT_INIT: [number, number, number][] = Array.from(
  { length: FLAT_POINTS },
  (_, i) => [(i / (FLAT_POINTS - 1)) * 8 - 4, 0, 0]
)

function SineWave3DScene() {
  const c = usePhysicsColors()
  const helixPositions = useMemo(() => new Float32Array(HELIX_POINTS * 3), [])
  const circlePositions = useMemo(() => new Float32Array(CIRCLE_POINTS * 3), [])
  const projPositions = useMemo(() => new Float32Array(2 * 3), [])

  const rmsPositions = useMemo(() => new Float32Array(2 * 3), [])

  const helixLine  = useRef<any>(null)
  const circleLine = useRef<any>(null)
  const projLine   = useRef<any>(null)
  const rmsLine    = useRef<any>(null)
  const dotRef     = useRef<THREE.Group>(null)
  const rmsTag     = useRef<THREE.Group>(null)
  const textRefs   = useRef<(HTMLSpanElement | null)[]>([]) // RMS readout, spec line
  const labelledFor = useRef<unknown>(null)

  const valuesRef = useSliderValues()
  const locale = useLocaleStore((s) => s.locale)
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime()
    const { amplitude = 1, frequency = 1, phase = 0 } = valuesRef.current
    const omega = 2 * Math.PI * frequency
    const phaseRad = (phase * Math.PI) / 180

    for (let i = 0; i < HELIX_POINTS; i++) {
      const frac = i / (HELIX_POINTS - 1)
      const tau = t - (1 - frac) * VISIBLE_DURATION
      helixPositions[i * 3]     = amplitude * Math.cos(omega * tau + phaseRad)
      helixPositions[i * 3 + 1] = amplitude * Math.sin(omega * tau + phaseRad)
      helixPositions[i * 3 + 2] = -(1 - frac) * 4
    }
    helixLine.current?.geometry.setPositions(helixPositions)
    setLineOpacity(helixLine.current, reveal[REV_HELIX])

    for (let i = 0; i < CIRCLE_POINTS; i++) {
      const theta = (i / (CIRCLE_POINTS - 1)) * 2 * Math.PI
      circlePositions[i * 3]     = amplitude * Math.cos(theta)
      circlePositions[i * 3 + 1] = amplitude * Math.sin(theta)
      circlePositions[i * 3 + 2] = 0.01
    }
    circleLine.current?.geometry.setPositions(circlePositions)
    circleLine.current?.computeLineDistances() // dashes are laid out from line distances

    const cx = amplitude * Math.cos(omega * t + phaseRad)
    const cy = amplitude * Math.sin(omega * t + phaseRad)
    if (dotRef.current) dotRef.current.position.set(cx, cy, 0.02)

    projPositions[0] = cx; projPositions[1] = cy; projPositions[2] = 0.02
    projPositions[3] = 0;  projPositions[4] = cy; projPositions[5] = 0.02
    projLine.current?.geometry.setPositions(projPositions)
    setLineOpacity(projLine.current, reveal[REV_PROJ])

    // The level a DC source would have to sit at to heat the same: Vm / √2
    const rms = amplitude * RMS_K
    rmsPositions[0] = 0; rmsPositions[1] = rms; rmsPositions[2] = 0.02
    rmsPositions[3] = 0; rmsPositions[4] = rms; rmsPositions[5] = -4
    rmsLine.current?.geometry.setPositions(rmsPositions)
    rmsLine.current?.computeLineDistances()
    setLineOpacity(rmsLine.current, reveal[REV_RMS])
    rmsTag.current?.position.set(0.1, rms + 0.18, 0.02)
    setOpacity(textRefs.current[0], reveal[REV_RMS])
    setOpacity(textRefs.current[1], reveal[REV_SPEC])

    if (labelledFor.current !== valuesRef.current && textRefs.current[1]) {
      labelledFor.current = valuesRef.current
      if (textRefs.current[0]) textRefs.current[0]!.textContent = `Vrms ${rms.toFixed(2)} V`
      textRefs.current[1]!.textContent = specLine(valuesRef.current)
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  return (
    <>
      {/* Backed off and aimed down the helix, so a full-amplitude wave clears the narration panel */}
      <PerspectiveCamera makeDefault position={[3.4, 2.2, 6.4]} fov={50} />
      <OrbitControls enablePan={false} target={[1.7, 0, -2]} />
      <TrailLine ref={helixLine} points={HELIX_INIT} lineWidth={3} color={c.voltage} />
      <Line ref={circleLine} points={CIRCLE_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      <Marker ref={dotRef} color={c.marker} size={0.08} />
      <Line ref={projLine} points={PROJ_INIT} lineWidth={1.5} color={c.marker} />
      <Line ref={rmsLine} points={RMS_INIT} lineWidth={1} color={c.power} dashed dashSize={0.1} gapSize={0.07} />
      <group ref={rmsTag}>
        <Html center>
          <span ref={(el) => { textRefs.current[0] = el }} className={labelClass} style={{ color: c.power }} />
        </Html>
      </group>
      <Html position={[0, -2.5, 0]} center>
        <span ref={(el) => { textRefs.current[1] = el }} className={labelClass} />
      </Html>
    </>
  )
}

function SineWave2DScene() {
  const c = usePhysicsColors()
  const positions = useMemo(() => new Float32Array(FLAT_POINTS * 3), [])
  const rmsPositions = useMemo(() => new Float32Array(2 * 3), [])
  const lineRef = useRef<any>(null)
  const rmsLine = useRef<any>(null)
  const dotRef  = useRef<THREE.Group>(null)
  const rmsTag  = useRef<THREE.Group>(null)
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<unknown>(null)

  const valuesRef = useSliderValues()
  const locale = useLocaleStore((s) => s.locale)
  const reveal = useStoryReveal(revealTargets, REV_COUNT)

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime()
    const { amplitude = 1, frequency = 1, phase = 0 } = valuesRef.current
    const omega = 2 * Math.PI * frequency
    const phaseRad = (phase * Math.PI) / 180

    for (let i = 0; i < FLAT_POINTS; i++) {
      const frac = i / (FLAT_POINTS - 1)
      const tau = t - (1 - frac) * VISIBLE_DURATION
      positions[i * 3]     = frac * 8 - 4
      positions[i * 3 + 1] = amplitude * Math.sin(omega * tau + phaseRad)
      positions[i * 3 + 2] = 0
    }
    lineRef.current?.geometry.setPositions(positions)
    setLineOpacity(lineRef.current, reveal[REV_HELIX]) // before its sentence there is only the bobbing dot

    if (dotRef.current) {
      dotRef.current.position.set(4, amplitude * Math.sin(omega * t + phaseRad), 0)
    }

    const rms = amplitude * RMS_K
    rmsPositions[0] = -4; rmsPositions[1] = rms; rmsPositions[2] = 0
    rmsPositions[3] = 4;  rmsPositions[4] = rms; rmsPositions[5] = 0
    rmsLine.current?.geometry.setPositions(rmsPositions)
    rmsLine.current?.computeLineDistances()
    setLineOpacity(rmsLine.current, reveal[REV_RMS])
    rmsTag.current?.position.set(-3.3, rms + 0.22, 0)
    setOpacity(textRefs.current[0], reveal[REV_RMS])
    setOpacity(textRefs.current[1], reveal[REV_SPEC])

    if (labelledFor.current !== valuesRef.current && textRefs.current[1]) {
      labelledFor.current = valuesRef.current
      if (textRefs.current[0]) textRefs.current[0]!.textContent = `Vrms ${rms.toFixed(2)} V`
      textRefs.current[1]!.textContent = specLine(valuesRef.current)
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  return (
    <>
      <TrailLine ref={lineRef} points={FLAT_INIT} lineWidth={3} color={c.voltage} />
      <Marker ref={dotRef} color={c.marker} size={0.08} />
      <Line ref={rmsLine} points={RMS_INIT} lineWidth={1} color={c.power} dashed dashSize={0.1} gapSize={0.07} />
      <group ref={rmsTag}>
        <Html center>
          <span ref={(el) => { textRefs.current[0] = el }} className={labelClass} style={{ color: c.power }} />
        </Html>
      </group>
      <Html position={[0, -2.5, 0]} center>
        <span ref={(el) => { textRefs.current[1] = el }} className={labelClass} />
      </Html>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'sine-wave',
  title: { en: 'Sinusoidal Wave', ko: '정현파' },
  subject: 'circuits',
  order: 1,
  description: {
    en: `## Sine Waves and Circular Motion

A sine wave is **the Y-axis projection of circular motion**.

**3D helix view:**
- Liquid helix: the rotating point's path over time
- Dashed circle: circular motion at the current slice
- Bright dot: the current rotating position
- Short line: Y-axis projection (this is the sine value)

Drag to rotate the view.
From the front you see a **sine wave**; from above, **circular motion**.

**Formula**: v(t) = Vm × sin(ωt + φ)`,
    ko: `## 정현파와 원운동

사인파는 **원운동의 Y축 투영**입니다.

**3D 나선(Helix) 시각화:**
- 액체 나선: 회전하는 점의 시간 궤적
- 점선 원: 현재 단면의 원운동
- 밝은 점: 현재 회전 위치
- 짧은 선: Y축 투영 (이것이 sin 값)

마우스로 시점을 회전해보세요.
정면에서 보면 **사인파**, 위에서 보면 **원운동**입니다.

**수식**: v(t) = Vm × sin(ωt + φ)`,
  },
  story: [
    {
      set: { amplitude: 1, frequency: 1, phase: 0 },
      text: {
        en: 'A dot going round a circle at a steady speed. No voltage, no current, no circuit. Just a dot.',
        ko: '점 하나가 일정한 속도로 원을 돕니다. 전압도, 전류도, 회로도 없습니다. 그냥 점입니다.',
      },
    },
    {
      set: { amplitude: 1, frequency: 1, phase: 0 },
      text: {
        en: 'Now ignore left and right. Watch only how high it is. Up, down, up, down — that one number is everything we care about.',
        ko: '이제 좌우는 잊으세요. 높이만 보세요. 올라가고, 내려가고, 올라가고, 내려가고. 우리가 관심 있는 건 그 숫자 하나뿐입니다.',
      },
    },
    {
      set: { amplitude: 1, frequency: 1, phase: 0 },
      text: {
        en: 'Drag that height out along time and there it is. A sine wave is circular motion seen from the side. That is the whole definition.',
        ko: '그 높이를 시간축으로 끌어내면 이게 나옵니다. 사인파는 원운동을 옆에서 본 것입니다. 정의는 이게 전부입니다.',
      },
    },
    {
      set: { amplitude: 2, frequency: 1, phase: 0 },
      text: {
        en: 'Three dials and the wave is pinned down. First: make the circle bigger. The radius is the peak value Vm — now 2 V.',
        ko: '다이얼 세 개면 파형이 결정됩니다. 첫째, 원을 키웁니다. 반지름이 최댓값 Vm입니다. 지금은 2 V.',
      },
    },
    {
      set: { amplitude: 2, frequency: 3, phase: 0 },
      text: {
        en: 'Second: spin it faster. Three laps a second is 3 Hz, so one lap takes 0.33 s — that is the period T. `ω = 2πf = 18.85 rad/s`',
        ko: '둘째, 더 빨리 돌립니다. 초당 세 바퀴면 3 Hz, 한 바퀴에 0.33초. 그게 주기 T입니다. `ω = 2πf = 18.85 rad/s`',
      },
    },
    {
      set: { amplitude: 2, frequency: 1, phase: 90 },
      text: {
        en: 'Third: start the dot somewhere else on the circle. The whole wave slides sideways. That head start is the phase, 90°.',
        ko: '셋째, 점을 원의 다른 자리에서 출발시킵니다. 파형 전체가 옆으로 밀립니다. 그 출발 각도가 위상입니다. 90°.',
      },
    },
    {
      set: { amplitude: 2, frequency: 1, phase: 0 },
      text: {
        en: 'Radius, speed, starting angle. Write those three down and you have written the wave. `v(t) = Vm × sin(ωt + φ)`',
        ko: '반지름, 속도, 출발 각도. 이 셋만 적으면 파형을 다 적은 겁니다. `v(t) = Vm × sin(ωt + φ)`',
      },
    },
    {
      set: { amplitude: 2, frequency: 1, phase: 0 },
      text: {
        en: 'But here is the catch: this thing is at its peak for an instant and at zero twice a cycle. So how much is it worth? Ask heat. The amber line is the DC level that would heat a resistor exactly as hard. `Vrms = Vm ÷ √2 = 1.41 V`',
        ko: '그런데 함정이 있습니다. 이 파형은 최댓값에 잠깐 머물고 한 주기에 두 번은 0입니다. 그럼 얼마짜리일까요. 열에게 물어보세요. 앰버색 선이 같은 저항을 똑같이 뜨겁게 만드는 직류 전압입니다. `Vrms = Vm ÷ √2 = 1.41 V`',
      },
    },
    {
      set: { amplitude: 2, frequency: 1, phase: 0 },
      text: {
        en: 'That is why the socket in your wall says 220 V and not 311 V. 220 is the heating value; the real voltage swings to `√2 × 220 = 311 V` and back, fifty times a second. The sliders are yours.',
        ko: '집 콘센트에 311 V가 아니라 220 V라고 적힌 이유가 이겁니다. 220은 열로 환산한 값이고, 실제 전압은 `√2 × 220 = 311 V`까지 올라갔다 내려오기를 초당 50번 반복합니다. 이제 슬라이더는 당신 것입니다.',
      },
    },
  ],
  sliderParams: [
    { key: 'amplitude', label: { en: 'Amplitude (Vm)', ko: '진폭 (Vm)' }, min: 0.1, max: 2,   step: 0.1, defaultValue: 1, unit: 'V'  },
    { key: 'frequency', label: { en: 'Frequency (f)',  ko: '주파수 (f)' }, min: 0.5, max: 5,   step: 0.5, defaultValue: 1, unit: 'Hz' },
    { key: 'phase',     label: { en: 'Phase (φ)',      ko: '위상 (φ)'   }, min: 0,   max: 360, step: 5,   defaultValue: 0, unit: '°'  },
  ],
  Scene: SineWave3DScene,
  Scene2D: SineWave2DScene,
}

export default concept
