import { useRef, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import type { LineMaterial } from 'three-stdlib'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { useLocaleStore } from '../../lib/i18n'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import { FloorCaption, PotentialRuler, useRiseIn } from '../../components/scene/PotentialLandscape'
import { Resistor, setResistorHeat } from '../../components/scene/Resistor'
import type { ConceptMeta } from '../../types/concept'

const V1_SRC = 12 // the driving source is fixed, so the sliders only reshape the box [V]
const Z_MIN = 2 // [Ω] slider floor for Z1 and Z2
const P_MAX = (V1_SRC * V1_SRC) / Z_MIN // hottest a resistor can get [W]

// Terminal condition the scene is showing. The 3D scene cycles through all three.
const LOADED = 0
const OPEN = 1
const SHORT = 2

// Pure physics for the built-in L-network: series Z1 from port 1, then shunt Z2 across port 2.
// I2 is defined leaving port 2, so I1 = I2 + V2 ÷ Z2 and V1 = V2 + I1 × Z1, which gives
// A = 1 + Z1 ÷ Z2, B = Z1, C = 1 ÷ Z2, D = 1, and AD − BC = 1 for any reciprocal network.
// Worked example: Z1 = 10 Ω, Z2 = 20 Ω → A = 1.5, B = 10 Ω, C = 0.05 S, D = 1,
// AD − BC = 1.5 × 1 − 10 × 0.05 = 1 exactly.
// With ZL = 40 Ω: Zin = (1.5 × 40 + 10) ÷ (0.05 × 40 + 1) = 70 ÷ 3 = 23.33 Ω,
// V2 ÷ V1 = 40 ÷ (1.5 × 40 + 10) = 40 ÷ 70 = 0.5714.
// At V1 = 12 V that is V2 = 6.857 V, I2 = 0.1714 A, I1 = 12 ÷ 23.333 = 0.5143 A,
// and the matrix agrees: I1 = C × V2 + D × I2 = 0.3429 + 0.1714 = 0.5143 A.
const OUT = {
  Z1: 0, Z2: 0, ZL: 0,
  A: 0, B: 0, C: 0, D: 0, det: 0,
  V1: 0, V2: 0, I1: 0, I2: 0, Ishunt: 0,
  Zin: 0, gain: 0,
}

/** Reuses one module-scope result object so nothing is allocated per frame. */
function solve({ Z1 = 10, Z2 = 20, ZL = 40 }: Record<string, number>, mode = LOADED) {
  const A = 1 + Z1 / Z2
  const B = Z1
  const C = 1 / Z2
  const D = 1
  // Open: I2 = 0 so V1 = A × V2. Short: V2 = 0 so V1 = B × I2. Loaded: I2 = V2 ÷ ZL.
  const V2 = mode === OPEN ? V1_SRC / A : mode === SHORT ? 0 : (V1_SRC * ZL) / (A * ZL + B)
  const I2 = mode === OPEN ? 0 : mode === SHORT ? V1_SRC / B : V2 / ZL
  OUT.Z1 = Z1; OUT.Z2 = Z2; OUT.ZL = ZL
  OUT.A = A; OUT.B = B; OUT.C = C; OUT.D = D
  OUT.det = A * D - B * C
  OUT.V1 = V1_SRC
  OUT.V2 = V2
  OUT.I2 = I2
  OUT.I1 = C * V2 + D * I2
  OUT.Ishunt = C * V2 // the part of I1 that the shunt swallows
  OUT.Zin = (A * ZL + B) / (C * ZL + D)
  OUT.gain = ZL / (A * ZL + B)
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

const fmtV = (x: number) => `${x.toFixed(2)} V`
const fmtA = (x: number) => `${x.toFixed(3)} A`
const fmtZ = (x: number) => `${x.toFixed(x < 10 ? 2 : 1)} Ω`
const fmtS = (x: number) => `${x.toFixed(4)} S`
const fmt3 = (x: number) => x.toFixed(3)
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const captionClass = 'pointer-events-none whitespace-nowrap text-[10px] text-muted-foreground'
const matrixClass = 'pointer-events-none whitespace-nowrap text-[15px] font-semibold tabular-nums text-foreground/90'
const headerClass = 'pointer-events-none whitespace-nowrap text-[12px] font-semibold tabular-nums text-foreground/85'

// ── 3D: the box as a potential landscape, so A is literally a ratio of two heights ──
const BASE_Y = -0.6
const HEIGHT_PER_VOLT = 1.6 / V1_SRC
const RULER_VOLTS = [0, 6, 12]
const Z_BACK = -1.0
const Z_FRONT = 1.0
const RZ: [number, number] = [-0.45, 0.45] // resistor span on a rung running front-to-back
const SEG_PTS = 16

type XZ = [number, number]
const SX = -2.9 // source
const XIN = -2.0 // port 1 terminal, left wall of the box
const Z1_A: XZ = [-1.8, Z_BACK]
const Z1_B: XZ = [-0.7, Z_BACK]
const XM = -0.25 // shunt node, electrically the same point as port 2
const XOUT = 0.55 // port 2 terminal, right wall of the box
const XL = 1.55 // load rung, outside the box

// Source loop and the series arm: carries I1
const MAIN_XZ: XZ[] = [
  [XM, Z_FRONT], [SX, Z_FRONT], [SX, Z_BACK], [XIN, Z_BACK], Z1_A, Z1_B, [XM, Z_BACK],
]
// Shunt arm Z2 inside the box: carries I1 − I2
const SHUNT_XZ: XZ[] = [[XM, Z_BACK], [XM, RZ[0]], [XM, RZ[1]], [XM, Z_FRONT]]
// Load loop outside the box: carries I2
const LOAD_XZ: XZ[] = [
  [XM, Z_BACK], [XOUT, Z_BACK], [XL, Z_BACK], [XL, RZ[0]], [XL, RZ[1]],
  [XL, Z_FRONT], [XOUT, Z_FRONT], [XM, Z_FRONT],
]

const mainV = new Float32Array(MAIN_XZ.length)
const shuntV = new Float32Array(SHUNT_XZ.length)
const loadV = new Float32Array(LOAD_XZ.length)
const LINE_XZ = [MAIN_XZ, SHUNT_XZ, LOAD_XZ]
const LINE_V = [mainV, shuntV, loadV]
const lineI = new Float32Array(3)

const pointCount = (xz: XZ[]) => (xz.length - 1) * SEG_PTS + 1
const initPoints = (xz: XZ[]): [number, number, number][] =>
  Array.from({ length: pointCount(xz) }, () => [0, 0, 0])

/** Sample a waypoint polyline; potential varies linearly along each segment and is flat on wires. */
function fillPath(buf: Float32Array, xz: XZ[], volts: Float32Array, k: number) {
  let n = 0
  for (let s = 0; s < xz.length - 1; s++) {
    for (let j = 0; j < SEG_PTS + (s === xz.length - 2 ? 1 : 0); j++) {
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

/** Seat a resistor between two waypoints; its filament thickens with power. */
function placeResistor(
  g: THREE.Group | null, tag: THREE.Group | null,
  a: XZ, va: number, b: XZ, vb: number, k: number, Z: number, P: number, tagDx: number
) {
  p0.set(a[0], BASE_Y + k * va, a[1])
  p1.set(b[0], BASE_Y + k * vb, b[1])
  dir.subVectors(p1, p0)
  if (g) {
    g.position.addVectors(p0, p1).multiplyScalar(0.5)
    g.quaternion.setFromUnitVectors(X_AXIS, unit.copy(dir).normalize())
    const r = 0.13 * Math.sqrt(Z_MIN / Math.max(Z, 0.2)) + 0.03 // thinner for larger Z
    g.scale.set(dir.length(), r, r)
  }
  setResistorHeat(g, Math.sqrt(P / P_MAX) * 1.6)
  tag?.position.set((p0.x + p1.x) / 2 + tagDx, (p0.y + p1.y) / 2 + 0.28, (p0.z + p1.z) / 2)
}

const widthFor = (I: number) => Math.min(1.2 + 1.6 * Math.sqrt(I), 8)
const flowFor = (I: number) => Math.min(1.2 * I, 8)

// How long each terminal condition stays on screen [s]
const T_LOADED = 4.4
const T_OPEN = 3.4
const T_SHORT = 3.4
const CYCLE = T_LOADED + T_OPEN + T_SHORT

function TwoPortScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const rise = useRiseIn()
  const bufs = useMemo(() => LINE_XZ.map((xz) => new Float32Array(pointCount(xz) * 3)), [])
  const inits = useMemo(() => LINE_XZ.map(initPoints), [])
  const lineRefs = useRef<any[]>([])
  const resRefs = useRef<(THREE.Group | null)[]>([]) // Z1, Z2, ZL
  const tagRefs = useRef<(THREE.Group | null)[]>([]) // 3 resistor tags, 2 port readouts
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const port1 = useRef<THREE.Group>(null)
  const port2 = useRef<THREE.Group>(null)
  const flows = useRef([0, 0, 0])
  const labelledFor = useRef<unknown>(null)
  const labelledMode = useRef(-1)

  useFrame(({ clock }) => {
    const phase = clock.getElapsedTime() % CYCLE
    const mode = phase < T_LOADED ? LOADED : phase < T_LOADED + T_OPEN ? OPEN : SHORT
    const s = solve(valuesRef.current, mode)
    const k = HEIGHT_PER_VOLT * rise.current

    // 0 → V1 up the source, down to V2 across Z1, then to 0 through the shunt and the load
    mainV[2] = mainV[3] = mainV[4] = s.V1
    mainV[5] = mainV[6] = s.V2
    shuntV[0] = shuntV[1] = s.V2
    loadV[0] = loadV[1] = loadV[2] = loadV[3] = s.V2

    lineI[0] = s.I1; lineI[1] = s.Ishunt; lineI[2] = s.I2
    for (let l = 0; l < 3; l++) {
      const line = lineRefs.current[l]
      if (!line) continue
      // Open circuit: the load branch is not there at all
      line.visible = !(l === 2 && mode === OPEN)
      fillPath(bufs[l], LINE_XZ[l], LINE_V[l], k)
      line.geometry.setPositions(bufs[l])
      ;(line.material as LineMaterial).linewidth = widthFor(lineI[l])
      flows.current[l] = flowFor(lineI[l])
    }

    placeResistor(resRefs.current[0], tagRefs.current[0], Z1_A, s.V1, Z1_B, s.V2, k, s.Z1, s.I1 * s.I1 * s.Z1, 0)
    placeResistor(resRefs.current[1], tagRefs.current[1],
      SHUNT_XZ[1], s.V2, SHUNT_XZ[2], 0, k, s.Z2, s.Ishunt * s.Ishunt * s.Z2, -0.42)
    placeResistor(resRefs.current[2], tagRefs.current[2],
      LOAD_XZ[3], s.V2, LOAD_XZ[4], 0, k, s.ZL, s.I2 * s.I2 * s.ZL, 0.42)
    // Open: no load component. Short: the branch is a bare wire, so the tube goes away too.
    const loadRes = resRefs.current[2]
    if (loadRes) loadRes.visible = mode === LOADED
    const loadTag = tagRefs.current[2]
    if (loadTag) loadTag.visible = mode === LOADED

    const y1 = BASE_Y + k * s.V1
    const y2 = BASE_Y + k * s.V2
    port1.current?.position.set(XIN, y1, Z_BACK)
    port2.current?.position.set(XOUT, y2, Z_BACK)
    tagRefs.current[3]?.position.set(XIN - 0.15, y1 + 0.3, Z_BACK)
    tagRefs.current[4]?.position.set(XOUT + 0.2, y2 + 0.3, Z_BACK)

    if ((labelledFor.current !== valuesRef.current || labelledMode.current !== mode) && textRefs.current[6]) {
      labelledFor.current = valuesRef.current
      labelledMode.current = mode
      const ko = locale === 'ko'
      const state = mode === OPEN
        ? `${ko ? '출력 개방' : 'Output open'} · I2 = 0 · A = V1 ÷ V2 = ${fmt3(s.A)} · C = I1 ÷ V2 = ${fmtS(s.C)}`
        : mode === SHORT
          ? `${ko ? '출력 단락' : 'Output shorted'} · V2 = 0 · B = V1 ÷ I2 = ${fmtZ(s.B)} · D = I1 ÷ I2 = ${fmt3(s.D)}`
          : `${ko ? '부하 연결' : 'Loaded'} · ZL ${fmtZ(s.ZL)} · Zin ${fmtZ(s.Zin)} · V2 ÷ V1 = ${s.gain.toFixed(4)}`
      const texts = [
        `Z1 ${fmtZ(s.Z1)}`,
        `Z2 ${fmtZ(s.Z2)}`,
        `ZL ${fmtZ(s.ZL)}`,
        `V1 ${fmtV(s.V1)} · I1 ${fmtA(s.I1)}`,
        `V2 ${fmtV(s.V2)} · I2 ${fmtA(s.I2)}`,
        state,
        `A ${fmt3(s.A)} · B ${fmtZ(s.B)} · C ${fmtS(s.C)} · D ${fmt3(s.D)} · AD − BC = ${s.det.toFixed(6)}`,
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  useEffect(() => { labelledMode.current = -1 }, [locale])

  const outlines = useMemo<[number, number, number][][]>(() => [
    // The box itself: everything inside is what the four numbers replace
    [[XIN, BASE_Y, Z_FRONT], [XIN, BASE_Y, Z_BACK], [XOUT, BASE_Y, Z_BACK], [XOUT, BASE_Y, Z_FRONT], [XIN, BASE_Y, Z_FRONT]],
    // Source loop and load loop, outside the box
    [[SX, BASE_Y, Z_FRONT], [SX, BASE_Y, Z_BACK], [XIN, BASE_Y, Z_BACK]],
    [[XOUT, BASE_Y, Z_BACK], [XL, BASE_Y, Z_BACK], [XL, BASE_Y, Z_FRONT], [XOUT, BASE_Y, Z_FRONT]],
  ], [])
  const ko = locale === 'ko'

  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />

      {outlines.map((pts, i) => (
        <Line key={i} points={pts} lineWidth={i === 0 ? 1.5 : 1} color={c.guide} dashed dashSize={0.08} gapSize={0.06} />
      ))}
      <FloorCaption position={[-0.7, BASE_Y, Z_FRONT + 0.85]} />
      <PotentialRuler x={SX - 0.4} z={Z_BACK} baseY={BASE_Y} volts={RULER_VOLTS} unitsPerVolt={() => HEIGHT_PER_VOLT * rise.current} />

      {inits.map((pts, l) => (
        <TrailLine
          key={l}
          ref={(line) => { lineRefs.current[l] = line }}
          points={pts}
          color={c.current}
          fade={false}
          flowSpeed={() => flows.current[l]}
        />
      ))}

      {[0, 1, 2].map((r) => (
        <Resistor key={r} ref={(g) => { resRefs.current[r] = g }} />
      ))}
      <Marker ref={port1} color={c.marker} size={0.07} />
      <Marker ref={port2} color={c.marker} size={0.07} />

      {[0, 1, 2, 3, 4].map((t) => (
        <group key={t} ref={(g) => { tagRefs.current[t] = g }}>
          <Html center>
            <span ref={(el) => { textRefs.current[t] = el }} className={labelClass} />
          </Html>
        </group>
      ))}

      <Html position={[(XIN + XOUT) / 2, BASE_Y - 0.14, Z_FRONT + 0.4]} center>
        <span className={captionClass}>{ko ? '4단자망 (상자 안이 L형 회로)' : 'Two-port network (an L-network inside)'}</span>
      </Html>
      <Html position={[(SX + XL) / 2, BASE_Y - 0.44, Z_FRONT + 0.4]} center>
        <span ref={(el) => { textRefs.current[5] = el }} className={labelClass} />
      </Html>
      <Html position={[(SX + XL) / 2, BASE_Y - 0.72, Z_FRONT + 0.4]} center>
        <span ref={(el) => { textRefs.current[6] = el }} className={labelClass} />
      </Html>
    </>
  )
}

// ── 2D: the four numbers themselves, the reciprocity check, and the two port voltages as bars ──
const MAT_L = -3.5
const MAT_R = -1.0
const MAT_T = 0.9
const MAT_B = -0.9
const TAB = 0.22
type P3 = [number, number, number]
const BRACKET_L: P3[] = [[MAT_L + TAB, MAT_T, 0], [MAT_L, MAT_T, 0], [MAT_L, MAT_B, 0], [MAT_L + TAB, MAT_B, 0]]
const BRACKET_R: P3[] = [[MAT_R - TAB, MAT_T, 0], [MAT_R, MAT_T, 0], [MAT_R, MAT_B, 0], [MAT_R - TAB, MAT_B, 0]]
const CELL_X = [-2.85, -1.65]
const CELL_Y = [0.42, -0.42]

const BAR_X0 = -0.1
const BAR_W = 3.7
const BAR_H = 0.52
const BAR_Y1 = 0.6
const BAR_Y2 = -0.35
const BAR_TRACK: P3[] = [
  [BAR_X0, BAR_Y2 - BAR_H / 2 - 0.12, 0], [BAR_X0, BAR_Y1 + BAR_H / 2 + 0.12, 0],
]

function AbcdMatrix2DScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()
  const barRefs = useRef<(THREE.Mesh | null)[]>([]) // V1, V2
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const labelledFor = useRef<unknown>(null)

  useFrame(() => {
    const s = solve(valuesRef.current, LOADED)
    // V1 is the reference bar at full width; V2 is the same bar scaled by the loaded gain
    const w2 = Math.max(BAR_W * s.gain, 1e-3)
    barRefs.current[0]?.position.set(BAR_X0 + BAR_W / 2, BAR_Y1, 0)
    barRefs.current[0]?.scale.set(BAR_W, BAR_H, 1)
    barRefs.current[1]?.position.set(BAR_X0 + w2 / 2, BAR_Y2, 0)
    barRefs.current[1]?.scale.set(w2, BAR_H, 1)

    if (labelledFor.current !== valuesRef.current && textRefs.current[7]) {
      labelledFor.current = valuesRef.current
      const ko = locale === 'ko'
      const texts = [
        fmt3(s.A),
        fmtZ(s.B),
        fmtS(s.C),
        fmt3(s.D),
        `AD − BC = ${s.det.toFixed(6)} · ${ko ? '가역 회로망' : 'reciprocal'}`,
        `V1 ${fmtV(s.V1)} · I1 ${fmtA(s.I1)}`,
        `V2 ${fmtV(s.V2)} · I2 ${fmtA(s.I2)} · V2 ÷ V1 = ${s.gain.toFixed(4)}`,
        `ZL ${fmtZ(s.ZL)} · Zin = (A·ZL + B) ÷ (C·ZL + D) = ${fmtZ(s.Zin)}`,
      ]
      texts.forEach((text, t) => { if (textRefs.current[t]) textRefs.current[t]!.textContent = text })
    }
  })

  useEffect(() => { labelledFor.current = null }, [locale])

  return (
    <>
      <Html position={[-1.1, 1.72, 0]} center>
        <span className={headerClass}>V1 = A·V2 + B·I2 　 I1 = C·V2 + D·I2</span>
      </Html>

      <Line points={BRACKET_L} lineWidth={1.5} color={c.guide} />
      <Line points={BRACKET_R} lineWidth={1.5} color={c.guide} />
      {[0, 1, 2, 3].map((i) => (
        <Html key={i} position={[CELL_X[i % 2], CELL_Y[i < 2 ? 0 : 1], 0]} center>
          <span ref={(el) => { textRefs.current[i] = el }} className={matrixClass} />
        </Html>
      ))}
      <Html position={[(MAT_L + MAT_R) / 2, -1.3, 0]} center>
        <span ref={(el) => { textRefs.current[4] = el }} className={labelClass} />
      </Html>

      <Line points={BAR_TRACK} lineWidth={1} color={c.guide} />
      {[0, 1].map((i) => (
        <mesh key={i} ref={(m) => { barRefs.current[i] = m }}>
          <planeGeometry args={[1, 1]} />
          <meshBasicMaterial color={c.voltage} transparent opacity={i === 0 ? 0.9 : 0.45} depthWrite={false} />
        </mesh>
      ))}
      <Html position={[BAR_X0, BAR_Y1 + BAR_H / 2 + 0.2, 0]}>
        <span ref={(el) => { textRefs.current[5] = el }} className={labelClass} />
      </Html>
      <Html position={[BAR_X0, BAR_Y2 + BAR_H / 2 + 0.2, 0]}>
        <span ref={(el) => { textRefs.current[6] = el }} className={labelClass} />
      </Html>
      <Html position={[BAR_X0, -1.15, 0]}>
        <span ref={(el) => { textRefs.current[7] = el }} className={labelClass} />
      </Html>
    </>
  )
}

const concept: ConceptMeta = {
  id: 'two-port-network',
  title: { en: 'Two-Port Networks (ABCD)', ko: '4단자망 (ABCD 파라미터)' },
  subject: 'circuits',
  order: 23,
  description: {
    en: `## Two-Port Networks and ABCD Parameters

**Once a network has an input pair and an output pair, you can forget its insides and carry just four numbers.**

The dashed box holds an L-network: a series Z1, then a shunt Z2. Drawn as a potential landscape, height is voltage, so A = V1 ÷ V2 is a ratio of two heights you can see.
- Bright dots: the port 1 and port 2 terminals
- Cyan liquid line: current, wider and faster where there is more of it. Watch it split at the shunt node
- Glass tubes: Z1, Z2 and the load ZL, thinner for larger impedance; the amber filament thickens with the heat

**Transmission form**: V1 = A·V2 + B·I2 and I1 = C·V2 + D·I2, with I2 leaving port 2.
**This box**: A = 1 + Z1 ÷ Z2, B = Z1, C = 1 ÷ Z2, D = 1
**Reciprocity**: AD − BC = 1, checked numerically on screen
**With a load**: V2 ÷ V1 = ZL ÷ (A·ZL + B), Zin = (A·ZL + B) ÷ (C·ZL + D)

The scene cycles through the three terminal conditions that define the four numbers. Output open gives I2 = 0, so A = V1 ÷ V2 and C = I1 ÷ V2. Output shorted gives V2 = 0, so B = V1 ÷ I2 and D = I1 ÷ I2. Loaded shows what the box does in service.

Defaults Z1 = 10 Ω, Z2 = 20 Ω give A = 1.500, B = 10.0 Ω, C = 0.0500 S, D = 1.000 and AD − BC = 1 exactly. At ZL = 40 Ω that is Zin = 70 ÷ 3 = 23.3 Ω and V2 ÷ V1 = 0.5714. Raise Z1 and both A and B grow, so more of V1 is lost on the way out. Raise Z2 and A falls toward 1 while C falls toward 0, until the box is almost a plain wire. ZL never touches the four numbers, only Zin and the output bar, which is exactly why cascading two blocks is just multiplying their matrices.

In 2D: the matrix with live numbers, the AD − BC check under it, and the two port voltages as bars.`,
    ko: `## 4단자망과 ABCD 파라미터

**입력 단자쌍과 출력 단자쌍만 정해지면 내부는 잊고 네 개의 숫자만 들고 다니면 됩니다.**

점선 상자 안에는 L형 회로가 들어 있습니다. 직렬 Z1 다음에 병렬 Z2입니다. 전위 지형으로 그렸기 때문에 높이가 곧 전압이고, A = V1 ÷ V2는 눈에 보이는 두 높이의 비입니다.
- 밝은 점: 1차 단자와 2차 단자
- 시안 액체 선: 전류. 클수록 굵고 빠릅니다. 병렬 마디에서 갈라지는 모습을 보세요
- 유리관: Z1, Z2, 부하 ZL. 임피던스가 클수록 가늘고, 열이 클수록 안쪽 앰버 필라멘트가 굵어집니다

**전송 파라미터 형식**: V1 = A·V2 + B·I2, I1 = C·V2 + D·I2 (I2는 2차 단자에서 나가는 방향)
**이 상자**: A = 1 + Z1 ÷ Z2, B = Z1, C = 1 ÷ Z2, D = 1
**가역 조건**: AD − BC = 1. 화면에서 수치로 확인합니다
**부하를 달면**: V2 ÷ V1 = ZL ÷ (A·ZL + B), Zin = (A·ZL + B) ÷ (C·ZL + D)

장면은 네 숫자를 정의하는 세 가지 단자 조건을 번갈아 보여줍니다. 출력 개방이면 I2 = 0이라 A = V1 ÷ V2, C = I1 ÷ V2입니다. 출력 단락이면 V2 = 0이라 B = V1 ÷ I2, D = I1 ÷ I2입니다. 부하 연결 상태는 실제로 쓰일 때의 모습입니다.

기본값 Z1 = 10 Ω, Z2 = 20 Ω이면 A = 1.500, B = 10.0 Ω, C = 0.0500 S, D = 1.000이고 AD − BC = 1이 정확히 성립합니다. ZL = 40 Ω에서 Zin = 70 ÷ 3 = 23.3 Ω, V2 ÷ V1 = 0.5714입니다. Z1을 키우면 A와 B가 함께 커져서 V1이 더 많이 깎인 채 나갑니다. Z2를 키우면 A는 1로, C는 0으로 내려가서 상자가 거의 맨 전선에 가까워집니다. ZL은 네 숫자를 전혀 건드리지 못하고 Zin과 출력 막대만 바꿉니다. 블록 두 개를 종속 접속하는 일이 행렬 곱셈으로 끝나는 이유가 바로 이것입니다.

2D: 숫자가 살아 있는 행렬, 그 아래 AD − BC 검산, 그리고 두 단자 전압을 막대로 보여줍니다.`,
  },
  sliderParams: [
    { key: 'Z1', label: { en: 'Series (Z1)', ko: '직렬 임피던스 (Z1)' }, unit: 'Ω', min: Z_MIN, max: 40,  step: 0.5, defaultValue: 10 },
    { key: 'Z2', label: { en: 'Shunt (Z2)',  ko: '병렬 임피던스 (Z2)' }, unit: 'Ω', min: Z_MIN, max: 60,  step: 0.5, defaultValue: 20 },
    { key: 'ZL', label: { en: 'Load (ZL)',   ko: '부하 임피던스 (ZL)' }, unit: 'Ω', min: 5,     max: 200, step: 1,   defaultValue: 40 },
  ],
  Scene: TwoPortScene,
  Scene2D: AbcdMatrix2DScene,
}

export default concept
