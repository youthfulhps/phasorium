import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import { useSliderStore } from '../../store/sliderStore'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { useLocaleStore } from '../../lib/i18n'
import { TrailLine } from '../../components/scene/TrailLine'
import { Marker } from '../../components/scene/Marker'
import { Resistor, setResistorHeat } from '../../components/scene/Resistor'
import type { ConceptMeta } from '../../types/concept'

const BETA = 100 // current gain hFE, fixed
const VCE_SAT = 0.2 // V, collector-emitter saturation voltage
const VZ = 5.1 // V, Zener breakdown voltage
const RS = 330 // Ω, series dropping resistor of the shunt regulator

const VCE_FULL = 20 // V across the full width of the BJT plot
const IC_FULL = 8e-3 // A up the full height of the BJT plot
const I_FULL = 48e-3 // A along the full length of the current-split bar
const P_FULL = 0.7 // W: filament fills its tube at this dissipation
const FLOW_K = 180 // sheen bands per second per amp, the same constant for every current drawn

const SOL = {
  IB: 0, IC: 0, IE: 0, alpha: 0, VCE: 0, ICsat: 0, sat: false, Rc: 0, Vcc: 0,
  Vout: 0, IL: 0, Itot: 0, Iz: 0, Pz: 0, PRs: 0, PRL: 0, reg: false, VinDrop: 0, RL: 0,
}

/*
 * Ground truth, both verified numerically before any scene code was written.
 *
 * BJT, common emitter:
 *   IC = beta * IB,  IE = IC + IB,  alpha = beta / (beta + 1),  load line VCE = Vcc - IC*Rc
 *   beta = 100, IB = 20 uA, Vcc = 12 V, Rc = 2.2 kohm
 *     -> IC = 2.000 mA, IE = 2.020 mA, alpha = 0.990099, VCE = 7.600 V (active: VCE > VCE_SAT)
 *   Saturation boundary: the load line allows at most ICsat = (Vcc - VCE_SAT) / Rc. Once beta*IB
 *   reaches it, IC stops following beta*IB and VCE sticks at VCE_SAT. At Vcc = 12 V, Rc = 6 kohm
 *   that ceiling is 11.8 / 6000 = 1.967 mA, so IB = 20 uA already saturates the transistor.
 *
 * Zener shunt regulator:
 *   IL = Vz / RL,  Itotal = (Vin - Vz) / Rs,  Iz = Itotal - IL,  Pz = Iz * Vz
 *   Vin = 12 V, Vz = 5.1 V, Rs = 330 ohm, RL = 1 kohm
 *     -> IL = 5.100 mA, Itotal = 20.909091 mA, Iz = 15.809091 mA, Pz = 80.626 mW
 *   Vin -> 15 V (RL unchanged): IL and Vout do not move at all, Itotal = 30.000 mA,
 *     so Iz absorbs the whole extra 9.0909 mA and becomes 24.900 mA.
 *   RL -> 500 ohm (Vin unchanged): Itotal stays 20.909091 mA, IL = 10.200 mA,
 *     so Iz gives up exactly that much and becomes 10.709091 mA. Iz + IL = Itotal, unchanged.
 *   Dropout: Iz reaches 0 when Vin*RL / (Rs + RL) falls to Vz, that is Vin = Vz*(Rs + RL) / RL
 *     (6.783 V at RL = 1 kohm, 13.515 V at RL = 200 ohm). Below it the Zener is off and
 *     Vout = Vin*RL / (Rs + RL) < Vz, so the regulated rail collapses into a plain divider.
 */
function solve({ IB_uA = 20, Rc_k = 2.2, Vin = 12, RL = 1000 }: Record<string, number>) {
  const IB = IB_uA * 1e-6
  const Rc = Rc_k * 1e3
  const ICsat = Math.max((Vin - VCE_SAT) / Rc, 0) // the most the load line leaves room for
  const ICact = BETA * IB
  const sat = ICact >= ICsat
  const IC = sat ? ICsat : ICact
  SOL.IB = IB
  SOL.IC = IC
  SOL.IE = IC + IB
  SOL.alpha = BETA / (BETA + 1)
  SOL.VCE = Vin - IC * Rc // equals VCE_SAT exactly once saturated
  SOL.ICsat = ICsat
  SOL.sat = sat
  SOL.Rc = Rc
  SOL.Vcc = Vin

  const Vdiv = (Vin * RL) / (RS + RL) // what the output would be with the Zener removed
  const reg = Vdiv > VZ
  const Vout = reg ? VZ : Vdiv
  const IL = Vout / RL
  const Itot = (Vin - Vout) / RS
  SOL.Vout = Vout
  SOL.IL = IL
  SOL.Itot = Itot
  SOL.Iz = reg ? Itot - IL : 0
  SOL.Pz = SOL.Iz * Vout
  SOL.PRs = (Vin - Vout) * Itot
  SOL.PRL = Vout * IL
  SOL.reg = reg
  SOL.VinDrop = (VZ * (RS + RL)) / RL
  SOL.RL = RL
  return SOL
}

function useSliderValues() {
  const valuesRef = useRef<Record<string, number>>(useSliderStore.getState().values)
  useEffect(
    () => useSliderStore.subscribe((s) => s.values, (v) => { valuesRef.current = v }),
    []
  )
  return valuesRef
}

const fmtV = (v: number) => `${v.toFixed(2)} V`
const fmtmA = (a: number) => `${(a * 1e3).toFixed(2)} mA`
const fmtuA = (a: number) => `${(a * 1e6).toFixed(0)} µA`
const fmtmW = (p: number) => `${(p * 1e3).toFixed(1)} mW`
const fmtR = (r: number) => (r >= 1000 ? `${(r / 1000).toFixed(2)} kΩ` : `${r.toFixed(0)} Ω`)

// Labels read in the foreground color; only the two accents (current, power) tint their text
const labelClass = 'pointer-events-none whitespace-nowrap text-[11px] font-semibold tabular-nums text-foreground/85'
const noteClass = 'pointer-events-none whitespace-nowrap text-[11px] tabular-nums text-muted-foreground'

const L10N = {
  en: {
    bias: 'BJT bias',
    rungs: 'one rung every ΔIB 10 µA = ΔIC 1.00 mA',
    active: 'active',
    sat: 'saturated',
    ceiling: 'saturation ceiling',
    loadline: 'load line',
    reg: 'regulating',
    drop: 'dropout',
    chars: 'Output characteristics',
    zreg: 'Zener regulation',
    below: (v: string) => `dropout below Vin ${v}`,
    nozener: 'dashed: Vout with no Zener',
  },
  ko: {
    bias: 'BJT 바이어스',
    rungs: '가로선 한 칸이 ΔIB 10 µA = ΔIC 1.00 mA',
    active: '활성',
    sat: '포화',
    ceiling: '포화 한계',
    loadline: '부하선',
    reg: '정전압 유지',
    drop: '드롭아웃',
    chars: '출력 특성 곡선',
    zreg: '제너 전압 조정',
    below: (v: string) => `Vin ${v} 아래에서는 드롭아웃`,
    nozener: '점선: 제너가 없을 때의 Vout',
  },
}

// ── shared geometry helpers ──────────────────────────────────────────────────
const X_AXIS = new THREE.Vector3(1, 0, 0)
const dirV = new THREE.Vector3()
const unitV = new THREE.Vector3()
const NODES = new Float64Array(8) // up to 4 (x, y) nodes of a branch path
const SEGLEN = new Float64Array(3)

const init = (n: number): [number, number, number][] =>
  Array.from({ length: n }, (): [number, number, number] => [0, 0, 0])

/** Even samples along the polyline held in NODES, written into buf. Allocates nothing. */
function fillPath(buf: Float32Array, count: number, nNodes: number, z: number) {
  let total = 0
  for (let k = 0; k < nNodes - 1; k++) {
    SEGLEN[k] = Math.hypot(NODES[(k + 1) * 2] - NODES[k * 2], NODES[(k + 1) * 2 + 1] - NODES[k * 2 + 1])
    total += SEGLEN[k]
  }
  for (let i = 0; i < count; i++) {
    let d = (i / (count - 1)) * total
    let k = 0
    while (k < nNodes - 2 && d > SEGLEN[k]) { d -= SEGLEN[k]; k++ }
    const t = SEGLEN[k] > 1e-9 ? Math.min(d / SEGLEN[k], 1) : 0
    const o = i * 3
    buf[o] = NODES[k * 2] + t * (NODES[(k + 1) * 2] - NODES[k * 2])
    buf[o + 1] = NODES[k * 2 + 1] + t * (NODES[(k + 1) * 2 + 1] - NODES[k * 2 + 1])
    buf[o + 2] = z
  }
}

/** Two-point segment into a drei Line. */
function setSeg(buf: Float32Array, line: any, x0: number, y0: number, x1: number, y1: number, z: number) {
  buf[0] = x0; buf[1] = y0; buf[2] = z
  buf[3] = x1; buf[4] = y1; buf[5] = z
  line?.geometry.setPositions(buf)
  line?.computeLineDistances()
}

/** Seat a unit-sized glass tube on the segment, covering `frac` of its length. */
function seat(g: THREE.Group | null, x0: number, y0: number, x1: number, y1: number, frac: number, r: number, z: number) {
  if (!g) return
  g.position.set((x0 + x1) / 2, (y0 + y1) / 2, z)
  dirV.set(x1 - x0, y1 - y0, 0)
  const len = dirV.length()
  if (len > 1e-6) g.quaternion.setFromUnitVectors(X_AXIS, unitV.copy(dirV).divideScalar(len))
  g.scale.set(Math.max(len * frac, 1e-3), r, r)
}

const heat = (p: number) => Math.sqrt(Math.min(Math.max(p, 0), P_FULL) / P_FULL)

// ── 3D layout: BJT load-line plot on the left, Zener shunt regulator on the right ──
// Everything stays inside x ∈ [−2.06, 1.88], y ∈ [−1.40, 0.98]: clear of the floating card
// (which projects to the right quarter of the viewport) and of the slider HUD at the bottom.
const Z_GUIDE = 0
const Z_LINE = 0.01
const Z_CUR = 0.02
const Z_DEV = 0.03
const Z_TXT = 0.04

const BX = -2.0
const BY = -1.08
const BW = 1.7
const BH = 1.68
const xv = (vce: number) => BX + (vce / VCE_FULL) * BW
const yi = (ic: number) => BY + (ic / IC_FULL) * BH

const ZX = 0.15 // supply node and left end of the ground rail
const ZXO = 1.0 // regulated output node, where the Zener hangs
const ZXL = 1.78 // load branch
const ZG = -0.62 // ground rail, 0 V
const KV = 0.095 // scene units per volt: height = potential in the right panel
const yv = (v: number) => ZG + v * KV

const BARY = -1.0 // the series current laid out as one bar, split at a marker
const KI = 1.72 / I_FULL
const bx = (i: number) => ZX + i * KI

const RUNG_IB = [10e-6, 20e-6, 30e-6, 40e-6, 50e-6, 60e-6]

const N_ICL = 28
const N_SER = 48
const N_ZBR = 24
const N_LBR = 32
const N_BAR = 24
const ICL_INIT = init(N_ICL)
const SER_INIT = init(N_SER)
const ZBR_INIT = init(N_ZBR)
const LBR_INIT = init(N_LBR)
const BAR_INIT = init(N_BAR)
const SEG_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0]]

const B_AXIS_X: [number, number, number][] = [[BX - 0.06, BY, Z_GUIDE], [BX + BW + 0.1, BY, Z_GUIDE]]
const B_AXIS_Y: [number, number, number][] = [[BX, BY - 0.06, Z_GUIDE], [BX, BY + BH + 0.1, Z_GUIDE]]
const GND_RAIL: [number, number, number][] = [[ZX - 0.08, ZG, Z_GUIDE], [ZXL + 0.1, ZG, Z_GUIDE]]
// Vz never moves, so its reference level is static geometry: the output node either sits on it or
// visibly drops below it in dropout.
const VZ_GUIDE: [number, number, number][] = [[ZX - 0.08, yv(VZ), Z_GUIDE], [ZXL + 0.1, yv(VZ), Z_GUIDE]]

// Which 3D labels follow geometry, and which of the two accents tints them
const DYN_3D: { i: number; tint?: 'current' | 'power' }[] = [
  { i: 1 }, { i: 2, tint: 'current' }, { i: 3 }, { i: 5 }, { i: 6 },
  { i: 7, tint: 'power' }, { i: 8, tint: 'current' }, { i: 9, tint: 'current' }, { i: 11 },
]

function TransistorZenerScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()

  const iclBuf = useMemo(() => new Float32Array(N_ICL * 3), [])
  const serBuf = useMemo(() => new Float32Array(N_SER * 3), [])
  const zbrBuf = useMemo(() => new Float32Array(N_ZBR * 3), [])
  const lbrBuf = useMemo(() => new Float32Array(N_LBR * 3), [])
  const bilBuf = useMemo(() => new Float32Array(N_BAR * 3), [])
  const bizBuf = useMemo(() => new Float32Array(N_BAR * 3), [])
  const seg = useMemo(() => new Float32Array(6), [])

  const curRefs = useRef<any[]>([]) // 0 IC, 1 series, 2 Zener branch, 3 load branch, 4 bar IL, 5 bar Iz
  const rungRefs = useRef<any[]>([])
  const activeRef = useRef<any>(null)
  const loadRef = useRef<any>(null)
  const satRef = useRef<any>(null)
  const tickRef = useRef<any>(null)
  const resRefs = useRef<(THREE.Group | null)[]>([]) // Rs, Zener, RL
  const qRef = useRef<THREE.Group>(null)
  const nodeRefs = useRef<(THREE.Group | null)[]>([]) // supply node, output node, bar divider
  const tagRefs = useRef<Record<number, THREE.Group | null>>({})
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const flow = useRef([0, 0, 0, 0, 0, 0])
  const builtFor = useRef<unknown>(null)
  const builtCol = useRef<unknown>(null)

  useFrame(() => {
    const vals = valuesRef.current
    const s = solve(vals)

    // Flow speed tracks the current itself, with one constant across both circuits: the base
    // current is practically frozen next to a collector current beta times larger.
    flow.current[0] = FLOW_K * s.IC
    flow.current[1] = FLOW_K * s.Itot
    flow.current[2] = FLOW_K * s.Iz
    flow.current[3] = FLOW_K * s.IL
    flow.current[4] = FLOW_K * s.IL
    flow.current[5] = FLOW_K * s.Iz

    if (builtFor.current === vals && builtCol.current === c) return
    builtFor.current = vals
    builtCol.current = c
    const t = L10N[locale]

    // ── left panel: the load line and the Q-point riding it ──
    const xQ = xv(s.VCE)
    const yQ = yi(s.IC)
    const xcc = xv(s.Vcc)
    for (let k = 0; k < RUNG_IB.length; k++) {
      const y = yi(BETA * RUNG_IB[k])
      setSeg(seg, rungRefs.current[k], xv(VCE_SAT), y, xcc, y, Z_GUIDE)
    }
    setSeg(seg, activeRef.current, xv(VCE_SAT), yi(BETA * s.IB), xcc, yi(BETA * s.IB), Z_LINE)
    // Load line from cutoff (VCE = Vcc, IC = 0) toward the IC axis, clipped at the top of the plot
    const icTop = Math.min(s.Vcc / s.Rc, IC_FULL)
    setSeg(seg, loadRef.current, xcc, yi(0), xv(s.Vcc - icTop * s.Rc), yi(icTop), Z_LINE)
    // The ceiling the load line imposes; off-scale ceilings are simply not drawn
    const satVisible = s.ICsat <= IC_FULL
    if (satRef.current) satRef.current.visible = satVisible
    if (satVisible) setSeg(seg, satRef.current, BX, yi(s.ICsat), xcc, yi(s.ICsat), Z_LINE)
    // Collector current read off the IC axis across to the Q-point
    NODES[0] = BX; NODES[1] = yQ
    NODES[2] = xQ; NODES[3] = yQ
    fillPath(iclBuf, N_ICL, 2, Z_CUR)
    curRefs.current[0]?.geometry.setPositions(iclBuf)
    qRef.current?.position.set(xQ, yQ, Z_DEV)

    // ── right panel: height = potential, so the output level either holds or falls ──
    const yVin = yv(s.Vcc)
    const yOut = yv(s.Vout)
    NODES[0] = ZX; NODES[1] = ZG
    NODES[2] = ZX; NODES[3] = yVin
    NODES[4] = ZXO; NODES[5] = yOut
    fillPath(serBuf, N_SER, 3, Z_CUR)
    curRefs.current[1]?.geometry.setPositions(serBuf)
    NODES[0] = ZXO; NODES[1] = yOut
    NODES[2] = ZXO; NODES[3] = ZG
    fillPath(zbrBuf, N_ZBR, 2, Z_CUR)
    curRefs.current[2]?.geometry.setPositions(zbrBuf)
    if (curRefs.current[2]) curRefs.current[2].visible = s.Iz > 1e-6
    NODES[0] = ZXO; NODES[1] = yOut
    NODES[2] = ZXL; NODES[3] = yOut
    NODES[4] = ZXL; NODES[5] = ZG
    fillPath(lbrBuf, N_LBR, 3, Z_CUR)
    curRefs.current[3]?.geometry.setPositions(lbrBuf)

    seat(resRefs.current[0], ZX, yVin, ZXO, yOut, 0.5, 0.055, Z_DEV)
    seat(resRefs.current[1], ZXO, yOut, ZXO, ZG, 0.55, 0.06, Z_DEV)
    seat(resRefs.current[2], ZXL, yOut, ZXL, ZG, 0.55, 0.06, Z_DEV)
    setResistorHeat(resRefs.current[0], heat(s.PRs))
    setResistorHeat(resRefs.current[1], heat(s.Pz))
    setResistorHeat(resRefs.current[2], heat(s.PRL))
    nodeRefs.current[0]?.position.set(ZX, yVin, Z_DEV)
    nodeRefs.current[1]?.position.set(ZXO, yOut, Z_DEV)

    // ── the split bar: IL first, then Iz, so moving RL slides the divider while the far end
    // holds, and moving Vin holds the divider while the far end slides. ──
    const xDiv = bx(s.IL)
    const xEnd = bx(s.Itot)
    NODES[0] = ZX; NODES[1] = BARY
    NODES[2] = xDiv; NODES[3] = BARY
    fillPath(bilBuf, N_BAR, 2, Z_CUR)
    curRefs.current[4]?.geometry.setPositions(bilBuf)
    NODES[0] = xDiv; NODES[1] = BARY
    NODES[2] = xEnd; NODES[3] = BARY
    fillPath(bizBuf, N_BAR, 2, Z_CUR)
    curRefs.current[5]?.geometry.setPositions(bizBuf)
    if (curRefs.current[5]) curRefs.current[5].visible = s.Iz > 1e-6
    nodeRefs.current[2]?.position.set(xDiv, BARY, Z_DEV)
    setSeg(seg, tickRef.current, xEnd, BARY - 0.1, xEnd, BARY + 0.1, Z_LINE)

    // ── labels, anchored to the geometry they name ──
    tagRefs.current[1]?.position.set(xQ, yQ + 0.21, Z_TXT)
    tagRefs.current[2]?.position.set((BX + xQ) / 2, yQ - 0.18, Z_TXT)
    tagRefs.current[3]?.position.set(xv(0) + 0.06, yi(s.ICsat) + 0.13, Z_TXT)
    tagRefs.current[5]?.position.set(ZX + 0.06, yVin + 0.21, Z_TXT)
    tagRefs.current[6]?.position.set(ZXO + 0.04, yOut + 0.22, Z_TXT)
    tagRefs.current[7]?.position.set(ZXO + 0.24, (yOut + ZG) / 2, Z_TXT)
    tagRefs.current[8]?.position.set((ZX + xDiv) / 2, BARY - 0.19, Z_TXT)
    tagRefs.current[9]?.position.set((xDiv + xEnd) / 2, BARY + 0.19, Z_TXT)
    tagRefs.current[11]?.position.set(ZXL - 0.27, (yOut + ZG) / 2, Z_TXT)

    const texts = [
      `${t.bias} · IC = β·IB, β ${BETA} · ${t.rungs}`,
      `Q · VCE ${fmtV(s.VCE)} · ${s.sat ? t.sat : t.active}`,
      `IC ${fmtmA(s.IC)} ← IB ${fmtuA(s.IB)}`,
      satVisible ? `${t.ceiling} · IC ${fmtmA(s.ICsat)}` : '',
      `VCE → · ${t.loadline}: Vcc ${fmtV(s.Vcc)}, Rc ${fmtR(s.Rc)}`,
      `Vin ${fmtV(s.Vcc)} → Rs ${fmtR(RS)}`,
      `Vout ${fmtV(s.Vout)} · Vz ${fmtV(VZ)}`,
      `Pz ${fmtmW(s.Pz)}`,
      `IL ${fmtmA(s.IL)}`,
      `Iz ${fmtmA(s.Iz)}`,
      `Itotal ${fmtmA(s.Itot)} = IL + Iz · ${s.reg ? t.reg : t.drop}`,
      `RL ${fmtR(s.RL)}`,
    ]
    texts.forEach((text, k) => { if (textRefs.current[k]) textRefs.current[k]!.textContent = text })
  })

  useEffect(() => { builtFor.current = null }, [locale])

  return (
    <>
      <PerspectiveCamera makeDefault position={[3, 2, 5]} fov={50} />
      <OrbitControls enablePan={false} />
      {/* Background, lights, grid and bloom come from the shared SceneStage */}

      {/* BJT plot: axes, the IC = β·IB rungs, the active rung, the load line, the saturation ceiling */}
      <Line points={B_AXIS_X} lineWidth={1.5} color={c.guide} />
      <Line points={B_AXIS_Y} lineWidth={1.5} color={c.guide} />
      {RUNG_IB.map((_, k) => (
        <Line
          key={k}
          ref={(l) => { rungRefs.current[k] = l }}
          points={SEG_INIT}
          lineWidth={1}
          color={c.guide}
          dashed
          dashSize={0.07}
          gapSize={0.06}
        />
      ))}
      <Line ref={activeRef} points={SEG_INIT} lineWidth={2} color={c.inductance} />
      <Line ref={loadRef} points={SEG_INIT} lineWidth={2.5} color={c.resistance} />
      <Line
        ref={satRef}
        points={SEG_INIT}
        lineWidth={1.5}
        color={c.resistance}
        dashed
        dashSize={0.14}
        gapSize={0.08}
      />

      {/* Zener panel: the 0 V rail and the Vz level the output is pinned to */}
      <Line points={GND_RAIL} lineWidth={1.5} color={c.guide} />
      <Line points={VZ_GUIDE} lineWidth={1.5} color={c.voltage} dashed dashSize={0.12} gapSize={0.08} />
      <Line ref={tickRef} points={SEG_INIT} lineWidth={1.5} color={c.guide} />

      {/* Rs, the Zener and RL as glass tubes: the amber filament thickens with the power each one
          takes, so the Zener's swells as it absorbs more and goes out entirely in dropout. */}
      <Resistor ref={(g) => { resRefs.current[0] = g }} />
      <Resistor ref={(g) => { resRefs.current[1] = g }} />
      <Resistor ref={(g) => { resRefs.current[2] = g }} />

      {/* Every current as a cyan liquid line, flowing at a speed proportional to itself */}
      {[0, 1, 2, 3, 4, 5].map((k) => (
        <TrailLine
          key={k}
          ref={(l) => { curRefs.current[k] = l }}
          points={[ICL_INIT, SER_INIT, ZBR_INIT, LBR_INIT, BAR_INIT, BAR_INIT][k]}
          lineWidth={k >= 4 ? 4 : 3}
          color={c.current}
          fade={false}
          flowSpeed={() => flow.current[k]}
        />
      ))}

      <Marker ref={qRef} color={c.marker} size={0.07} />
      {[0, 1, 2].map((k) => (
        <Marker key={k} ref={(g) => { nodeRefs.current[k] = g }} color={c.marker} size={0.05} />
      ))}

      <Html position={[BX + 0.02, BY + BH + 0.2, Z_TXT]}>
        <span ref={(el) => { textRefs.current[0] = el }} className={noteClass} />
      </Html>
      <Html position={[BX + 0.02, BY - 0.14, Z_TXT]}>
        <span ref={(el) => { textRefs.current[4] = el }} className={noteClass} />
      </Html>
      <Html position={[ZX, BARY - 0.4, Z_TXT]}>
        <span ref={(el) => { textRefs.current[10] = el }} className={noteClass} />
      </Html>
      {DYN_3D.map(({ i, tint }) => (
        <group key={i} ref={(g) => { tagRefs.current[i] = g }}>
          <Html center>
            <span
              ref={(el) => { textRefs.current[i] = el }}
              className={labelClass}
              style={tint ? { color: tint === 'current' ? c.current : c.power } : undefined}
            />
          </Html>
        </group>
      ))}
    </>
  )
}

// ── 2D: the textbook pair of characteristic curves ──
// The BJT output characteristics with the load line across them, and beside them the regulator's
// Vout against Vin. The regulation curve is a sweep over Vin, which no single slider setting can
// show in 3D, so this view carries information the 3D scene cannot.
const PX = -3.7
const PY = -1.25
const PH = 2.35
const PWB = 2.7 // BJT box width
const PWZ = 3.3 // Zener box width
const ZPX = -0.3
const cx = (vce: number) => PX + (vce / VCE_FULL) * PWB
const cy = (ic: number) => PY + (ic / IC_FULL) * PH
const zx2 = (vin: number) => ZPX + (vin / VCE_FULL) * PWZ
const zy2 = (v: number) => PY + (v / VCE_FULL) * PH

const N_CURVE = 72
const CURVE_INIT = init(N_CURVE)
const C_AXIS_X: [number, number, number][] = [[PX - 0.06, PY, 0], [PX + PWB + 0.1, PY, 0]]
const C_AXIS_Y: [number, number, number][] = [[PX, PY - 0.06, 0], [PX, PY + PH + 0.1, 0]]
const Z_AXIS_X: [number, number, number][] = [[ZPX - 0.06, PY, 0], [ZPX + PWZ + 0.1, PY, 0]]
const Z_AXIS_Y: [number, number, number][] = [[ZPX, PY - 0.06, 0], [ZPX, PY + PH + 0.1, 0]]
const VZ2_GUIDE: [number, number, number][] = [
  [ZPX, zy2(VZ), 0], [ZPX + PWZ + 0.1, zy2(VZ), 0],
]
const REG_INIT: [number, number, number][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
const KNEE = 0.08 // V: width of the knee where the device curve leaves the origin

const DYN_2D: { i: number; tint?: 'current' }[] = [{ i: 1 }, { i: 4 }, { i: 5 }]

function TransistorZener2DScene() {
  const c = usePhysicsColors()
  const locale = useLocaleStore((s) => s.locale)
  const valuesRef = useSliderValues()

  const curveBufs = useMemo(
    () => RUNG_IB.map(() => new Float32Array(N_CURVE * 3)),
    []
  )
  const regBuf = useMemo(() => new Float32Array(9), [])
  const seg = useMemo(() => new Float32Array(6), [])
  const curveRefs = useRef<any[]>([])
  const loadRef = useRef<any>(null)
  const regRef = useRef<any>(null)
  const divRef = useRef<any>(null)
  const kneeRef = useRef<any>(null)
  const qRef = useRef<THREE.Group>(null)
  const opRef = useRef<THREE.Group>(null)
  const tagRefs = useRef<Record<number, THREE.Group | null>>({})
  const textRefs = useRef<(HTMLSpanElement | null)[]>([])
  const builtFor = useRef<unknown>(null)
  const builtCol = useRef<unknown>(null)

  useFrame(() => {
    const vals = valuesRef.current
    if (builtFor.current === vals && builtCol.current === c) return
    builtFor.current = vals
    builtCol.current = c
    const s = solve(vals)
    const t = L10N[locale]

    // Output characteristics: each curve climbs out of the origin over about VCE_SAT, then holds at
    // IC = β·IB. The Early effect is left out, so the active part is drawn flat.
    for (let k = 0; k < RUNG_IB.length; k++) {
      const top = BETA * RUNG_IB[k]
      const buf = curveBufs[k]
      for (let i = 0; i < N_CURVE; i++) {
        const u = i / (N_CURVE - 1)
        const vce = VCE_FULL * u * u * u // dense near the knee, sparse along the flat part
        const o = i * 3
        buf[o] = cx(vce)
        buf[o + 1] = cy(top * (1 - Math.exp(-vce / KNEE)))
        buf[o + 2] = 0
      }
      curveRefs.current[k]?.geometry.setPositions(buf)
    }
    const icTop = Math.min(s.Vcc / s.Rc, IC_FULL)
    setSeg(seg, loadRef.current, cx(s.Vcc), cy(0), cx(s.Vcc - icTop * s.Rc), cy(icTop), 0)
    qRef.current?.position.set(cx(s.VCE), cy(s.IC), 0.01)

    // Regulation curve: a plain divider up to the dropout point, then flat at Vz
    const vd = Math.min(s.VinDrop, VCE_FULL)
    regBuf[0] = zx2(0); regBuf[1] = zy2(0); regBuf[2] = 0
    regBuf[3] = zx2(vd); regBuf[4] = zy2(Math.min(VZ, (vd * s.RL) / (RS + s.RL))); regBuf[5] = 0
    regBuf[6] = zx2(VCE_FULL); regBuf[7] = zy2(VZ); regBuf[8] = 0
    regRef.current?.geometry.setPositions(regBuf)
    // What the output would be with the Zener removed: the regulator holds it down to Vz
    setSeg(seg, divRef.current, zx2(0), zy2(0), zx2(VCE_FULL), zy2((VCE_FULL * s.RL) / (RS + s.RL)), 0)
    setSeg(seg, kneeRef.current, zx2(vd), zy2(0), zx2(vd), zy2(VZ), 0)
    opRef.current?.position.set(zx2(s.Vcc), zy2(s.Vout), 0.01)

    tagRefs.current[1]?.position.set(cx(s.VCE), cy(s.IC) + 0.22, 0.02)
    tagRefs.current[4]?.position.set(zx2(vd) + 0.02, zy2(VZ) + 0.46, 0.02)
    tagRefs.current[5]?.position.set(zx2(s.Vcc), zy2(s.Vout) + 0.22, 0.02)

    const texts = [
      `${t.chars} · IB 10 … 60 µA, β ${BETA}`,
      `Q · IC ${fmtmA(s.IC)} · VCE ${fmtV(s.VCE)} · ${s.sat ? t.sat : t.active}`,
      `VCE (V) → · ${t.loadline}: Vcc ${fmtV(s.Vcc)}, Rc ${fmtR(s.Rc)}`,
      `${t.zreg} · Vz ${fmtV(VZ)}, Rs ${fmtR(RS)}, RL ${fmtR(s.RL)}`,
      t.below(fmtV(s.VinDrop)),
      `Vin ${fmtV(s.Vcc)} → Vout ${fmtV(s.Vout)}`,
      `Vin (V) → · ${t.nozener}`,
    ]
    texts.forEach((text, k) => { if (textRefs.current[k]) textRefs.current[k]!.textContent = text })
  })

  useEffect(() => { builtFor.current = null }, [locale])

  return (
    <>
      <Line points={C_AXIS_X} lineWidth={1.5} color={c.guide} />
      <Line points={C_AXIS_Y} lineWidth={1.5} color={c.guide} />
      <Line points={Z_AXIS_X} lineWidth={1.5} color={c.guide} />
      <Line points={Z_AXIS_Y} lineWidth={1.5} color={c.guide} />
      <Line points={VZ2_GUIDE} lineWidth={1.5} color={c.voltage} dashed dashSize={0.12} gapSize={0.08} />

      {RUNG_IB.map((_, k) => (
        <Line
          key={k}
          ref={(l) => { curveRefs.current[k] = l }}
          points={CURVE_INIT}
          lineWidth={2}
          color={c.inductance}
        />
      ))}
      <Line ref={loadRef} points={SEG_INIT} lineWidth={2.5} color={c.resistance} />
      <Line ref={regRef} points={REG_INIT} lineWidth={3} color={c.voltage} />
      <Line ref={divRef} points={SEG_INIT} lineWidth={1.5} color={c.guide} dashed dashSize={0.1} gapSize={0.08} />
      <Line ref={kneeRef} points={SEG_INIT} lineWidth={1} color={c.guide} dashed dashSize={0.07} gapSize={0.06} />

      <Marker ref={qRef} color={c.marker} size={0.07} />
      <Marker ref={opRef} color={c.marker} size={0.07} />

      <Html position={[PX + 0.02, PY + PH + 0.22, 0.02]}>
        <span ref={(el) => { textRefs.current[0] = el }} className={noteClass} />
      </Html>
      <Html position={[PX + 0.02, PY - 0.16, 0.02]}>
        <span ref={(el) => { textRefs.current[2] = el }} className={noteClass} />
      </Html>
      <Html position={[ZPX + 0.02, PY + PH + 0.22, 0.02]}>
        <span ref={(el) => { textRefs.current[3] = el }} className={noteClass} />
      </Html>
      <Html position={[ZPX + 0.02, PY - 0.16, 0.02]}>
        <span ref={(el) => { textRefs.current[6] = el }} className={noteClass} />
      </Html>
      {DYN_2D.map(({ i }) => (
        <group key={i} ref={(g) => { tagRefs.current[i] = g }}>
          <Html center>
            <span ref={(el) => { textRefs.current[i] = el }} className={labelClass} />
          </Html>
        </group>
      ))}
    </>
  )
}

const concept: ConceptMeta = {
  id: 'transistor-zener',
  title: { en: 'Transistor Bias & Zener Regulation', ko: '트랜지스터 바이어스와 제너 전압 조정' },
  subject: 'electronics',
  order: 3,
  description: {
    en: `## Transistor Bias & Zener Regulation

**Two devices, one idea: a small controlling quantity holds a much larger current in place.**

Left panel, the BJT. Up the plot is collector current IC, across it is collector-emitter voltage VCE.
- Faint dashed rungs: the active-region levels IC = β·IB, one rung per 10 µA of base current
- Light neutral line: the rung the base current is actually asking for right now
- Mid neutral line: the load line VCE = Vcc − IC·Rc, set by the supply and Rc alone
- Marker: the Q-point, where what the transistor asks for meets what the load line allows
- Cyan liquid line: the collector current, flowing faster the larger it is

Right panel, the Zener shunt regulator, drawn with height = potential. The supply node rides up and down with Vin while the output node is pinned to the dashed Vz level and refuses to move. Below the circuit the same series current is laid out as one bar, split at a marker: IL on the left, Iz on the right.
- Cyan liquid lines: the three currents, each flowing at its own speed
- Amber filaments: the power each device takes, thickest in the Zener when it is absorbing the most

**BJT**: IC = β·IB, IE = IC + IB, α = β ÷ (β + 1), load line VCE = Vcc − IC·Rc
**Saturation**: once IC·Rc reaches Vcc − VCE(sat) the load line has no room left, IC stops following β·IB, and VCE sticks at about 0.2 V
**Zener shunt**: IL = Vz ÷ RL, Itotal = (Vin − Vz) ÷ Rs, Iz = Itotal − IL, Pz = Iz·Vz

With β = 100, IB = 20 µA, Vcc = 12 V and Rc = 2.2 kΩ: IC = 2.000 mA, IE = 2.020 mA, α = 0.9901 and VCE = 7.600 V, safely active. Raise Rc to 6 kΩ and the load line allows at most (12 − 0.2) ÷ 6 kΩ = 1.967 mA, so the transistor saturates: the rung for 20 µA now floats above a Q-point that can no longer climb to it.

With Vin = 12 V, Vz = 5.1 V, Rs = 330 Ω and RL = 1 kΩ: IL = 5.100 mA, Itotal = 20.909 mA, Iz = 15.809 mA and Pz = 80.63 mW. Raise Vin to 15 V and Itotal becomes 30.000 mA while IL and Vout do not budge, so Iz swallows the whole extra 9.091 mA and climbs to 24.900 mA. Leave Vin alone and drop RL to 500 Ω instead: Itotal stays at 20.909 mA, IL doubles to 10.200 mA, and Iz gives up exactly that much, falling to 10.709 mA. The bar shows both invariants at once. Moving RL leaves its total length untouched and only slides the split, because Iz + IL = Itotal and Itotal does not depend on RL. Moving Vin leaves the IL segment untouched and only lengthens the Iz segment.

The regulator has a floor. Iz reaches zero when Vin·RL ÷ (Rs + RL) falls to Vz, that is at Vin = Vz·(Rs + RL) ÷ RL, which is 6.78 V at RL = 1 kΩ and 13.52 V at RL = 200 Ω. Below it the Zener is off: its amber filament goes out, the Iz segment of the bar vanishes, and the output node drops below the dashed Vz level and simply tracks Vin through the divider.

In 2D: the BJT output characteristics with the load line drawn across them and the Q-point at their intersection, and beside them the regulator's Vout against Vin, flat above dropout and sloping below. That second plot is a sweep over Vin, something no single setting of the sliders can show in 3D, which is why it earns the space.

Try it. Walk IB up from zero and watch the Q-point slide along the load line from cutoff at the right end up to the saturation ceiling, then stop dead while the rungs keep climbing past it. Raise Rc to tilt the load line and bring that ceiling down to meet the Q-point sooner. On the right, move Vin and watch only the Iz segment grow; move RL and watch the split slide while the total end stays put; push either far enough and the output falls off the Vz level.`,
    ko: `## 트랜지스터 바이어스와 제너 전압 조정

**소자는 둘이지만 생각은 하나입니다. 작은 제어량이 훨씬 큰 전류를 제자리에 붙잡아 둡니다.**

왼쪽은 BJT입니다. 세로축은 컬렉터 전류 IC, 가로축은 컬렉터·이미터 전압 VCE입니다.
- 희미한 점선 가로선들: 활성 영역의 IC = β·IB 높이. 베이스 전류 10 µA마다 한 칸입니다
- 밝은 중성색 선: 지금의 베이스 전류가 요구하는 높이
- 중간 중성색 선: 부하선 VCE = Vcc − IC·Rc. 전원과 Rc만으로 정해집니다
- 마커: 동작점 Q. 트랜지스터가 요구하는 값과 부하선이 허용하는 값이 만나는 곳입니다
- 시안 액체 선: 컬렉터 전류. 클수록 빠르게 흐릅니다

오른쪽은 제너 병렬(션트) 정전압 회로이며, 높이가 전위입니다. 전원 쪽 노드는 Vin에 따라 오르내리지만, 출력 노드는 점선으로 표시한 Vz 높이에 고정되어 꼼짝하지 않습니다. 회로 아래에는 같은 직렬 전류를 하나의 막대로 펼쳐 놓았고, 마커가 그 막대를 둘로 나눕니다. 왼쪽이 IL, 오른쪽이 Iz입니다.
- 시안 액체 선: 세 전류. 각자 자기 크기에 맞는 속도로 흐릅니다
- 호박색 필라멘트: 각 소자가 받는 전력. 제너가 가장 많이 흡수할 때 가장 굵어집니다

**BJT**: IC = β·IB, IE = IC + IB, α = β ÷ (β + 1), 부하선 VCE = Vcc − IC·Rc
**포화**: IC·Rc가 Vcc − VCE(sat)에 이르면 부하선에 더 이상 여유가 없어, IC는 β·IB를 따라가지 못하고 VCE는 약 0.2 V에 붙어 버립니다
**제너 션트**: IL = Vz ÷ RL, Itotal = (Vin − Vz) ÷ Rs, Iz = Itotal − IL, Pz = Iz·Vz

β = 100, IB = 20 µA, Vcc = 12 V, Rc = 2.2 kΩ일 때 IC = 2.000 mA, IE = 2.020 mA, α = 0.9901, VCE = 7.600 V로 넉넉히 활성 영역입니다. Rc를 6 kΩ으로 올리면 부하선이 허용하는 최대 전류가 (12 − 0.2) ÷ 6 kΩ = 1.967 mA뿐이므로 트랜지스터는 포화합니다. 20 µA에 해당하는 가로선은 이제 Q점보다 위에 떠 있고, Q점은 거기까지 올라갈 수 없습니다.

Vin = 12 V, Vz = 5.1 V, Rs = 330 Ω, RL = 1 kΩ일 때 IL = 5.100 mA, Itotal = 20.909 mA, Iz = 15.809 mA, Pz = 80.63 mW입니다. Vin을 15 V로 올리면 IL과 Vout은 전혀 움직이지 않은 채 Itotal이 30.000 mA가 되므로, 늘어난 9.091 mA를 제너가 모두 삼켜 Iz가 24.900 mA로 올라갑니다. Vin은 그대로 두고 RL만 500 Ω으로 낮추면 Itotal은 20.909 mA에 머물고 IL이 10.200 mA로 두 배가 되며, 제너는 정확히 그만큼을 내놓아 Iz가 10.709 mA로 줄어듭니다. 막대 하나가 이 두 불변량을 동시에 보여 줍니다. RL을 움직이면 Iz + IL = Itotal이고 Itotal은 RL과 무관하므로 막대의 전체 길이는 그대로이고 나누는 지점만 움직입니다. Vin을 움직이면 IL 구간은 그대로이고 Iz 구간만 길어집니다.

이 정전압 회로에는 바닥이 있습니다. Vin·RL ÷ (Rs + RL)이 Vz까지 내려오면 Iz가 0이 되는데, 그 경계는 Vin = Vz·(Rs + RL) ÷ RL입니다. RL = 1 kΩ에서 6.78 V, RL = 200 Ω에서 13.52 V입니다. 그 아래에서는 제너가 꺼집니다. 호박색 필라멘트가 사라지고, 막대의 Iz 구간도 없어지며, 출력 노드는 점선 Vz 높이 아래로 내려가 그냥 분압기처럼 Vin을 따라갑니다.

2D에서는 BJT 출력 특성 곡선 위에 부하선을 그리고 그 교점에 Q점을 찍었습니다. 그 옆은 정전압 특성, 즉 Vin에 대한 Vout으로, 드롭아웃 위에서는 평평하고 아래에서는 기울어집니다. 이 두 번째 그림은 Vin을 훑어 본 결과이고, 3D에서는 슬라이더를 어떻게 맞춰도 한 번에 볼 수 없는 정보이기에 자리를 차지할 값이 있습니다.

직접 해 보세요. IB를 0에서 올리면 Q점이 부하선을 따라 오른쪽 끝의 차단 상태에서 포화 한계까지 미끄러져 올라가다가, 가로선들은 계속 올라가는데도 딱 멈춰 섭니다. Rc를 키우면 부하선이 기울어 그 한계가 내려오고 Q점이 더 빨리 그것을 만납니다. 오른쪽에서는 Vin을 움직여 Iz 구간만 늘어나는 것을, RL을 움직여 전체 끝은 그대로인 채 나누는 지점만 미끄러지는 것을 보세요. 어느 쪽이든 충분히 밀면 출력이 Vz 높이에서 떨어져 나갑니다.`,
  },
  sliderParams: [
    { key: 'IB_uA', label: { en: 'Base current (IB)',       ko: '베이스 전류 (IB)' },    unit: 'µA', min: 0,   max: 60,   step: 2,   defaultValue: 20   },
    { key: 'Rc_k',  label: { en: 'Collector resistor (Rc)', ko: '컬렉터 저항 (Rc)' },    unit: 'kΩ', min: 0.5, max: 6,    step: 0.1, defaultValue: 2.2  },
    { key: 'Vin',   label: { en: 'Supply voltage (Vcc = Vin)', ko: '공급 전압 (Vcc = Vin)' }, unit: 'V', min: 6, max: 20, step: 0.5, defaultValue: 12 },
    { key: 'RL',    label: { en: 'Load resistance (RL)',    ko: '부하 저항 (RL)' },      unit: 'Ω',  min: 200, max: 5000, step: 50,  defaultValue: 1000 },
  ],
  Scene: TransistorZenerScene,
  Scene2D: TransistorZener2DScene,
}

export default concept
