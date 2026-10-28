import { useMemo, useRef, type MutableRefObject } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line } from '@react-three/drei'
import type { Group } from 'three'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'
import { UI, useLocaleStore } from '../../lib/i18n'

// Helpers for "potential landscape" scenes: the circuit lies flat on the floor (0 V), and every
// point is raised by its potential, so a voltage between two points is literally a height difference.

const RISE_DELAY = 0.5 // s the flat circuit stays visible before rising
const RISE_DURATION = 1.4 // s
const easeInOutCubic = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2)

/**
 * 0 → 1 over the first seconds after mount: multiply heights by it so the learner first sees the
 * ordinary circuit, then watches each point lift to its potential.
 */
export function useRiseIn(): MutableRefObject<number> {
  const rise = useRef(0)
  const start = useRef<number | null>(null)
  useFrame(({ clock }) => {
    const now = clock.getElapsedTime()
    start.current ??= now
    const t = (now - start.current - RISE_DELAY) / RISE_DURATION
    rise.current = easeInOutCubic(Math.min(Math.max(t, 0), 1))
  })
  return rise
}

type Point = [number, number, number]
const TICK = 0.12

/**
 * A vertical voltage scale standing at (x, z) with its 0 V mark on the floor.
 * `unitsPerVolt` is read every frame, so it can follow sliders and the rise-in animation.
 */
export function PotentialRuler({
  x, z, baseY, volts, unitsPerVolt,
}: { x: number; z: number; baseY: number; volts: number[]; unitsPerVolt: () => number }) {
  const c = usePhysicsColors()
  const axisRef = useRef<any>(null)
  const tickRefs = useRef<(Group | null)[]>([])
  const labelRefs = useRef<(HTMLSpanElement | null)[]>([])
  const axisBuf = useMemo(() => new Float32Array(6), [])
  const init = useMemo<Point[]>(() => [[x, baseY, z], [x, baseY, z]], [x, baseY, z])
  const tickPoints = useMemo<Point[]>(() => [[-TICK / 2, 0, 0], [TICK / 2, 0, 0]], [])
  const lo = Math.min(...volts)
  const hi = Math.max(...volts)

  useFrame(() => {
    const k = unitsPerVolt()
    axisBuf[0] = x; axisBuf[1] = baseY + lo * k; axisBuf[2] = z
    axisBuf[3] = x; axisBuf[4] = baseY + hi * k; axisBuf[5] = z
    axisRef.current?.geometry.setPositions(axisBuf)
    for (let i = 0; i < volts.length; i++) {
      tickRefs.current[i]?.position.set(x, baseY + volts[i] * k, z)
      // Hide non-zero marks while they are still bunched up on the floor
      const el = labelRefs.current[i]
      if (el) el.style.opacity = volts[i] === 0 || Math.abs(volts[i] * k) > 0.15 ? '1' : '0'
    }
  })

  return (
    <>
      <Line ref={axisRef} points={init} lineWidth={1} color={c.guide} />
      {volts.map((v, i) => (
        <group key={v} ref={(g) => { tickRefs.current[i] = g }}>
          <Line points={tickPoints} lineWidth={1} color={c.guide} />
          <Html position={[-0.12, 0, 0]} center>
            <span
              ref={(el) => { labelRefs.current[i] = el }}
              className="pointer-events-none whitespace-nowrap text-[10px] tabular-nums text-muted-foreground transition-opacity"
              style={{ transform: 'translateX(-50%)' }}
            >
              {v} V
            </span>
          </Html>
        </group>
      ))}
    </>
  )
}

/** Caption on the floor explaining that the dashed outline is the ordinary circuit at 0 V. */
export function FloorCaption({ position }: { position: Point }) {
  const t = UI[useLocaleStore((s) => s.locale)]
  return (
    <Html position={position} center>
      <span className="pointer-events-none whitespace-nowrap text-[10px] text-muted-foreground">
        {t.floorCaption}
      </span>
    </Html>
  )
}
