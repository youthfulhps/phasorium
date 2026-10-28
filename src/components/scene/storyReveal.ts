import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import type { Material } from 'three'
import { useStoryStepRef } from '../../store/storyStore'

const REVEAL_RATE = 5 // per second, as a fraction of the remaining gap

/**
 * Eased 0 → 1 weights for whatever the current narration step asks to be on screen, so a scene can
 * fade elements in and out inside useFrame without a React re-render. `targets` writes the step's
 * wanted weights into the array it is handed, and the hook eases the live weights toward them.
 * Register it before the scene's own useFrame so the weights are fresh when that frame reads them.
 */
export function useStoryReveal(targets: (step: number, out: Float32Array) => void, count: number) {
  const stepRef = useStoryStepRef()
  const reveal = useMemo(() => new Float32Array(count), [count])
  const target = useMemo(() => new Float32Array(count), [count])
  useFrame((_, delta) => {
    targets(stepRef.current, target)
    const k = Math.min(1, delta * REVEAL_RATE)
    for (let i = 0; i < count; i++) reveal[i] += (target[i] - reveal[i]) * k
  })
  return reveal
}

/**
 * For a scene whose readouts quote numbers that the current step changes. Hands back the step's
 * *settled* target weights — not the eased live ones — so a quoted number appears together with the
 * sentence that quotes it instead of creeping up behind it, and only when something has actually
 * changed, so the text is rebuilt at most once per slider drag or step.
 *
 *     const labels = useSettledLabels(revealTargets, REV_COUNT)
 *     // inside useFrame, once the <Html> spans exist:
 *     const w = textRefs.current[4] ? labels.settled(valuesRef.current) : null
 *     if (w) { const q = solve(valuesRef.current, w[REV_L], w[REV_C]); ... }
 *
 * `deps` invalidates the cache on anything outside the sliders that the text depends on, typically
 * the locale.
 */
export function useSettledLabels(
  targets: (step: number, out: Float32Array) => void,
  count: number,
  deps: unknown[] = []
) {
  const stepRef = useStoryStepRef()
  const weights = useMemo(() => new Float32Array(count), [count])
  const seenValues = useRef<unknown>(null)
  const seenStep = useRef(-1)
  useEffect(() => { seenValues.current = null }, deps)
  return useMemo(() => ({
    settled(values: unknown): Float32Array | null {
      if (seenValues.current === values && seenStep.current === stepRef.current) return null
      seenValues.current = values
      seenStep.current = stepRef.current
      targets(stepRef.current, weights)
      return weights
    },
  }), [targets, weights, stepRef])
}

/** Fade an <Html> readout along with the thing it labels. */
export function setOpacity(el: HTMLElement | null, v: number) {
  if (el) el.style.opacity = `${v}`
}

/**
 * Fade anything that owns a material by mutating it: a drei Line, a TrailLine which forwards one,
 * or a plain Mesh. A Mesh's `material` may be an array, in which case only the first slot is
 * touched — no scene here uses more than one.
 */
export function setLineOpacity(
  line: { material?: Material | Material[] } | null,
  v: number
) {
  const own = line?.material
  const m = Array.isArray(own) ? own[0] : own
  if (!m) return
  m.transparent = true
  m.opacity = v
}
