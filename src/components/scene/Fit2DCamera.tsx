import { useLayoutEffect } from 'react'
import { useThree } from '@react-three/fiber'
import type { PerspectiveCamera } from 'three'

// 2D plots live in x ∈ [−4, 4], y ∈ [−2, 2] (DESIGN.md); fit that plus a margin
const CONTENT_W = 8.6
const CONTENT_H = 4.6
// Screen space taken by the overlays (px): collapsed sidebar, docked card, top bar, slider HUD
// ponytail: fixed reservations; on phone-width screens the card will overlap the plot
const RESERVED = { left: 200, right: 340, top: 60, bottom: 110 }

/**
 * Moves the default camera so the 2D plot fills the space the overlays leave free.
 * `rightReserved` overrides the docked-card width for wider right-hand overlays,
 * such as the narration (StoryNarration), which is broader than ConceptCard2D.
 */
export function Fit2DCamera({ rightReserved = RESERVED.right }: { rightReserved?: number }) {
  const camera = useThree((s) => s.camera) as PerspectiveCamera
  const { width, height } = useThree((s) => s.size)

  useLayoutEffect(() => {
    const freeW = Math.max(width - RESERVED.left - rightReserved, 1)
    const freeH = Math.max(height - RESERVED.top - RESERVED.bottom, 1)
    const pxPerUnit = Math.min(freeW / CONTENT_W, freeH / CONTENT_H)
    const tan = Math.tan((camera.fov * Math.PI) / 360)
    const z = height / (2 * pxPerUnit * tan)
    // Shift the camera so the plot's center lands in the middle of the free area
    const dxPx = RESERVED.left + freeW / 2 - width / 2
    const dyPx = RESERVED.top + freeH / 2 - height / 2
    camera.position.set(-dxPx / pxPerUnit, dyPx / pxPerUnit, z)
    camera.updateProjectionMatrix()
  }, [camera, width, height, rightReserved])

  return null
}
