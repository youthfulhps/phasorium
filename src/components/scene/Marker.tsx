import { forwardRef } from 'react'
import type { Group } from 'three'

/** "You are here" dot with a soft halo. Position it through the forwarded group ref. */
export const Marker = forwardRef<Group, { color: string; size?: number }>(function Marker(
  { color, size = 0.07 },
  ref
) {
  return (
    <group ref={ref}>
      <mesh>
        <sphereGeometry args={[size, 24, 24]} />
        <meshBasicMaterial color={color} />
      </mesh>
      <mesh>
        <sphereGeometry args={[size * 1.9, 24, 24]} />
        <meshBasicMaterial color={color} transparent opacity={0.12} depthWrite={false} />
      </mesh>
    </group>
  )
})
