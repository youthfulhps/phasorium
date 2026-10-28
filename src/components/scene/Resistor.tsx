import { forwardRef } from 'react'
import { BackSide, type Group } from 'three'
import { usePhysicsColors } from '../../hooks/usePhysicsColors'

/**
 * A resistor as a faint glass tube with an amber filament inside. Heat is shown by the filament's
 * thickness, never by blending gray toward amber (that midpoint reads as brown).
 * The group is unit-sized along local X: set position/quaternion, then
 * scale.set(length, radius, radius), and call setResistorHeat() every frame.
 */
export const Resistor = forwardRef<Group>(function Resistor(_, ref) {
  const c = usePhysicsColors()
  return (
    <group ref={ref}>
      <mesh rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[1, 1, 1, 24]} />
        {/* Back faces only: the glass sits behind the filament instead of tinting it */}
        <meshBasicMaterial color={c.resistance} transparent opacity={0.35} depthWrite={false} side={BackSide} />
      </mesh>
      <mesh rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[1, 1, 0.96, 24]} />
        <meshBasicMaterial color={c.power} />
      </mesh>
    </group>
  )
})

/** heat ∈ [0, 1] (e.g. √(P / Pmax)): 0 = cold, empty tube; 1 = filament fills the tube. */
export function setResistorHeat(group: Group | null, heat: number) {
  const filament = group?.children[1]
  if (!filament) return
  const r = Math.min(Math.max(heat, 0), 1) * 0.9
  filament.visible = r > 0.02
  filament.scale.set(r, 1, r) // cylinder is rotated: its local x/z are the tube's radial axes
}
