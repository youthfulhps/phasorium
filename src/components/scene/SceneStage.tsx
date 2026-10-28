import { Grid } from '@react-three/drei'
import { Bloom, EffectComposer } from '@react-three/postprocessing'
import { useIsDark, usePhysicsColors } from '../../hooks/usePhysicsColors'

/**
 * Shared backdrop for every concept: page-matching background, lights, a fading floor grid in 3D,
 * and bloom in dark mode only (glow is invisible on a white background).
 */
export function SceneStage({ is3D }: { is3D: boolean }) {
  const c = usePhysicsColors()
  const dark = useIsDark()

  return (
    <>
      <color attach="background" args={[c.backdrop]} />
      <ambientLight intensity={0.8} />
      <pointLight position={[5, 5, 5]} intensity={0.5} />
      {is3D && (
        <Grid
          position={[0, -2.2, 0]}
          infiniteGrid
          cellSize={0.5}
          sectionSize={2}
          cellThickness={0.5}
          sectionThickness={0.9}
          cellColor={c.guide}
          sectionColor={c.guide}
          fadeDistance={16}
          fadeStrength={2}
        />
      )}
      {dark && (
        <EffectComposer>
          <Bloom mipmapBlur intensity={0.9} luminanceThreshold={0.15} luminanceSmoothing={0.35} />
        </EffectComposer>
      )}
    </>
  )
}
