import { forwardRef, useImperativeHandle, useLayoutEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Line } from '@react-three/drei'
import { AdditiveBlending, Color, InstancedBufferAttribute, NormalBlending } from 'three'
import type { Line2, LineMaterial } from 'three-stdlib'
import { useIsDark } from '../../hooks/usePhysicsColors'

type Point = [number, number, number]

// Liquid look (gloss, flowing sheen, oil-slick hue drift), done in shading only —
// the curve's shape is physics and must never wobble.
// vUv.x runs −1…1 across the line's width; vProgress runs 0 (tail) … 1 (head).
const VERTEX_DECL = /* glsl */ `
attribute float instanceProgressStart;
attribute float instanceProgressEnd;
varying float vProgress;
`
const VERTEX_MAIN = /* glsl */ `
  vProgress = ( position.y < 0.5 ) ? instanceProgressStart : instanceProgressEnd;
`
const FRAGMENT_DECL = /* glsl */ `
uniform float uTime;
uniform float uFlow;   // accumulated sheen phase (bands travelled)
uniform float uFade;   // 1 = trail fades toward its tail, 0 = uniform (closed loops)
varying float vProgress;

// Hue drifts this far (fraction of the color wheel, ≈ ±15°) around the base color —
// a subtle shimmer, so each quantity clearly keeps its own color.
#define HUE_SPREAD 0.04

vec3 rgb2hsv( vec3 c ) {
  vec4 K = vec4( 0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0 );
  vec4 p = mix( vec4( c.bg, K.wz ), vec4( c.gb, K.xy ), step( c.b, c.g ) );
  vec4 q = mix( vec4( p.xyw, c.r ), vec4( c.r, p.yzx ), step( p.x, c.r ) );
  float d = q.x - min( q.w, q.y );
  return vec3( abs( q.z + ( q.w - q.y ) / ( 6.0 * d + 1e-10 ) ), d / ( q.x + 1e-10 ), q.x );
}

vec3 hsv2rgb( vec3 c ) {
  vec3 p = abs( fract( c.xxx + vec3( 1.0, 2.0 / 3.0, 1.0 / 3.0 ) ) * 6.0 - 3.0 );
  return c.z * mix( vec3( 1.0 ), clamp( p - 1.0, 0.0, 1.0 ), c.y );
}
`
const FRAGMENT_SHADE = /* glsl */ `
  {
    float x = clamp( vUv.x, -1.0, 1.0 );

    // Oil-slick color: hue swirls along the line over time and shifts slightly across its width
    vec3 hsv = rgb2hsv( diffuseColor.rgb );
    float swirl = sin( ( vProgress * 2.5 - uTime * 0.25 ) * 6.2831853 ) + 0.6 * sin( ( vProgress * 5.0 + uTime * 0.4 ) * 6.2831853 );
    hsv.x = fract( hsv.x + HUE_SPREAD * ( 0.6 * swirl + 0.5 * x ) );
    diffuseColor.rgb = hsv2rgb( hsv );

    float body = sqrt( 1.0 - x * x );                                      // round tube cross-section
    float spec = pow( max( 0.0, 1.0 - abs( x + 0.35 ) * 2.2 ), 3.0 );      // off-center gloss
    float wave = 0.5 + 0.5 * sin( ( vProgress * 14.0 - uFlow ) * 6.2831853 );
    float flow = pow( wave, 10.0 ) * mix( 1.0, smoothstep( 0.15, 0.9, vProgress ), uFade );  // sheen flowing tail -> head
    vec3 tint = mix( diffuseColor.rgb, vec3( 1.0 ), 0.65 );
    diffuseColor.rgb = diffuseColor.rgb * ( 0.6 + 0.4 * body ) + tint * ( 0.5 * spec + 0.45 * flow );
    diffuseColor.a *= 0.75 + 0.25 * body;                                  // softer edges
  }
`

type Uniforms = { uTime: { value: number }; uFlow: { value: number }; uFade: { value: number } }

function applyLiquidShader(material: LineMaterial, uniforms: Uniforms) {
  material.defines.USE_LINE_COLOR_ALPHA = ''
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms)
    shader.vertexShader = shader.vertexShader
      .replace('void main() {', `${VERTEX_DECL}\nvoid main() {\n${VERTEX_MAIN}`)
    shader.fragmentShader = shader.fragmentShader
      .replace('void main() {', `${FRAGMENT_DECL}\nvoid main() {`)
      .replace('#include <logdepthbuf_fragment>', `${FRAGMENT_SHADE}\n#include <logdepthbuf_fragment>`)
  }
  material.customProgramCacheKey = () => 'liquid-trail'
  material.needsUpdate = true
}

/** Per-segment 0…1 position along the trail, fed to the shader as vProgress. */
function setProgress(line: Line2, count: number) {
  const segments = count - 1
  const start = new Float32Array(segments)
  const end = new Float32Array(segments)
  for (let i = 0; i < segments; i++) {
    start[i] = i / segments
    end[i] = (i + 1) / segments
  }
  line.geometry.setAttribute('instanceProgressStart', new InstancedBufferAttribute(start, 1))
  line.geometry.setAttribute('instanceProgressEnd', new InstancedBufferAttribute(end, 1))
}

type TrailLineProps = {
  points: Point[]
  color: string
  lineWidth?: number
  /** Fade the tail (index 0) out. Turn off for closed loops such as a circuit. Default true. */
  fade?: boolean
  /** Sheen bands per second travelling tail → head; a function is read every frame (e.g. ∝ current). */
  flowSpeed?: number | (() => number)
}

/**
 * A glossy "liquid" line. The head (last index) is full color, the tail optionally fades out, and a
 * sheen flows from tail to head. Additive in dark mode so it glows with the bloom.
 * Update it like any drei Line: ref.current.geometry.setPositions(buffer).
 */
export const TrailLine = forwardRef<Line2, TrailLineProps>(
  function TrailLine({ points, color, lineWidth = 2.5, fade = true, flowSpeed = 1.6 }, ref) {
    const dark = useIsDark()
    const lineRef = useRef<Line2>(null!)
    const uniforms = useMemo<Uniforms>(() => ({ uTime: { value: 0 }, uFlow: { value: 0 }, uFade: { value: 1 } }), [])
    const count = points.length
    uniforms.uFade.value = fade ? 1 : 0

    useImperativeHandle(ref, () => lineRef.current)

    const vertexColors = useMemo(() => {
      const { r, g, b } = new Color(color)
      return Array.from({ length: count }, (_, i): [number, number, number, number] => [
        r, g, b, fade ? (i / (count - 1)) ** 1.6 : 1,
      ])
    }, [color, count, fade])

    // drei rebuilds the geometry when colors change, so re-attach progress each time
    useLayoutEffect(() => {
      setProgress(lineRef.current, count)
      applyLiquidShader(lineRef.current.material as LineMaterial, uniforms)
    }, [vertexColors, count, uniforms])

    useFrame(({ clock }, dt) => {
      uniforms.uTime.value = clock.getElapsedTime()
      uniforms.uFlow.value += dt * (typeof flowSpeed === 'function' ? flowSpeed() : flowSpeed)
    })

    return (
      <Line
        ref={lineRef}
        points={points}
        vertexColors={vertexColors}
        lineWidth={lineWidth}
        transparent
        depthWrite={false}
        blending={dark ? AdditiveBlending : NormalBlending}
      />
    )
  }
)
