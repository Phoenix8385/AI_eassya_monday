/**
 * The hero's signature moment — and the only 3D in the app.
 *
 * A page of prose rendered as a particle field, with a scan band sweeping down
 * it. Points inside the band brighten to amber and lift toward the camera;
 * everything else sits at low-contrast ink. It is the tool's own process shown
 * literally: text being passed over and measured, line by line.
 *
 * **All animation happens on the GPU.** Each frame updates three uniforms and
 * nothing else — no per-particle JavaScript, no geometry rebuilds, no React
 * state. That is what keeps ~8,000 points cheap enough to hold 60fps on
 * integrated graphics, and it is why the particle budget is the only knob that
 * needs turning if a device struggles.
 *
 * This module is the lazy-loaded chunk: three, @react-three/fiber and drei are
 * imported here and nowhere else, so nothing above pays for WebGL.
 */

import { useMemo, useRef } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import {
  buildScanField,
  FIELD_BOTTOM,
  FIELD_TOP,
} from './scanFieldGeometry'

/** Ink and amber, matching the CSS tokens exactly. */
const INK = new THREE.Color('#2A2F38')
const AMBER = new THREE.Color('#B8720B')
const AMBER_HOT = new THREE.Color('#F0A93C')

const SWEEP_SECONDS = 7.5
/** Half-height of the bright band, in world units. */
const BAND = 0.26

const vertexShader = /* glsl */ `
  uniform float uTime;
  uniform float uScanY;
  uniform float uBand;
  uniform float uSize;
  uniform float uPixelRatio;

  attribute float aSeed;

  varying float vIntensity;

  void main() {
    vec3 pos = position;

    // Constant low-amplitude drift so the field is alive even between sweeps.
    float phase = uTime * 0.35 + aSeed * 6.2831853;
    pos.z += sin(phase) * 0.035;
    pos.x += cos(phase * 0.7) * 0.006;

    // Distance from the scan band, 1 at its centre and 0 outside it.
    float d = abs(pos.y - uScanY);
    float intensity = 1.0 - smoothstep(0.0, uBand, d);
    // Sharpen so the band has a defined edge rather than a soft haze.
    intensity = pow(intensity, 1.7);

    // Scanned points lift toward the camera — the band reads as a physical
    // pass over the page rather than a colour change painted on it.
    pos.z += intensity * 0.42;

    vIntensity = intensity;

    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
    gl_Position = projectionMatrix * mvPosition;

    // Manual size attenuation: bigger when scanned, smaller with distance.
    float size = uSize * (1.0 + intensity * 2.4);
    gl_PointSize = size * uPixelRatio * (1.0 / -mvPosition.z);
  }
`

const fragmentShader = /* glsl */ `
  uniform vec3 uInk;
  uniform vec3 uAmber;
  uniform vec3 uAmberHot;

  varying float vIntensity;

  void main() {
    // Round points. Square particles are the giveaway of an untouched default.
    vec2 offset = gl_PointCoord - vec2(0.5);
    float dist = dot(offset, offset);
    if (dist > 0.25) discard;

    // Soft edge, so points don't alias into hard squares at small sizes.
    float edge = 1.0 - smoothstep(0.16, 0.25, dist);

    vec3 color = mix(uInk, uAmber, clamp(vIntensity * 1.5, 0.0, 1.0));
    color = mix(color, uAmberHot, smoothstep(0.65, 1.0, vIntensity));

    float alpha = (0.34 + vIntensity * 0.66) * edge;
    gl_FragColor = vec4(color, alpha);
  }
`

interface FieldProps {
  maxPoints: number
  onFrame?: () => void
}

function ScanField({ maxPoints, onFrame }: FieldProps) {
  const materialRef = useRef<THREE.ShaderMaterial>(null)
  const groupRef = useRef<THREE.Group>(null)
  const { positions, seeds, count } = useMemo(
    () => buildScanField({ maxPoints }),
    [maxPoints],
  )

  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    g.setAttribute('aSeed', new THREE.BufferAttribute(seeds, 1))
    return g
  }, [positions, seeds])

  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uScanY: { value: FIELD_TOP },
      uBand: { value: BAND },
      uSize: { value: 13.0 },
      uPixelRatio: { value: 1 },
      uInk: { value: INK },
      uAmber: { value: AMBER },
      uAmberHot: { value: AMBER_HOT },
    }),
    [],
  )

  const { viewport, pointer } = useThree()
  uniforms.uPixelRatio.value = Math.min(viewport.dpr, 2)

  useFrame((state, delta) => {
    onFrame?.()
    const material = materialRef.current
    if (material) {
      material.uniforms.uTime.value += delta
      // Sweep top to bottom, then restart. A wrapping sawtooth rather than a
      // ping-pong: a scan that runs backwards up the page reads as an error.
      const t = (material.uniforms.uTime.value % SWEEP_SECONDS) / SWEEP_SECONDS
      material.uniforms.uScanY.value =
        FIELD_TOP - t * (FIELD_TOP - FIELD_BOTTOM)
    }

    // Pointer parallax, damped. Rotating the page slightly is enough; moving
    // the camera makes the layout feel unstable.
    const group = groupRef.current
    if (group) {
      const targetY = pointer.x * 0.18
      const targetX = -pointer.y * 0.1
      group.rotation.y += (targetY - group.rotation.y) * Math.min(delta * 2.5, 1)
      group.rotation.x += (targetX - group.rotation.x) * Math.min(delta * 2.5, 1)
    }
    void state
  })

  return (
    <group ref={groupRef} rotation={[0, -0.18, 0]}>
      <points geometry={geometry} frustumCulled={false}>
        <shaderMaterial
          ref={materialRef}
          uniforms={uniforms}
          vertexShader={vertexShader}
          fragmentShader={fragmentShader}
          transparent
          depthWrite={false}
          blending={THREE.NormalBlending}
        />
      </points>
      <PointCountProbe count={count} />
    </group>
  )
}

/** Exposes the realised point count for the FPS probe, without a re-render. */
function PointCountProbe({ count }: { count: number }) {
  const { gl } = useThree()
  ;(gl.domElement as HTMLCanvasElement & { dataset: DOMStringMap }).dataset.points =
    String(count)
  return null
}

export interface ScanFieldSceneProps {
  /** Particle budget. Lowered on small viewports by the caller. */
  maxPoints?: number
  onFrame?: () => void
}

export default function ScanFieldScene({
  maxPoints = 8000,
  onFrame,
}: ScanFieldSceneProps) {
  return (
    <Canvas
      // Capped device pixel ratio: a 3x phone screen would otherwise render
      // ~9x the fragments for no visible gain on a field of soft points.
      dpr={[1, 1.75]}
      camera={{ position: [0, 0, 5.2], fov: 42 }}
      gl={{
        antialias: false, // round soft points don't need it; it costs fill rate
        powerPreference: 'high-performance',
        alpha: true,
      }}
      style={{ pointerEvents: 'none' }}
      aria-hidden="true"
    >
      <ScanField maxPoints={maxPoints} onFrame={onFrame} />
    </Canvas>
  )
}
