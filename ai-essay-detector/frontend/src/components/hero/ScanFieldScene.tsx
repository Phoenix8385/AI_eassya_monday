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
 * state. The whole field is one draw call, measured at ~3.2ms per frame for
 * 7,500 points on a *software* rasterizer (no GPU at all) — roughly a fifth of
 * a 60fps budget, which is the headroom that makes real hardware a non-issue.
 * Re-measure with /bench.html; the particle budget is the knob to turn first.
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
import {
  BAND,
  COLOR_AMBER,
  COLOR_AMBER_HOT,
  COLOR_INK,
  fragmentShader,
  POINT_SIZE,
  SWEEP_SECONDS,
  vertexShader,
} from './scanFieldShaders'

const INK = new THREE.Color(COLOR_INK)
const AMBER = new THREE.Color(COLOR_AMBER)
const AMBER_HOT = new THREE.Color(COLOR_AMBER_HOT)


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
      uSize: { value: POINT_SIZE },
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
  /**
   * Ceiling on particles, not a target. The page layout yields ~7,500 at full
   * size, so raising this above that changes nothing; lowering it is what
   * trims the field on small or touch devices.
   */
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
