/**
 * Perf harness for the hero scan field. Served at /bench.html by `npm run dev`.
 *
 * Exists because "test the frame rate on a mid-range laptop" cannot be answered
 * on the machine that wrote the code. This renders the *shipping* geometry and
 * shaders — imported from the same modules the hero uses, never re-typed — so
 * the number it reports is the number the hero gets.
 *
 * Two measurements, because they answer different questions:
 *
 *   Sustained FPS (rAF)   what a person actually sees. Only meaningful with the
 *                         tab focused and visible; browsers suspend rAF in
 *                         background tabs, which is reported rather than hidden.
 *   Median frame time     cost of one render() call, driven directly. Survives
 *                         a throttled tab, so it still says something useful in
 *                         automation where rAF never fires.
 */

import * as THREE from 'three'
import { buildScanField, FIELD_BOTTOM, FIELD_TOP } from '../components/hero/scanFieldGeometry'
import {
  BAND,
  COLOR_AMBER,
  COLOR_AMBER_HOT,
  COLOR_INK,
  fragmentShader,
  POINT_SIZE,
  SWEEP_SECONDS,
  vertexShader,
} from '../components/hero/scanFieldShaders'

const TARGET_FPS = 30 // Phase D's floor: below this, cut particles first.

/** Most recently built harness, so the freeze button can re-render it. */
let current: Harness | null = null
const RUN_MS = 6000

function el<T extends HTMLElement>(id: string): T {
  return document.getElementById(id) as T
}

interface Harness {
  renderer: THREE.WebGLRenderer
  scene: THREE.Scene
  camera: THREE.PerspectiveCamera
  material: THREE.ShaderMaterial
  points: number
  dispose: () => void
}

function build(maxPoints: number): Harness {
  const stage = el('stage')
  stage.innerHTML = ''

  const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75))
  renderer.setSize(window.innerWidth, window.innerHeight)
  stage.appendChild(renderer.domElement)

  const scene = new THREE.Scene()
  const camera = new THREE.PerspectiveCamera(
    42,
    window.innerWidth / window.innerHeight,
    0.1,
    100,
  )
  camera.position.set(0, 0, 5.2)

  const { positions, seeds, count } = buildScanField({ maxPoints })
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geometry.setAttribute('aSeed', new THREE.BufferAttribute(seeds, 1))

  const material = new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uScanY: { value: FIELD_TOP },
      uBand: { value: BAND },
      uSize: { value: POINT_SIZE },
      uPixelRatio: { value: Math.min(window.devicePixelRatio, 1.75) },
      uInk: { value: new THREE.Color(COLOR_INK) },
      uAmber: { value: new THREE.Color(COLOR_AMBER) },
      uAmberHot: { value: new THREE.Color(COLOR_AMBER_HOT) },
    },
    vertexShader,
    fragmentShader,
    transparent: true,
    depthWrite: false,
  })

  const cloud = new THREE.Points(geometry, material)
  cloud.frustumCulled = false
  cloud.rotation.y = -0.18
  scene.add(cloud)

  return {
    renderer,
    scene,
    camera,
    material,
    points: count,
    dispose: () => {
      geometry.dispose()
      material.dispose()
      renderer.dispose()
      stage.innerHTML = ''
    },
  }
}

function advance(material: THREE.ShaderMaterial, dt: number) {
  material.uniforms.uTime.value += dt
  const t = (material.uniforms.uTime.value % SWEEP_SECONDS) / SWEEP_SECONDS
  material.uniforms.uScanY.value = FIELD_TOP - t * (FIELD_TOP - FIELD_BOTTOM)
}

function percentile(sorted: number[], p: number): number {
  if (!sorted.length) return 0
  const i = Math.min(sorted.length - 1, Math.floor(sorted.length * p))
  return sorted[i]
}

/**
 * Frame cost including GPU execution.
 *
 * Timing `renderer.render()` alone measures how long it took to *queue* the
 * draw, not to perform it — with one draw call that is near zero and tells you
 * nothing. Reading a single pixel back forces the pipeline to finish first, so
 * the elapsed time includes the GPU actually doing the work.
 *
 * The readback stall is why this runs as its own pass rather than inside the
 * FPS loop: it would depress the very frame rate it is meant to characterise.
 */
function measureSyncedFrameCost(h: Harness, frames: number): number[] {
  const gl = h.renderer.getContext()
  const pixel = new Uint8Array(4)
  const samples: number[] = []

  for (let i = 0; i < frames; i += 1) {
    advance(h.material, 1 / 60)
    const t0 = performance.now()
    h.renderer.render(h.scene, h.camera)
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel)
    samples.push(performance.now() - t0)
  }
  return samples
}

function rendererName(renderer: THREE.WebGLRenderer): string {
  try {
    const gl = renderer.getContext()
    const dbg = gl.getExtension('WEBGL_debug_renderer_info')
    if (dbg) return String(gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL))
    return String(gl.getParameter(gl.RENDERER))
  } catch {
    return '(unavailable)'
  }
}

async function run(maxPoints: number) {
  const h = build(maxPoints)
  current = h
  el('r-points').textContent = h.points.toLocaleString()
  el('r-gpu').textContent = rendererName(h.renderer)
  el('verdict').textContent = 'Measuring…'
  el('verdict').className = 'verdict'

  const frameTimes: number[] = []
  let visibleThroughout = document.visibilityState === 'visible'
  let frames = 0
  const start = performance.now()
  let last = start

  await new Promise<void>((resolve) => {
    let rafSeen = false
    const tick = () => {
      const now = performance.now()
      const dt = (now - last) / 1000
      last = now
      advance(h.material, dt)

      const t0 = performance.now()
      h.renderer.render(h.scene, h.camera)
      frameTimes.push(performance.now() - t0)

      frames += 1
      rafSeen = true
      if (document.visibilityState !== 'visible') visibleThroughout = false

      if (now - start < RUN_MS) requestAnimationFrame(tick)
      else resolve()
    }
    requestAnimationFrame(tick)

    // Fallback for a throttled/hidden tab, where rAF never fires: drive
    // render() directly so frame cost is still measurable. FPS is NOT reported
    // from this path — it would be meaningless.
    setTimeout(() => {
      if (rafSeen) return
      visibleThroughout = false
      for (let i = 0; i < 300; i += 1) {
        advance(h.material, 1 / 60)
        const t0 = performance.now()
        h.renderer.render(h.scene, h.camera)
        frameTimes.push(performance.now() - t0)
      }
      resolve()
    }, 1200)
  })

  const elapsed = (performance.now() - start) / 1000
  const fps = frames > 0 ? frames / elapsed : 0

  // Frame cost comes from the synced pass, not from the FPS loop above: the
  // unsynced timings there only cover command submission.
  const sorted = measureSyncedFrameCost(h, 120).sort((a, b) => a - b)
  const median = percentile(sorted, 0.5)
  const p95 = percentile(sorted, 0.95)
  void frameTimes

  el('r-fps').textContent = frames > 0 ? `${fps.toFixed(1)} fps` : 'n/a (rAF suspended)'
  el('r-median').textContent = `${median.toFixed(2)} ms`
  el('r-p95').textContent = `${p95.toFixed(2)} ms`
  el('r-calls').textContent = String(h.renderer.info.render.calls)
  el('r-vis').textContent = visibleThroughout ? 'yes' : 'no — FPS unreliable'

  const verdict = el('verdict')
  if (frames > 0 && visibleThroughout) {
    const ok = fps >= TARGET_FPS
    verdict.className = `verdict ${ok ? 'ok' : 'bad'}`
    verdict.textContent = ok
      ? `PASS — ${fps.toFixed(0)} fps sustained at ${h.points.toLocaleString()} points, above the ${TARGET_FPS} fps floor.`
      : `BELOW FLOOR — ${fps.toFixed(0)} fps at ${h.points.toLocaleString()} points. Reduce the particle budget in HeroVisual.tsx before polishing anything else.`
  } else {
    verdict.className = 'verdict'
    verdict.textContent =
      `Frame cost measured at ${median.toFixed(2)} ms median (${h.points.toLocaleString()} points), ` +
      `but the tab was not visible so sustained FPS could not be measured. ` +
      `A frame budget of 16.7 ms is 60 fps and 33.3 ms is 30 fps — re-run with this tab focused for the real number.`
  }

  // Keep the last scene on screen so the harness shows what it measured.
  void h.dispose
}

/**
 * Paint one rendered frame into a 2D canvas that survives repaint.
 *
 * A WebGL drawing buffer is discarded after compositing unless
 * `preserveDrawingBuffer` is set, so once the render loop stops (or the tab is
 * backgrounded and rAF is suspended) the canvas goes blank — which makes "does
 * the field actually look right?" impossible to answer from a screenshot.
 * Reading the pixels back immediately after a render captures the real frame
 * without paying for preserveDrawingBuffer on every frame.
 */
function freezeFrame(h: Harness, sweep: number) {
  const gl = h.renderer.getContext()
  const size = h.renderer.getDrawingBufferSize(new THREE.Vector2())
  const w = size.x
  const ht = size.y

  h.material.uniforms.uTime.value = sweep * SWEEP_SECONDS
  const t = sweep
  h.material.uniforms.uScanY.value = FIELD_TOP - t * (FIELD_TOP - FIELD_BOTTOM)
  h.renderer.render(h.scene, h.camera)

  const pixels = new Uint8Array(w * ht * 4)
  gl.readPixels(0, 0, w, ht, gl.RGBA, gl.UNSIGNED_BYTE, pixels)

  const out = document.createElement('canvas')
  out.width = w
  out.height = ht
  out.style.cssText =
    'position:fixed;inset:0;width:100%;height:100%;z-index:2;background:#0b0d10'
  const ctx = out.getContext('2d')!
  const image = ctx.createImageData(w, ht)
  // readPixels is bottom-up; ImageData is top-down.
  for (let y = 0; y < ht; y += 1) {
    const src = (ht - 1 - y) * w * 4
    image.data.set(pixels.subarray(src, src + w * 4), y * w * 4)
  }
  ctx.putImageData(image, 0, 0)
  document.querySelectorAll('.frozen').forEach((n) => n.remove())
  out.className = 'frozen'
  document.body.appendChild(out)
}

el<HTMLButtonElement>('run').addEventListener('click', () => {
  document.querySelectorAll('.frozen').forEach((n) => n.remove())
  const count = Number(el<HTMLInputElement>('count').value) || 8000
  void run(count)
})

el<HTMLButtonElement>('freeze').addEventListener('click', () => {
  if (current) freezeFrame(current, 0.5)
})

void run(8000)
