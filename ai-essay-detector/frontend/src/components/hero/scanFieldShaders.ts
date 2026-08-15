/**
 * The scan-field shaders and their tuning constants.
 *
 * Split out from ScanFieldScene so the perf harness (`bench.html`) measures the
 * shader that actually ships rather than a copy of it that can drift. If the
 * benchmark and the hero disagree, the benchmark is worthless.
 */

/** Ink and amber, matching the CSS tokens exactly. */
/**
 * Unscanned points. Bright enough that the page of text reads as a shape on
 * #0B0D10 without the scan band — the band is meant to reveal structure that
 * is already there, not to be the only thing on screen.
 */
export const COLOR_INK = '#5D6B7F'
export const COLOR_AMBER = '#B8720B'
export const COLOR_AMBER_HOT = '#F0A93C'

export const SWEEP_SECONDS = 7.5
/** Half-height of the bright band, in world units. */
export const BAND = 0.26
/**
 * Base point size before distance attenuation.
 *
 * Raised from 13 after looking at a frozen frame: at 13 the unscanned page was
 * too sparse to read as text on #0B0D10, so the scan band was the only thing on
 * screen. Bigger points cost fill rate, which is the one thing that scales
 * badly here — re-check /bench.html after changing this.
 */
export const POINT_SIZE = 19.0

export const vertexShader = /* glsl */ `
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

export const fragmentShader = /* glsl */ `
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

    float alpha = (0.62 + vIntensity * 0.38) * edge;
    gl_FragColor = vec4(color, alpha);
  }
`
