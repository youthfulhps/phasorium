# DESIGN.md

How concepts look. Read this before building or changing a scene or UI.

## Principle

One scene, one idea. Every visual element should map to a physical quantity or a guide the learner needs; if it doesn't explain something, leave it out. Every slider must produce a change the learner can see.

## Layout

- Scene fills the screen; panels float over it.
- Left: `ConceptSidebar`, a curved wheel of concept titles (`OptionWheel` from React Bits, installed with `pnpm dlx shadcn@latest add https://reactbits.dev/r/OptionWheel-TS-TW` — don't edit it; adjust through props or the wrapper, and re-run the command to update). It shrinks Dock-style after a pick or on pointer leave and springs open on hover, scroll, or focus.
- In the scene: `ConceptCard`, the subject + description floating in 3D (drei `<Html transform sprite>`, always faces the camera). Default spot `CARD_POSITION_3D` = `[2.6, 0.6, -1.5]`, to the camera's right. Keep that area clear, or set `cardPosition` on the concept. The card stops wheel/pointer events so it scrolls instead of moving the camera.
- In 2D: `ConceptCard2D`, the same card always open, docked right and vertically centered. `Fit2DCamera` moves the camera so the plot fills the space left between the sidebar, card, top bar, and HUD.
- For a concept with a `story`, `StoryNarration` takes the card's place in both 2D and 3D: no surface, no border, just the sentence set large over the scene on the right, one at a time, each word fading up out of a blur (`.story-word` in `index.css`, the React Bits ScrollReveal effect driven by the step rather than page scroll). It reserves about 480px, so keep the scene clear of that band — a camera and orbit target sharing an x offset slides a 3D scene left without distorting it.
- Bottom center: slider HUD (`SliderControls`), fixed on screen so sliders never move while you use them.
- Top center: 2D/3D toggle. Top right: locale/theme toggles.
- Floating surfaces use React Bits `GlassSurface` (installed via `pnpm dlx shadcn@latest add https://reactbits.dev/r/GlassSurface-TS-TW`; don't edit it). Liquid refraction is Chromium-only; other browsers get its plain blur fallback. It follows the OS color scheme, not the in-app theme toggle. Used by the slider HUD and the floating 3D card.
- Spring easing for panel motion: `cubic-bezier(0.34, 1.56, 0.64, 1)`, 500ms; always pair with `motion-reduce:transition-none`.

## Typography

`font-sans` = Geist (Latin) → Pretendard (Hangul) fallback, set in `--font-sans` in `src/index.css`. Don't set `font-family` anywhere else.

## Color = physical quantity (restrained)

Modern and quiet: neutrals carry structure and most quantities (told apart by lightness), and only two accents exist — **cool for current** (drawn as a liquid line) and **warm for power/heat**. Name quantities with labels in the scene, not by hue; descriptions refer to labels ("VL"), not color names, except the two accents.

| Token | Means | Tone |
|---|---|---|
| `voltage` | voltage, EMF, source waveform | strongest neutral |
| `current` | current | **cool accent** (cyan) |
| `resistance` | resistor / resistive component (e.g. VR) | mid neutral |
| `inductance` | inductor / inductive component (e.g. VL) | light neutral |
| `capacitance` | capacitor / capacitive component (e.g. VC) | dark neutral |
| `power` | power, energy, heat | **warm accent** (amber) |
| `e-field` | electric field | neutral |
| `b-field` | magnetic field / flux | cool accent, deeper |
| `guide` | axes, reference circles, grids, helper lines | faint neutral |
| `backdrop` | scene background (= page background as hex) | white / near-black |
| `marker` | "current position" dots and their projections | foreground |

- Label text in scenes uses the foreground color; only the two accents (current, power) tint their labels. Dark neutrals are fine for strokes but unreadable as text.
- Don't introduce new hues in a scene. A new quantity gets a neutral token; a third accent needs a design decision first.
- One hue lives outside this palette: `--story-mark`, the highlighter behind a narration formula. It is UI chrome, not a quantity, and is deliberately greener and lighter than `--power`'s amber so a highlighted formula never reads as a heat cue. Never use it in a scene.
- Values live in `src/index.css` (`:root` for light, `.dark` for dark), as **hex only** — Three.js cannot parse `oklch()`.
- In scenes: `const c = usePhysicsColors()` then `color={c.voltage}`. It re-reads on theme change.
- In UI: Tailwind classes such as `text-voltage`, `bg-current`.
- Liquid lines shimmer ±15° around their hue (invisible on neutrals, subtle on the accents).
- Need a quantity that isn't listed? Add a token to both themes in `index.css`, the `@theme` mapping, `PHYSICS_COLORS` in `usePhysicsColors.ts`, and this table.

## Scene conventions

- **Stage is shared.** `SceneStage` (rendered by `SceneRenderer`) sets the background to `backdrop`, adds lights, a fading floor grid at y = −2.2 in 3D, and bloom in dark mode only. Concepts must not add their own background, lights, grid, or post-processing.
- **Use the shared primitives** in `src/components/scene/`:
  - `TrailLine` for anything that has a history (time traces, helices, waveforms) and for current flowing in a loop (`fade={false}`, `flowSpeed` ∝ I): a glossy "liquid" line — tube-like shading, a sheen flowing tail → head, a subtle hue shimmer of about ±15° around the token color, tail (index 0) fading out, head (last index) full color. The liquid effect is shading only; never distort a curve's shape for looks, the shape is the physics.
  - `Marker` for "current position" dots (solid core + soft halo).
  - Plain drei `Line` for phasor arrows and guides; guides are thin (`lineWidth` ≈ 1) and may be `dashed` (call `computeLineDistances()` after `setPositions`).
- **Resistors** are `scene/Resistor.tsx`: a faint glass tube (thinner for larger R) whose amber filament thickens with power (`setResistorHeat`). Never tint a neutral toward amber to show heat — the midpoint reads as brown.
- **Potential landscape** (circuits drawn with height = potential, as in Ohm's law and series RLC): use `scene/PotentialLandscape.tsx` — `useRiseIn()` (multiply heights by it so the flat circuit rises into the landscape on open), `PotentialRuler` (0 V on the floor, voltage ticks), and `FloorCaption` (says the dashed outline is the ordinary circuit at 0 V). Current is a `TrailLine` with `fade={false}`.
- Canvas renders `flat` (no tone mapping) so token colors appear exactly as defined.
- **Units**: 1 scene unit ≈ one display amplitude. Keep content within about ±2 on X/Y so it fits the default camera.
- **Axes**: X = real / horizontal, Y = imaginary / value, Z = time depth (history recedes toward −Z).
- **Phasors**: rotate counter-clockwise, current as the 0° reference unless the concept says otherwise.
- **3D scene**: `PerspectiveCamera` at about `[3, 2, 5]`, `OrbitControls` with `enablePan={false}`. Keep content above the floor grid (y > −2.2).
- **2D scene** (optional `Scene2D`): the flattened time-domain or front view of the same idea, no camera or camera controls. Keep content within x ∈ [−4, 4], y ∈ [−2, 2]; `Fit2DCamera` frames exactly that.
- **Line weights**: main quantity 2.5–3, secondary 2, guides 1–1.5. `Marker` size 0.05–0.08.
- **Motion**: animate with `clock.getElapsedTime()`; slow real frequencies down to a readable speed (≈ 0.5–2 cycles per second on screen) and say so in the description if it matters.
- **Scale honestly**: when normalizing amplitudes for display, keep ratios between quantities true.

## Narration

- One idea per sentence, in the voice you'd use explaining it to a bright high-schooler: plain words, concrete numbers, no formula dumps. The formula arrives last, as the summary of what the learner has already seen.
- Every sentence must be matched by a visible change. If a step changes nothing on screen, merge it into its neighbour.
- Quote a number in a sentence only if the step's `set` pins every slider that number depends on; otherwise an earlier drag makes the text lie.
- Don't claim something the camera hides. A sentence that says two things are level has to be checked in a screenshot: perspective puts points at the same potential at different screen heights, so point at the readouts or the guide line instead.
- A drei `<Html>` without `transform` is a DOM overlay, not scene geometry, so nothing you do to its parent group reaches it: neither `visible = false` nor `scale.setScalar(0)` hides the span. Give the span its own ref and `setOpacity` it. The same group's meshes and lines do respond, which is exactly why the bug hides — the label is the one thing left on screen.
- Watch for clamped tails. A curve flattened by a display clamp is a horizontal line, and perspective tilts a horizontal line, so "this one keeps rising" reads as false at the end of the plot. Pick a scale that doesn't need the clamp — `resonance` draws XL and XC in decades against its logarithmic frequency axis, which turns the pair into a clean X with nothing cut off. Where the scale has to stay fixed, trim the domain instead of the value: `capacitance-dielectric` starts its 1 ÷ d curve at whatever gap first fits on the plot, so the curve runs off the top edge where it really does rather than flattening against it.
- The narration floats over the right of the frame, so a story scene has to live in the left two thirds. Check it with a screenshot at the widest slider values, and if a readout runs under the panel, aim `OrbitControls` right of centre (`target={[1, 0, 0]}`) or back the camera off rather than moving labels one at a time.

## Descriptions

- Markdown: one `##` heading, a one-sentence core idea in **bold**, a color legend list, the formula(s), and a hint about what to try with the sliders.
- Keep formulas and symbols identical in `en` and `ko`.

## Sliders

- Label with the name and symbol (`Amplitude (Vm)`), SI unit in `unit`.
- Defaults should show the interesting behavior immediately (e.g. near but not at resonance).
- Ranges should cover realistic values and the interesting edge (resonance, saturation, etc.).
