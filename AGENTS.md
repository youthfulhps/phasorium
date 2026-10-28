# AGENTS.md

**Phasorium** — interactive 3D visualizations that make electrical engineering intuitive, in English and Korean.
Each concept is a Three.js scene plus sliders: the learner moves a parameter and watches the physics respond.

Scope is electrical engineering broadly (electromagnetics, circuits, electronics, power systems, machines, control, installations). Exam material such as the Korean 전기기사 license is welcome, but concepts must stand on their own for any learner.

## Stack

React 18 · TypeScript · Vite · @react-three/fiber + drei · zustand · Tailwind v4 (CSS-first, no `tailwind.config`) · shadcn/ui · pnpm

## Commands

- `pnpm dev` — dev server
- `pnpm build` — runs `scripts/validate-concepts.ts`, then `tsc`, then `vite build`

**Done means `pnpm build` passes.** For scene changes, also look at it in the running app when a browser is available.

## Layout

```
src/concepts/{id}/index.tsx   one concept = one file (meta + Scene + optional Scene2D), auto-discovered
src/types/concept.ts          ConceptMeta, SliderParam, SubjectType
src/lib/i18n.ts               Locale, Localized, UI strings, SUBJECT_LABELS, useLocaleStore
src/hooks/usePhysicsColors.ts theme-aware colors for physical quantities
src/hooks/useConceptRegistry.ts  import.meta.glob over concepts, sorted by subject then order
src/store/sliderStore.ts      slider values, read by scenes
src/components/scene/         shared scene primitives: SceneStage, TrailLine, Marker
src/store/storyStore.ts       narration step, read by scenes through useStoryStepRef()
src/components/scene/storyReveal.ts  useStoryReveal(): eased 0 → 1 weight per revealed element
src/components/               app shell (ConceptSidebar wheel, SceneRenderer + floating ConceptCard, SliderControls HUD)
scripts/validate-concepts.ts  prebuild checks on concept files
```

Adding a concept needs no registration: create the folder and file, and it appears in the sidebar wheel.
Use the `add-concept` skill for that workflow.

## Rules

- **Narration over description.** A concept may define `story: StoryStep[]` — one sentence per step, `{ en, ko }`, revealed word by word as the learner scrolls. The step is the scene's script: a scene declares which elements each step wants, gets eased 0 → 1 weights back from `useStoryReveal()`, and fades and moves things by those weights in `useFrame`. A step's optional `set` tweens sliders into the state its sentence describes; a step that quotes a number must `set` every slider that number depends on, because the progress bar lets the learner jump straight to it. Wrap a formula inside a sentence in `backticks` and it is lifted onto its own line under a marker-pen band (`.story-formula`); `en` and `ko` must mark the same number of them, which the prebuild checks. A weight may feed the physics and not just the visuals: `rlc-series` scales XL and XC by theirs, so a step that has not introduced the coil yet solves the circuit that is actually on screen. A story replaces the description card, so the last step should carry the formula. `description` stays required either way. A readout that quotes a number the step changes gets it from `useSettledLabels()`, which hands back the step's settled target weights rather than the eased live ones, so the number lands with its own sentence. See `src/concepts/coulomb-field/index.tsx` for the plain case, `rlc-series` and `kirchhoff-laws` for a weight that feeds the physics, `rc-rl-transient` for a shared row component that fades as one unit, `transformer` for a weight that opens a circuit (no load, so no secondary current), `phasor-impedance` for two weights that take the coil and the capacitor out of the impedance, `rlc-parallel` for weights that simply leave a branch unconnected, `resonance` for a story that reveals one layer of a sweep at a time without touching the physics, and `magnetic-circuit` for a story that simply assembles the scene one part at a time, in the order the circuit is reasoned about. `grep -l 'story:' src/concepts/*/index.tsx` lists them all.
- **Bilingual everywhere.** Concept text is `{ en, ko }` inside the concept file; app UI text goes in `UI` in `src/lib/i18n.ts`. Never hard-code a user-visible string in one language. Write English first, then natural Korean (not word-for-word).
- **Physics must be correct.** Verify formulas, units, and sign/phase conventions against a reliable source and a worked numeric example before writing scene code.
- **Colors come from `usePhysicsColors()`**, never hex literals in concept files (prebuild rejects them). Read `DESIGN.md` before building a scene or UI.
- **Per-frame work stays out of React state.** Scenes read sliders through a ref subscribed to `useSliderStore` and mutate buffers in `useFrame`.
- A new subject needs updates in 4 places: `SubjectType`, `SUBJECT_LABELS`, `SUBJECT_ORDER` (`useConceptRegistry.ts`), `VALID_SUBJECTS` (`validate-concepts.ts`).
- If a rule matters, enforce it in `scripts/validate-concepts.ts` rather than only documenting it.
- No i18n library for now (hand-rolled is enough at 2 locales and a handful of UI strings).

## Git

The repo is initialized locally only. Do not commit or push unless the user asks.
