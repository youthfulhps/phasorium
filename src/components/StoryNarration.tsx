import { useCallback, useEffect, useRef, type WheelEvent } from 'react'
import { SUBJECT_LABELS, UI, useLocaleStore } from '../lib/i18n'
import { useSliderStore } from '../store/sliderStore'
import { useStoryStore } from '../store/storyStore'
import type { ConceptMeta, StoryStep } from '../types/concept'

const TWEEN_MS = 700
const WHEEL_COOLDOWN_MS = 420
const WHEEL_THRESHOLD = 40 // accumulated deltaY, so trackpad momentum doesn't skip steps
const WORD_STAGGER_MS = 26

// Keys the narration handles only when nothing that owns them is focused
const KEY_OWNERS = 'input, textarea, select, button, a, [contenteditable], [role="slider"], [role="listbox"], [role="option"]'

const smoothstep = (t: number) => t * t * (3 - 2 * t)
const prefersReducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches

/** Eases the sliders named by the step into the values its sentence talks about. */
function useSliderTween(set: StoryStep['set']) {
  useEffect(() => {
    if (!set) return
    const { setSliderValue } = useSliderStore.getState()
    const snap = () => Object.keys(set).forEach((k) => setSliderValue(k, set[k]))

    const from = useSliderStore.getState().values
    const keys = Object.keys(set).filter((k) => from[k] !== set[k])
    if (keys.length === 0) return
    if (prefersReducedMotion()) {
      snap()
      return
    }

    const start = performance.now()
    const startValues = keys.map((k) => from[k] ?? set[k])
    const written: Record<string, number> = {}
    let raf = 0
    let settled = false

    const tick = (now: number) => {
      // If a slider drag wrote over us, stop rather than fight the user for the thumb
      const live = useSliderStore.getState().values
      if (keys.some((k) => k in written && live[k] !== written[k])) {
        settled = true
        return
      }
      const e = smoothstep(Math.min(1, (now - start) / TWEEN_MS))
      keys.forEach((k, i) => {
        const v = e === 1 ? set[k] : startValues[i] + (set[k] - startValues[i]) * e
        written[k] = v
        setSliderValue(k, v) // re-renders the HUD thumbs; that is the point of the tween
      })
      if (e < 1) raf = requestAnimationFrame(tick)
      else settled = true
    }
    raf = requestAnimationFrame(tick)

    // Scrolling on before the tween ends would strand the sliders between two
    // sentences' numbers, so jump them the rest of the way.
    return () => {
      cancelAnimationFrame(raf)
      if (!settled) snap()
    }
  }, [set])
}

// `…` in a step's sentence marks a formula. The capture group makes String.split keep the
// formulas, so the odd entries of the result are exactly the marked ones.
const FORMULA = /`([^`]+)`/
const stripMarks = (text: string) => text.replace(/`/g, '')
// Beats the formula's sweep waits after the words before it, so it lands as its own moment
const FORMULA_BEATS = 3
// A formula is lifted out of the middle of its sentence, so the words after it would otherwise
// open with the punctuation that used to follow it — a stray comma or full stop starting a line.
// The formula's own line is the break, so that punctuation is dropped here. The sr-only copy of
// the sentence keeps it, since a screen reader is read the sentence whole.
const ORPHAN_PUNCT = /^[\s.,;:·]+/

/**
 * The sentence, one word at a time, each fading up out of a blur (see .story-word), with any
 * marked formula lifted onto its own line under a marker-pen band (see .story-formula).
 */
function RevealText({ text }: { text: string }) {
  let beat = 0
  return (
    <div
      aria-hidden
      className="text-[21px] font-medium leading-[1.75] tracking-tight text-foreground"
    >
      {text.split(FORMULA).map((part, i) => {
        const run = i > 0 ? part.replace(ORPHAN_PUNCT, '') : part
        const words = i % 2 === 0 ? run.trim().split(/\s+/).filter(Boolean) : []
        // Two formulas in a row leave an empty run of words between them
        if (i % 2 === 0 && words.length === 0) return null
        const delay = `${beat * WORD_STAGGER_MS}ms`
        beat += i % 2 === 0 ? words.length : FORMULA_BEATS
        return i % 2 === 0 ? (
          <p key={i}>
            {words.map((word, w) => (
              <span key={w}>
                <span
                  className="story-word"
                  style={{ animationDelay: `${(beat - words.length + w) * WORD_STAGGER_MS}ms` }}
                >
                  {word}
                </span>{' '}
              </span>
            ))}
          </p>
        ) : (
          <p key={i} className="my-3">
            <span className="story-formula" style={{ animationDelay: delay }}>
              {part}
            </span>
          </p>
        )
      })}
    </div>
  )
}

/**
 * Narration floating over the right of the scene: one sentence at a time, revealed
 * word by word, advanced by scrolling. Replaces ConceptCard for concepts with a `story`.
 */
export function StoryNarration({ meta }: { meta: ConceptMeta }) {
  const story = meta.story!
  const locale = useLocaleStore((s) => s.locale)
  const t = UI[locale]
  const last = story.length - 1
  // Clamped: a concept switch resets the step (ConceptSidebar.activate), but a shorter
  // story must not index past its end on the render before that lands.
  const step = Math.min(useStoryStore((s) => s.step), last)

  const go = useCallback(
    (next: number) => useStoryStore.getState().setStep(Math.max(0, Math.min(last, next))),
    [last]
  )

  useSliderTween(story[step]?.set)

  const wheelAcc = useRef(0)
  const wheelLockUntil = useRef(0)
  const onWheel = (e: WheelEvent<HTMLDivElement>) => {
    const now = performance.now()
    if (now < wheelLockUntil.current) {
      wheelAcc.current = 0 // swallow the momentum tail of the gesture we just acted on
      return
    }
    wheelAcc.current += e.deltaY
    if (Math.abs(wheelAcc.current) < WHEEL_THRESHOLD) return
    go(step + Math.sign(wheelAcc.current))
    wheelAcc.current = 0
    wheelLockUntil.current = now + WHEEL_COOLDOWN_MS
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Leave the keys to whatever owns them: the concept wheel (which preventDefaults
      // its own arrows), buttons, sliders. Space stays a button-activation key.
      if (e.defaultPrevented || e.repeat || e.metaKey || e.ctrlKey || e.altKey) return
      if ((e.target as HTMLElement)?.closest(KEY_OWNERS)) return
      if (e.key === 'ArrowDown' || e.key === 'PageDown') go(step + 1)
      else if (e.key === 'ArrowUp' || e.key === 'PageUp') go(step - 1)
      else return
      e.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [go, step])

  return (
    <section
      aria-label={meta.title[locale]}
      className="absolute right-10 top-1/2 z-10 w-[27rem] max-w-[calc(100vw-4rem)] -translate-y-1/2 py-6 pl-8"
      onWheel={onWheel}
    >
      <p className="text-[10px] font-medium uppercase tracking-[0.2em] text-muted-foreground/70">
        {SUBJECT_LABELS[meta.subject][locale]}
      </p>
      <h2 className="mb-6 text-[17px] font-semibold tracking-tight text-foreground/90">
        {meta.title[locale]}
      </h2>

      {/* The animated words are split into spans, which screen readers read one at a
          time, so the sentence is announced once from here instead. */}
      <p aria-live="polite" className="sr-only">
        {stripMarks(story[step].text[locale])}
      </p>

      {/* Keyed by step so every sentence replays the word stagger from the start */}
      <div className="min-h-[13rem]">
        <RevealText key={`${step}:${locale}`} text={story[step].text[locale]} />
      </div>

      <div className="mt-4 flex items-center gap-3">
        <div className="flex flex-1 items-center gap-1.5">
          {story.map((_, i) => (
            <button
              key={i}
              onClick={() => go(i)}
              aria-label={`${t.storyStepLabel} ${i + 1}`}
              aria-current={i === step ? 'step' : undefined}
              className="group flex h-6 flex-1 items-center rounded-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
            >
              <span
                className={`h-0.5 w-full rounded-full transition-colors ${
                  i <= step ? 'bg-foreground/50' : 'bg-foreground/15'
                } group-hover:bg-foreground/70`}
              />
            </button>
          ))}
        </div>
        <span className="shrink-0 text-[11px] tabular-nums text-muted-foreground">
          {step + 1} / {story.length}
        </span>
      </div>

      <div className="mt-3 flex items-center justify-between gap-2">
        <p className="text-[11px] leading-tight text-muted-foreground">{t.storyHint}</p>
        <div className="flex shrink-0 gap-1.5">
          <NavButton label={t.storyBack} disabled={step === 0} onClick={() => go(step - 1)}>
            ↑
          </NavButton>
          {step === last ? (
            <NavButton label={t.storyRestart} onClick={() => go(0)}>
              ↺
            </NavButton>
          ) : (
            <NavButton label={t.storyForward} onClick={() => go(step + 1)}>
              ↓
            </NavButton>
          )}
        </div>
      </div>
    </section>
  )
}

function NavButton({
  label,
  disabled,
  onClick,
  children,
}: {
  label: string
  disabled?: boolean
  onClick: () => void
  children: string
}) {
  return (
    <button
      type="button"
      title={label}
      aria-label={label}
      // aria-disabled, not disabled: a disabled button loses focus mid-keyboard-navigation
      aria-disabled={disabled}
      onClick={() => !disabled && onClick()}
      className={`flex h-7 w-7 items-center justify-center rounded-md text-[12px] transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring ${
        disabled ? 'text-muted-foreground/30' : 'text-muted-foreground hover:bg-muted hover:text-foreground'
      }`}
    >
      {children}
    </button>
  )
}
