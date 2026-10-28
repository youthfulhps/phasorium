import { useEffect, useMemo, useRef, useState } from 'react'
import OptionWheel from './OptionWheel'
import { useConceptStore } from '../store/conceptStore'
import { useStoryStore } from '../store/storyStore'
import { useSliderStore } from '../store/sliderStore'
import { UI, useLocaleStore } from '../lib/i18n'
import type { ConceptMeta } from '../types/concept'

// Wait for the wheel to settle before swapping scenes, so scrolling past items doesn't load each one
const ACTIVATE_DELAY_MS = 120

function activate(meta: ConceptMeta) {
  useSliderStore.getState().resetForConcept(meta.sliderParams)
  useStoryStore.getState().setStep(0)
  useConceptStore.getState().setActiveConcept(meta.id)
}

/**
 * Concept picker. Collapses Dock-style (scales down toward the left edge) after a pick
 * or when the pointer leaves, and springs back open on hover, scroll, or keyboard focus.
 */
export function ConceptSidebar({ concepts }: { concepts: ConceptMeta[] | null }) {
  const locale = useLocaleStore((s) => s.locale)
  const t = UI[locale]
  const [expanded, setExpanded] = useState(true)
  const timerRef = useRef<ReturnType<typeof setTimeout>>()

  const titles = useMemo(() => concepts?.map((c) => c.title[locale]) ?? [], [concepts, locale])

  useEffect(() => {
    if (concepts?.[0]) activate(concepts[0])
  }, [concepts])

  useEffect(() => () => clearTimeout(timerRef.current), [])

  function handleChange(index: number) {
    clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => {
      const meta = concepts?.[index]
      if (meta) activate(meta)
    }, ACTIVATE_DELAY_MS)
  }

  return (
    <nav
      className={`fixed left-0 top-0 z-10 h-full transition-[width] duration-500 ease-[cubic-bezier(0.34,1.56,0.64,1)] motion-reduce:transition-none ${
        expanded ? 'w-80' : 'w-48'
      }`}
      onMouseEnter={() => setExpanded(true)}
      onMouseLeave={() => setExpanded(false)}
      onWheel={() => setExpanded(true)}
      onFocus={() => setExpanded(true)}
      onBlur={() => setExpanded(false)}
      // Clicking an option picks it; collapse right away like the Dock after a launch
      onClick={(e) => {
        if ((e.target as HTMLElement).closest('[role="option"]')) setExpanded(false)
      }}
    >
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-r from-background via-background/70 to-transparent" />

      <p className="absolute left-6 top-5 z-10 text-[10px] font-medium uppercase tracking-[0.2em] text-muted-foreground/60">
        {t.appName}
      </p>

      {concepts === null || concepts.length === 0 ? (
        <p className="absolute left-6 top-1/2 text-sm text-muted-foreground/60">
          {concepts === null ? t.loading : t.noConcepts}
        </p>
      ) : (
        <div
          className={`absolute inset-0 origin-left transition-transform duration-500 ease-[cubic-bezier(0.34,1.56,0.64,1)] motion-reduce:transition-none ${
            expanded ? 'scale-100' : 'scale-[0.62]'
          }`}
        >
          <OptionWheel
            items={titles}
            defaultSelected={0}
            onChange={handleChange}
            textColor="var(--muted-foreground)"
            activeColor="var(--foreground)"
            fontSize={1.35}
            spacing={1.9}
            tilt={9}
            blur={1}
            fade={0.28}
            inset={24}
          />
        </div>
      )}
    </nav>
  )
}
