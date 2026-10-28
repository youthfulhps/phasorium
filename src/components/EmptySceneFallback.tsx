import { UI, useLocaleStore } from '../lib/i18n'

export function EmptySceneFallback() {
  const t = UI[useLocaleStore((s) => s.locale)]
  return (
    <div className="flex flex-col items-center justify-center h-full gap-3">
      <p className="text-xl font-semibold tracking-tight text-foreground/40">
        {t.selectConcept}
      </p>
      <p className="text-sm text-muted-foreground/60">
        {t.selectConceptHint}
      </p>
    </div>
  )
}
