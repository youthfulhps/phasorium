import type { FallbackProps } from 'react-error-boundary'
import { UI, useLocaleStore } from '../lib/i18n'

export function SceneErrorFallback({ error }: FallbackProps) {
  const t = UI[useLocaleStore((s) => s.locale)]
  return (
    <div className="flex flex-col items-center justify-center h-full gap-3 p-6">
      <h3 className="text-lg font-semibold tracking-tight text-foreground/70">
        {t.sceneError}
      </h3>
      <pre className="text-sm font-mono text-red-400/70 mt-2">
        {error.message}
      </pre>
    </div>
  )
}
