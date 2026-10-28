import { Canvas } from '@react-three/fiber'
import { Stats } from '@react-three/drei'
import { Suspense, lazy, useState } from 'react'
import { ErrorBoundary } from 'react-error-boundary'
import { EmptySceneFallback } from './EmptySceneFallback'
import { SceneErrorFallback } from './SceneErrorFallback'
import { UI, useLocaleStore } from '../lib/i18n'
import { ConceptCard, ConceptCard2D } from './ConceptCard'
import { StoryNarration } from './StoryNarration'
import { SceneStage } from './scene/SceneStage'
import { Fit2DCamera } from './scene/Fit2DCamera'
import type { ConceptMeta } from '../types/concept'

// Width the narration takes on the right (w-27rem + right-10), for 2D framing
const STORY_RESERVED_PX = 480

const sceneCache = new Map<string, ReturnType<typeof lazy>>()

if (import.meta.hot) {
  import.meta.hot.accept(
    Object.keys(import.meta.glob('../concepts/*/index.tsx')),
    () => window.location.reload()
  )
}

function getLazyScene(conceptId: string, is3D: boolean) {
  const key = `${conceptId}:${is3D ? '3d' : '2d'}`
  if (!sceneCache.has(key)) {
    sceneCache.set(
      key,
      lazy(() =>
        import(`../concepts/${conceptId}/index.tsx`).then((m) => ({
          default: is3D ? m.default.Scene : (m.default.Scene2D ?? m.default.Scene),
        }))
      )
    )
  }
  return sceneCache.get(key)!
}

function SceneContent({ meta }: { meta: ConceptMeta }) {
  const conceptId = meta.id
  const [is3D, setIs3D] = useState(true)
  const t = UI[useLocaleStore((s) => s.locale)]
  const LazyScene = getLazyScene(conceptId, is3D)

  return (
    <div className="relative h-full">
      <ErrorBoundary key={`${conceptId}:${is3D}`} FallbackComponent={SceneErrorFallback}>
        <Suspense fallback={<div className="p-6 text-muted-foreground">{t.sceneLoading}</div>}>
          <Canvas flat camera={{ position: [0, 0, 5], fov: 50 }} style={{ height: '100%' }}>
            {import.meta.env.DEV && <Stats />}
            <SceneStage is3D={is3D} />
            <LazyScene />
            {is3D ? !meta.story && <ConceptCard meta={meta} /> : <Fit2DCamera rightReserved={meta.story ? STORY_RESERVED_PX : undefined} />}
          </Canvas>
        </Suspense>
      </ErrorBoundary>

      {meta.story ? <StoryNarration meta={meta} /> : !is3D && <ConceptCard2D meta={meta} />}

      <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 flex overflow-hidden rounded-md border border-border text-xs font-mono">
        <button
          onClick={() => setIs3D(false)}
          className={`px-3 py-1 transition-colors ${!is3D ? 'bg-primary text-primary-foreground' : 'bg-background text-foreground hover:bg-muted'}`}
        >
          2D
        </button>
        <button
          onClick={() => setIs3D(true)}
          className={`px-3 py-1 transition-colors ${is3D ? 'bg-primary text-primary-foreground' : 'bg-background text-foreground hover:bg-muted'}`}
        >
          3D
        </button>
      </div>
    </div>
  )
}

export function SceneRenderer({ meta }: { meta: ConceptMeta | null }) {
  if (!meta) return <EmptySceneFallback />

  return <SceneContent key={meta.id} meta={meta} />
}
