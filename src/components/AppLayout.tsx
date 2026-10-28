import { ConceptSidebar } from './ConceptSidebar'
import { SceneRenderer } from './SceneRenderer'
import { SliderControls } from './SliderControls'
import { LocaleToggle } from './LocaleToggle'
import { ThemeToggle } from './ThemeToggle'
import { useConceptRegistry } from '../hooks/useConceptRegistry'
import { useConceptStore } from '../store/conceptStore'

export function AppLayout() {
  const concepts = useConceptRegistry()
  const activeId = useConceptStore((s) => s.activeConceptId)
  const activeMeta = concepts?.find((c) => c.id === activeId) ?? null

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-background">
      <div className="absolute inset-0">
        <SceneRenderer meta={activeMeta} />
      </div>
      <ConceptSidebar concepts={concepts} />
      <div className="fixed right-3 top-3 z-10 flex items-center">
        <LocaleToggle />
        <ThemeToggle />
      </div>
      {activeMeta && <SliderControls meta={activeMeta} />}
    </div>
  )
}
