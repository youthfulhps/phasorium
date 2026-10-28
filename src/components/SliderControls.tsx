import { Slider } from '@/components/ui/slider'
import { useSliderStore } from '../store/sliderStore'
import type { ConceptMeta, SliderParam } from '../types/concept'
import { useLocaleStore } from '../lib/i18n'
import GlassSurface from './GlassSurface'

function SingleSlider({ param }: { param: SliderParam }) {
  const value = useSliderStore((s) => s.values[param.key] ?? param.defaultValue)
  const locale = useLocaleStore((s) => s.locale)

  return (
    <div className="flex w-44 flex-col gap-1.5">
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-[11px] text-muted-foreground">{param.label[locale]}</span>
        <span className="text-[11px] font-semibold tabular-nums text-foreground">
          {value.toFixed(2)}
          {param.unit && <span className="ml-0.5 font-normal text-muted-foreground">{param.unit}</span>}
        </span>
      </div>
      <Slider
        min={param.min}
        max={param.max}
        step={param.step}
        value={[value]}
        onValueChange={(vals) => useSliderStore.getState().setSliderValue(param.key, vals[0])}
      />
    </div>
  )
}

/** Bottom HUD: sliders stay put on screen so they can be tweaked while watching the scene. */
export function SliderControls({ meta }: { meta: ConceptMeta }) {
  return (
    <div className="fixed bottom-[calc(1.25rem+env(safe-area-inset-bottom,0px))] left-1/2 z-10 max-w-[calc(100vw-2rem)] -translate-x-1/2">
      <GlassSurface width="auto" height="auto" borderRadius={16}>
        <div className="flex flex-wrap justify-center gap-x-8 gap-y-4 px-4 py-2">
          {meta.sliderParams.map((p) => (
            <SingleSlider key={p.key} param={p} />
          ))}
        </div>
      </GlassSurface>
    </div>
  )
}
