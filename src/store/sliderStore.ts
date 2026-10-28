import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'
import type { SliderParam } from '../types/concept'

interface SliderStore {
  values: Record<string, number>
  setSliderValue: (key: string, value: number) => void
  resetForConcept: (params: SliderParam[]) => void
}

export const useSliderStore = create<SliderStore>()(
  subscribeWithSelector((set) => ({
    values: {},
    setSliderValue: (key, value) =>
      set((s) => ({ values: { ...s.values, [key]: value } })),
    resetForConcept: (params) =>
      set({ values: Object.fromEntries(params.map((p) => [p.key, p.defaultValue])) }),
  }))
)
