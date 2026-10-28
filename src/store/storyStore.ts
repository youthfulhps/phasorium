import { useEffect, useRef } from 'react'
import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'

interface StoryStore {
  /** Index into the active concept's `story`. */
  step: number
  setStep: (step: number) => void
}

export const useStoryStore = create<StoryStore>()(
  subscribeWithSelector((set) => ({
    step: 0,
    setStep: (step) => set({ step }),
  }))
)

/** The current narration step in a ref, for scenes that read it inside useFrame. */
export function useStoryStepRef() {
  const stepRef = useRef(useStoryStore.getState().step)
  useEffect(
    () => useStoryStore.subscribe((s) => s.step, (v) => { stepRef.current = v }),
    []
  )
  return stepRef
}
