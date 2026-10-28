import { create } from 'zustand'

interface ConceptStore {
  activeConceptId: string | null
  setActiveConcept: (id: string) => void
}

export const useConceptStore = create<ConceptStore>()((set) => ({
  activeConceptId: null,
  setActiveConcept: (id) => set({ activeConceptId: id }),
}))
