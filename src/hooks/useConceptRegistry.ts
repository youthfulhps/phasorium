import { useState, useEffect } from 'react'
import type { ConceptMeta, SubjectType } from '../types/concept'

const SUBJECT_ORDER: SubjectType[] = [
  'electromagnetics',
  'circuits',
  'electronics',
  'power-systems',
  'machines',
  'control',
  'installations',
]

const modules = import.meta.glob<{ default: ConceptMeta }>(
  '../concepts/*/index.tsx',
  { eager: false }
)

export function useConceptRegistry() {
  const [concepts, setConcepts] = useState<ConceptMeta[] | null>(null)

  useEffect(() => {
    Promise.all(Object.values(modules).map((l) => l()))
      .then((loaded) => {
        const metas = loaded.map((m) => m.default)
        metas.sort((a, b) => {
          const si =
            SUBJECT_ORDER.indexOf(a.subject) - SUBJECT_ORDER.indexOf(b.subject)
          if (si !== 0) return si
          return (
            (a.order ?? Infinity) - (b.order ?? Infinity) ||
            a.id.localeCompare(b.id)
          )
        })
        setConcepts(metas)
      })
      .catch((err) => {
        console.error('Failed to load concepts:', err)
        setConcepts([])
      })
  }, [])

  return concepts
}
