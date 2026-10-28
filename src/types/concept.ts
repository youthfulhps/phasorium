import type { ComponentType } from 'react'
import type { Localized } from '../lib/i18n'

/** Field of electrical engineering. Labels live in SUBJECT_LABELS (src/lib/i18n.ts). */
export type SubjectType =
  | 'electromagnetics'
  | 'circuits'
  | 'electronics'
  | 'power-systems'
  | 'machines'
  | 'control'
  | 'installations'

export interface SliderParam {
  key: string
  label: Localized
  min: number
  max: number
  step: number
  defaultValue: number
  unit?: string
}

/**
 * One line of narration plus the scene state it describes.
 * The learner scrolls through the steps; `set` tweens those sliders so the
 * visualization moves into the state the sentence is talking about.
 */
export interface StoryStep {
  text: Localized
  set?: Record<string, number>
}

export interface ConceptMeta {
  id: string
  title: Localized
  subject: SubjectType
  order?: number
  description: Localized
  sliderParams: SliderParam[]
  /** Scroll-through narration. When present it replaces the description card. */
  story?: StoryStep[]
  /** Where the floating description card sits in the 3D scene. Defaults to CARD_POSITION_3D. */
  cardPosition?: [number, number, number]
  Scene: ComponentType
  Scene2D?: ComponentType
}
