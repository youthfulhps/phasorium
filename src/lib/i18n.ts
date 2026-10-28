import { create } from 'zustand'
import type { SubjectType } from '../types/concept'

export const LOCALES = ['en', 'ko'] as const
export type Locale = (typeof LOCALES)[number]

/** Text that must exist in every supported locale. */
export type Localized = Record<Locale, string>

export const UI = {
  en: {
    appName: 'Phasorium',
    tagline: 'Interactive 3D visualizations that make electricity intuitive',
    loading: 'Loading…',
    noConcepts: 'No concepts yet.',
    selectConcept: 'Select a concept',
    selectConceptHint: 'Scroll or click to start the visualization',
    sceneLoading: 'Loading scene…',
    sceneError: 'Failed to load scene',
    toggleTheme: 'Toggle theme',
    toggleLocale: '한국어로 보기',
    floorCaption: 'Floor: the circuit seen from above · 0 V',
    storyHint: 'Scroll here, or press ↓, for the next line',
    storyBack: 'Previous line',
    storyForward: 'Next line',
    storyStepLabel: 'Line',
    storyRestart: 'Start over',
  },
  ko: {
    appName: 'Phasorium',
    tagline: '전기 개념을 눈으로 이해하는 인터랙티브 3D 시각화',
    loading: '로딩 중…',
    noConcepts: '개념이 없습니다.',
    selectConcept: '개념을 선택하세요',
    selectConceptHint: '스크롤하거나 클릭하면 시각화가 시작됩니다',
    sceneLoading: '씬 로드 중…',
    sceneError: '씬 로드 실패',
    toggleTheme: '테마 전환',
    toggleLocale: 'View in English',
    floorCaption: '바닥: 위에서 본 회로도 · 0 V',
    storyHint: '여기서 스크롤하거나 ↓ 키를 누르면 다음 문장',
    storyBack: '이전 문장',
    storyForward: '다음 문장',
    storyStepLabel: '문장',
    storyRestart: '처음부터',
  },
} satisfies Record<Locale, Record<string, string>>

export const SUBJECT_LABELS: Record<SubjectType, Localized> = {
  electromagnetics: { en: 'Electromagnetics', ko: '전자기학' },
  circuits: { en: 'Circuits', ko: '회로이론' },
  electronics: { en: 'Electronics', ko: '전자회로' },
  'power-systems': { en: 'Power Systems', ko: '전력공학' },
  machines: { en: 'Electric Machines', ko: '전기기기' },
  control: { en: 'Control Systems', ko: '제어공학' },
  installations: { en: 'Electrical Installations', ko: '전기설비' },
}

function getInitialLocale(): Locale {
  const stored = localStorage.getItem('locale')
  if (stored === 'en' || stored === 'ko') return stored
  return navigator.language.startsWith('ko') ? 'ko' : 'en'
}

interface LocaleStore {
  locale: Locale
  toggleLocale: () => void
}

export const useLocaleStore = create<LocaleStore>()((set) => ({
  locale: getInitialLocale(),
  toggleLocale: () => set((s) => ({ locale: s.locale === 'en' ? 'ko' : 'en' })),
}))

function applyLocale(locale: Locale) {
  localStorage.setItem('locale', locale)
  document.documentElement.lang = locale
  document.title = `${UI[locale].appName} — ${UI[locale].tagline}`
  document.querySelector('meta[name="description"]')?.setAttribute('content', UI[locale].tagline)
}

applyLocale(useLocaleStore.getState().locale)
useLocaleStore.subscribe((s) => applyLocale(s.locale))
