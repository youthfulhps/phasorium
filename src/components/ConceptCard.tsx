import type { SyntheticEvent } from 'react'
import { Html } from '@react-three/drei'
import ReactMarkdown from 'react-markdown'
import { SUBJECT_LABELS, useLocaleStore } from '../lib/i18n'
import type { ConceptMeta } from '../types/concept'
import GlassSurface from './GlassSurface'

// To the camera's right of the origin for the default 3D camera at [3, 2, 5]
export const CARD_POSITION_3D: [number, number, number] = [2.6, 0.6, -1.5]
// drei Html transform: 1 CSS px = distanceFactor / 400 world units → the 260px card is ~2 units wide
const DISTANCE_FACTOR = 3

// The card lives inside the canvas container, where OrbitControls listens for wheel/drag.
// Stop those events at the card so it scrolls and selects text instead of moving the camera.
const keepEvent = (e: SyntheticEvent) => e.stopPropagation()

/** 3D: concept description floating in the scene, always turned toward the camera. Render inside <Canvas>. */
export function ConceptCard({ meta }: { meta: ConceptMeta }) {
  return (
    <Html transform sprite distanceFactor={DISTANCE_FACTOR} position={meta.cardPosition ?? CARD_POSITION_3D}>
      <div onWheel={keepEvent} onPointerDown={keepEvent}>
        <GlassSurface key={meta.id} width={260} height="auto" borderRadius={16} className="animate-in fade-in duration-300">
          <div className="max-h-[340px] w-full select-text overflow-y-auto overscroll-contain p-2 text-[12px] leading-[1.7]">
            <CardBody meta={meta} />
          </div>
        </GlassSurface>
      </div>
    </Html>
  )
}

/** 2D: the same card, always open, docked to the right edge and vertically centered. Render outside <Canvas>. */
export function ConceptCard2D({ meta }: { meta: ConceptMeta }) {
  return (
    <div className="absolute right-6 top-1/2 z-10 w-72 -translate-y-1/2">
      <GlassSurface key={meta.id} width="100%" height="auto" borderRadius={16} className="animate-in fade-in duration-300">
        <div className="max-h-[60vh] w-full select-text overflow-y-auto overscroll-contain p-2 text-[12px] leading-[1.7]">
          <CardBody meta={meta} />
        </div>
      </GlassSurface>
    </div>
  )
}

function CardBody({ meta }: { meta: ConceptMeta }) {
  const locale = useLocaleStore((s) => s.locale)
  return (
    <>
      <p className="mb-1 text-[9px] font-medium uppercase tracking-[0.2em] text-muted-foreground/70">
        {SUBJECT_LABELS[meta.subject][locale]}
      </p>
      <ReactMarkdown
        components={{
          h2: ({ children }) => (
            <h2 className="mb-2 text-[15px] font-semibold tracking-tight text-foreground">{children}</h2>
          ),
          h3: ({ children }) => (
            <h3 className="mb-1 mt-3 text-[12px] font-medium text-foreground/80">{children}</h3>
          ),
          p: ({ children }) => <p className="mb-2 text-muted-foreground last:mb-0">{children}</p>,
          strong: ({ children }) => <strong className="font-medium text-foreground/85">{children}</strong>,
          ul: ({ children }) => <ul className="mb-2 list-none space-y-0.5 p-0">{children}</ul>,
          li: ({ children }) => (
            <li className="flex gap-2 text-muted-foreground">
              <span className="mt-px shrink-0 select-none text-muted-foreground/30">—</span>
              <span>{children}</span>
            </li>
          ),
        }}
      >
        {meta.description[locale]}
      </ReactMarkdown>
    </>
  )
}
