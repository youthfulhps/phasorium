import { useEffect, useState } from 'react'

/** Semantic scene colors. Values live in src/index.css; meanings in DESIGN.md. */
export const PHYSICS_COLORS = [
  'voltage',
  'current',
  'resistance',
  'inductance',
  'capacitance',
  'power',
  'e-field',
  'b-field',
  'guide',
  'marker',
  'backdrop',
] as const

export type PhysicsColor = (typeof PHYSICS_COLORS)[number]

function readColors(): Record<PhysicsColor, string> {
  const style = getComputedStyle(document.documentElement)
  return Object.fromEntries(
    PHYSICS_COLORS.map((name) => [name, style.getPropertyValue(`--${name}`).trim()])
  ) as Record<PhysicsColor, string>
}

const readIsDark = () => document.documentElement.classList.contains('dark')

/** Re-evaluates `read` whenever the light/dark theme class on <html> changes. */
function useThemeValue<T>(read: () => T) {
  const [value, setValue] = useState(read)

  useEffect(() => {
    const observer = new MutationObserver(() => setValue(read()))
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
    return () => observer.disconnect()
  }, [read])

  return value
}

/** Theme-aware scene colors as hex strings (Three.js can't parse oklch). */
export const usePhysicsColors = () => useThemeValue(readColors)

export const useIsDark = () => useThemeValue(readIsDark)
