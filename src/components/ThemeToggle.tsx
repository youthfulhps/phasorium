import { Moon, Sun } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useTheme } from '@/hooks/useTheme'
import { UI, useLocaleStore } from '@/lib/i18n'

export function ThemeToggle() {
  const { theme, toggleTheme } = useTheme()
  const t = UI[useLocaleStore((s) => s.locale)]
  return (
    <Button variant="ghost" size="icon" onClick={toggleTheme} aria-label={t.toggleTheme}>
      {theme === 'dark' ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
    </Button>
  )
}
