import { Button } from '@/components/ui/button'
import { UI, useLocaleStore } from '@/lib/i18n'

export function LocaleToggle() {
  const locale = useLocaleStore((s) => s.locale)
  const toggleLocale = useLocaleStore((s) => s.toggleLocale)
  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={toggleLocale}
      aria-label={UI[locale].toggleLocale}
      className="text-xs font-medium"
    >
      {locale === 'en' ? '한' : 'EN'}
    </Button>
  )
}
