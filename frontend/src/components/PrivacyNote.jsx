import { Lock } from 'lucide-react'
import { useEffect, useState } from 'react'
import { loadHealth } from '../lib/privacy'
import { privacyText } from '../lib/privacy'

export default function PrivacyNote({ feature }) {
  const [ai, setAi] = useState(null)
  useEffect(() => {
    let live = true
    loadHealth().then((h) => live && setAi(h?.ai ?? null))
    return () => {
      live = false
    }
  }, [])
  return (
    <p className="flex gap-2 text-xs text-ink-2">
      <Lock className="mt-0.5 size-3.5 shrink-0" strokeWidth={1.75} aria-hidden="true" />
      <span>{feature === 'caption' || ai ? privacyText(feature, ai) : 'Checking where data goes…'}</span>
    </p>
  )
}
