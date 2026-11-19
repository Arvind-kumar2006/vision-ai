import { RefreshCw } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { getPlantUses } from '../api/client'
import { strings } from '../lib/i18n'
import { Banner, Button, Source } from './ui'

/** Medicinal uses of the identified plant, fetched after the result so identification never waits for the AI. */
export default function PlantUses({ name, commonNames, guide, lang = 'en' }) {
  const [attempt, setAttempt] = useState(0)
  const request = `${name}#${lang}#${attempt}`
  const [state, setState] = useState({ request: null })
  const s = strings(lang)
  const local = guide.translations?.[lang] ?? guide

  useEffect(() => {
    const ctrl = new AbortController()
    getPlantUses(name, commonNames, { lang, signal: ctrl.signal })
      .then((data) => setState({ request, data }))
      .catch((err) => err.name !== 'AbortError' && setState({ request, error: err.message }))
    return () => ctrl.abort()
  }, [name, commonNames, lang, request])

  const retry = useCallback(() => setAttempt((a) => a + 1), [])
  // Anything stored for an earlier plant or attempt counts as still loading.
  const loading = state.request !== request
  const { data, error } = loading ? {} : state
  const ok = data?.status === 'ok'
  // Asked for Hindi/Kannada but the backend could only give English.
  const untranslated = ok && lang !== 'en' && data.lang !== lang

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <h2 id="use-h" className="text-lg font-medium">
          {s.usesTitle}
        </h2>
        {ok && data.source && (
          <Source>
            {s.writtenByAI} · {data.source.provider}
          </Source>
        )}
      </div>

      {guide.in_guide && (
        <div className="rounded-control border border-hairline p-3 text-sm">
          <p className="mb-1 flex flex-wrap items-center gap-2 font-medium">
            {local.common_name} <Source>{s.fromGuide}</Source>
          </p>
          <p className="text-ink-2">{local.use}</p>
          <p className="text-ink-2">{local.safety}</p>
        </div>
      )}

      {loading && (
        <div className="space-y-3" role="status" aria-label={s.loadingUses}>
          <div className="h-4 w-5/6 rounded-tag bg-raised" />
          <div className="h-4 w-2/3 rounded-tag bg-raised" />
          <div className="h-4 w-3/4 rounded-tag bg-raised" />
        </div>
      )}

      {(error || (data && !ok)) && (
        <Banner kind={data?.status === 'unavailable' ? 'offline' : 'limit'} title={s.usesFailed}>
          <p>{data?.status === 'unavailable' ? s.noAI : error || s.busy}</p>
          {data?.status !== 'unavailable' && (
            <Button variant="text" onClick={retry}>
              <RefreshCw className="size-4" strokeWidth={1.75} aria-hidden="true" />
              {s.tryAgain}
            </Button>
          )}
        </Banner>
      )}

      {untranslated && (
        <Banner kind="limit" title={s.notTranslated}>
          <Button variant="text" onClick={retry}>
            <RefreshCw className="size-4" strokeWidth={1.75} aria-hidden="true" />
            {s.tryAgain}
          </Button>
        </Banner>
      )}

      {ok && (
        <div lang={data.lang ?? 'en'} className="space-y-4">
          {data.toxic && (
            <Banner kind="error" title={s.toxicTitle}>
              {s.toxicBody}
            </Banner>
          )}
          {data.local_name && s.localName && (
            <p className="text-sm">
              <span className="font-medium">{s.localName}: </span>
              {data.local_name}
            </p>
          )}
          {data.summary && <p>{data.summary}</p>}
          {data.uses.length > 0 ? (
            <ul className="divide-y divide-hairline border-y border-hairline">
              {data.uses.map((u) => (
                <li key={u.use} className="grid gap-1 py-3 sm:grid-cols-[1fr_auto] sm:gap-4">
                  <div>
                    <p className="font-medium">{u.use}</p>
                    {u.detail && <p className="text-sm text-ink-2">{u.detail}</p>}
                  </div>
                  <div>
                    <Source>{s.evidence[u.evidence] ?? u.evidence}</Source>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-ink-2">{s.noUse}</p>
          )}
          {data.parts_used.length > 0 && (
            <p className="text-sm">
              <span className="font-medium">{s.partsUsed}</span>
              <span className="text-ink-2">{data.parts_used.join(', ')}</span>
            </p>
          )}
          {data.safety.length > 0 && (
            <Banner kind="warning" title={s.safety}>
              <ul className="list-disc space-y-1 pl-5">
                {data.safety.map((s) => (
                  <li key={s}>{s}</li>
                ))}
              </ul>
            </Banner>
          )}
        </div>
      )}
    </div>
  )
}
