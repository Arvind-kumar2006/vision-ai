import { Eye, FileText, Info } from 'lucide-react'
import { useEffect, useState } from 'react'
import { interpretProductLabel } from '../api/client'
import PhotoPicker from '../components/PhotoPicker'
import PrivacyNote from '../components/PrivacyNote'
import Allergens from '../components/product-label/Allergens'
import Explanation from '../components/product-label/Explanation'
import NutriScoreStrip from '../components/product-label/NutriScore'
import NutritionTable from '../components/product-label/NutritionTable'
import { Banner, Button, Card, CardBody, Disclaimer, Disclosure, Source, Spinner } from '../components/ui'
import { photoProblem } from '../lib/photos'
import { verdict } from '../lib/verdict'

const NO_INGREDIENTS = /doesn't show the ingredients list/

const COMPONENT_NAMES = {
  energy: 'Energy',
  sugars: 'Sugars',
  saturated_fat: 'Saturated fat',
  saturates_to_lipids: 'Saturated fat share of fat',
  salt: 'Salt',
  non_nutritive_sweeteners: 'Sweeteners',
  protein: 'Protein',
  fibre: 'Fibre',
  fruit_veg_legumes: 'Fruit, vegetables and legumes',
}

function ScoreBreakdown({ ns }) {
  const v = (b) => (b.value == null ? 'not on the label' : `${Number(Number(b.value).toFixed(2))} ${b.unit}`)
  return (
    <div className="space-y-3">
      <table className="w-full border-collapse text-sm">
        <caption className="sr-only">How the Nutri-Score was worked out</caption>
        <thead>
          <tr className="text-xs text-ink-2">
            <th scope="col" className="border-b border-control pb-2 text-left font-medium">Part</th>
            <th scope="col" className="border-b border-control pb-2 text-left font-medium">Per 100 g</th>
            <th scope="col" className="border-b border-control pb-2 text-right font-medium">Points</th>
          </tr>
        </thead>
        <tbody>
          {ns.breakdown.map((b) => (
            <tr key={b.component}>
              <th scope="row" className="border-b border-hairline py-2 text-left font-medium">
                {COMPONENT_NAMES[b.component] || b.component}
              </th>
              <td className="border-b border-hairline py-2 text-ink-2">{v(b)}</td>
              <td className="border-b border-hairline py-2 text-right whitespace-nowrap">
                {b.note?.startsWith('not counted') ? 'not counted' : `${b.kind === 'negative' && b.points > 0 ? '+' : b.points > 0 ? '−' : ''}${b.points}`}
              </td>
            </tr>
          ))}
          <tr>
            <th scope="row" className="py-2 text-left font-medium">Total</th>
            <td />
            <td className="py-2 text-right font-medium whitespace-nowrap">
              {ns.points} points = {ns.grade}
            </td>
          </tr>
        </tbody>
      </table>
      {ns.notes?.length > 0 && (
        <ul className="list-disc space-y-1 pl-5 text-sm text-ink-2">
          {ns.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}
      <p className="text-sm text-ink-2">Fewer points is better. Nutri-Score 2023 algorithm.</p>
    </div>
  )
}

function Result({ result }) {
  const ns = result.nutriscore || {}
  const v = verdict(ns, result.nutrition_per_100)
  const allergenUnknown = !(result.allergens?.contains?.length || result.allergens?.may_contain?.length) && result.allergens?.status !== 'none_detected'
  const photoNotes = (result.photo_notes ?? []).filter((n) => !(allergenUnknown && NO_INGREDIENTS.test(n)))
  const aiValues = Object.values(result.field_sources ?? {}).filter((s) => s === 'ai_from_text').length
  // per 100 g couldn't be worked out (no serving weight): show what is printed per serving instead
  const perServing = !Object.keys(result.nutrition_per_100 ?? {}).length && Object.keys(result.nutrition_per_serving ?? {}).length > 0
  const fibreNote = ns.notes?.find((n) => n.startsWith('Fibre not available'))
  const quality = result.sanity_notes ?? []

  return (
    <div className="space-y-4">

      <Card aria-labelledby="verdict-h">
        <CardBody>
          {result.product_name && (
            <p className="flex flex-wrap items-center gap-2">
              <span className="text-lg font-medium">{result.product_name}</span>
              <Source>name you typed</Source>
            </p>
          )}
          <p className="flex flex-wrap items-center gap-2 text-sm text-ink-2">
            <Source>from your photo</Source>
            {aiValues > 0 && <Source>{aiValues} value{aiValues === 1 ? '' : 's'} read by AI from the text</Source>}
          </p>
          {v ? (
            <div className="grid items-center gap-4 sm:grid-cols-[auto_1fr] sm:gap-6">
              <NutriScoreStrip grade={v.grade} />
              <div>
                <h2 id="verdict-h" className="text-2xl font-medium tracking-tight">
                  {v.title}
                </h2>
                {v.reason && <p className="mt-1">{v.reason}</p>}
              </div>
            </div>
          ) : (
            <div className="grid items-center gap-4 sm:grid-cols-[auto_1fr] sm:gap-6">
              <NutriScoreStrip grade={null} />
              <div>
                <h2 id="verdict-h" className="text-2xl font-medium tracking-tight">
                  No health score yet
                </h2>
                <p className="mt-1">
                  {perServing
                    ? 'Only per-serving values are printed, without the serving’s weight, so the score can’t be worked out.'
                    : 'The nutrition table wasn’t read, so the score can’t be worked out.'}
                </p>
              </div>
            </div>
          )}
          {fibreNote && (
            <p className="flex gap-2 text-sm text-ink-2">
              <Info className="mt-0.5 size-4 shrink-0" strokeWidth={1.75} aria-hidden="true" />
              Fibre isn’t on the label, so it was counted as 0. The real score could be slightly better.
            </p>
          )}
          {photoNotes.map((n) => (
            <p key={n} className="flex gap-2 text-sm text-ink-2">
              <Info className="mt-0.5 size-4 shrink-0" strokeWidth={1.75} aria-hidden="true" />
              {n}
            </p>
          ))}
        </CardBody>
      </Card>

      {quality.length > 0 && (
        <Banner kind="warning" title="Some values may be misread">
          <ul className="list-disc space-y-1 pl-5">
            {quality.map((q) => (
              <li key={q}>{q}</li>
            ))}
          </ul>
        </Banner>
      )}

      <Allergens allergens={result.allergens} />

      <Card>
        <CardBody>
          <Explanation key={result.session_id} result={result} />
          {result.tips?.length > 0 && (
            <ul className="list-disc space-y-1 pl-5 text-sm">
              {result.tips.map((t) => (
                <li key={t}>{t}</li>
              ))}
            </ul>
          )}
          <div>
            {perServing ? (
              <Disclosure title="Nutrition per serving" hint={`${Object.keys(result.nutrition_per_serving).length} values`} icon={FileText} defaultOpen>
                <NutritionTable per100={result.nutrition_per_serving} basis="per serving" />
              </Disclosure>
            ) : (
              <Disclosure title="Nutrition per 100 g" hint={`${Object.keys(result.nutrition_per_100 || {}).length} values`} icon={FileText} defaultOpen>
                <NutritionTable per100={result.nutrition_per_100} basis={result.nutrition_basis} sources={result.field_sources} flags={result.nutrition_flags} />
              </Disclosure>
            )}
            {ns.breakdown?.length > 0 && ns.grade && (
              <Disclosure title="How the score was worked out" hint={`${ns.points} points`} icon={Info}>
                <ScoreBreakdown ns={ns} />
              </Disclosure>
            )}
            {(result.ingredients_text || result.additives?.length > 0) && (
              <Disclosure title="Ingredients and additives" hint={result.additives?.length ? `${result.additives.length} additive${result.additives.length === 1 ? '' : 's'}` : ''} icon={FileText}>
                {result.ingredients_text && <p className="text-sm">{result.ingredients_text}</p>}
                {result.additives?.length > 0 && (
                  <dl className="mt-3 grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-sm">
                    {result.additives.map((a) => (
                      <div key={a.code} className="contents">
                        <dt className="font-mono text-ink-2">{a.code}</dt>
                        <dd>
                          {a.name || 'Unknown additive'}
                          {a.classes?.length > 0 && <span className="text-ink-2"> · {a.classes.join(', ')}</span>}
                        </dd>
                      </div>
                    ))}
                  </dl>
                )}
              </Disclosure>
            )}
            {result.transcript?.length > 0 && (
              <Disclosure title="Text read from the photo" icon={Eye}>
                <pre className="overflow-x-auto rounded-control bg-raised p-3 font-mono text-xs whitespace-pre-wrap">{result.transcript.join('\n')}</pre>
              </Disclosure>
            )}
          </div>
          <Disclaimer>Informational only, not a substitute for a dietitian. If you have an allergy, always check the printed pack.</Disclaimer>
        </CardBody>
      </Card>

    </div>
  )
}

export default function FoodLabel() {
  const [photo, setPhoto] = useState(null) // {file, url}
  const [productName, setProductName] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => () => photo && URL.revokeObjectURL(photo.url), [photo])

  function onFile(f) {
    const problem = photoProblem(f)
    if (problem) return setError(problem)
    setError(null)
    setResult(null)
    setPhoto({ file: f, url: URL.createObjectURL(f) })
  }

  async function check() {
    if (!photo) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      setResult(await interpretProductLabel({ file: photo.file, productName }))
    } catch (err) {
      setError(err.message || 'The label couldn’t be read. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const ok = result?.status === 'ok'
  const unreadable = result && !ok ? result.message : null

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h1 className="text-xl font-medium">Check a food label</h1>
        <p className="text-sm text-ink-2">Photograph the back of a pack to see how healthy it is and what’s in it</p>
      </div>

      <div className="grid items-start gap-5 lg:grid-cols-[minmax(0,380px)_minmax(0,1fr)]">
        <Card>
          <CardBody>
            <PhotoPicker hasPhoto={Boolean(photo)} onFile={onFile} onClear={() => (setPhoto(null), setResult(null))} disabled={loading}>
              <img src={photo?.url} alt="Your label photo" className="mx-auto block max-h-[420px] w-auto max-w-full" />
            </PhotoPicker>
            <div className="grid gap-1.5">
              <label htmlFor="product-name" className="text-sm font-medium">
                Product name <span className="font-normal text-ink-2">(optional)</span>
              </label>
              <input
                id="product-name"
                type="text"
                value={productName}
                onChange={(e) => {
                  setProductName(e.target.value)
                  if (ok) setResult(null)
                }}
                maxLength={100}
                disabled={loading}
                autoComplete="off"
                placeholder="For example, Maggi 2-Minute Noodles"
                aria-describedby="product-name-hint"
                className="min-h-11 rounded-control border border-control bg-card px-3 text-base text-ink placeholder:text-ink-3 disabled:opacity-60"
              />
              <p id="product-name-hint" className="text-xs text-ink-2">
                Names the result and lets the explanation refer to it. The score uses only the label.
              </p>
            </div>
            {!ok && (
              <Button variant="primary" className="w-full" onClick={check} disabled={!photo || loading}>
                {loading ? (
                  <>
                    <Spinner />
                    Reading the label…
                  </>
                ) : (
                  'Check this label'
                )}
              </Button>
            )}
            <PrivacyNote feature="label" />
          </CardBody>
        </Card>

        <div className="space-y-4">
          {error && (
            <Banner kind="error" role="alert" title="That didn’t work">
              {error}
            </Banner>
          )}
          {unreadable && (
            <Banner kind="warning" role="alert" title="This photo couldn’t be read">
              {unreadable}
            </Banner>
          )}
          {ok ? (
            <Result result={result} />
          ) : (
            <Card className="hidden lg:block">
              <CardBody>
                <h2 className="text-lg font-medium">{loading ? 'Reading your label' : 'What you’ll get'}</h2>
                {loading ? (
                  <div className="space-y-3" aria-hidden="true">
                    <div className="h-14 w-48 rounded-control bg-raised" />
                    <div className="h-7 w-2/3 rounded-tag bg-raised" />
                  </div>
                ) : (
                  <ul className="list-disc space-y-1 pl-5 text-sm text-ink-2">
                    <li>A health score from A to E, with the main reason</li>
                    <li>Allergens it contains or may contain</li>
                    <li>The nutrition table per 100 g</li>
                    <li>Answers to your questions about it</li>
                  </ul>
                )}
              </CardBody>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
