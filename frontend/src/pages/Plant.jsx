import { Languages, Leaf, Plus, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { identifyPlant } from '../api/client'
import PhotoPicker from '../components/PhotoPicker'
import PlantUses from '../components/PlantUses'
import PrivacyNote from '../components/PrivacyNote'
import { Banner, Button, Card, CardBody, Disclaimer, Disclosure, Source, Spinner } from '../components/ui'
import { LANGS, saveLang, savedLang, strings } from '../lib/i18n'
import { photoProblem } from '../lib/photos'

const MAX_PHOTOS = 5
const ORGANS = [
  ['auto', 'Not sure'],
  ['leaf', 'Leaf'],
  ['flower', 'Flower'],
  ['fruit', 'Fruit'],
  ['bark', 'Bark'],
]

const pct = (s) => `${Math.round(s * 1000) / 10}%`

function OrganPicker({ value, onChange, disabled }) {
  return (
    <fieldset className="min-w-0" disabled={disabled}>
      <legend className="mb-2 text-sm font-medium">What do the photos show?</legend>
      <div className="flex flex-wrap gap-2">
        {ORGANS.map(([id, label]) => {
          const on = id === value
          return (
            <label
              key={id}
              className={`flex min-h-11 cursor-pointer items-center rounded-control border px-3 text-sm transition-colors duration-150 has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-[var(--focus)] ${
                on ? 'border-accent bg-accent-soft font-medium' : 'border-control bg-card hover:bg-raised'
              } ${disabled ? 'cursor-not-allowed opacity-60' : ''}`}
            >
              <input type="radio" name="plant-organ" value={id} checked={on} onChange={() => onChange(id)} className="sr-only" />
              {label}
            </label>
          )
        })}
      </div>
    </fieldset>
  )
}

/** Thumbnails of the chosen photos, each removable, plus "Add another" up to five. */
function PhotoStrip({ photos, onAdd, onRemove, disabled }) {
  const inputRef = useRef(null)
  return (
    <div className="space-y-2">
      <ul className="grid grid-cols-3 gap-2">
        {photos.map((p, i) => (
          <li key={p.url} className="relative aspect-square overflow-hidden rounded-control border border-hairline bg-raised">
            <img src={p.url} alt={`Plant photo ${i + 1}`} className="size-full object-cover" />
            <button
              type="button"
              onClick={() => onRemove(i)}
              disabled={disabled}
              aria-label={`Remove photo ${i + 1}`}
              className="absolute top-1 right-1 grid size-8 place-items-center rounded-full bg-card/90 text-ink shadow-card hover:bg-card"
            >
              <X className="size-4" strokeWidth={1.75} aria-hidden="true" />
            </button>
          </li>
        ))}
        {photos.length < MAX_PHOTOS && (
          <li>
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              disabled={disabled}
              className="grid aspect-square w-full place-content-center justify-items-center gap-1 rounded-control border-2 border-dashed border-control text-sm text-ink-2 transition-colors duration-150 hover:bg-raised"
            >
              <Plus className="size-5" strokeWidth={1.75} aria-hidden="true" />
              Add another
            </button>
          </li>
        )}
      </ul>
      <p className="text-xs text-ink-3">
        {photos.length} of {MAX_PHOTOS} photos. Other views of the same plant (whole plant, leaf, flower) help.
      </p>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        multiple
        className="sr-only"
        tabIndex={-1}
        aria-hidden="true"
        onChange={(e) => {
          onAdd([...(e.target.files ?? [])])
          e.target.value = ''
        }}
      />
    </div>
  )
}

/** Language of the result, top-right of the result card. */
function LangPicker({ value, onChange }) {
  return (
    <label className="relative flex items-center">
      <span className="sr-only">{strings(value).language}</span>
      <Languages className="pointer-events-none absolute left-3 size-4 text-ink-2" strokeWidth={1.75} aria-hidden="true" />
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="min-h-11 cursor-pointer rounded-control border border-control bg-card pr-3 pl-9 text-sm text-ink hover:bg-raised focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--focus)]"
      >
        {LANGS.map(([id, name]) => (
          <option key={id} value={id} lang={id}>
            {name}
          </option>
        ))}
      </select>
    </label>
  )
}

function Result({ result, lang, onLang }) {
  const { best, guide, candidates } = result
  const others = candidates.slice(1)
  const s = strings(lang)
  const commonName = guide.translations?.[lang]?.common_name ?? guide.common_name
  return (
    <div className="space-y-4" lang={lang}>
      <Card aria-labelledby="plant-h">
        <CardBody>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="flex flex-wrap items-center gap-2 text-sm text-ink-2">
              <Source>{s.fromPlantNet}</Source>
              {s.bestMatch(result.images)}
            </p>
            <LangPicker value={lang} onChange={onLang} />
          </div>
          <div className="flex flex-wrap items-end justify-between gap-x-6 gap-y-2">
            <div className="min-w-0">
              <h2 id="plant-h" lang="la" className="text-2xl font-medium tracking-tight italic">
                {best.scientific_name}
              </h2>
              <p className="mt-1">{commonName}</p>
              {best.family && (
                <p className="text-sm text-ink-2">
                  {s.family} <span lang="la">{best.family}</span>
                </p>
              )}
            </div>
            <div className="text-right">
              <p className="font-mono text-2xl">{pct(best.score)}</p>
              <p className="text-xs text-ink-2">{s.confidence}</p>
            </div>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-raised" aria-hidden="true">
            <div
              className={`h-full rounded-full ${result.low_confidence ? 'bg-warning' : 'bg-success'}`}
              style={{ width: `${Math.max(2, best.score * 100)}%` }}
            />
          </div>
          {result.low_confidence && (
            <Banner kind="warning" title={s.lowTitle}>
              {s.lowBody}
            </Banner>
          )}
        </CardBody>
      </Card>

      <Card aria-labelledby="use-h">
        <CardBody>
          <PlantUses name={best.scientific_name} commonNames={best.common_names} guide={guide} lang={lang} />
          {others.length > 0 && (
            <Disclosure title={s.others} hint={`${others.length}`} icon={Leaf}>
              <table className="w-full border-collapse text-sm">
                <caption className="sr-only">{s.othersCaption}</caption>
                <tbody>
                  {others.map((c) => (
                    <tr key={c.scientific_name}>
                      <th scope="row" className="border-t border-hairline py-2 text-left font-normal">
                        <span lang="la" className="italic">
                          {c.scientific_name}
                        </span>
                        {c.common_names.length > 0 && <span className="text-ink-2"> · {c.common_names.slice(0, 2).join(', ')}</span>}
                      </th>
                      <td className="border-t border-hairline py-2 text-right font-mono whitespace-nowrap">{pct(c.score)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Disclosure>
          )}
          <Disclaimer>{s.disclaimer}</Disclaimer>
        </CardBody>
      </Card>
    </div>
  )
}

export default function Plant() {
  const [photos, setPhotos] = useState([]) // [{file, url}]
  const [organ, setOrgan] = useState('auto')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [lang, setLang] = useState(savedLang)

  function changeLang(l) {
    setLang(l)
    saveLang(l)
  }

  // Revoke object URLs when the page closes.
  const photosRef = useRef(photos)
  useEffect(() => {
    photosRef.current = photos
  }, [photos])
  useEffect(() => () => photosRef.current.forEach((p) => URL.revokeObjectURL(p.url)), [])

  function add(files) {
    const room = MAX_PHOTOS - photos.length
    const problem = files.map(photoProblem).find(Boolean)
    if (problem) return setError(problem)
    setError(files.length > room ? `Only ${MAX_PHOTOS} photos can be used; the extra ones were left out.` : null)
    setResult(null)
    setPhotos((ps) => [...ps, ...files.slice(0, room).map((file) => ({ file, url: URL.createObjectURL(file) }))])
  }

  function remove(i) {
    URL.revokeObjectURL(photos[i].url)
    setPhotos((ps) => ps.filter((_, j) => j !== i))
    setResult(null)
  }

  async function identify() {
    if (!photos.length) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      setResult(await identifyPlant(photos.map((p) => p.file), organ))
    } catch (err) {
      setError(err.message || 'The plant couldn’t be identified. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const ok = result?.status === 'ok'

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h1 className="text-xl font-medium">Identify a plant</h1>
        <p className="text-sm text-ink-2">Photograph a plant to find out what it is, and what it’s used for in medicine</p>
      </div>

      <div className="grid items-start gap-5 lg:grid-cols-[minmax(0,380px)_minmax(0,1fr)]">
        <Card>
          <CardBody>
            {photos.length ? (
              <PhotoStrip photos={photos} onAdd={add} onRemove={remove} disabled={loading} />
            ) : (
              <PhotoPicker hasPhoto={false} onFile={(f) => add([f])} onClear={() => {}} disabled={loading} />
            )}
            <OrganPicker value={organ} onChange={(o) => (setOrgan(o), setResult(null))} disabled={loading} />
            <Button variant="primary" className="w-full" onClick={identify} disabled={!photos.length || loading}>
              {loading ? (
                <>
                  <Spinner />
                  Identifying the plant…
                </>
              ) : (
                'Identify this plant'
              )}
            </Button>
            <PrivacyNote feature="plant" />
          </CardBody>
        </Card>

        <div className="space-y-4">
          {error && (
            <Banner kind="error" role="alert" title="That didn’t work">
              {error}
            </Banner>
          )}
          {result && !ok && (
            <Banner kind="warning" role="alert" title="No plant matched">
              PlantNet couldn’t match these photos to a plant. Try a sharper photo of a leaf or flower, taken close up.
            </Banner>
          )}
          {ok ? (
            <Result result={result} lang={lang} onLang={changeLang} />
          ) : (
            <Card className="hidden lg:block">
              <CardBody>
                <h2 className="text-lg font-medium">{loading ? 'Identifying your plant' : 'What you’ll get'}</h2>
                {loading ? (
                  <div className="space-y-3" aria-hidden="true">
                    <div className="h-7 w-2/3 rounded-tag bg-raised" />
                    <div className="h-2 w-full rounded-full bg-raised" />
                  </div>
                ) : (
                  <ul className="list-disc space-y-1 pl-5 text-sm text-ink-2">
                    <li>The most likely plant, with its scientific and common names</li>
                    <li>How confident PlantNet is, and other possible plants</li>
                    <li>Its medicinal uses, how well each is supported, and safety notes</li>
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
