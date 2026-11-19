import { Check, Copy } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { captionImage, getHealth } from '../api/client'
import { CAPTION_MODELS, heatmapLabel, modelName } from '../lib/captionModels'
import AttentionImage from '../components/AttentionImage'
import PhotoPicker from '../components/PhotoPicker'
import PrivacyNote from '../components/PrivacyNote'
import { Banner, Button, Card, CardBody, Disclaimer, Spinner } from '../components/ui'
import { photoProblem } from '../lib/photos'


// Function words have diffuse attention, so start the overlay on the first content word.
const STOP_WORDS = new Set(['a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'with', 'and', 'is', 'are'])
function firstContentWord(words = []) {
  const i = words.findIndex((w) => !STOP_WORDS.has(w))
  return i >= 0 ? i : words.length ? 0 : -1
}

function ModelPicker({ value, onChange, disabled }) {
  return (
    <fieldset className="min-w-0" disabled={disabled}>
      <legend className="mb-2 text-sm font-medium">Model</legend>
      <div className="grid gap-2">
        {CAPTION_MODELS.map((m) => {
          const on = m.id === value
          return (
            <label
              key={m.id}
              className={`flex cursor-pointer items-start gap-3 rounded-control border px-3 py-2.5 transition-colors duration-150 has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-[var(--focus)] ${
                on ? 'border-accent bg-accent-soft' : 'border-hairline bg-card hover:bg-raised'
              } ${disabled ? 'cursor-not-allowed opacity-60' : ''}`}
            >
              <input
                type="radio"
                name="caption-model"
                value={m.id}
                checked={on}
                onChange={() => onChange(m.id)}
                className="mt-1 size-4 accent-[var(--accent)]"
              />
              <span className="grid">
                <span className="text-sm font-medium">{m.name}</span>
                <span className="text-xs text-ink-2">{m.hint}</span>
              </span>
            </label>
          )
        })}
      </div>
    </fieldset>
  )
}

function WordChips({ words, active, onActive }) {
  useEffect(() => {
    if (!words.length) return undefined
    function onKey(e) {
      const tag = e.target?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return
      if (e.key === 'ArrowRight') onActive(Math.min(active + 1, words.length - 1))
      if (e.key === 'ArrowLeft') onActive(Math.max(active - 1, 0))
      if (e.key === 'Escape') onActive(-1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [words.length, active, onActive])

  return (
    <div role="group" aria-label="Words in the caption" className="flex flex-wrap gap-2">
      {words.map((word, i) => {
        const on = i === active
        return (
          <button
            key={`${word}-${i}`}
            type="button"
            aria-pressed={on}
            onClick={() => onActive(on ? -1 : i)}
            className={`min-h-11 min-w-11 rounded-control border px-3 text-base transition-colors duration-150 ${
              on ? 'border-accent bg-accent font-medium text-on-accent' : 'border-control bg-card hover:bg-raised'
            }`}
          >
            {word}
          </button>
        )
      })}
    </div>
  )
}

export default function Caption() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [model, setModel] = useState('blip')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [active, setActive] = useState(-1)
  const [copied, setCopied] = useState(false)

  // Follow the server's default model (BLIP) unless the visitor picks another.
  useEffect(() => {
    let live = true
    getHealth()
      .then((h) => live && h?.caption_model && setModel((m) => (m === 'blip' ? h.caption_model : m)))
      .catch(() => {})
    return () => {
      live = false
    }
  }, [])

  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview])

  const onFile = useCallback((f) => {
    const problem = photoProblem(f)
    if (problem) return setError(problem)
    setError(null)
    setResult(null)
    setFile(f)
    setPreview(URL.createObjectURL(f))
  }, [])

  async function describe() {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await captionImage(file, { model })
      setActive(firstContentWord(data.words))
      setResult(data)
    } catch (err) {
      setError(err.message || 'The photo couldn’t be described. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(result.caption)
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    } catch {
      // clipboard blocked: nothing to do
    }
  }

  const words = result?.words ?? []
  const grid = result?.alphas?.[active] ?? null
  const usedModel = result?.model ?? model
  const caption = result?.caption ? result.caption.charAt(0).toUpperCase() + result.caption.slice(1) : ''

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h1 className="text-xl font-medium">Caption a photo</h1>
        <p className="text-sm text-ink-2">A one-line description of any photo, and where the model looked for each word</p>
      </div>

      <div className="grid items-start gap-5 lg:grid-cols-[minmax(0,420px)_minmax(0,1fr)]">
        <Card>
          <CardBody>
            <PhotoPicker hasPhoto={Boolean(preview)} onFile={onFile} onClear={() => (setFile(null), setPreview(null), setResult(null))} disabled={loading}>
              <AttentionImage src={preview} grid={grid} alt="Your photo" />
            </PhotoPicker>
            {result && words.length > 0 && (
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-ink-2">
                <span>
                  {heatmapLabel(usedModel)}
                  {active >= 0 && (
                    <>
                      {' '}for “<span className="font-medium text-ink">{words[active]}</span>”
                    </>
                  )}
                </span>
                <span className="flex items-center gap-2">
                  Less
                  <i className="block h-2 w-16 rounded-full bg-[linear-gradient(90deg,rgb(0_0_0/0.45),#ffd600_50%,#e63e11)]" aria-hidden="true" />
                  More
                </span>
              </div>
            )}
            <ModelPicker value={model} onChange={(m) => (setModel(m), setResult(null))} disabled={loading} />
            <Button variant="primary" className="w-full" onClick={describe} disabled={!file || loading}>
              {loading ? (
                <>
                  <Spinner />
                  Describing the photo…
                </>
              ) : (
                'Describe this photo'
              )}
            </Button>
            <PrivacyNote feature="caption" />
          </CardBody>
        </Card>

        <div className="space-y-4">
          {error && (
            <Banner kind="error" role="alert" title="That didn’t work">
              {error}
            </Banner>
          )}
          {result ? (
            <Card aria-labelledby="caption-answer">
              <CardBody>
                <p className="text-sm text-ink-2">Caption by {modelName(usedModel)}</p>
                <h2 id="caption-answer" className="text-2xl font-medium tracking-tight">
                  {caption}
                </h2>
                <div className="flex flex-wrap gap-2">
                  <Button onClick={copy}>
                    {copied ? (
                      <Check className="size-4" strokeWidth={1.75} aria-hidden="true" />
                    ) : (
                      <Copy className="size-4" strokeWidth={1.75} aria-hidden="true" />
                    )}
                    {copied ? 'Copied' : 'Copy caption'}
                  </Button>
                </div>
                {words.length > 0 && (
                  <div className="space-y-3 border-t border-hairline pt-4">
                    <h3 className="font-medium">{heatmapLabel(usedModel)}</h3>
                    <p className="text-sm text-ink-2">
                      Choose a word to see which parts of the photo the model paid attention to when it wrote it.
                      <span className="hidden sm:inline">
                        {' '}Use <kbd className="rounded-tag border border-b-2 border-control px-1 font-mono text-xs">←</kbd>{' '}
                        <kbd className="rounded-tag border border-b-2 border-control px-1 font-mono text-xs">→</kbd> to move between words.
                      </span>
                    </p>
                    <WordChips words={words} active={active} onActive={setActive} />
                  </div>
                )}
                <Disclaimer>
                  Captions are written by a computer model and can be wrong. The attention map shows what the model
                  focused on, not why.
                </Disclaimer>
              </CardBody>
            </Card>
          ) : (
            <Card className="hidden lg:block">
              <CardBody>
                <h2 className="text-lg font-medium">{loading ? 'Describing your photo' : 'What you’ll get'}</h2>
                {loading ? (
                  <div className="space-y-3" aria-hidden="true">
                    <div className="h-7 w-3/4 rounded-tag bg-raised" />
                    <div className="h-4 w-1/2 rounded-tag bg-raised" />
                  </div>
                ) : (
                  <ul className="list-disc space-y-1 pl-5 text-sm text-ink-2">
                    <li>A one-line description of the photo</li>
                    <li>For each word, where the model looked in the photo</li>
                    <li>Three models to compare</li>
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
