import { Bookmark, CheckCircle2, ChevronDown, CircleHelp, Eye, Info, XCircle } from 'lucide-react'
import { Fragment, useCallback, useEffect, useState } from 'react'
import { analyzePrescription } from '../api/client'
import PhotoPicker from '../components/PhotoPicker'
import PrivacyNote from '../components/PrivacyNote'
import { Banner, Button, Card, CardBody, Disclaimer, Disclosure, Source, Spinner } from '../components/ui'
import { photoProblem } from '../lib/photos'
import { LEVELS, visibleColumns } from '../lib/prescriptionTable'


const STATUS = {
  matched: { label: 'Matched', icon: CheckCircle2, cls: 'border-success-line bg-success-soft text-success' },
  known_brand: { label: 'Known brand', icon: Bookmark, cls: 'border-hairline bg-accent-soft text-accent' },
  verify: { label: 'Verify', icon: CircleHelp, cls: 'border-warning-line bg-warning-soft text-warning' },
  unmatched: { label: 'Not found', icon: XCircle, cls: 'border-danger-line bg-danger-soft text-danger' },
}

function StatusBadge({ status, level }) {
  const s = STATUS[status] ?? STATUS.unmatched
  const Icon = s.icon
  return (
    <span className="inline-flex flex-col items-start gap-1">
      <span className={`inline-flex items-center gap-1.5 whitespace-nowrap rounded-tag border px-2 py-0.5 text-xs font-medium ${s.cls}`}>
        <Icon className="size-3.5" strokeWidth={2} aria-hidden="true" />
        {s.label}
      </span>
      {level && (
        <span className={`whitespace-nowrap text-xs ${level === 'low' ? 'text-danger' : level === 'medium' ? 'text-warning' : 'text-ink-2'}`}>
          {LEVELS[level]}
        </span>
      )}
    </span>
  )
}

/** The written strength with its "check" marker (compare with the paper). Nothing is ever guessed. */
function Strength({ med }) {
  if (!med.strength) {
    const n = med.strength_variants
    // One strength (or none) in the database: nothing is missing. Several: the written name doesn't say which.
    if (n > 1) return <span className="text-xs text-warning">Not written: {n} strengths exist. Check the prescription.</span>
    return <span className="text-ink-3">{n === 1 ? 'Not needed (one strength only)' : 'Not written'}</span>
  }
  return (
    <span className="inline-flex flex-wrap items-center gap-1">
      <span>{med.strength}</span>
      {med.checks?.includes('strength') && <Source tone="warning">check</Source>}
    </span>
  )
}

const needsCheck = (m) => m.match_status === 'verify' || m.match_status === 'unmatched' || m.in_ocr_text === false

const NO_DESCRIPTION = 'No description available. Ask your pharmacist what this medicine is for.'
const USES_CREDIT = '“250k Medicines Usage, Side Effects and Substitutes” by Shudhanshu Singh (Kaggle), CC BY-SA 4.0'

/** "What it's for": database text only, never written by AI. A Verify row shows it after a medicine is chosen. */
function WhatFor({ med, pick }) {
  const confirmed = med.match_status === 'matched' || med.match_status === 'known_brand'
  const chosen = !confirmed && pick ? (med.candidates ?? []).find((c) => c.name === pick) : null
  if (chosen) med = { ...med, what_for: chosen.what_for }
  if (!confirmed && !chosen) return null
  return (
    <span className="mt-1 block max-w-72 text-sm">
      {med.what_for ? (
        <>
          <span className="text-ink-2">What it’s for:</span> {med.what_for}{' '}
          <Source className="mt-0.5">from medicine database</Source>
        </>
      ) : (
        <span className="text-ink-2">{NO_DESCRIPTION}</span>
      )}
    </span>
  )
}

function Composition({ med, pick, onPick }) {
  const choice = pick ?? ''
  const candidates = med.candidates ?? []
  if (med.match_status === 'verify' && candidates.length > 0) {
    const chosen = candidates.find((c) => c.name === choice)
    return (
      <div className="max-w-80 space-y-1">
        <label className="sr-only" htmlFor={`pick-${med.read_as}`}>
          Which medicine is “{med.read_as || med.name}”?
        </label>
        <select
          id={`pick-${med.read_as}`}
          value={choice}
          onChange={(e) => onPick(e.target.value)}
          className="min-h-11 w-full rounded-control border border-control bg-card px-3 text-sm text-ink"
        >
          <option value="">Choose the right medicine</option>
          {candidates.map((c) => (
            <option key={c.name} value={c.name}>
              {c.name}
            </option>
          ))}
        </select>
        <p className="text-xs text-ink-2">
          {chosen
            ? [chosen.composition, chosen.manufacturer, chosen.pack_size].filter(Boolean).join(' · ')
            : `${candidates.length} close matches. Check the spelling on the prescription.`}
        </p>
      </div>
    )
  }
  const comp = med.db_info?.composition || med.known_brand?.composition
  if (comp) return <span>{comp}</span>
  if (med.non_medicine) return <span className="text-ink-2">Not a medicine (oral care product or device)</span>
  return <span className="text-ink-2">Not in the medicine list. Ask your pharmacist.</span>
}

const EFFECT = { good: 'text-success', fair: 'text-warning', bad: 'text-danger', neutral: 'text-ink-2' }

/** Why the row has its confidence level: the factors in words, never a percentage. */
function ConfidenceFactors({ med }) {
  if (!med.confidence_factors?.length) return null
  return (
    <div className="rounded-control border border-hairline bg-card p-4">
      <p className="mb-2 text-sm font-medium">{LEVELS[med.confidence_level]}: how it was worked out</p>
      <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-sm">
        {med.confidence_factors.map((f) => (
          <div key={f.factor} className="contents">
            <dt className="text-ink-2">{f.factor}</dt>
            <dd className={EFFECT[f.effect] ?? ''}>{f.value}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-2 text-xs text-ink-3">
        High: matched, no close other brand, handwriting clear, and either the written strength confirms it (name 95 or
        more) or the medicine has one strength only (name must match exactly). Low: any factor in red, including no
        strength written when the medicine comes in 4 or more strengths.
      </p>
    </div>
  )
}

function DbDetails({ med }) {
  const info = med.db_info ?? {}
  const facts = [info.manufacturer, info.pack_size, info.type].filter(Boolean).join(' · ')
  const sameIngredient = (med.candidates ?? []).filter((c) => c.name !== med.suggested_name).slice(0, 3)
  const fromUses = med.what_for_source?.startsWith('250k')
  return (
    <div className="rounded-control border border-hairline bg-card p-4">
      <p className="mb-3 flex items-center gap-2 text-sm font-medium">
        From medicine database <Source>Indian Medicine Dataset</Source>
      </p>
      <div className="grid gap-4 text-sm md:grid-cols-3">
        {(med.drug_class || med.other_uses?.length > 0) && (
          <div>
            <h4 className="mb-1 text-xs text-ink-2">Other uses listed</h4>
            {med.other_uses?.length > 0 ? (
              <ul className="list-disc pl-5">
                {med.other_uses.map((u) => (
                  <li key={u}>{u}</li>
                ))}
              </ul>
            ) : (
              <p className="text-ink-2">None listed</p>
            )}
            {med.drug_class && <p className="mt-1 text-xs text-ink-2">Group: {med.drug_class}</p>}
          </div>
        )}
        {!info.side_effects?.length && med.common_side_effects?.length > 0 && (
          <div>
            <h4 className="mb-1 text-xs text-ink-2">Common side effects</h4>
            <ul className="list-disc pl-5">
              {med.common_side_effects.map((s) => (
                <li key={s}>{s}</li>
              ))}
            </ul>
          </div>
        )}
        {sameIngredient.length > 0 && (
          <div>
            <h4 className="mb-1 text-xs text-ink-2">Similar names in the database</h4>
            <ul className="list-disc pl-5">
              {sameIngredient.map((c) => (
                <li key={c.name}>
                  {c.name}
                  {c.composition ? <span className="text-ink-2"> · {c.composition}</span> : ''}
                </li>
              ))}
            </ul>
          </div>
        )}
        {info.description && (
          <div>
            <h4 className="mb-1 text-xs text-ink-2">About this medicine</h4>
            <p>{info.description.length > 260 ? `${info.description.slice(0, 257)}…` : info.description}</p>
          </div>
        )}
        {info.side_effects?.length > 0 && (
          <div>
            <h4 className="mb-1 text-xs text-ink-2">Common side effects</h4>
            <ul className="list-disc pl-5">
              {info.side_effects.map((s) => (
                <li key={s}>{s}</li>
              ))}
            </ul>
          </div>
        )}
        {info.interactions?.length > 0 && (
          <div>
            <h4 className="mb-1 text-xs text-ink-2">Interactions</h4>
            <ul className="list-disc pl-5">
              {info.interactions.map((it, i) => (
                <li key={i}>
                  {it.drug}
                  {it.effect ? <span className="text-ink-2"> · {it.effect.toLowerCase()}</span> : ''}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
      {(facts || info.variant_note) && <p className="mt-3 text-xs text-ink-2">{[facts, info.variant_note].filter(Boolean).join(' · ')}</p>}
      {fromUses && <p className="mt-2 text-xs text-ink-3">Uses and side effects: {USES_CREDIT}</p>}
    </div>
  )
}

const HEADINGS = {
  medicine: 'Medicine',
  contents: 'What’s in it',
  strength: 'Strength',
  status: 'Status',
}

function MedicineTable({ medicines, picks, onPick, columns }) {
  const [open, setOpen] = useState(null)
  const th = 'border-b border-control px-3 pb-2 text-left text-xs font-medium text-ink-2'
  const td = 'border-b border-hairline px-3 py-3 align-top text-sm'
  const show = (c) => columns.includes(c)
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[560px] border-collapse">
        <caption className="sr-only">Medicines on the prescription</caption>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c} scope="col" className={th}>
                {HEADINGS[c]}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {medicines.map((m, i) => {
            const readAs = m.read_as || m.name
            const isOpen = open === i
            // Only a confirmed match may be shown as the medicine's name; a guess stays in the dropdown.
            const confirmed = m.match_status === 'matched' || m.match_status === 'known_brand'
            const title = confirmed ? m.suggested_name || readAs : readAs
            return (
              <Fragment key={`${readAs}-${i}`}>
                <tr className={isOpen ? 'bg-raised' : ''}>
                  <th scope="row" className={`${td} text-left font-normal`}>
                    <span className="block text-base font-medium">{title}</span>
                    {confirmed && title !== readAs ? (
                      <span className="mt-0.5 flex flex-wrap items-center gap-1 text-xs text-ink-2">
                        <Source>as written</Source> {readAs}
                      </span>
                    ) : (
                      !confirmed && <Source className="mt-1">as written</Source>
                    )}
                    {m.alternatives?.length > 0 && (
                      <span className="mt-1 block text-xs text-ink-2">Also written: {m.alternatives.join(', ')}</span>
                    )}
                    <WhatFor med={m} pick={picks[i]} />
                    <button
                      type="button"
                      aria-expanded={isOpen}
                      onClick={() => setOpen(isOpen ? null : i)}
                      className="mt-1 inline-flex min-h-9 items-center gap-1 text-xs font-medium text-accent underline underline-offset-4"
                    >
                      {isOpen ? 'Hide details' : 'Details'}
                      <ChevronDown className={`size-3.5 ${isOpen ? 'rotate-180' : ''}`} strokeWidth={2} aria-hidden="true" />
                    </button>
                  </th>
                  {show('contents') && (
                    <td className={td}>
                      <Composition med={m} pick={picks[i]} onPick={(name) => onPick(i, name)} />
                    </td>
                  )}
                  {show('strength') && (
                    <td className={td}>
                      <Strength med={m} />
                    </td>
                  )}
                  <td className={td}>
                    <StatusBadge status={m.match_status} level={m.confidence_level} />
                    {m.in_ocr_text === false && <p className="mt-1 text-xs text-danger">Not found in the text that was read</p>}
                  </td>
                </tr>
                {isOpen && (
                  <tr className="bg-raised">
                    <td colSpan={columns.length} className="space-y-3 border-b border-hairline px-3 pb-4">
                      <ConfidenceFactors med={m} />
                      {confirmed && (m.db_info || m.what_for) && <DbDetails med={m} />}
                    </td>
                  </tr>
                )}
              </Fragment>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function Result({ result }) {
  // Verify rows: the medicine the person chose stays in this browser.
  const [picks, setPicks] = useState({})
  const meds = result.medicines ?? []
  const columns = visibleColumns(meds)
  const toCheck = meds.filter(needsCheck).length
  const identified = meds.filter((m) => m.match_status === 'matched' || m.match_status === 'known_brand').length
  const lines = result.ocr_lines ?? []
  const busy = result.status === 'ai_busy'
  const answer = meds.length
    ? `${identified} of ${meds.length} medicine${meds.length === 1 ? '' : 's'} identified${toCheck ? `. ${toCheck} need${toCheck === 1 ? 's' : ''} checking` : ''}`
    : 'No medicines found'

  return (
    <>
      <Card aria-labelledby="rx-answer">
        <CardBody>
          <p className="flex flex-wrap items-center gap-2 text-sm text-ink-2">
            <Source>read on this computer</Source>
            {result.lines_sent_to_ai > 0 && <Source>medicine lines to AI</Source>}
          </p>
          {busy && (
            <Banner kind="limit" role="alert" title="AI busy">
              The AI couldn’t help this time, so fewer medicines may be found. Try again in a few minutes. Below are the
              medicine names found in the text that was read, checked against the medicine list.
            </Banner>
          )}
          <h2 id="rx-answer" className="text-2xl font-medium tracking-tight">
            {answer}
          </h2>
          {toCheck > 0 && (
            <Banner kind="warning" title="Some handwriting was hard to read">
              Check the rows marked Verify or Not found against the prescription before you rely on this list.
            </Banner>
          )}
          {meds.length > 0 ? (
            <>
              <MedicineTable
              medicines={meds}
              picks={picks}
              onPick={(i, name) => setPicks((p) => ({ ...p, [i]: name }))}
              columns={columns}
            />
              {meds.some((m) => m.what_for) && (
                <p className="text-xs text-ink-3">
                  “What it’s for” is database text by ingredient, never written by AI. Source: {USES_CREDIT}; where it has
                  none, the Indian Medicine Dataset.
                </p>
              )}
            </>
          ) : (
            !busy && <p className="text-ink-2">{result.message || 'No medicine lines were found. A flat photo of the whole page works best.'}</p>
          )}
          {result.already_taking?.length > 0 && (
            <div className="text-sm">
              <p className="text-ink-2">Already taking (not part of this prescription)</p>
              <ul className="mt-1 list-disc pl-5">
                {result.already_taking.map((name) => (
                  <li key={name}>{name}</li>
                ))}
              </ul>
            </div>
          )}
          <div>
            {result.review_notes?.length > 0 && (
              <Disclosure title="Review notes" hint={`${result.review_notes.length}`} icon={Info} defaultOpen>
                <ul className="list-disc space-y-1 pl-5 text-sm">
                  {result.review_notes.map((f, i) => (
                    <li key={i}>{f}</li>
                  ))}
                </ul>
              </Disclosure>
            )}
            <Disclosure title="Text read from the photo" icon={Eye}>
              {lines.length ? (
                <ol className="space-y-1 font-mono text-xs">
                  {lines.map((l, i) => (
                    <li key={i} className={l.sent_to_ai ? '' : 'text-ink-3'}>
                      {l.text}
                      {!l.sent_to_ai && <span className="sr-only"> (kept on this computer)</span>}
                    </li>
                  ))}
                </ol>
              ) : (
                <p className="text-sm text-ink-2">No text was read.</p>
              )}
            </Disclosure>
          </div>
          <Disclaimer>
            This is a reading aid, not medical advice. Handwriting is often misread: confirm every medicine and strength with
            your doctor or pharmacist, and follow their instructions on how to take it.
          </Disclaimer>
        </CardBody>
      </Card>
    </>
  )
}

export default function Prescription() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview])

  const onFile = useCallback((f) => {
    const problem = photoProblem(f)
    if (problem) return setError(problem)
    setError(null)
    setResult(null)
    setFile(f)
    setPreview(URL.createObjectURL(f))
  }, [])

  async function read() {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      setResult(await analyzePrescription(file))
    } catch (err) {
      setError(err.message || 'The prescription couldn’t be read. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const photoCard = (
    <Card>
      <CardBody>
        <PhotoPicker hasPhoto={Boolean(preview)} onFile={onFile} onClear={() => (setFile(null), setPreview(null), setResult(null))} disabled={loading}>
          <img src={preview} alt="Your prescription photo" className="mx-auto block max-h-[420px] w-auto max-w-full" />
        </PhotoPicker>
        {(!result || result.status === 'unreadable') && (
          <Button variant="primary" className="w-full" onClick={read} disabled={!file || loading}>
            {loading ? (
              <>
                <Spinner />
                Reading the prescription…
              </>
            ) : (
              'Read this prescription'
            )}
          </Button>
        )}
        <PrivacyNote feature="prescription" />
      </CardBody>
    </Card>
  )

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h1 className="text-xl font-medium">Read a prescription</h1>
        <p className="text-sm text-ink-2">Find out which medicines a prescription lists, and what each one is for</p>
      </div>
      {error && (
        <Banner kind="error" role="alert" title="That didn’t work">
          {error}
        </Banner>
      )}
      {result?.status === 'unreadable' && (
        <Banner kind="warning" role="alert" title="This photo couldn’t be read">
          {result.message}
        </Banner>
      )}
      {result && result.status !== 'unreadable' ? (
        // Keep the source photo visible while the long result remains independently scrollable.
        <div className="grid items-start gap-5 lg:grid-cols-[minmax(0,420px)_minmax(0,1fr)]">
          <div className="lg:sticky lg:top-5">{photoCard}</div>
          <div className="min-w-0 lg:max-h-[calc(100vh-8rem)] lg:overflow-y-auto lg:pr-2">
            <Result key={result.id} result={result} />
          </div>
        </div>
      ) : (
        <div className="grid items-start gap-5 lg:grid-cols-[minmax(0,420px)_minmax(0,1fr)]">
          {photoCard}
          <Card className="hidden lg:block">
            <CardBody>
              <h2 className="text-lg font-medium">{loading ? 'Reading your prescription' : 'What you’ll get'}</h2>
              {loading ? (
                <div className="space-y-3" aria-hidden="true">
                  <div className="h-7 w-2/3 rounded-tag bg-raised" />
                  <div className="h-24 w-full rounded-tag bg-raised" />
                </div>
              ) : (
                <ul className="list-disc space-y-1 pl-5 text-sm text-ink-2">
                  <li>Each medicine identified against a list of 240,000 Indian medicines</li>
                  <li>Its strength, what’s in it, and what it’s for</li>
                  <li>Clear marks on anything that needs checking</li>
                </ul>
              )}
            </CardBody>
          </Card>
        </div>
      )}
    </div>
  )
}
