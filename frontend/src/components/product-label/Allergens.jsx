import { AlertCircle, AlertTriangle, CircleHelp } from 'lucide-react'
import { Card, CardBody } from '../ui'

const FROM = {
  ingredients: 'ingredients',
  contains_statement: '“Contains” line',
  may_contain_statement: '“May contain” line',
}

function Chip({ a, kind }) {
  const why = `Found “${a.trigger}” in the ${FROM[a.source] || a.source}`
  const cls =
    kind === 'contains'
      ? 'border-danger-line bg-danger-soft text-danger'
      : 'border-dashed border-warning-line bg-warning-soft text-warning'
  const Icon = kind === 'contains' ? AlertCircle : AlertTriangle
  return (
    // Focusable, so keyboard users get the same "found …" explanation as the hover title.
    <li>
      <span tabIndex={0} title={why} className={`inline-flex min-h-9 items-center gap-1.5 rounded-full border px-3 text-sm font-medium ${cls}`}>
        <Icon className="size-4" strokeWidth={2} aria-hidden="true" />
        {a.label}
        <span className="sr-only">. {why}</span>
      </span>
    </li>
  )
}

/** Found / none detected / unknown. "None" is only said when the ingredients were actually read. */
export default function Allergens({ allergens }) {
  const contains = allergens?.contains ?? []
  const may = allergens?.may_contain ?? []
  const found = contains.length > 0 || may.length > 0
  const triggers = [...contains, ...may].map((a) => `“${a.trigger}”`)

  return (
    <Card aria-labelledby="allergens-h">
      <CardBody>
        <h2 id="allergens-h" className="text-lg font-medium">
          Allergens
        </h2>
        {found ? (
          <>
            <dl className="grid gap-x-4 gap-y-2 sm:grid-cols-[7rem_1fr]">
              {contains.length > 0 && (
                <>
                  <dt className="text-sm text-ink-2 sm:pt-1.5">Contains</dt>
                  <dd>
                    <ul className="flex flex-wrap gap-2">
                      {contains.map((a) => (
                        <Chip key={a.allergen} a={a} kind="contains" />
                      ))}
                    </ul>
                  </dd>
                </>
              )}
              {may.length > 0 && (
                <>
                  <dt className="text-sm text-ink-2 sm:pt-1.5">May contain</dt>
                  <dd>
                    <ul className="flex flex-wrap gap-2">
                      {may.map((a) => (
                        <Chip key={a.allergen} a={a} kind="may" />
                      ))}
                    </ul>
                  </dd>
                </>
              )}
            </dl>
            <p className="text-sm text-ink-2">Found in the text: {[...new Set(triggers)].join(', ')}</p>
            {allergens?.ingredients_read === false && (
              <p className="text-sm text-warning">The ingredients list couldn’t be read, so other allergens may be present. Check the pack.</p>
            )}
          </>
        ) : allergens?.status === 'none_detected' ? (
          <p className="text-ink-2">No common allergens were found in the ingredients. Always check the printed pack.</p>
        ) : (
          <div className="flex gap-3 rounded-control border border-hairline bg-raised p-3 sm:p-4" role="status">
            <CircleHelp className="mt-0.5 size-5 shrink-0 text-ink-2" strokeWidth={1.75} aria-hidden="true" />
            <div className="space-y-2">
              <p className="font-medium">Allergens unknown</p>
              <p className="text-sm">The ingredients list wasn’t read from this photo, so allergens can’t be checked. Check the pack, or take a photo of the ingredients list.</p>
            </div>
          </div>
        )}
      </CardBody>
    </Card>
  )
}
