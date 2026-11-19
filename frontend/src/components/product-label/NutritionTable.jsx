import { AlertTriangle } from 'lucide-react'
import { SOURCE_TAGS } from '../../lib/verdict'
import { Source } from '../ui'
import { nutritionRows } from './nutritionRows'

const fmt = (v) => Number(Number(v).toFixed(2))

export default function NutritionTable({ per100, basis, sources = {}, flags = {} }) {
  const rows = nutritionRows(per100)
  if (!rows.length) return <p className="text-sm text-ink-2">No nutrition values were found.</p>
  const th = 'border-b border-control pb-2 text-xs font-medium text-ink-2'
  return (
    <table className="w-full border-collapse text-sm">
      <caption className="sr-only">Nutrition {basis || 'per 100 g'}, with where each value came from</caption>
      <thead>
        <tr>
          <th scope="col" className={`${th} text-left`}>
            {basis ? basis.charAt(0).toUpperCase() + basis.slice(1) : 'Per 100 g'}
          </th>
          <th scope="col" className={`${th} pr-3 text-right`}>
            Amount
          </th>
          <th scope="col" className={`${th} text-right`}>
            Source
          </th>
        </tr>
      </thead>
      <tbody>
        {rows.map(({ key, name, unit, child, missing }) => {
          const rowFlags = flags[key] || []
          return (
            <tr key={key}>
              <th scope="row" className={`border-b border-hairline py-2.5 pr-3 text-left ${child ? 'pl-6 font-normal' : 'font-medium'}`}>
                {child ? (
                  <>
                    <span className="text-ink-2">of which </span>
                    {name.replace(/^of which /, '')}
                  </>
                ) : (
                  name
                )}
              </th>
              <td className={`border-b border-hairline py-2.5 pr-3 text-right whitespace-nowrap ${missing ? 'text-ink-2' : ''}`}>
                {missing ? 'not read' : `${fmt(per100[key])} ${unit}`}
                {rowFlags.length > 0 && (
                  <span title={rowFlags.join('; ')} className="ml-1 inline-flex align-[-2px] text-warning">
                    <AlertTriangle className="size-3.5" strokeWidth={2} aria-hidden="true" />
                    <span className="sr-only">{rowFlags.join('; ')}</span>
                  </span>
                )}
              </td>
              <td className="border-b border-hairline py-2.5 text-right">
                {!missing && sources[key] && <Source>{SOURCE_TAGS[sources[key]] || sources[key]}</Source>}
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}
