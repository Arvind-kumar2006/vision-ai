import { useEffect, useState } from 'react'
import { getProductExplanation } from '../../api/client'
import { Spinner } from '../ui'

/** "In plain words": fetched after the results, so it never delays them. */
export default function Explanation({ result }) {
  const ready = result.explanation_status === 'ready'
  const [state, setState] = useState(
    ready
      ? { status: 'ready', text: result.explanation, note: result.explanation_note }
      : { status: result.explanation_status === 'pending' ? 'loading' : 'none' },
  )

  useEffect(() => {
    if (result.explanation_status !== 'pending') return undefined
    const controller = new AbortController()
    getProductExplanation(result.session_id, { signal: controller.signal })
      .then((r) => setState({ status: 'ready', text: r.explanation, note: r.note }))
      .catch((err) => {
        if (err.name !== 'AbortError') setState({ status: 'error' })
      })
    return () => controller.abort()
  }, [result.session_id, result.explanation_status])

  if (state.status === 'none') return null
  return (
    <div className="space-y-1" aria-busy={state.status === 'loading'}>
      <h3 className="font-medium">In plain words</h3>
      <div role="status" aria-live="polite">
        {state.status === 'loading' && (
          <p className="flex items-center gap-2 text-ink-2">
            <Spinner className="size-4" />
            Writing a short explanation…
          </p>
        )}
        {state.status === 'ready' && (
          <>
            <p>{state.text}</p>
            {state.note && <p className="text-sm text-ink-2">Short summary: the detailed explanation isn’t available right now.</p>}
          </>
        )}
        {state.status === 'error' && <p className="text-sm text-ink-2">The explanation couldn’t be loaded. Everything else above is complete.</p>}
      </div>
    </div>
  )
}
