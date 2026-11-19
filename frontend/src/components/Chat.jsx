import { MessageSquare, Send } from 'lucide-react'
import { useId, useState } from 'react'
import { askProductQuestion } from '../../api/client'
import { Button, Card, CardBody, Spinner } from '../ui'

const SUGGESTIONS = ['What makes it this grade?', 'Is it high in salt?']

/** Session chat. Defaults are the food-label ones; the prescription page passes its own. */
export default function Chat({
  sessionId,
  disabledNote,
  ask: askApi = askProductQuestion,
  title = 'Ask about this product',
  suggestions = SUGGESTIONS,
  inputLabel = 'Your question about this product',
  footer = 'Answers use only this product’s label and database data',
}) {
  const [question, setQuestion] = useState('')
  const [history, setHistory] = useState([])
  const [busy, setBusy] = useState(false)
  const inputId = useId()

  async function ask(q) {
    q = q.trim()
    if (!q || busy) return
    setBusy(true)
    setQuestion('')
    try {
      const r = await askApi(sessionId, q)
      setHistory((h) => [...h, { q, a: r.answer }])
    } catch (err) {
      setHistory((h) => [...h, { q, a: `That question couldn’t be answered (${err.message}).` }])
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card aria-labelledby="chat-h">
      <CardBody>
        <h2 id="chat-h" className="flex items-center gap-2 text-lg font-medium">
          <MessageSquare className="size-[18px] text-ink-3" strokeWidth={1.75} aria-hidden="true" />
          {title}
        </h2>
        {history.length > 0 && (
          <ul className="grid gap-3" aria-live="polite">
            {history.map((m, i) => (
              <li key={i} className="grid gap-2">
                <p className="max-w-[85%] justify-self-end rounded-control rounded-br-sm bg-accent-soft px-4 py-2.5 text-sm">{m.q}</p>
                <p className="max-w-[85%] justify-self-start rounded-control rounded-bl-sm bg-raised px-4 py-2.5 text-sm">{m.a}</p>
              </li>
            ))}
          </ul>
        )}
        {disabledNote ? (
          <p className="text-sm text-ink-2">{disabledNote}</p>
        ) : (
          history.length === 0 && (
            <div className="flex flex-wrap gap-2">
              {suggestions.map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => ask(s)}
                  disabled={busy}
                  className="min-h-9 rounded-full border border-control bg-card px-3 text-sm transition-colors duration-150 hover:bg-raised"
                >
                  {s}
                </button>
              ))}
            </div>
          )
        )}
        <form
          className="flex gap-2"
          onSubmit={(e) => {
            e.preventDefault()
            ask(question)
          }}
        >
          <label htmlFor={inputId} className="sr-only">
            {inputLabel}
          </label>
          <input
            id={inputId}
            type="text"
            maxLength={500}
            value={question}
            placeholder="Ask a question"
            onChange={(e) => setQuestion(e.target.value)}
            disabled={busy || Boolean(disabledNote)}
            className="min-h-11 min-w-0 flex-1 rounded-control border border-control bg-card px-3 text-base text-ink placeholder:text-ink-3 disabled:bg-raised"
          />
          <Button type="submit" disabled={busy || !question.trim() || Boolean(disabledNote)}>
            {busy ? <Spinner className="size-4" /> : <Send className="size-4" strokeWidth={1.75} aria-hidden="true" />}
            Ask
          </Button>
        </form>
        <p className="text-xs text-ink-2">{footer}</p>
      </CardBody>
    </Card>
  )
}
