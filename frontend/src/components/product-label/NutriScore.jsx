const GRADES = [
  ['A', 'bg-ns-a text-white'],
  ['B', 'bg-ns-b text-[#111827]'],
  ['C', 'bg-ns-c text-[#111827]'],
  ['D', 'bg-ns-d text-[#111827]'],
  ['E', 'bg-ns-e text-white'],
]

/** The official A–E strip: one of the two places with rich colour. The chosen grade is bigger and outlined. */
export default function NutriScoreStrip({ grade }) {
  return (
    <div
      role="img"
      aria-label={grade ? `Nutri-Score ${grade}, on a scale from A (healthiest) to E` : 'Nutri-Score not calculated'}
      className="inline-flex items-center gap-0.5 self-start justify-self-start rounded-[10px] border border-hairline bg-raised p-1"
    >
      {GRADES.map(([g, cls]) => {
        const on = g === grade
        return (
          <span
            key={g}
            className={`grid place-items-center rounded-[5px] font-semibold ${cls} ${
              on ? 'h-14 w-12 text-[1.75rem] outline-2 outline-offset-1 outline-ink' : 'h-10 w-8 text-lg opacity-35'
            }`}
          >
            {g}
          </span>
        )
      })}
    </div>
  )
}
