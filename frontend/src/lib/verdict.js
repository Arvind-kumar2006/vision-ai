/** Plain-language verdict from the Nutri-Score result: "D: less healthy, mainly sugar" + the reason. */
const GRADE_WORDS = {
  A: 'a healthier choice',
  B: 'a healthier choice',
  C: 'middle of the range',
  D: 'less healthy',
  E: 'least healthy',
}
const NAMES = {
  energy: 'energy',
  sugars: 'sugar',
  saturated_fat: 'saturated fat',
  saturates_to_lipids: 'saturated fat',
  salt: 'salt',
  non_nutritive_sweeteners: 'sweeteners',
  protein: 'protein',
  fibre: 'fibre',
  fruit_veg_legumes: 'fruit and vegetables',
}

const num = (v) => (typeof v === 'number' ? Number(v.toFixed(1)).toString() : String(v))

function amount(b, per100) {
  if (b.component === 'energy') {
    return per100?.energy_kcal != null ? `${num(per100.energy_kcal)} kcal` : `${num(b.value)} ${b.unit}`
  }
  return b.value == null ? null : `${num(b.value)} ${b.unit}`
}

export function verdict(nutriscore, per100) {
  const ns = nutriscore || {}
  if (!ns.grade) return null
  const breakdown = ns.breakdown ?? []
  const bad = breakdown.filter((b) => b.kind === 'negative' && b.points > 0).sort((a, b) => b.points - a.points)
  const good = breakdown.filter((b) => b.kind === 'positive' && b.points > 0).sort((a, b) => b.points - a.points)
  const lead = ns.grade <= 'B' ? good : bad
  const main = lead[0] ? NAMES[lead[0].component] : null
  const title = `${ns.grade}: ${GRADE_WORDS[ns.grade]}${main && ns.grade > 'B' ? `, mainly ${main}` : ''}`
  const parts = lead.slice(0, 2).map((b) => {
    const a = amount(b, per100)
    return a ? `${NAMES[b.component]} (${a} per 100 g)` : NAMES[b.component]
  })
  const reason = parts.length
    ? `${ns.grade <= 'B' ? 'Helped by' : 'Mainly because of'} ${parts.join(' and ')}`
    : null
  return { grade: ns.grade, title, reason }
}

// Where each value came from: read from the photo by the OCR, or read by the AI from the OCR text.
export const SOURCE_TAGS = {
  read_on_device: 'read from photo',
  ai_from_text: 'AI read the text',
}
