// Display order (EU declaration order). Sub-rows ("of which …") are nested under their parent BY KEY,
// never by whichever row happens to come before them. A sub-row whose parent wasn't read still gets
// its parent row, shown as "not read", so "of which saturated" can never appear under Fibre.
const GROUPS = [
  { key: 'energy_kcal', name: 'Energy', unit: 'kcal' },
  { key: 'energy_kj', name: 'Energy', unit: 'kJ' },
  {
    key: 'fat_g', name: 'Fat', unit: 'g',
    children: [
      ['saturated_fat_g', 'of which saturated', 'g'],
      ['trans_fat_g', 'of which trans', 'g'],
    ],
  },
  {
    key: 'carbohydrate_g', name: 'Carbohydrate', unit: 'g',
    children: [
      ['sugars_g', 'of which sugars', 'g'],
      ['added_sugars_g', 'of which added sugars', 'g'],
    ],
  },
  { key: 'fibre_g', name: 'Fibre', unit: 'g' },
  { key: 'protein_g', name: 'Protein', unit: 'g' },
  { key: 'salt_g', name: 'Salt', unit: 'g' },
  { key: 'sodium_mg', name: 'Sodium', unit: 'mg' },
  { key: 'cholesterol_mg', name: 'Cholesterol', unit: 'mg' },
]

const has = (per100, k) => per100?.[k] !== undefined && per100?.[k] !== null

/** Rows to render: {key, name, unit, child, missing}. Tested in nutritionRows.test.js (npm test). */
export function nutritionRows(per100) {
  const rows = []
  const known = new Set()
  for (const g of GROUPS) {
    known.add(g.key)
    const kids = (g.children || []).filter(([k]) => {
      known.add(k)
      return has(per100, k)
    })
    if (has(per100, g.key) || kids.length) {
      rows.push({ key: g.key, name: g.name, unit: g.unit, child: false, missing: !has(per100, g.key) })
      for (const [k, name, unit] of kids) rows.push({ key: k, name, unit, child: true, missing: false })
    }
  }
  for (const k of Object.keys(per100 || {})) {
    if (!known.has(k) && has(per100, k)) {
      rows.push({
        key: k, name: k.replace(/_(g|mg|kcal|kj)$/, '').replace(/_/g, ' '), unit: k.split('_').pop(),
        child: false, missing: false,
      })
    }
  }
  return rows
}

