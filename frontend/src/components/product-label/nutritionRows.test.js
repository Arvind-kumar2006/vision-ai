// Run: npm test   (from frontend/)
import assert from 'node:assert/strict'
import test from 'node:test'
import { nutritionRows } from './nutritionRows.js'

const view = (per100) => nutritionRows(per100).map((r) => `${r.child ? '  ' : ''}${r.name}${r.missing ? ' [not read]' : ''}`)

test('sub-rows nest under their parent by key, not under the previous row', () => {
  // the real photo's result: fat missing, fibre present, saturated present
  const rows = view({ energy_kcal: 256, protein_g: 4.4, carbohydrate_g: 41, fibre_g: 8.7, saturated_fat_g: 0.8 })
  assert.deepEqual(rows, ['Energy', 'Fat [not read]', '  of which saturated', 'Carbohydrate', 'Fibre', 'Protein'])
})

test('sugars sit under carbohydrate and saturated under fat when all are present', () => {
  const rows = view({ fat_g: 6.2, saturated_fat_g: 0.8, carbohydrate_g: 41, sugars_g: 5.8, fibre_g: 8.7, salt_g: 0.91 })
  assert.deepEqual(rows, ['Fat', '  of which saturated', 'Carbohydrate', '  of which sugars', 'Fibre', 'Salt'])
})

test('sugars without carbohydrate shows "Carbohydrate: not read"', () => {
  assert.deepEqual(view({ sugars_g: 5 }), ['Carbohydrate [not read]', '  of which sugars'])
})

test('no "not read" parent row when no sub-row is present', () => {
  assert.deepEqual(view({ protein_g: 3 }), ['Protein'])
})
