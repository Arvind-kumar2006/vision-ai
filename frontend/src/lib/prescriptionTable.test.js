import assert from 'node:assert/strict'
import { test } from 'node:test'
import { visibleColumns } from './prescriptionTable.js'

test('identification columns: contents and strength only when some row has them', () => {
  const meds = [
    { strength: '650mg', candidates: [{}] },
    { strength: null, candidates: [] },
  ]
  assert.deepEqual(visibleColumns(meds), ['medicine', 'contents', 'strength', 'status'])
  assert.deepEqual(visibleColumns([{ name: 'x' }, { name: 'y' }]), ['medicine', 'status'])
  assert.deepEqual(visibleColumns([{ name: 'Colgate', non_medicine: true }]), ['medicine', 'contents', 'status'])
  for (const gone of ['dose', 'frequency', 'duration', 'instructions']) assert.ok(!visibleColumns(meds).includes(gone))
})
