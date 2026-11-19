/** Table layout rules for the prescription result (pure functions, tested in prescriptionTable.test.js). */

/** Which columns to show. Medicine and Status always; What's in it and Strength only when some row has one. */
export function visibleColumns(medicines = []) {
  const any = (f) => medicines.some(f)
  const cols = ['medicine']
  if (any((m) => m.db_info?.composition || m.known_brand?.composition || m.candidates?.length || m.non_medicine)) cols.push('contents')
  if (any((m) => m.strength)) cols.push('strength')
  cols.push('status')
  return cols
}

export const LEVELS = {
  high: 'High confidence',
  medium: 'Medium confidence',
  low: 'Low confidence',
}
