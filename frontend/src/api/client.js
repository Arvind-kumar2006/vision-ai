const API_BASE = '/api'

async function request(path, options) {
  const res = await fetch(`${API_BASE}${path}`, options)
  if (!res.ok) {
    let detail = res.statusText
    try {
      const data = await res.json()
      detail = data.detail ?? detail
    } catch {
      // keep the status text
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return res.json()
}

export function getHealth() {
  return request('/health')
}

export function captionImage(file, { model, maxLen } = {}) {
  const form = new FormData()
  form.append('file', file)
  const params = new URLSearchParams()
  if (model) params.set('model', model)
  if (maxLen) params.set('max_len', String(maxLen))
  const query = params.toString() ? `?${params}` : ''
  return request(`/caption${query}`, { method: 'POST', body: form })
}

export function analyzePrescription(file) {
  const form = new FormData()
  form.append('file', file)
  return request('/prescription', { method: 'POST', body: form })
}

// ---- Product Label Interpreter -------------------------------------------
export function interpretProductLabel({ file, productName } = {}) {
  const form = new FormData()
  form.append('image', file)
  if (productName?.trim()) form.append('product_name', productName.trim())
  return request('/product-label/interpret', { method: 'POST', body: form })
}

export function getProductExplanation(sessionId, { signal } = {}) {
  return request(`/product-label/${encodeURIComponent(sessionId)}/explanation`, { signal })
}


// ---- Plant identification (PlantNet) -------------------------------------
/** 1-5 photos of the same plant; ``organ`` = auto | leaf | flower | fruit | bark. */
export function identifyPlant(files, organ = 'auto') {
  const form = new FormData()
  files.forEach((f) => form.append('images', f))
  form.append('organ', organ)
  return request('/plant/identify', { method: 'POST', body: form })
}

/** Medicinal uses written by the AI from the plant's name only (no photo); ``lang`` = en | hi | kn. */
export function getPlantUses(scientificName, commonNames = [], { lang = 'en', signal } = {}) {
  return request('/plant/uses', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scientific_name: scientificName, common_names: commonNames, lang }),
    signal,
  })
}
