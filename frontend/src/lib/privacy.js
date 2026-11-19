import { getHealth } from '../api/client'

// One /api/health request per page load; every privacy note shares it.
let healthPromise = null
export const loadHealth = () => (healthPromise ??= getHealth().catch(() => null))

/** Where each feature's data goes, with the real AI provider names from the server. */
export function privacyText(feature, ai) {
  if (feature === 'caption') return 'Read on this device. Your photo isn’t sent to any other service or kept.'
  const providers = ai?.providers ?? []
  const via = providers.map((p) => `${p.provider} (${p.model})`).join(', or if busy ')
  if (feature === 'plant') {
    const names = providers.length ? ` Only the plant’s name is sent to ${via} to describe its medicinal uses.` : ''
    return `Your photos are sent to PlantNet (Pl@ntNet, France) to identify the plant.${names} Nothing is kept on this server.`
  }
  if (!providers.length) return 'Read on this computer only. No AI service is set up, so nothing leaves it.'
  const free = providers.some((p) => p.provider === 'OpenRouter')
    ? ' OpenRouter’s free models may use what they receive to improve their models.'
    : ''
  if (feature === 'prescription')
    return `The photo is read on this computer and never sent anywhere. Only the medicine lines (no patient, doctor or clinic details) are sent as text to ${via}.${free}`
  return `The photo is read on this computer and never sent anywhere. Only if a printed value couldn’t be read is the label’s text sent to ${via}; explanations send the values and any product name you typed. Results are kept for up to an hour for the explanation.${free}`
}
