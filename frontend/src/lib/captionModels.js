/** Captioning models offered by the backend (`?model=`). BLIP is the default (most accurate on our photos). */
const ALL_CAPTION_MODELS = [
  { id: 'sat', name: 'Show, Attend & Tell', hint: 'ResNet-101 + LSTM · COCO 2014 · PRD model' },
  { id: 'catr', name: 'CATR', hint: 'ResNet-101 + Transformer · COCO 2017' },
  { id: 'blip', name: 'BLIP-base', hint: 'ViT + text decoder · larger model · slower first load' },
]

// A deployment can restrict the picker with VITE_CAPTION_MODELS (comma-separated ids,
// set at build time by deploy/deploy_ec2.sh). With no variable, all three are shown.
const allowed = (import.meta.env.VITE_CAPTION_MODELS || '')
  .split(',')
  .map((id) => id.trim())
  .filter(Boolean)

export const CAPTION_MODELS = allowed.length
  ? ALL_CAPTION_MODELS.filter((m) => allowed.includes(m.id))
  : ALL_CAPTION_MODELS

/** SAT has one real attention map per word; CATR and BLIP maps are averaged over heads and layers. */
export function heatmapLabel(modelId) {
  return modelId === 'sat' ? 'Where the model looked' : 'Approximate attention (averaged across heads)'
}

export function modelName(modelId) {
  return CAPTION_MODELS.find((m) => m.id === modelId)?.name ?? modelId ?? 'the model'
}
