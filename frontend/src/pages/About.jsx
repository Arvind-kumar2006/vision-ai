import { useEffect, useState } from 'react'
import { getHealth } from '../api/client'
import { loadHealth, privacyText } from '../lib/privacy'
import { CAPTION_MODELS } from '../lib/captionModels'
import { Card, CardBody, Disclosure } from '../components/ui'

const MODEL_CREDITS = {
  sat: {
    by: 'Pretrained checkpoint from sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning by Sagar Vinodababu',
    licence: 'MIT',
    url: 'https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning',
    note: 'The model described in the project’s PRD: one attention map per word.',
  },
  catr: {
    by: 'saahiluppal/catr (parts from Facebook’s DETR)',
    licence: 'Apache-2.0',
    url: 'https://github.com/saahiluppal/catr',
    note: 'For comparison. Its heatmap is approximate (averaged across attention heads).',
  },
  blip: {
    by: 'Salesforce BLIP, loaded through Hugging Face transformers',
    licence: 'BSD-3-Clause',
    url: 'https://github.com/salesforce/BLIP',
    note: 'Default and most accurate on our test photos. Its heatmap is approximate (averaged across heads).',
  },
}

const DATA = [
  ['MS-COCO 2014 and 2017', 'Training data of the three caption models', 'Annotations CC BY 4.0', 'https://cocodataset.org'],
  ['Indian Medicine Dataset', 'Medicine names, compositions, side effects and interactions (about 240,000 medicines)', 'MIT', null],
  ['250k Medicines Usage, Side Effects and Substitutes (Shudhanshu Singh)', '“What it’s for”, other uses and common side effects, by ingredient', 'CC BY-SA 4.0', 'https://www.kaggle.com/datasets/shudhanshusingh/250k-medicines-usage-side-effects-and-substitutes'],
  ['Open Food Facts additives taxonomy', 'Names and classes of food additives (E numbers / INS)', 'ODbL', 'https://world.openfoodfacts.org'],
  ['PaddleOCR models (via RapidOCR)', 'Reading the text in prescription and label photos, on this computer', 'Apache-2.0', 'https://github.com/RapidAI/RapidOCR'],
  ['Pl@ntNet API', 'Identifying plants from photos (the photos are sent to Pl@ntNet)', 'Pl@ntNet API terms', 'https://my.plantnet.org'],
  ['Nutri-Score (2023 update)', 'Health score thresholds, from Santé publique France', 'Official specification', null],
  ['openfoodfacts/nutrient-detection-layout', 'Nutrition-table photos used only to measure accuracy', 'CC BY-SA 3.0', 'https://huggingface.co/datasets/openfoodfacts/nutrient-detection-layout'],
]

const FEATURES = [
  ['caption', 'Caption a photo'],
  ['prescription', 'Read a prescription'],
  ['label', 'Check a food label'],
  ['plant', 'Identify a plant'],
]

function WhereDataGoes() {
  const [ai, setAi] = useState(undefined)
  useEffect(() => {
    let live = true
    loadHealth().then((h) => live && setAi(h?.ai ?? null))
    return () => {
      live = false
    }
  }, [])
  return (
    <dl className="space-y-3 text-sm">
      {FEATURES.map(([k, title]) => (
        <div key={k}>
          <dt className="font-medium">{title}</dt>
          <dd className="text-ink-2">{k === 'caption' || ai !== undefined ? privacyText(k, ai) : 'Checking…'}</dd>
        </div>
      ))}
    </dl>
  )
}

function ServiceStatus() {
  const [health, setHealth] = useState(undefined)
  useEffect(() => {
    let live = true
    getHealth()
      .then((h) => live && setHealth(h))
      .catch(() => live && setHealth(null))
    return () => {
      live = false
    }
  }, [])
  if (health === undefined) return <p className="text-sm text-ink-2">Checking…</p>
  if (health === null) return <p className="text-sm text-danger">The server isn’t responding.</p>
  const rows = [
    ['Server', health.status],
    ['Device', health.device],
    ['Default caption model', health.caption_model],
    ...Object.entries(health.caption_models || {}).map(([k, v]) => [`Caption model ${k}`, v]),
    ...Object.entries(health.modes || {}).map(([k, v]) => [`Mode ${k}`, v]),
  ]
  return (
    <dl className="grid grid-cols-[max-content_1fr] gap-x-6 gap-y-1 font-mono text-xs">
      {rows.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-ink-2">{k}</dt>
          <dd className="break-words">{String(v)}</dd>
        </div>
      ))}
    </dl>
  )
}

export default function About() {
  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div className="space-y-2">
        <h1 className="text-2xl font-medium">About and credits</h1>
        <p className="text-ink-2">
          Vision AI is a student project. It reads photos to answer one question: what is this, and what should I know?
          It is not a medical device, and every answer shows where it came from.
        </p>
      </div>

      <Card aria-labelledby="models-h">
        <CardBody>
          <h2 id="models-h" className="text-lg font-medium">
            Caption models
          </h2>
          <ul className="divide-y divide-hairline">
            {CAPTION_MODELS.map((m) => {
              const c = MODEL_CREDITS[m.id]
              return (
                <li key={m.id} className="space-y-1 py-3 first:pt-0 last:pb-0">
                  <p className="font-medium">{m.name}</p>
                  <p className="text-sm text-ink-2">{m.hint}</p>
                  <p className="text-sm text-ink-2">{c.note}</p>
                  <p className="text-sm">
                    <a className="text-accent underline underline-offset-4" href={c.url} target="_blank" rel="noreferrer">
                      {c.by}
                    </a>{' '}
                    <span className="text-ink-2">· {c.licence}</span>
                  </p>
                </li>
              )
            })}
          </ul>
        </CardBody>
      </Card>

      <Card aria-labelledby="data-h">
        <CardBody>
          <h2 id="data-h" className="text-lg font-medium">
            Data and licences
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[520px] border-collapse text-sm">
              <thead>
                <tr className="text-xs text-ink-2">
                  <th scope="col" className="border-b border-control pb-2 text-left font-medium">Source</th>
                  <th scope="col" className="border-b border-control pb-2 text-left font-medium">Used for</th>
                  <th scope="col" className="border-b border-control pb-2 text-left font-medium">Licence</th>
                </tr>
              </thead>
              <tbody>
                {DATA.map(([name, use, licence, url]) => (
                  <tr key={name}>
                    <th scope="row" className="border-b border-hairline py-2.5 pr-4 text-left align-top font-medium">
                      {url ? (
                        <a className="text-accent underline underline-offset-4" href={url} target="_blank" rel="noreferrer">
                          {name}
                        </a>
                      ) : (
                        name
                      )}
                    </th>
                    <td className="border-b border-hairline py-2.5 pr-4 align-top text-ink-2">{use}</td>
                    <td className="border-b border-hairline py-2.5 align-top whitespace-nowrap">{licence}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardBody>
      </Card>

      <Card aria-labelledby="eval-h">
        <CardBody>
          <h2 id="eval-h" className="text-lg font-medium">
            How well it works
          </h2>
          <dl className="space-y-4 text-sm">
            <div>
              <dt className="font-medium">Food labels</dt>
              <dd className="text-ink-2">
                On 16 checked nutrition-table photos, 106 of 112 values were within 5 % of the printed value. 11 of the 16
                photos were read fully correctly. A test on Indian label photos is still to come.
              </dd>
            </div>
            <div>
              <dt className="font-medium">Captions</dt>
              <dd className="text-ink-2">
                Show, Attend &amp; Tell scores BLEU-4 ≈ 0.33 on COCO, as reported by its authors (not re-measured here). On
                four of our own photos, Show, Attend &amp; Tell got 2 right, CATR 0 and BLIP-base 3. That is a small sample,
                not a benchmark.
              </dd>
            </div>
            <div>
              <dt className="font-medium">Prescriptions</dt>
              <dd className="text-ink-2">
                Checked on one real prescription only: all 3 medicines and the medicine already being taken were found, and
                5 of 8 doses and timings were right. Handwriting is often misread, which is why uncertain values are marked
                “check” and uncertain rows Verify or Not found.
              </dd>
            </div>
          </dl>
        </CardBody>
      </Card>

      <Card aria-labelledby="privacy-h">
        <CardBody>
          <h2 id="privacy-h" className="text-lg font-medium">
            Where your photos go
          </h2>
          <WhereDataGoes />
        </CardBody>
      </Card>

      <Card className="px-5 sm:px-6">
        <Disclosure title="Service status" hint="for troubleshooting">
          <ServiceStatus />
        </Disclosure>
      </Card>
    </div>
  )
}
