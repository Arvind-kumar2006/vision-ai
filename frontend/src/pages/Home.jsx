import { ArrowRight, Image as ImageIcon, Leaf, Package, Pill } from 'lucide-react'
import NutriScoreStrip from '../components/product-label/NutriScore'
import { Card, Source } from '../components/ui'

// Real outputs from this project, so the page promises nothing the tools don't do.
const TOOLS = [
  {
    href: '#/product-label',
    icon: Package,
    title: 'Check a food label',
    text: 'Photograph the back of a pack. Get a health score from A to E, the allergens it contains or may contain, and the nutrition table per 100 g.',
    example: 'D: less healthy, mainly sugar',
    source: 'from label',
  },
  {
    href: '#/prescription',
    icon: Pill,
    title: 'Read a prescription',
    text: 'Photograph a prescription. Get each medicine with its dose, how often and for how long, checked against a list of Indian medicines.',
    example: 'Dolo 650 Tablet · Paracetamol 650 mg',
    source: 'from database',
  },
  {
    href: '#/plant',
    icon: Leaf,
    title: 'Identify a plant',
    text: 'Photograph a plant. Get its most likely name with PlantNet’s confidence, and traditional use and safety notes for common herbs.',
    example: 'Ocimum tenuiflorum · Tulsi / Holy basil',
    source: 'from PlantNet',
  },
  {
    href: '#/caption',
    icon: ImageIcon,
    title: 'Caption a photo',
    text: 'Upload any photo. Get a one-line description and see where the model looked for each word.',
    example: 'a dog running with a tennis ball in its mouth',
    source: 'example caption',
  },
]

/** The one rich moment on the page: a real result, built in code (393 kcal label photo, Nutri-Score D, 15 points). */
function ExampleResult() {
  const rows = [
    ['Energy', '393 kcal'],
    ['Sugars', '37.0 g'],
    ['Saturated fat', '0.9 g'],
    ['Salt', '0.34 g'],
  ]
  return (
    <Card className="p-5 shadow-float sm:p-6" aria-label="Example result for a food label">
      <p className="mb-4 flex items-center gap-2 text-sm text-ink-2">
        <Source>from your photo</Source>
        Example result
      </p>
      <div className="grid items-center gap-4 sm:grid-cols-[auto_1fr]">
        <NutriScoreStrip grade="D" />
        <div>
          <p className="text-xl font-medium">D: less healthy, mainly sugar</p>
          <p className="mt-1 text-sm text-ink-2">Mainly because of sugar (37 g per 100 g) and energy (393 kcal per 100 g)</p>
        </div>
      </div>
      <table className="mt-5 w-full border-collapse text-sm">
        <caption className="sr-only">Nutrition per 100 g in the example</caption>
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k}>
              <th scope="row" className="border-t border-hairline py-2 text-left font-normal text-ink-2">
                {k}
              </th>
              <td className="border-t border-hairline py-2 pr-3 text-right">{v}</td>
              <td className="w-px border-t border-hairline py-2 text-right">
                <Source>from label</Source>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  )
}

export default function Home() {
  return (
    <div className="space-y-12 sm:space-y-16">
      <section className="grid items-center gap-8 pt-2 lg:grid-cols-[minmax(0,1fr)_minmax(0,460px)] lg:gap-12 lg:pt-8">
        <div className="space-y-5">
          <h1 className="text-[2.25rem] leading-[1.12] font-medium tracking-[-0.02em] sm:text-3xl">
            Plain answers from a photo of a food pack, a prescription, a plant or anything else
          </h1>
          <p className="max-w-xl text-lg text-ink-2">
            Check how healthy a packaged food is, read the medicines on a prescription, identify a plant, or get a photo described. Every
            value shows where it came from.
          </p>
          <div className="flex flex-wrap items-center gap-3">
            <a
              href="#/product-label"
              className="inline-flex min-h-11 items-center gap-2 rounded-control bg-accent px-5 font-medium text-on-accent transition-colors duration-150 hover:bg-accent-hover"
            >
              Check a food label
              <ArrowRight className="size-[18px]" strokeWidth={1.75} aria-hidden="true" />
            </a>
            <a href="#/about" className="min-h-11 px-1 py-2.5 text-accent underline underline-offset-4">
              How it works
            </a>
          </div>
        </div>
        <ExampleResult />
      </section>

      <section aria-labelledby="tools-h" className="space-y-4">
        <h2 id="tools-h" className="text-lg font-medium">
          Four tools
        </h2>
        <ul className="divide-y divide-hairline overflow-hidden rounded-card border border-hairline bg-card shadow-card">
          {TOOLS.map(({ href, icon: Icon, title, text, example, source }) => (
            <li key={href}>
              <a href={href} className="group grid gap-3 p-5 transition-colors duration-150 hover:bg-raised sm:grid-cols-[auto_1fr_auto] sm:items-center sm:gap-6 sm:p-6">
                <Icon className="size-6 text-ink-3" strokeWidth={1.5} aria-hidden="true" />
                <div className="space-y-1">
                  <p className="text-lg font-medium">{title}</p>
                  <p className="text-sm text-ink-2">{text}</p>
                  <p className="flex flex-wrap items-center gap-2 pt-1 text-sm">
                    <span className="font-mono text-ink">{example}</span>
                    <Source>{source}</Source>
                  </p>
                </div>
                <ArrowRight className="hidden size-5 text-ink-3 transition-colors group-hover:text-accent sm:block" strokeWidth={1.75} aria-hidden="true" />
              </a>
            </li>
          ))}
        </ul>
      </section>
    </div>
  )
}
