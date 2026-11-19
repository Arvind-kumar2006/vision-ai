import { Eye, Image as ImageIcon, Leaf, Moon, Package, Pill, Sun } from 'lucide-react'
import { useEffect, useState } from 'react'
import About from './pages/About'
import Caption from './pages/Caption'
import FoodLabel from './pages/FoodLabel'
import Home from './pages/Home'
import Plant from './pages/Plant'
import Prescription from './pages/Prescription'

// Hash routes, so a page can be linked and survives a reload (no router needed).
const ROUTES = {
  '#/caption': 'caption',
  '#/prescription': 'prescription',
  '#/product-label': 'label',
  '#/plant': 'plant',
  '#/about': 'about',
}
const pageFromHash = () => ROUTES[window.location.hash] ?? 'home'

const TOOLS = [
  { page: 'caption', href: '#/caption', label: 'Caption a photo', short: 'Caption', icon: ImageIcon },
  { page: 'prescription', href: '#/prescription', label: 'Read a prescription', short: 'Prescription', icon: Pill },
  { page: 'label', href: '#/product-label', label: 'Check a food label', short: 'Food label', icon: Package },
  { page: 'plant', href: '#/plant', label: 'Identify a plant', short: 'Plant', icon: Leaf },
]

function useTheme() {
  const [theme, setTheme] = useState(() => (document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light'))
  useEffect(() => {
    if (theme === 'dark') document.documentElement.dataset.theme = 'dark'
    else delete document.documentElement.dataset.theme
    try {
      localStorage.setItem('theme', theme)
    } catch {
      // storage blocked: the choice lasts for this visit only
    }
  }, [theme])
  return [theme, () => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))]
}

function Header({ page, theme, onToggleTheme }) {
  return (
    <header className="border-b border-hairline bg-card">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 px-4 sm:px-6">
        <a href="#/" className="flex min-h-14 items-center gap-2 text-lg font-semibold text-ink no-underline">
          <span className="grid size-7 place-items-center rounded-[7px] bg-accent text-on-accent">
            <Eye className="size-[18px]" strokeWidth={2} aria-hidden="true" />
          </span>
          Vision AI
        </a>
        <nav
          aria-label="Tools"
          className="order-3 -mx-4 flex w-[calc(100%+2rem)] border-t border-hairline sm:order-none sm:mx-0 sm:w-auto sm:border-t-0"
        >
          {TOOLS.map(({ page: p, href, label, short, icon: Icon }) => {
            const active = p === page
            return (
              <a
                key={p}
                href={href}
                aria-current={active ? 'page' : undefined}
                className={`flex min-h-12 flex-1 items-center justify-center gap-2 border-b-2 px-3 text-sm font-medium transition-colors duration-150 sm:min-h-14 sm:flex-none ${
                  active ? 'border-accent text-ink' : 'border-transparent text-ink-2 hover:text-ink'
                }`}
              >
                <Icon className="size-[18px]" strokeWidth={1.75} aria-hidden="true" />
                <span className="sm:hidden">{short}</span>
                <span className="hidden sm:inline">{label}</span>
              </a>
            )
          })}
        </nav>
        <div className="ml-auto flex items-center gap-1">
          <a
            href="#/about"
            aria-current={page === 'about' ? 'page' : undefined}
            className={`rounded-control px-2 py-2 text-sm ${page === 'about' ? 'text-ink' : 'text-ink-2 hover:text-ink'}`}
          >
            About
          </a>
          <button
            type="button"
            onClick={onToggleTheme}
            aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
            className="grid size-11 place-items-center rounded-control text-ink-2 transition-colors duration-150 hover:bg-raised hover:text-ink"
          >
            {theme === 'dark' ? (
              <Sun className="size-[18px]" strokeWidth={1.75} aria-hidden="true" />
            ) : (
              <Moon className="size-[18px]" strokeWidth={1.75} aria-hidden="true" />
            )}
          </button>
        </div>
      </div>
    </header>
  )
}

function Footer() {
  return (
    <footer className="mx-auto mt-8 flex w-full max-w-6xl flex-wrap gap-x-6 gap-y-1 border-t border-hairline px-4 py-5 text-xs text-ink-2 sm:px-6">
      <span>Vision AI is a student project, not a medical device</span>
      <a href="#/about" className="text-accent underline underline-offset-4">
        About and credits
      </a>
      <span>Additive names from Open Food Facts (ODbL)</span>
    </footer>
  )
}

export default function App() {
  const [page, setPage] = useState(pageFromHash)
  const [theme, toggleTheme] = useTheme()

  useEffect(() => {
    const onHash = () => {
      setPage(pageFromHash())
      window.scrollTo(0, 0)
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  return (
    <div className="flex min-h-dvh flex-col">
      {/* A button, not href="#main": the hash is the router */}
      <button
        type="button"
        onClick={() => document.getElementById('main')?.focus()}
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-2 focus:z-10 focus:rounded-control focus:bg-card focus:px-3 focus:py-2"
      >
        Skip to content
      </button>
      <Header page={page} theme={theme} onToggleTheme={toggleTheme} />
      <main id="main" tabIndex={-1} className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 sm:px-6 sm:py-8">
        {page === 'home' && <Home />}
        {page === 'caption' && <Caption />}
        {page === 'prescription' && <Prescription />}
        {page === 'label' && <FoodLabel />}
        {page === 'plant' && <Plant />}
        {page === 'about' && <About />}
      </main>
      <Footer />
    </div>
  )
}
