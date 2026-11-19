import { AlertCircle, AlertTriangle, ChevronDown, Clock, CloudOff, Info } from 'lucide-react'

/** Shared building blocks. Quiet chrome: neutrals, one ink-blue accent, hairlines, no glow. */

const BTN = 'inline-flex min-h-11 items-center justify-center gap-2 rounded-control px-5 text-base font-medium transition-colors duration-150 ease-out disabled:cursor-not-allowed'
const VARIANTS = {
  primary: 'bg-accent text-on-accent hover:bg-accent-hover disabled:bg-raised disabled:text-ink-3',
  secondary: 'border border-control bg-card text-ink hover:bg-raised disabled:border-hairline disabled:text-ink-3',
  text: 'min-h-11 px-1 text-accent underline underline-offset-4 hover:text-accent-hover',
}

export function Button({ variant = 'secondary', className = '', children, ...props }) {
  return (
    <button type="button" className={`${BTN} ${VARIANTS[variant]} ${className}`} {...props}>
      {children}
    </button>
  )
}

export function Card({ as: Tag = 'section', className = '', children, ...props }) {
  return (
    <Tag className={`rounded-card border border-hairline bg-card shadow-card ${className}`} {...props}>
      {children}
    </Tag>
  )
}

export function CardBody({ className = '', children }) {
  return <div className={`space-y-4 p-5 sm:p-6 ${className}`}>{children}</div>
}

/** The signature: where a value or answer came from, in small mono text. */
export function Source({ children, tone = 'neutral', className = '' }) {
  const tones = {
    neutral: 'border-hairline text-ink-3',
    warning: 'border-warning-line bg-warning-soft text-warning',
  }
  return (
    <span className={`inline-flex items-center whitespace-nowrap rounded-tag border px-1.5 font-mono text-xs leading-5 ${tones[tone]} ${className}`}>
      {children}
    </span>
  )
}

const BANNERS = {
  info: { icon: Info, cls: 'border-hairline bg-raised', ic: 'text-accent' },
  warning: { icon: AlertTriangle, cls: 'border-warning-line bg-warning-soft', ic: 'text-warning' },
  error: { icon: AlertCircle, cls: 'border-danger-line bg-danger-soft', ic: 'text-danger' },
  limit: { icon: Clock, cls: 'border-warning-line bg-warning-soft', ic: 'text-warning' },
  offline: { icon: CloudOff, cls: 'border-warning-line bg-warning-soft', ic: 'text-warning' },
}

export function Banner({ kind = 'info', title, children, role = 'status', className = '' }) {
  const b = BANNERS[kind]
  const Icon = b.icon
  return (
    <div role={role} className={`flex gap-3 rounded-control border p-3 text-sm sm:px-4 ${b.cls} ${className}`}>
      <Icon className={`mt-0.5 size-[18px] shrink-0 ${b.ic}`} strokeWidth={1.75} aria-hidden="true" />
      <div className="min-w-0 space-y-1">
        {title && <p className="font-medium text-ink">{title}</p>}
        {children && <div className="text-ink">{children}</div>}
      </div>
    </div>
  )
}

export function Disclosure({ title, hint, icon: Icon, defaultOpen = false, children }) {
  return (
    <details className="group border-t border-hairline" open={defaultOpen}>
      <summary className="flex min-h-12 items-center gap-2 py-2 font-medium">
        {Icon && <Icon className="size-[18px] text-ink-3" strokeWidth={1.75} aria-hidden="true" />}
        <span>{title}</span>
        {hint && <span className="text-sm font-normal text-ink-3">{hint}</span>}
        <ChevronDown className="chev ml-auto size-[18px] text-ink-3" strokeWidth={1.75} aria-hidden="true" />
      </summary>
      <div className="pb-4">{children}</div>
    </details>
  )
}

export function Disclaimer({ children }) {
  return (
    <p className="flex gap-2 rounded-control bg-raised p-3 text-xs text-ink-2">
      <Info className="mt-0.5 size-4 shrink-0" strokeWidth={1.75} aria-hidden="true" />
      <span>{children}</span>
    </p>
  )
}

export function Spinner({ className = 'size-[18px]' }) {
  return (
    <span
      className={`inline-block animate-spin rounded-full border-2 border-current border-r-transparent ${className}`}
      aria-hidden="true"
    />
  )
}

export function Muted({ children, className = '' }) {
  return <p className={`text-sm text-ink-2 ${className}`}>{children}</p>
}
