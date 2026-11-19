import { Camera, ImageUp, RefreshCw, X } from 'lucide-react'
import { useRef, useState } from 'react'
import { Button } from './ui'

/**
 * Empty: a drop zone with "Take or upload a photo" (on phones the button opens the camera or photos).
 * With a photo: `children` renders it (plain image or attention view) with Change / Remove.
 */
export default function PhotoPicker({ hasPhoto, onFile, onClear, children, disabled = false, label = 'Take or upload a photo' }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  function pick(files) {
    const file = files?.[0]
    if (file) onFile(file)
    if (inputRef.current) inputRef.current.value = '' // choosing the same file again still fires
  }

  const input = (
    <input
      ref={inputRef}
      type="file"
      accept="image/*"
      className="sr-only"
      tabIndex={-1}
      aria-hidden="true"
      onChange={(e) => pick(e.target.files)}
    />
  )

  if (hasPhoto) {
    return (
      <div className="space-y-3">
        <div className="overflow-hidden rounded-control border border-hairline bg-raised">{children}</div>
        <div className="flex gap-2">
          <Button className="flex-1" onClick={() => inputRef.current?.click()} disabled={disabled}>
            <RefreshCw className="size-4" strokeWidth={1.75} aria-hidden="true" />
            Change
          </Button>
          <Button className="flex-1" onClick={onClear} disabled={disabled}>
            <X className="size-4" strokeWidth={1.75} aria-hidden="true" />
            Remove
          </Button>
        </div>
        {input}
      </div>
    )
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        pick(e.dataTransfer.files)
      }}
      className={`grid justify-items-center gap-3 rounded-control border-2 border-dashed px-5 py-8 text-center transition-colors duration-150 ${
        dragging ? 'border-accent bg-accent-soft' : 'border-control bg-card'
      }`}
    >
      <ImageUp className="size-7 text-ink-3" strokeWidth={1.5} aria-hidden="true" />
      <p className="hidden text-ink-2 sm:block">Drag a photo here, or</p>
      <Button variant="primary" onClick={() => inputRef.current?.click()} disabled={disabled}>
        <Camera className="size-[18px]" strokeWidth={1.75} aria-hidden="true" />
        {label}
      </Button>
      <p className="text-xs text-ink-3">JPG, PNG or WebP, up to 10 MB</p>
      {input}
    </div>
  )
}
