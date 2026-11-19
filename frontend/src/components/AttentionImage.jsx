import { useCallback, useEffect, useRef } from 'react'

/**
 * A photo with the model's attention for one word painted on top: one of the two places with rich colour.
 * `grid` is rows x cols scaled 0-1 by the backend (14x14 SAT, 24x24 BLIP, image-shaped CATR), or null.
 * Low attention is dimmed, high attention runs yellow to red. The <img> box equals the visible picture,
 * so the grid lines up exactly.
 */
function heat(v) {
  // 0 -> dim (black, 45 %), 0.5 -> yellow (35 %), 1 -> red-orange (62 %)
  if (v < 0.5) {
    const t = v / 0.5
    return [Math.round(255 * t), Math.round(214 * t), 0, Math.round(255 * (0.45 - 0.1 * t))]
  }
  const t = (v - 0.5) / 0.5
  return [Math.round(255 - 25 * t), Math.round(214 - 152 * t), Math.round(17 * t), Math.round(255 * (0.35 + 0.27 * t))]
}

export default function AttentionImage({ src, grid, alt }) {
  const imgRef = useRef(null)
  const canvasRef = useRef(null)

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img) return
    const w = img.clientWidth
    const h = img.clientHeight
    const dpr = window.devicePixelRatio || 1
    canvas.width = Math.max(1, Math.round(w * dpr))
    canvas.height = Math.max(1, Math.round(h * dpr))
    canvas.style.width = `${w}px`
    canvas.style.height = `${h}px`
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    if (!grid || !grid.length) return
    const rows = grid.length
    const cols = grid[0]?.length ?? 0
    if (!cols) return
    const small = document.createElement('canvas')
    small.width = cols
    small.height = rows
    const sctx = small.getContext('2d')
    const pixels = sctx.createImageData(cols, rows)
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const i = (y * cols + x) * 4
        const [r, g, b, a] = heat(Math.max(0, Math.min(1, grid[y][x])))
        pixels.data[i] = r
        pixels.data[i + 1] = g
        pixels.data[i + 2] = b
        pixels.data[i + 3] = a
      }
    }
    sctx.putImageData(pixels, 0, 0)
    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(small, 0, 0, canvas.width, canvas.height)
  }, [grid])

  useEffect(() => {
    draw()
    const img = imgRef.current
    if (!img || typeof ResizeObserver === 'undefined') return undefined
    const observer = new ResizeObserver(draw)
    observer.observe(img)
    return () => observer.disconnect()
  }, [draw])

  return (
    <div className="relative mx-auto w-fit max-w-full">
      <img ref={imgRef} src={src} alt={alt} onLoad={draw} className="block h-auto max-h-[520px] w-auto max-w-full" />
      <canvas ref={canvasRef} aria-hidden="true" className="pointer-events-none absolute left-0 top-0" />
    </div>
  )
}
