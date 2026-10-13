import React, { useCallback, useRef, useState } from 'react'

export function ImageMount({ label, src, empty = 'Nothing here yet' }) {
  return (
    <figure className="flex flex-col gap-2">
      <figcaption className="text-xs text-muted">{label}</figcaption>
      <div className="aspect-square w-full overflow-hidden rounded border border-rule bg-panel">
        {src ? (
          <img src={src} alt={label} className="h-full w-full object-contain" />
        ) : (
          <div className="flex h-full items-center justify-center px-4 text-center text-xs text-muted">
            {empty}
          </div>
        )}
      </div>
    </figure>
  )
}

export function Dropzone({ file, onFile, hint }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  const accept = useCallback(
    (candidate) => {
      if (!candidate) return
      if (!candidate.type.startsWith('image/')) return
      onFile(candidate)
    },
    [onFile],
  )

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => { e.preventDefault(); setDragging(false); accept(e.dataTransfer.files[0]) }}
      className={`rounded border border-dashed p-4 transition-colors ${
        dragging ? 'border-teal bg-panel' : 'border-rule'
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => accept(e.target.files[0])}
      />
      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="rounded bg-teal px-3 py-1.5 text-sm font-medium text-ink hover:bg-teal/90"
        >
          Choose image
        </button>
        <span className="text-xs text-muted">
          {file ? file.name : hint || 'or drop a file here'}
        </span>
      </div>
    </div>
  )
}

export function Choice({ label, value, options, onChange }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-xs text-muted">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded border border-rule bg-panel px-2.5 py-1.5 text-sm text-mount"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </label>
  )
}

export function Readout({ label, value, tone = 'default' }) {
  const tones = { default: 'text-mount', teal: 'text-teal', amber: 'text-amber' }
  return (
    <div className="rounded border border-rule bg-panel px-3 py-2">
      <div className="text-xs text-muted">{label}</div>
      <div className={`font-mono text-sm ${tones[tone]}`}>{value}</div>
    </div>
  )
}

export function WeightBar({ label, value, highlight = false }) {
  const pct = Math.round(value * 1000) / 10
  return (
    <div className="flex items-center gap-3">
      <span className="w-44 shrink-0 text-xs text-muted">{label}</span>
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-rule">
        <div
          className={`h-full rounded-full ${highlight ? 'bg-amber' : 'bg-teal'}`}
          style={{ width: `${Math.max(pct, 0.5)}%` }}
        />
      </div>
      <span className="w-14 shrink-0 text-right font-mono text-xs">{pct.toFixed(1)}%</span>
    </div>
  )
}

export function Notice({ kind = 'error', children }) {
  if (!children) return null
  const styles = {
    error: 'border-alert/50 bg-alert/10 text-alert',
    info: 'border-rule bg-panel text-muted',
  }
  return (
    <div className={`rounded border px-3 py-2 text-sm ${styles[kind]}`}>{children}</div>
  )
}

export function DownloadLink({ src, filename }) {
  if (!src) return null
  return (
    <a
      href={src}
      download={filename}
      className="inline-block rounded border border-rule px-3 py-1.5 text-sm hover:border-teal hover:text-teal"
    >
      Download result
    </a>
  )
}
