import React, { useState } from 'react'
import { restore } from '../api'
import { Choice, DownloadLink, Dropzone, ImageMount, Notice } from './Panels'

const CORRUPTIONS = [
  { value: 'none', label: 'Use image as uploaded' },
  { value: 'salt_pepper', label: 'Salt & pepper noise' },
  { value: 'blur', label: 'Gaussian blur' },
  { value: 'occlusion', label: 'Rectangular occlusion' },
]

const SEVERITIES = [
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High' },
]

/**
 * Shared shell for Tasks 1-3: the upload, the corruption controls and the
 * before/after mount are identical, only the diagnostics below differ.
 */
export default function RestorationWorkspace({ title, description, endpoint,
                                               filename, renderDiagnostics }) {
  const [file, setFile] = useState(null)
  const [corruption, setCorruption] = useState('salt_pepper')
  const [severity, setSeverity] = useState('medium')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  async function run() {
    if (!file) { setError('Choose an image first.'); return }
    setBusy(true); setError(''); setResult(null)
    try {
      setResult(await restore(endpoint, file, { corruption, severity, seed: 42 }))
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const applied = result?.applied_corruption

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1.5">
        <h2 className="text-xl font-semibold text-mount">{title}</h2>
        <p className="max-w-2xl text-sm leading-relaxed text-muted">{description}</p>
      </header>

      <section className="flex flex-col gap-4 rounded border border-rule bg-panel/40 p-4">
        <Dropzone file={file} onFile={setFile} />
        <div className="grid gap-3 sm:grid-cols-3">
          <Choice label="Corruption to apply" value={corruption}
                  options={CORRUPTIONS} onChange={setCorruption} />
          <Choice label="Severity" value={severity} options={SEVERITIES}
                  onChange={setSeverity} />
          <div className="flex items-end">
            <button
              type="button"
              onClick={run}
              disabled={busy}
              className="w-full rounded bg-teal px-4 py-2 text-sm font-medium text-ink disabled:opacity-50"
            >
              {busy ? 'Restoring…' : 'Restore image'}
            </button>
          </div>
        </div>
        <Notice>{error}</Notice>
      </section>

      <section className="grid gap-5 sm:grid-cols-3">
        <ImageMount label="Model input" src={result?.input}
                    empty="Upload an image and run the model" />
        <ImageMount label="Restored output" src={result?.output} empty="—" />
        <ImageMount label="Difference from input" src={result?.error_map} empty="—" />
      </section>

      {result && (
        <section className="flex flex-col gap-4">
          {renderDiagnostics?.(result)}
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded border border-rule bg-panel px-3 py-2">
              <div className="text-xs text-muted">Inference time</div>
              <div className="font-mono text-sm text-teal">{result.inference_ms} ms</div>
            </div>
            <div className="rounded border border-rule bg-panel px-3 py-2">
              <div className="text-xs text-muted">Corruption applied</div>
              <div className="font-mono text-xs leading-relaxed">
                {applied?.kind}
                {applied?.severity ? ` · ${applied.severity}` : ''}
                {applied?.probability !== undefined ? ` · p=${applied.probability}` : ''}
                {applied?.kernel ? ` · k=${applied.kernel}, σ=${applied.sigma}` : ''}
                {applied?.target_area ? ` · ${Math.round(applied.target_area * 100)}% masked` : ''}
              </div>
            </div>
          </div>
          <DownloadLink src={result.output} filename={filename} />
        </section>
      )}
    </div>
  )
}
