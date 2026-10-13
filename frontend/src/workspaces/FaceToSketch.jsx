import React, { useEffect, useRef, useState } from 'react'
import { generateSketch } from '../api'
import { DownloadLink, Dropzone, ImageMount, Notice } from '../components/Panels'

const STYLES = ['Style 1', 'Style 2', 'Style 3']

function useWebcam(active) {
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!active) {
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
      return undefined
    }
    let cancelled = false
    navigator.mediaDevices
      .getUserMedia({ video: { width: 512, height: 512 } })
      .then((stream) => {
        if (cancelled) { stream.getTracks().forEach((t) => t.stop()); return }
        streamRef.current = stream
        if (videoRef.current) videoRef.current.srcObject = stream
      })
      .catch(() => setError('The camera is unavailable. Upload a photograph instead.'))
    return () => {
      cancelled = true
      streamRef.current?.getTracks().forEach((t) => t.stop())
    }
  }, [active])

  return { videoRef, error }
}

export default function FaceToSketch() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [style, setStyle] = useState(0)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [cameraOn, setCameraOn] = useState(false)
  const { videoRef, error: cameraError } = useWebcam(cameraOn)
  const canvasRef = useRef(null)

  function chooseFile(f) {
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
  }

  function capture() {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return
    const size = Math.min(video.videoWidth, video.videoHeight)
    canvas.width = size
    canvas.height = size
    canvas.getContext('2d').drawImage(
      video,
      (video.videoWidth - size) / 2, (video.videoHeight - size) / 2, size, size,
      0, 0, size, size,
    )
    canvas.toBlob((blob) => {
      chooseFile(new File([blob], 'capture.png', { type: 'image/png' }))
      setCameraOn(false)
    }, 'image/png')
  }

  async function run() {
    if (!file) { setError('Choose or capture a photograph first.'); return }
    setBusy(true); setError(''); setResult(null)
    try {
      setResult(await generateSketch(file, style))
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1.5">
        <h2 className="text-xl font-semibold text-mount">Face-to-Sketch Generator</h2>
        <p className="max-w-2xl text-sm leading-relaxed text-muted">
          A conditional GAN trained on paired FS2K photographs and sketches. Pick one of
          the three sketch styles; the style is a learned embedding inside the generator,
          not a label on the interface, so the three options produce genuinely different
          drawings from the same face.
        </p>
      </header>

      <section className="flex flex-col gap-4 rounded border border-rule bg-panel/40 p-4">
        <Dropzone file={file} onFile={chooseFile} hint="or use the camera below" />

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={() => setCameraOn((v) => !v)}
            className="rounded border border-rule px-3 py-1.5 text-sm hover:border-teal hover:text-teal"
          >
            {cameraOn ? 'Close camera' : 'Use camera'}
          </button>
          {cameraOn && (
            <button
              type="button"
              onClick={capture}
              className="rounded bg-amber px-3 py-1.5 text-sm font-medium text-ink"
            >
              Capture photograph
            </button>
          )}
        </div>

        {cameraOn && (
          <video ref={videoRef} autoPlay playsInline muted
                 className="w-full max-w-sm rounded border border-rule" />
        )}
        <canvas ref={canvasRef} className="hidden" />
        <Notice>{cameraError}</Notice>

        <div className="flex flex-col gap-2">
          <span className="text-xs text-muted">Sketch style</span>
          <div className="flex flex-wrap gap-2">
            {STYLES.map((name, index) => (
              <button
                key={name}
                type="button"
                onClick={() => setStyle(index)}
                className={`rounded border px-3 py-1.5 text-sm ${
                  style === index
                    ? 'border-teal bg-teal/15 text-teal'
                    : 'border-rule text-muted hover:border-teal/50'
                }`}
              >
                {name}
              </button>
            ))}
          </div>
        </div>

        <button
          type="button"
          onClick={run}
          disabled={busy}
          className="w-full rounded bg-teal px-4 py-2 text-sm font-medium text-ink disabled:opacity-50 sm:w-48"
        >
          {busy ? 'Generating…' : 'Generate sketch'}
        </button>
        <Notice>{error}</Notice>
      </section>

      <section className="grid gap-5 sm:grid-cols-2">
        <ImageMount label="Photograph" src={result?.input || preview}
                    empty="Upload or capture a face" />
        <ImageMount label="Generated sketch" src={result?.output} empty="—" />
      </section>

      {result && (
        <section className="flex flex-col gap-3">
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded border border-rule bg-panel px-3 py-2">
              <div className="text-xs text-muted">Inference time</div>
              <div className="font-mono text-sm text-teal">{result.inference_ms} ms</div>
            </div>
            <div className="rounded border border-rule bg-panel px-3 py-2">
              <div className="text-xs text-muted">Style condition</div>
              <div className="font-mono text-sm">{result.style}</div>
            </div>
          </div>
          <DownloadLink src={result.output} filename="face-to-sketch.png" />
        </section>
      )}
    </div>
  )
}
