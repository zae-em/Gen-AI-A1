const BASE = import.meta.env.VITE_API_BASE || '/api'

async function post(path, formData) {
  const response = await fetch(`${BASE}${path}`, { method: 'POST', body: formData })
  if (!response.ok) {
    let message = `Request failed (${response.status})`
    try {
      const body = await response.json()
      if (body.detail) message = body.detail
    } catch { /* response had no JSON body */ }
    throw new Error(message)
  }
  return response.json()
}

export function health() {
  return fetch(`${BASE}/health`).then((r) => {
    if (!r.ok) throw new Error('Backend is not responding')
    return r.json()
  })
}

export function restore(path, file, { corruption, severity, seed } = {}) {
  const form = new FormData()
  form.append('file', file)
  if (corruption) form.append('corruption', corruption)
  if (severity) form.append('severity', severity)
  if (seed !== undefined) form.append('seed', String(seed))
  return post(path, form)
}

export function generateSketch(file, style) {
  const form = new FormData()
  form.append('file', file)
  form.append('style', String(style))
  return post('/face-to-sketch', form)
}
