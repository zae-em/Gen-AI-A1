import React, { useEffect, useState } from 'react'
import { health } from './api'
import UniversalRestoration from './workspaces/UniversalRestoration'
import HardRouted from './workspaces/HardRouted'
import SoftMoE from './workspaces/SoftMoE'
import FaceToSketch from './workspaces/FaceToSketch'

const WORKSPACES = [
  { id: 'universal', name: 'Universal Restoration', Component: UniversalRestoration },
  { id: 'hard', name: 'Hard-Routed Restoration', Component: HardRouted },
  { id: 'soft', name: 'Soft Mixture-of-Experts', Component: SoftMoE },
  { id: 'sketch', name: 'Face-to-Sketch Generator', Component: FaceToSketch },
]

function StatusDot({ status }) {
  const colour = { ok: 'bg-teal', degraded: 'bg-amber', down: 'bg-alert' }[status]
  return <span className={`inline-block h-2 w-2 rounded-full ${colour}`} />
}

export default function App() {
  const [active, setActive] = useState('universal')
  const [status, setStatus] = useState('down')
  const [models, setModels] = useState({})

  useEffect(() => {
    health()
      .then((data) => { setStatus(data.status); setModels(data.models) })
      .catch(() => setStatus('down'))
  }, [])

  const missing = Object.entries(models).filter(([, present]) => !present).map(([k]) => k)
  const Active = WORKSPACES.find((w) => w.id === active).Component

  return (
    <div className="min-h-screen lg:flex">
      <aside className="border-b border-rule bg-panel/50 lg:flex lg:h-screen lg:w-64 lg:shrink-0 lg:flex-col lg:border-b-0 lg:border-r">
        <div className="flex flex-col gap-1 border-b border-rule px-5 py-5">
          <h1 className="text-base font-semibold tracking-tight text-mount">
            Restoration Bench
          </h1>
          <p className="text-xs text-muted">Four models, one interface</p>
        </div>

        <nav className="flex gap-1 overflow-x-auto p-3 lg:flex-col lg:overflow-visible">
          {WORKSPACES.map((w) => (
            <button
              key={w.id}
              type="button"
              onClick={() => setActive(w.id)}
              className={`whitespace-nowrap rounded px-3 py-2 text-left text-sm lg:whitespace-normal ${
                active === w.id
                  ? 'bg-teal/15 text-teal'
                  : 'text-muted hover:bg-panel hover:text-mount'
              }`}
            >
              {w.name}
            </button>
          ))}
        </nav>

        <div className="border-t border-rule px-5 py-4 text-xs text-muted lg:mt-auto">
          <div className="flex items-center gap-2">
            <StatusDot status={status} />
            <span>
              {status === 'ok' && 'All models loaded'}
              {status === 'degraded' && `${missing.length} model(s) missing`}
              {status === 'down' && 'Backend unreachable'}
            </span>
          </div>
          {status === 'degraded' && (
            <p className="mt-2 leading-relaxed">
              Export the missing ONNX files into models/onnx, then restart the backend.
            </p>
          )}
          {status === 'down' && (
            <p className="mt-2 leading-relaxed">
              Start the stack with docker compose up --build.
            </p>
          )}
        </div>
      </aside>

      <main className="flex-1 overflow-y-auto px-5 py-8 lg:px-10">
        <div className="mx-auto max-w-4xl">
          <Active />
        </div>
      </main>
    </div>
  )
}
