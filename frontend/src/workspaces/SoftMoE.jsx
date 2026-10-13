import React from 'react'
import RestorationWorkspace from '../components/RestorationWorkspace'
import { Readout, WeightBar } from '../components/Panels'

function Diagnostics(result) {
  const entries = Object.entries(result.weights || {})
  const top = entries.reduce((a, b) => (b[1] > a[1] ? b : a), ['', 0])
  return (
    <div className="flex flex-col gap-4">
      <div className="grid gap-3 sm:grid-cols-3">
        <Readout label="Strongest branch" value={result.dominant_branch} tone="teal" />
        <Readout label="Its weight"
                 value={`${(result.dominant_weight * 100).toFixed(1)}%`} />
        <Readout
          label="Routing"
          value={result.is_blended
            ? `blended with ${result.second_branch}`
            : 'effectively single-branch'}
          tone={result.is_blended ? 'amber' : 'default'}
        />
      </div>
      <div className="flex flex-col gap-2.5 rounded border border-rule bg-panel/40 p-4">
        <span className="text-xs text-muted">
          Every branch runs; these weights decide how much each one contributes
        </span>
        {entries.map(([name, value]) => (
          <WeightBar key={name} label={name} value={value} highlight={name === top[0]} />
        ))}
      </div>
    </div>
  )
}

export default function SoftMoE() {
  return (
    <RestorationWorkspace
      title="Soft Mixture-of-Experts Restoration"
      description="Instead of picking one expert, a gating network weighs all four branches and blends their outputs. Because the blend is differentiable, the gate and the experts were fine-tuned together. Images that sit between two corruption types get a genuine mixture rather than a coin flip."
      endpoint="/soft-moe"
      filename="soft-moe-restoration.png"
      renderDiagnostics={Diagnostics}
    />
  )
}
