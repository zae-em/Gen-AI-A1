import React from 'react'
import RestorationWorkspace from '../components/RestorationWorkspace'
import { Readout, WeightBar } from '../components/Panels'

function Diagnostics(result) {
  const entries = Object.entries(result.probabilities || {})
  const labels = {
    clean: 'Clean', salt_pepper: 'Salt & pepper', blur: 'Blur', occlusion: 'Occlusion',
  }
  return (
    <div className="flex flex-col gap-4">
      <div className="grid gap-3 sm:grid-cols-3">
        <Readout label="Predicted corruption"
                 value={labels[result.predicted_corruption] || result.predicted_corruption}
                 tone="teal" />
        <Readout label="Confidence"
                 value={`${(result.confidence * 100).toFixed(1)}%`} />
        <Readout label="Expert selected" value={result.expert_used} tone="amber" />
      </div>
      <div className="flex flex-col gap-2.5 rounded border border-rule bg-panel/40 p-4">
        <span className="text-xs text-muted">Classifier probabilities</span>
        {entries.map(([name, value]) => (
          <WeightBar key={name} label={labels[name] || name} value={value}
                     highlight={name === result.predicted_corruption} />
        ))}
      </div>
    </div>
  )
}

export default function HardRouted() {
  return (
    <RestorationWorkspace
      title="Hard-Routed Restoration"
      description="A classifier decides what kind of damage the image has, then one specialist autoencoder does the repair. A clean image skips restoration entirely through the identity bypass. When the classifier is wrong, the wrong expert runs - the probabilities below show how close the call was."
      endpoint="/hard-routed"
      filename="hard-routed-restoration.png"
      renderDiagnostics={Diagnostics}
    />
  )
}
