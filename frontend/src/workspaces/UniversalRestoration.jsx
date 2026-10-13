import React from 'react'
import RestorationWorkspace from '../components/RestorationWorkspace'

export default function UniversalRestoration() {
  return (
    <RestorationWorkspace
      title="Universal Restoration"
      description="One autoencoder handles all four input conditions. It is never told which corruption was applied - the same encoder, bottleneck and decoder reconstruct a clean image from noise, blur, occlusion or an already-clean photograph."
      endpoint="/universal"
      filename="universal-restoration.png"
    />
  )
}
