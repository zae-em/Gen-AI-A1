import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In development the dev server proxies /api to the backend container.
// In production nginx does the same thing (see nginx.conf), so the frontend
// code never needs to know the backend's address.
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_API_TARGET || 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
