import process from 'node:process'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      // Forward API calls to the FastAPI backend (same PORT as run.py: PORT=8010 npm run dev).
      '/api': `http://127.0.0.1:${process.env.PORT || 8000}`,
    },
  },
})
