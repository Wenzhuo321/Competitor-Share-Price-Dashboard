import { defineConfig } from 'vite'

export default defineConfig(({ mode }) => ({
  server: {
    port: 5173,
    host: '0.0.0.0',
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  define: {
    // Expose backend URL to frontend for production builds
    __BACKEND_URL__: JSON.stringify(process.env.VITE_BACKEND_URL || ''),
  },
}))
