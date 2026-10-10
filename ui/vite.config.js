import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The UI always calls the API under /api.
//  - development: this proxy forwards /api to the local FastAPI service
//  - production: any static server that rewrites /api to the API origin, or a
//    build with VITE_API_BASE pointing at the API (see README, Serving the built UI)
const apiTarget = process.env.VITE_API_TARGET || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: apiTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
  test: {
    environment: "node",
  },
});
