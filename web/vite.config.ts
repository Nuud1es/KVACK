/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the UI runs on :5173 and forwards API calls to the FastAPI service, so the
// browser sees one origin (no CORS). In production FastAPI serves web/dist itself.
const API = process.env.QA_API_URL ?? "http://localhost:8000";
const apiPaths = ["ask", "ingest", "documents", "certificates", "agent", "feedback", "compliance", "health", "me", "auth", "review", "admin", "export"];

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: Object.fromEntries(apiPaths.map((p) => [`/${p}`, { target: API, changeOrigin: true }])),
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["src/test-setup.ts"],
  },
});
