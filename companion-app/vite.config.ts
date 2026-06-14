import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Backend target. Defaults to the local backend on :8000; override with
// DANCE_API_TARGET (e.g. a second backend on another port) without editing
// this file.
const API_TARGET = process.env.DANCE_API_TARGET ?? "http://localhost:8000";
const WS_TARGET = API_TARGET.replace(/^http/, "ws");

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: Number(process.env.PORT) || 5173,
    proxy: {
      "/api": API_TARGET,
      "/ws": { target: WS_TARGET, ws: true },
    },
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./tests/setup.ts"],
    css: false,
  },
});
