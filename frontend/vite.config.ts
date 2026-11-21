import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // TensorFlow.js is most of the bundle (~220 KB gzipped); expected.
  build: { chunkSizeWarningLimit: 1400 },
  server: {
    port: 5173,
  },
});
