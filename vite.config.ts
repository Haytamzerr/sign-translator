import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // ONNX Runtime loads its own WebAssembly files; pre-bundling would break that.
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  server: { port: 5173, open: true },
  test: { environment: 'node', include: ['src/**/*.test.ts'] },
});
