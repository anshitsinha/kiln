import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  build: {
    target: "es2022",
    rollupOptions: {
      input: {
        index: resolve(__dirname, "index.html"),
        matmul: resolve(__dirname, "bench/matmul.html"),
        decode: resolve(__dirname, "bench/decode.html"),
        attention: resolve(__dirname, "bench/attention.html"),
      },
    },
  },
});
