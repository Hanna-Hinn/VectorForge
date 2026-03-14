import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
import mockApiPlugin from "./src/mock/vite-plugin-mock-api";

export default defineConfig(({ mode }) => {
  const useMock = mode === "mock";
  return {
    plugins: [react(), ...(useMock ? [mockApiPlugin()] : [])],
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
    server: {
      port: 5173,
      // When mocking, skip the proxy so the plugin handles /api requests
      ...(useMock
        ? {}
        : {
            proxy: {
              "/api": {
                target: "http://127.0.0.1:8000",
                changeOrigin: true,
              },
            },
          }),
    },
  };
});
