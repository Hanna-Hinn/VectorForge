import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#f0f5ff",
          100: "#e0ebff",
          200: "#c2d6ff",
          300: "#94b8ff",
          400: "#6699ff",
          500: "#3d7aff",
          600: "#1a5cff",
          700: "#0044e6",
          800: "#0036b8",
          900: "#002d99",
        },
      },
    },
  },
  plugins: [],
} satisfies Config;
