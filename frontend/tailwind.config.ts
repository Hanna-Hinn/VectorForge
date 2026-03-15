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
      keyframes: {
        "slide-in": {
          "0%": { opacity: "0", transform: "translateY(0.5rem)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "animate-in": {
          "0%": { opacity: "0", transform: "scale(0.95)" },
          "100%": { opacity: "1", transform: "scale(1)" },
        },
      },
      animation: {
        "slide-in": "slide-in 200ms ease-out",
        in: "animate-in 150ms ease-out",
      },
    },
  },
  plugins: [],
} satisfies Config;
