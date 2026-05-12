// 文件说明：该文件属于前端工程配置，集中实现 tailwind.config 相关逻辑。
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#07111f",
        panel: "rgba(10, 27, 47, 0.84)",
        "panel-strong": "rgba(14, 35, 61, 0.92)",
        ink: "#f5fbff",
        "ink-soft": "rgba(216, 235, 255, 0.78)",
        accent: "#68d8ff",
        "accent-2": "#4f7cff",
        line: "rgba(141, 188, 255, 0.18)",
      },
      boxShadow: {
        command: "0 26px 80px rgba(0, 8, 20, 0.38)",
      },
      fontFamily: {
        sans: ['"Manrope"', '"Noto Sans SC"', "sans-serif"],
      },
    },
  },
  plugins: [],
};
