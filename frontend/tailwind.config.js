/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#10151c',
        panel: '#19212b',
        rule: '#2a3541',
        mount: '#e8eae4',
        muted: '#8a99a8',
        teal: '#4fa3a8',
        amber: '#d2a24c',
        alert: '#c9635b',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
    },
  },
  plugins: [],
}
