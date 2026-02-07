/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // VSCode Dark+ inspired palette
        editor: {
          bg: '#1e1e1e',
          fg: '#d4d4d4',
          line: '#2d2d2d',
          selection: '#264f78',
          cursor: '#aeafad',
        },
        sidebar: {
          bg: '#252526',
          fg: '#cccccc',
          hover: '#2a2d2e',
          active: '#37373d',
          border: '#3c3c3c',
        },
        activitybar: {
          bg: '#333333',
          fg: '#858585',
          active: '#ffffff',
          badge: '#007acc',
        },
        statusbar: {
          bg: '#007acc',
          fg: '#ffffff',
          hover: '#1f8ad2',
          nobg: '#68217a',
        },
        titlebar: {
          bg: '#3c3c3c',
          fg: '#cccccc',
        },
        input: {
          bg: '#3c3c3c',
          fg: '#cccccc',
          border: '#3c3c3c',
          focus: '#007acc',
        },
        list: {
          hover: '#2a2d2e',
          active: '#094771',
          focus: '#062f4a',
        },
        accent: {
          blue: '#007acc',
          green: '#4ec9b0',
          yellow: '#dcdcaa',
          orange: '#ce9178',
          red: '#f14c4c',
          purple: '#c586c0',
        },
        token: {
          keyword: '#569cd6',
          string: '#ce9178',
          number: '#b5cea8',
          function: '#dcdcaa',
          variable: '#9cdcfe',
          type: '#4ec9b0',
          comment: '#6a9955',
        },
      },
      fontFamily: {
        mono: ['Consolas', 'Monaco', 'Courier New', 'monospace'],
        sans: ['Segoe UI', 'system-ui', 'sans-serif'],
      },
      fontSize: {
        '2xs': '0.65rem',
      },
      spacing: {
        '13': '3.25rem',
        '15': '3.75rem',
      },
      animation: {
        'fade-in': 'fadeIn 0.15s ease-out',
        'slide-up': 'slideUp 0.15s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}
