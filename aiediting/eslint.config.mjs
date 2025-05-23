import { defineConfig } from 'eslint-define-config';
export default defineConfig({
  root: true,
  env: { browser: true, es2021: true },
  extends: [
    'plugin:react/recommended',
    'plugin:@typescript-eslint/recommended',
    'next/core-web-vitals',
  ],
  parser: '@typescript-eslint/parser',
  plugins: ['react', '@typescript-eslint'],
  rules: {},
});
