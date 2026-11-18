import { defineConfig } from 'astro/config';

// Static build for Cloudflare Pages (build command: npm run build, output: dist).
export default defineConfig({
  site: 'https://qiskit-pes-26.pages.dev',
  markdown: { shikiConfig: { theme: 'github-light' } },
});
