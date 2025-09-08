import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';

// https://astro.build/config
export default defineConfig({
  // For GitHub Pages deployment
  site: 'https://gianlucamazza.github.io',
  base: process.env.NODE_ENV === 'production' ? '/website_ai-knowledge' : '/',
  integrations: [
    mdx(),
    sitemap()
  ],
  markdown: {
    shikiConfig: {
      theme: 'github-light',
      wrap: true
    }
  },
  build: {
    format: 'directory'
  },
  compilerOptions: {
    types: ['astro/client']
  }
});