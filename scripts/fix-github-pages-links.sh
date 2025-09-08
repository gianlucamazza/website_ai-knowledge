#!/bin/bash

# Fix links for GitHub Pages deployment
echo "🔧 Fixing links for GitHub Pages deployment..."

# Update all page files to use BASE_URL
cd apps/site/src/pages

# Fix index.astro
if [ -f "index.astro" ]; then
  echo "Updating index.astro..."
  sed -i '' 's|href="/glossary/|href={\`${import.meta.env.BASE_URL}glossary/|g' index.astro
  sed -i '' 's|href="/glossary"|href={\`${import.meta.env.BASE_URL}glossary\`}|g' index.astro
  sed -i '' 's|">Explore|"\`}>Explore|g' index.astro
fi

# Fix glossary/index.astro
if [ -f "glossary/index.astro" ]; then
  echo "Updating glossary/index.astro..."
  sed -i '' 's|href={\`/glossary/${entry.id}\`}|href={\`${import.meta.env.BASE_URL}glossary/${entry.id}\`}|g' glossary/index.astro
fi

# Fix glossary/[...slug].astro
if [ -f "glossary/[...slug].astro" ]; then
  echo "Updating glossary/[...slug].astro..."
  sed -i '' 's|href="/glossary/|href={\`${import.meta.env.BASE_URL}glossary/|g' glossary/[...slug].astro
  sed -i '' 's|">|"\`}>|g' glossary/[...slug].astro
fi

# Fix articles/index.astro
if [ -f "articles/index.astro" ]; then
  echo "Updating articles/index.astro..."
  sed -i '' 's|href="/glossary/|href={\`${import.meta.env.BASE_URL}glossary/|g' articles/index.astro
  sed -i '' 's|">View|"\`}>View|g' articles/index.astro
fi

echo "✅ Links fixed for GitHub Pages deployment!"