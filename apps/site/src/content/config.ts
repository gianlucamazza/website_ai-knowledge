import { defineCollection, z } from 'astro:content';

// Source schema for attribution
const sourceSchema = z.object({
  source_url: z.string().url(),
  source_title: z.string(),
  license: z.enum(['cc0', 'cc-by', 'cc-by-sa', 'mit', 'apache-2.0', 'proprietary', 'unknown']),
  accessed_date: z.string().optional(),
  author: z.string().optional(),
});

// Base schema for all content
const baseContentSchema = z.object({
  title: z.string().min(1).max(120),
  summary: z.string().min(120).max(500), // Enforced summary length
  tags: z.array(z.string()).min(1).max(10),
  updated: z.string().regex(/^\d{4}-\d{2}-\d{2}$/), // YYYY-MM-DD format
  sources: z.array(sourceSchema).optional(),
  draft: z.boolean().optional().default(false),
});

// Glossary-specific schema
const glossarySchema = baseContentSchema.extend({
  aliases: z.array(z.string()).optional(),
  related: z.array(z.string()).optional(), // Array of slugs
  difficulty: z.enum(['beginner', 'intermediate', 'advanced']).optional().default('beginner'),
  category: z.enum([
    'fundamentals',
    'machine-learning',
    'deep-learning',
    'nlp',
    'computer-vision',
    'robotics',
    'ethics',
    'applications',
    'tools',
    'research'
  ]).optional(),
});

// Article schema
const articleSchema = baseContentSchema.extend({
  author: z.string().optional(),
  publishDate: z.string().regex(/^\d{4}-\d{2}-\d{2}$/).optional(),
  readingTime: z.number().optional(), // in minutes
  featured: z.boolean().optional().default(false),
  series: z.string().optional(), // For multi-part articles
  seriesOrder: z.number().optional(),
  relatedGlossary: z.array(z.string()).optional(), // Array of glossary slugs
});

// Define collections
export const collections = {
  'glossary': defineCollection({
    type: 'content',
    schema: glossarySchema,
  }),
  'articles': defineCollection({
    type: 'content', 
    schema: articleSchema,
  }),
};

// Export types for use in components
export type GlossaryEntry = z.infer<typeof glossarySchema>;
export type Article = z.infer<typeof articleSchema>;
export type Source = z.infer<typeof sourceSchema>;