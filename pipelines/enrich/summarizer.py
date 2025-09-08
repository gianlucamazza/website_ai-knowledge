"""
Content summarization using AI models for generating article summaries
and extracting key insights.
"""

import asyncio
import logging
import re
from typing import Dict, List

import numpy as np
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from ..config import config

logger = logging.getLogger(__name__)


class ContentSummarizer:
    """AI-powered content summarization with multiple provider support."""

    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.enrichment_config = config.enrichment

        # Initialize clients if API keys are available
        if self.enrichment_config.openai_api_key:
            self.openai_client = AsyncOpenAI(
                api_key=self.enrichment_config.openai_api_key
            )
        else:
            self.openai_client = None

        if self.enrichment_config.anthropic_api_key:
            self.anthropic_client = AsyncAnthropic(
                api_key=self.enrichment_config.anthropic_api_key
            )
        else:
            self.anthropic_client = None

    async def generate_summary(
        self, content: str, title: str = "", summary_type: str = "executive", 
        provider: str = None, max_length: int = None
    ) -> str:
        """
        Generate a summary of the content using AI (test-compatible API).
        
        Args:
            content: Full text content to summarize
            title: Optional title for context
            summary_type: Type of summary ('executive', 'technical', 'brief')
            provider: AI provider to use ('openai', 'anthropic', or None for auto)
            max_length: Maximum length of summary
            
        Returns:
            Summary text string
        """
        try:
            # Handle provider selection
            if provider == 'extractive':
                result = await self._extractive_summarization(content, title, max_length)
            else:
                # Call the main summarization method
                result = await self.summarize_content(content, title, summary_type)
            
            summary = result.get("summary", "")
            
            # Apply length limit if specified
            if max_length and len(summary) > max_length:
                summary = summary[:max_length].rsplit(' ', 1)[0] + "..."
            
            return summary
        except Exception as e:
            logger.error(f"Error in generate_summary: {e}")
            return ""

    async def summarize_content(
        self, content: str, title: str = "", summary_type: str = "executive"
    ) -> Dict:
        """
        Generate a summary of the content using AI.

        Args:
            content: Full text content to summarize
            title: Optional title for context
            summary_type: Type of summary ('executive', 'technical', 'brief')

        Returns:
            Dict with summary text and metadata
        """
        try:
            # Validate input - for very short content, return it as-is
            if not content:
                return self._empty_summary_result()
            
            word_count = len(content.split())
            if word_count < 50:
                logger.warning("Content too short for summarization, returning as-is")
                # Return the content itself as the summary for very short content
                return {
                    "summary": content,
                    "provider": "passthrough",
                    "model": "none",
                    "tokens_used": 0,
                    "summary_type": summary_type,
                    "method": "passthrough",
                }

            # Choose summarization approach based on content length
            word_count = len(content.split())

            if word_count > 4000:
                # Use chunked summarization for very long content
                return await self._chunked_summarization(content, title, summary_type)
            else:
                # Use direct summarization
                return await self._direct_summarization(content, title, summary_type)

        except Exception as e:
            logger.error(f"Error summarizing content: {e}")
            return self._empty_summary_result()

    async def _direct_summarization(
        self, content: str, title: str, summary_type: str
    ) -> Dict:
        """Direct summarization for shorter content."""
        try:
            # Try OpenAI first, then Anthropic as fallback
            if self.openai_client:
                result = await self._summarize_with_openai(content, title, summary_type)
                if result["summary"]:
                    return result

            if self.anthropic_client:
                result = await self._summarize_with_anthropic(
                    content, title, summary_type
                )
                if result["summary"]:
                    return result

            # Fallback to extractive summarization
            return await self._extractive_summarization(content, title, max_length=None)

        except Exception as e:
            logger.error(f"Error in direct summarization: {e}")
            return self._empty_summary_result()

    async def _chunked_summarization(
        self, content: str, title: str, summary_type: str
    ) -> Dict:
        """Chunked summarization for longer content."""
        try:
            # Split content into chunks
            chunks = self._split_into_chunks(content, max_chunk_size=2000)

            if len(chunks) == 1:
                return await self._direct_summarization(content, title, summary_type)

            # Summarize each chunk
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                logger.debug(f"Summarizing chunk {i+1}/{len(chunks)}")

                chunk_result = await self._direct_summarization(
                    chunk, f"{title} (Part {i+1})", "brie"
                )

                if chunk_result["summary"]:
                    chunk_summaries.append(chunk_result["summary"])

                # Add delay to respect rate limits
                await asyncio.sleep(0.5)

            if not chunk_summaries:
                return self._empty_summary_result()

            # Combine chunk summaries into final summary
            combined_content = "\n\n".join(chunk_summaries)
            final_result = await self._direct_summarization(
                combined_content, title, summary_type
            )

            # Add metadata about chunked processing
            final_result["chunks_processed"] = len(chunks)
            final_result["method"] = "chunked"

            return final_result

        except Exception as e:
            logger.error(f"Error in chunked summarization: {e}")
            return self._empty_summary_result()

    async def _summarize_with_openai(
        self, content: str, title: str, summary_type: str
    ) -> Dict:
        """Summarize content using OpenAI API."""
        try:
            if not self.openai_client:
                return self._empty_summary_result()
                
            system_prompt = self._get_system_prompt(summary_type)
            user_prompt = self._get_user_prompt(content, title, summary_type)

            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=min(500, self.enrichment_config.max_summary_length),
                temperature=0.3,
                presence_penalty=0.1,
            )

            summary = response.choices[0].message.content.strip()

            return {
                "summary": summary,
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "tokens_used": response.usage.total_tokens if response.usage else 0,
                "summary_type": summary_type,
                "method": "direct",
            }

        except Exception as e:
            logger.error(f"OpenAI summarization failed: {e}")
            return self._empty_summary_result()

    async def _summarize_with_anthropic(
        self, content: str, title: str, summary_type: str
    ) -> Dict:
        """Summarize content using Anthropic Claude API."""
        try:
            system_prompt = self._get_system_prompt(summary_type)
            user_prompt = self._get_user_prompt(content, title, summary_type)

            response = await self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=min(500, self.enrichment_config.max_summary_length),
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            summary = response.content[0].text.strip()

            return {
                "summary": summary,
                "provider": "anthropic",
                "model": "claude-3-haiku",
                "tokens_used": response.usage.input_tokens
                + response.usage.output_tokens,
                "summary_type": summary_type,
                "method": "direct",
            }

        except Exception as e:
            logger.error(f"Anthropic summarization failed: {e}")
            return self._empty_summary_result()

    async def _extractive_summarization(self, content: str, title: str, max_length: int = None) -> Dict:
        """Fallback extractive summarization using TF-IDF."""
        try:
            import numpy as np
            from sklearn.feature_extraction.text import TfidfVectorizer

            # Split into sentences
            sentences = self._split_into_sentences(content)

            if len(sentences) < 3:
                return {"summary": content[:300], "method": "truncation"}

            # Calculate TF-IDF for sentences
            vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
            tfidf_matrix = vectorizer.fit_transform(sentences)

            # Calculate sentence scores based on TF-IDF
            sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)

            # Get top sentences (aim for max_length characters or ~150 words)
            if max_length:
                # Convert character limit to approximate word count (avg 5 chars per word)
                target_words = max_length // 5
            else:
                target_words = 150
            selected_sentences = []
            current_words = 0

            # Sort sentences by score, then by original order
            scored_sentences = [
                (score, i, sent)
                for i, (score, sent) in enumerate(zip(sentence_scores, sentences))
            ]
            scored_sentences.sort(key=lambda x: x[0], reverse=True)

            if max_length:
                # Use character limit directly
                current_chars = 0
                for score, orig_idx, sentence in scored_sentences:
                    sentence_chars = len(sentence)
                    if current_chars + sentence_chars <= max_length:
                        selected_sentences.append((orig_idx, sentence))
                        current_chars += sentence_chars
                    
                    if current_chars >= max_length * 0.9:  # Stop at 90% to leave some margin
                        break
            else:
                # Use word count
                for score, orig_idx, sentence in scored_sentences:
                    sentence_words = len(sentence.split())
                    if current_words + sentence_words <= target_words:
                        selected_sentences.append((orig_idx, sentence))
                        current_words += sentence_words

                    if current_words >= target_words:
                        break

            # Sort selected sentences by original order
            selected_sentences.sort(key=lambda x: x[0])
            summary = " ".join([sent for _, sent in selected_sentences])

            return {
                "summary": summary,
                "provider": "extractive",
                "model": "tfid",
                "sentences_selected": len(selected_sentences),
                "method": "extractive",
            }

        except Exception as e:
            logger.error(f"Extractive summarization failed: {e}")
            return {"summary": content[:300], "method": "truncation"}

    def _split_into_chunks(self, content: str, max_chunk_size: int = 2000) -> List[str]:
        """Split content into chunks for processing."""
        words = content.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            current_chunk.append(word)
            current_size += 1

            if current_size >= max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0

        # Add remaining words
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _split_into_sentences(self, content: str) -> List[str]:
        """Split content into sentences."""
        # Simple sentence splitting - could be improved with NLTK
        sentences = re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return sentences
    
    def _score_sentences(self, sentences: List[str]) -> List[float]:
        """
        Score sentences for extractive summarization.
        
        Args:
            sentences: List of sentences to score
            
        Returns:
            List of scores for each sentence
        """
        try:
            if not sentences:
                return []
            
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Calculate TF-IDF for sentences
            vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate sentence scores based on TF-IDF
            scores = np.sum(tfidf_matrix.toarray(), axis=1)
            
            # Boost scores for sentences with important terms
            important_terms = [
                'artificial intelligence', 'machine learning', 'deep learning',
                'neural network', 'algorithm', 'model', 'system', 'technology'
            ]
            
            for i, sentence in enumerate(sentences):
                sentence_lower = sentence.lower()
                for term in important_terms:
                    if term in sentence_lower:
                        scores[i] *= 1.2  # Boost by 20% for each important term
            
            return scores.tolist()
            
        except Exception as e:
            logger.warning(f"Error scoring sentences: {e}")
            # Return equal scores as fallback
            return [1.0] * len(sentences)

    def _get_system_prompt(self, summary_type: str) -> str:
        """Get system prompt based on summary type."""
        prompts = {
            "executive": (
                "You are an expert content summarizer. Create concise, executive-level "
                "summaries that capture the key insights and main points. Focus on "
                "actionable information and high-level takeaways."
            ),
            "technical": (
                "You are a technical content expert. Create detailed summaries that "
                "preserve important technical details, methodologies, and specific "
                "findings. Maintain technical accuracy and include key terminology."
            ),
            "brie": (
                "You are a content summarization expert. Create brief, clear summaries "
                "that capture the essence of the content in 2-3 sentences. Focus on "
                "the most important points only."
            ),
        }

        return prompts.get(summary_type, prompts["executive"])

    def _get_user_prompt(self, content: str, title: str, summary_type: str) -> str:
        """Generate user prompt for summarization."""
        prompt = "Please summarize the following content"

        if title:
            prompt += f" titled '{title}'"

        if summary_type == "executive":
            prompt += " in 3-4 sentences, focusing on key insights and implications:"
        elif summary_type == "technical":
            prompt += " in 4-5 sentences, preserving important technical details:"
        elif summary_type == "brie":
            prompt += " in 2-3 sentences, capturing only the most essential points:"

        prompt += f"\n\n{content}\n\nSummary:"

        return prompt

    async def enhance_metadata(
        self, content: str, original_metadata: Dict
    ) -> Dict:
        """
        Enhance article metadata using AI analysis.
        
        Args:
            content: Article content to analyze
            original_metadata: Original metadata dict with title, tags, etc.
            
        Returns:
            Enhanced metadata dict with additional/improved fields
        """
        try:
            # Preserve original metadata
            enhanced = original_metadata.copy()
            
            # If no AI clients available, return original
            if not self.openai_client and not self.anthropic_client:
                return enhanced
            
            # Extract content insights for metadata enhancement
            content_preview = content[:2000] if len(content) > 2000 else content
            
            prompt = f"""Analyze this content and enhance its metadata:
Title: {original_metadata.get('title', 'Unknown')}
Current Tags: {original_metadata.get('tags', [])}
Content Preview: {content_preview}

Provide enhanced metadata in JSON format with:
- suggested_tags: 5-10 relevant tags
- category: primary category
- complexity_level: beginner/intermediate/advanced
- target_audience: general/technical/academic
- key_topics: main topics covered
"""
            
            try:
                if self.openai_client:
                    # Use structured output for metadata
                    response = await self._generate_metadata_with_ai(prompt, "openai")
                elif self.anthropic_client:
                    response = await self._generate_metadata_with_ai(prompt, "anthropic")
                else:
                    response = {}
                
                # Merge enhanced metadata
                if response:
                    enhanced.update(response)
                    
                    # Combine original and suggested tags
                    original_tags = set(original_metadata.get('tags', []))
                    suggested_tags = set(response.get('suggested_tags', []))
                    enhanced['tags'] = list(original_tags | suggested_tags)[:10]
                    
                return enhanced
                
            except Exception as e:
                logger.warning(f"AI metadata enhancement failed: {e}")
                return enhanced
                
        except Exception as e:
            logger.error(f"Error enhancing metadata: {e}")
            return original_metadata
    
    async def _generate_metadata_with_ai(self, prompt: str, provider: str) -> Dict:
        """Generate metadata using AI provider."""
        try:
            import json
            
            if provider == "openai" and self.openai_client:
                # Use new OpenAI API v1.0+
                response = await self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a metadata extraction assistant. Always respond with valid JSON."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.3
                )
                
                content = response.choices[0].message.content.strip()
                return json.loads(content)
                
            elif provider == "anthropic" and self.anthropic_client:
                response = await self.anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=500,
                    temperature=0.3,
                    system="You are a metadata extraction assistant. Always respond with valid JSON.",
                    messages=[{"role": "user", "content": prompt}]
                )
                
                content = response.content[0].text.strip()
                return json.loads(content)
                
            return {}
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse AI metadata response: {e}")
            return {}

    async def generate_key_points(self, content: str, max_points: int = 5) -> List[str]:
        """Extract key points from content."""
        try:
            if not self.openai_client and not self.anthropic_client:
                return []

            prompt = (
                f"Extract the {max_points} most important key points from the following content. "
                f"Return each point as a single, clear sentence:\n\n{content}\n\n"
                "Key points:"
            )

            if self.openai_client:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You extract key points from content.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=300,
                    temperature=0.3,
                )

                points_text = response.choices[0].message.content.strip()

                # Parse points (assuming numbered or bulleted list)
                points = []
                for line in points_text.split("\n"):
                    line = line.strip()
                    if line and (
                        line[0].isdigit()
                        or line.startswith("-")
                        or line.startswith("•")
                    ):
                        # Remove numbering/bullets
                        point = re.sub(r"^\d+\.?\s*|^[-•]\s*", "", line).strip()
                        if point:
                            points.append(point)

                return points[:max_points]

            return []

        except Exception as e:
            logger.error(f"Error generating key points: {e}")
            return []

    def _empty_summary_result(self) -> Dict:
        """Return empty summary result."""
        return {
            "summary": "",
            "provider": "none",
            "model": "none",
            "tokens_used": 0,
            "summary_type": "none",
            "method": "failed",
        }
