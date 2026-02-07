"""
AI/ML Helper Tools for OpenAgentFlow

Pure Python implementations of common AI/ML utilities including:
- Token counting and text truncation
- Text chunking and keyword extraction
- Similarity calculations
- Placeholder embeddings for testing
- Prompt formatting
- LLM cost estimation

All implementations are pure Python with no external dependencies.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any

from openagentflow.core.tool import tool


# Model pricing (per 1M tokens) - Updated as of 2025
MODEL_PRICING = {
    # OpenAI models
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    "gpt-3.5-turbo-instruct": {"input": 1.5, "output": 2.0},
    # Anthropic models
    "claude-opus-4": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-haiku-4": {"input": 0.8, "output": 4.0},
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    # Google models
    "gemini-1.5-pro": {"input": 1.25, "output": 5.0},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.3},
    # Default fallback
    "default": {"input": 5.0, "output": 15.0},
}


@tool
def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Estimate token count for a given text and model.

    Uses character-based estimation:
    - GPT models: ~4 characters per token
    - Claude models: ~3.5 characters per token
    - Default: ~4 characters per token

    Args:
        text: The text to count tokens for
        model: Model name (affects estimation ratio)

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    # Determine chars per token based on model family
    if "claude" in model.lower():
        chars_per_token = 3.5
    elif "gpt" in model.lower():
        chars_per_token = 4.0
    elif "gemini" in model.lower():
        chars_per_token = 3.8
    else:
        chars_per_token = 4.0

    # Count characters
    char_count = len(text)

    # Estimate tokens
    token_count = math.ceil(char_count / chars_per_token)

    return token_count


@tool
def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-4") -> str:
    """
    Truncate text to fit within a token limit.

    Preserves word boundaries and adds ellipsis if truncated.

    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        model: Model name (affects token estimation)

    Returns:
        Truncated text that fits within token limit
    """
    if not text or max_tokens <= 0:
        return ""

    # Check if already under limit
    current_tokens = count_tokens(text, model)
    if current_tokens <= max_tokens:
        return text

    # Determine chars per token
    if "claude" in model.lower():
        chars_per_token = 3.5
    elif "gpt" in model.lower():
        chars_per_token = 4.0
    elif "gemini" in model.lower():
        chars_per_token = 3.8
    else:
        chars_per_token = 4.0

    # Estimate target character count (leave room for ellipsis)
    target_chars = int((max_tokens - 10) * chars_per_token)

    if target_chars <= 0:
        return "..."

    # Truncate to target
    truncated = text[:target_chars]

    # Find last complete word
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]

    # Add ellipsis
    truncated = truncated.rstrip() + "..."

    return truncated


@tool
def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    """
    Split text into chunks of approximately equal token size.

    Preserves word boundaries and includes optional overlap between chunks.

    Args:
        text: Text to split
        chunk_size: Target token count per chunk
        overlap: Number of tokens to overlap between chunks

    Returns:
        List of text chunks
    """
    if not text or chunk_size <= 0:
        return []

    # Split into words to preserve boundaries
    words = text.split()

    if not words:
        return []

    chunks = []
    current_chunk = []
    current_tokens = 0

    for word in words:
        word_tokens = count_tokens(word + " ")

        if current_tokens + word_tokens > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(" ".join(current_chunk))

            # Calculate overlap
            if overlap > 0:
                overlap_words = []
                overlap_tokens = 0

                # Take words from end of current chunk for overlap
                for w in reversed(current_chunk):
                    w_tokens = count_tokens(w + " ")
                    if overlap_tokens + w_tokens <= overlap:
                        overlap_words.insert(0, w)
                        overlap_tokens += w_tokens
                    else:
                        break

                current_chunk = overlap_words
                current_tokens = overlap_tokens
            else:
                current_chunk = []
                current_tokens = 0

        current_chunk.append(word)
        current_tokens += word_tokens

    # Add final chunk if not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


@tool
def extract_keywords(text: str, top_n: int = 10) -> list[str]:
    """
    Extract keywords from text using TF-IDF-like scoring.

    Uses a simple frequency-based approach with stopword filtering
    and length-based importance scoring.

    Args:
        text: Text to extract keywords from
        top_n: Number of keywords to return

    Returns:
        List of top N keywords
    """
    if not text or top_n <= 0:
        return []

    # Common stopwords
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "will", "with", "the", "this", "but", "they", "have",
        "had", "what", "when", "where", "who", "which", "why", "how",
        "can", "could", "would", "should", "may", "might", "must",
        "i", "you", "we", "us", "our", "your", "their", "them"
    }

    # Normalize and tokenize
    text = text.lower()
    # Remove punctuation but keep hyphens in words
    text = re.sub(r'[^\w\s-]', ' ', text)
    words = text.split()

    # Filter and score words
    word_scores = {}

    for word in words:
        # Skip stopwords and very short words
        if word in stopwords or len(word) < 3:
            continue

        # Skip if mostly numbers
        if sum(c.isdigit() for c in word) > len(word) / 2:
            continue

        # Calculate score: frequency * length_factor
        # Longer words are often more meaningful
        length_factor = min(len(word) / 5.0, 2.0)

        if word in word_scores:
            word_scores[word] += length_factor
        else:
            word_scores[word] = length_factor

    # Sort by score and return top N
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, score in sorted_words[:top_n]]

    return keywords


@tool
def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between two texts.

    Jaccard similarity = |intersection| / |union| of word sets.
    Returns a value between 0.0 (no similarity) and 1.0 (identical).

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not text1 or not text2:
        return 0.0

    # Normalize and tokenize
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    # Calculate Jaccard similarity
    intersection = words1.intersection(words2)
    union = words1.union(words2)

    if not union:
        return 0.0

    similarity = len(intersection) / len(union)

    return round(similarity, 4)


@tool
def generate_embeddings_placeholder(texts: list[str], dimensions: int = 128) -> list[list[float]]:
    """
    Generate simple hash-based embeddings for testing purposes.

    WARNING: These are NOT real semantic embeddings. Use only for:
    - Testing embedding pipelines
    - Placeholder data
    - Development without API access

    For production, use proper embedding models (OpenAI, Cohere, etc.).

    Args:
        texts: List of texts to generate embeddings for
        dimensions: Embedding dimension size (default 128)

    Returns:
        List of embedding vectors (one per text)
    """
    embeddings = []

    for text in texts:
        # Create a deterministic hash
        text_hash = hashlib.sha256(text.encode()).digest()

        # Generate embedding from hash
        embedding = []
        for i in range(dimensions):
            # Use different slices of the hash for each dimension
            byte_idx = (i * 2) % len(text_hash)
            value = int.from_bytes(text_hash[byte_idx:byte_idx+2], byteorder='big')
            # Normalize to [-1, 1]
            normalized = (value / 32768.0) - 1.0
            embedding.append(round(normalized, 6))

        embeddings.append(embedding)

    return embeddings


@tool
def format_prompt(template: str, **kwargs: Any) -> str:
    """
    Format a prompt template with variables.

    Supports multiple formatting styles:
    - {variable} - Python str.format style
    - {{variable}} - Double braces (literal)
    - $variable or ${variable} - Shell-style

    Args:
        template: Prompt template string
        **kwargs: Variables to substitute

    Returns:
        Formatted prompt string
    """
    if not template:
        return ""

    result = template

    # Handle Python format style {variable}
    try:
        result = result.format(**kwargs)
    except KeyError:
        # Some variables might not be provided, that's ok
        pass

    # Handle shell-style $variable and ${variable}
    for key, value in kwargs.items():
        # ${variable}
        result = result.replace(f"${{{key}}}", str(value))
        # $variable (but not $$)
        result = re.sub(
            f"\\${key}(?![a-zA-Z0-9_])",
            str(value),
            result
        )

    return result


@tool
def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4"
) -> float:
    """
    Estimate LLM API cost based on token counts and model.

    Includes pricing for major models from:
    - OpenAI (GPT-4, GPT-4o, GPT-3.5)
    - Anthropic (Claude Opus, Sonnet, Haiku)
    - Google (Gemini)

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name (e.g., "gpt-4", "claude-sonnet-4")

    Returns:
        Estimated cost in USD
    """
    if input_tokens < 0 or output_tokens < 0:
        return 0.0

    # Normalize model name
    model_lower = model.lower()

    # Find matching pricing
    pricing = None
    for key in MODEL_PRICING:
        if key in model_lower or model_lower in key:
            pricing = MODEL_PRICING[key]
            break

    # Use default if not found
    if pricing is None:
        pricing = MODEL_PRICING["default"]

    # Calculate cost (pricing is per 1M tokens)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost

    # Round to 6 decimal places
    return round(total_cost, 6)


# Export all tools
__all__ = [
    "count_tokens",
    "truncate_to_tokens",
    "split_into_chunks",
    "extract_keywords",
    "calculate_similarity",
    "generate_embeddings_placeholder",
    "format_prompt",
    "estimate_cost",
]
