"""Text processing tools for openagentflow.

This module provides a comprehensive set of text manipulation and analysis tools
including string transformations, pattern extraction, language detection, and statistical analysis.
"""

import re
from collections import Counter
from typing import Any, Dict, List
from openagentflow import tool


@tool
def reverse_words(text: str) -> str:
    """Reverse the order of words in a text string.

    Args:
        text: Input text string

    Returns:
        Text with words in reversed order

    Examples:
        >>> reverse_words("Hello world from Python")
        "Python from world Hello"
    """
    words = text.split()
    return ' '.join(reversed(words))


@tool
def extract_emails(text: str) -> List[str]:
    """Extract all email addresses from text using regex pattern matching.

    Args:
        text: Input text to search for email addresses

    Returns:
        List of email addresses found in the text

    Examples:
        >>> extract_emails("Contact us at info@example.com or support@test.org")
        ['info@example.com', 'support@test.org']
    """
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(email_pattern, text)


@tool
def extract_urls(text: str) -> List[str]:
    """Extract all URLs from text using regex pattern matching.

    Args:
        text: Input text to search for URLs

    Returns:
        List of URLs found in the text

    Examples:
        >>> extract_urls("Visit https://example.com or http://test.org/page")
        ['https://example.com', 'http://test.org/page']
    """
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, text)


@tool
def extract_phone_numbers(text: str, country: str = "US") -> List[str]:
    """Extract phone numbers from text based on country format.

    Args:
        text: Input text to search for phone numbers
        country: Country code for phone format (default: "US")

    Returns:
        List of phone numbers found in the text

    Examples:
        >>> extract_phone_numbers("Call 555-123-4567 or (555) 987-6543")
        ['555-123-4567', '(555) 987-6543']
    """
    if country == "US":
        # Matches formats: (555) 123-4567, 555-123-4567, 555.123.4567, 5551234567
        phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    elif country == "UK":
        # Matches UK formats
        phone_pattern = r'(?:\+44|0)[-.\s]?\d{4}[-.\s]?\d{6}|\d{5}[-.\s]?\d{6}'
    else:
        # Generic international format
        phone_pattern = r'\+?\d{1,4}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}'

    return re.findall(phone_pattern, text)


@tool
def text_to_slug(text: str) -> str:
    """Convert text to a URL-safe slug format.

    Args:
        text: Input text to convert to slug

    Returns:
        URL-safe slug (lowercase, hyphen-separated)

    Examples:
        >>> text_to_slug("Hello World! This is a Test.")
        "hello-world-this-is-a-test"
    """
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and replace spaces with hyphens
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    # Remove leading/trailing hyphens
    return text.strip('-')


@tool
def count_words(text: str) -> Dict[str, int]:
    """Count the frequency of each word in the text.

    Args:
        text: Input text to analyze

    Returns:
        Dictionary mapping words to their occurrence counts

    Examples:
        >>> count_words("hello world hello")
        {'hello': 2, 'world': 1}
    """
    # Remove punctuation and convert to lowercase
    words = re.findall(r'\b\w+\b', text.lower())
    return dict(Counter(words))


@tool
def detect_language(text: str) -> str:
    """Detect the language of text using character analysis.

    This is a simple heuristic-based detector that identifies common languages
    based on character patterns and common words.

    Args:
        text: Input text to analyze

    Returns:
        Detected language code (e.g., 'en', 'es', 'fr', 'de', 'unknown')

    Examples:
        >>> detect_language("Hello world")
        'en'
    """
    text_lower = text.lower()

    # Common words in different languages
    english_words = {'the', 'is', 'and', 'or', 'in', 'to', 'a', 'of', 'for'}
    spanish_words = {'el', 'la', 'de', 'que', 'y', 'en', 'los', 'es', 'por'}
    french_words = {'le', 'de', 'un', 'et', 'être', 'à', 'il', 'que', 'ne'}
    german_words = {'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'ist'}

    words = set(re.findall(r'\b\w+\b', text_lower))

    # Count matches
    scores = {
        'en': len(words & english_words),
        'es': len(words & spanish_words),
        'fr': len(words & french_words),
        'de': len(words & german_words)
    }

    # Check for special characters
    if re.search(r'[áéíóúñ]', text_lower):
        scores['es'] += 2
    if re.search(r'[àâçéèêëîïôùûü]', text_lower):
        scores['fr'] += 2
    if re.search(r'[äöüß]', text_lower):
        scores['de'] += 2

    max_score = max(scores.values())
    if max_score > 0:
        return max(scores, key=scores.get)

    return 'unknown'


@tool
def remove_html_tags(html: str) -> str:
    """Remove all HTML tags from a string, leaving only text content.

    Args:
        html: HTML string to process

    Returns:
        Plain text with HTML tags removed

    Examples:
        >>> remove_html_tags("<p>Hello <b>world</b>!</p>")
        "Hello world!"
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', html)
    # Decode common HTML entities
    entities = {
        '&nbsp;': ' ',
        '&lt;': '<',
        '&gt;': '>',
        '&amp;': '&',
        '&quot;': '"',
        '&#39;': "'"
    }
    for entity, char in entities.items():
        text = text.replace(entity, char)
    return text.strip()


@tool
def markdown_to_text(markdown: str) -> str:
    """Convert markdown formatted text to plain text.

    Args:
        markdown: Markdown formatted string

    Returns:
        Plain text with markdown syntax removed

    Examples:
        >>> markdown_to_text("# Header\\n**bold** and *italic*")
        "Header\\nbold and italic"
    """
    text = markdown

    # Remove headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Remove bold and italic
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)  # Bold + italic
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)      # Bold
    text = re.sub(r'\*(.+?)\*', r'\1', text)          # Italic
    text = re.sub(r'__(.+?)__', r'\1', text)          # Bold
    text = re.sub(r'_(.+?)_', r'\1', text)            # Italic

    # Remove links [text](url)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Remove images ![alt](url)
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', text)

    # Remove inline code
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Remove code blocks
    text = re.sub(r'```[^\n]*\n.*?```', '', text, flags=re.DOTALL)

    # Remove blockquotes
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)

    # Remove horizontal rules
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)

    # Remove list markers
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    return text.strip()


@tool
def text_to_morse(text: str) -> str:
    """Convert text to Morse code.

    Args:
        text: Input text to convert

    Returns:
        Morse code representation (dots and dashes separated by spaces)

    Examples:
        >>> text_to_morse("SOS")
        "... --- ..."
    """
    morse_code = {
        'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
        'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
        'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
        'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
        'Y': '-.--', 'Z': '--..', '0': '-----', '1': '.----', '2': '..---',
        '3': '...--', '4': '....-', '5': '.....', '6': '-....', '7': '--...',
        '8': '---..', '9': '----.', ' ': '/'
    }

    result = []
    for char in text.upper():
        if char in morse_code:
            result.append(morse_code[char])
        elif char == ' ':
            result.append('/')

    return ' '.join(result)


@tool
def morse_to_text(morse: str) -> str:
    """Convert Morse code back to text.

    Args:
        morse: Morse code string (dots and dashes separated by spaces)

    Returns:
        Decoded text string

    Examples:
        >>> morse_to_text("... --- ...")
        "SOS"
    """
    morse_code_reverse = {
        '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E', '..-.': 'F',
        '--.': 'G', '....': 'H', '..': 'I', '.---': 'J', '-.-': 'K', '.-..': 'L',
        '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P', '--.-': 'Q', '.-.': 'R',
        '...': 'S', '-': 'T', '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X',
        '-.--': 'Y', '--..': 'Z', '-----': '0', '.----': '1', '..---': '2',
        '...--': '3', '....-': '4', '.....': '5', '-....': '6', '--...': '7',
        '---..': '8', '----.': '9', '/': ' '
    }

    result = []
    for code in morse.split(' '):
        if code in morse_code_reverse:
            result.append(morse_code_reverse[code])

    return ''.join(result)


@tool
def pig_latin(text: str) -> str:
    """Convert text to Pig Latin.

    Rules:
    - Words starting with vowels: add "way" at the end
    - Words starting with consonants: move consonants before first vowel to end and add "ay"

    Args:
        text: Input text to convert

    Returns:
        Text converted to Pig Latin

    Examples:
        >>> pig_latin("hello world")
        "ellohay orldway"
    """
    def convert_word(word):
        if not word:
            return word

        # Preserve case
        is_capitalized = word[0].isupper()
        word_lower = word.lower()

        vowels = 'aeiou'

        # Check if starts with vowel
        if word_lower[0] in vowels:
            result = word_lower + 'way'
        else:
            # Find first vowel
            first_vowel = 0
            for i, char in enumerate(word_lower):
                if char in vowels:
                    first_vowel = i
                    break
            else:
                # No vowel found
                return word

            result = word_lower[first_vowel:] + word_lower[:first_vowel] + 'ay'

        # Restore capitalization
        if is_capitalized:
            result = result.capitalize()

        return result

    # Split text and convert each word
    words = re.findall(r'\b\w+\b|\W+', text)
    return ''.join(convert_word(word) if word.isalnum() else word for word in words)


@tool
def text_statistics(text: str) -> Dict[str, Any]:
    """Generate comprehensive statistics about the text.

    Args:
        text: Input text to analyze

    Returns:
        Dictionary containing various text statistics

    Examples:
        >>> text_statistics("Hello world! This is a test.")
        {'characters': 29, 'words': 6, 'sentences': 2, ...}
    """
    # Character count
    char_count = len(text)
    char_count_no_spaces = len(text.replace(' ', ''))

    # Word count
    words = re.findall(r'\b\w+\b', text)
    word_count = len(words)

    # Sentence count
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])

    # Line count
    line_count = len(text.split('\n'))

    # Average word length
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0

    # Most common words
    word_freq = Counter(word.lower() for word in words)
    most_common = dict(word_freq.most_common(5))

    # Unique words
    unique_words = len(set(word.lower() for word in words))

    return {
        'characters': char_count,
        'characters_no_spaces': char_count_no_spaces,
        'words': word_count,
        'unique_words': unique_words,
        'sentences': sentence_count,
        'lines': line_count,
        'average_word_length': round(avg_word_length, 2),
        'most_common_words': most_common,
        'lexical_diversity': round(unique_words / word_count, 2) if word_count > 0 else 0
    }


@tool
def find_palindromes(text: str) -> List[str]:
    """Find all palindrome words in the text.

    A palindrome is a word that reads the same forwards and backwards.

    Args:
        text: Input text to search

    Returns:
        List of unique palindrome words found (case-insensitive, minimum 2 characters)

    Examples:
        >>> find_palindromes("A man, a plan, a canal: Panama! Racecar level.")
        ['a', 'racecar', 'level']
    """
    words = re.findall(r'\b\w+\b', text.lower())
    palindromes = []

    for word in words:
        # Check if word is a palindrome (minimum 2 characters)
        if len(word) >= 2 and word == word[::-1]:
            if word not in palindromes:
                palindromes.append(word)

    return sorted(palindromes)


@tool
def anagram_check(word1: str, word2: str) -> bool:
    """Check if two words are anagrams of each other.

    Anagrams are words formed by rearranging the letters of another word.
    Comparison is case-insensitive and ignores spaces.

    Args:
        word1: First word
        word2: Second word

    Returns:
        True if the words are anagrams, False otherwise

    Examples:
        >>> anagram_check("listen", "silent")
        True
        >>> anagram_check("hello", "world")
        False
    """
    # Remove spaces and convert to lowercase
    cleaned1 = word1.replace(' ', '').lower()
    cleaned2 = word2.replace(' ', '').lower()

    # Sort characters and compare
    return sorted(cleaned1) == sorted(cleaned2)
