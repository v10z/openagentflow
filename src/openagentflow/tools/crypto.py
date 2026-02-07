"""
Cryptography tools for OpenAgentFlow.

Provides tools for hashing, encryption, password generation, and encoding.
Uses Python's built-in hashlib, secrets, and uuid modules.
"""

import hashlib
import secrets
import string
import uuid

from openagentflow.core.tool import tool


@tool
def hash_md5(text: str) -> str:
    """
    Calculate MD5 hash of input text.

    Args:
        text: The text to hash

    Returns:
        Hexadecimal MD5 hash string
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()


@tool
def hash_sha256(text: str) -> str:
    """
    Calculate SHA-256 hash of input text.

    Args:
        text: The text to hash

    Returns:
        Hexadecimal SHA-256 hash string
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


@tool
def hash_sha512(text: str) -> str:
    """
    Calculate SHA-512 hash of input text.

    Args:
        text: The text to hash

    Returns:
        Hexadecimal SHA-512 hash string
    """
    return hashlib.sha512(text.encode('utf-8')).hexdigest()


@tool
def generate_uuid() -> str:
    """
    Generate a random UUID (UUID4).

    Returns:
        Random UUID string in standard format
    """
    return str(uuid.uuid4())


@tool
def generate_password(length: int = 16, special: bool = True) -> str:
    """
    Generate a secure random password.

    Args:
        length: Length of the password (default: 16)
        special: Include special characters (default: True)

    Returns:
        Randomly generated secure password
    """
    if length < 1:
        raise ValueError("Password length must be at least 1")

    # Define character sets
    chars = string.ascii_letters + string.digits
    if special:
        chars += string.punctuation

    # Generate password using secrets for cryptographic strength
    password = ''.join(secrets.choice(chars) for _ in range(length))
    return password


@tool
def caesar_cipher(text: str, shift: int) -> str:
    """
    Apply Caesar cipher encryption/decryption to text.

    Shifts each letter by the specified amount. Non-letter characters
    are preserved as-is.

    Args:
        text: The text to encode/decode
        shift: Number of positions to shift (positive or negative)

    Returns:
        Caesar cipher encrypted/decrypted text
    """
    result = []

    for char in text:
        if char.isalpha():
            # Determine if uppercase or lowercase
            ascii_offset = ord('A') if char.isupper() else ord('a')
            # Shift character and wrap around alphabet
            shifted = (ord(char) - ascii_offset + shift) % 26
            result.append(chr(shifted + ascii_offset))
        else:
            # Keep non-alphabetic characters unchanged
            result.append(char)

    return ''.join(result)


@tool
def rot13(text: str) -> str:
    """
    Apply ROT13 encoding to text.

    ROT13 is a special case of Caesar cipher with shift of 13.
    It's its own inverse (applying twice returns original text).

    Args:
        text: The text to encode/decode

    Returns:
        ROT13 encoded/decoded text
    """
    return caesar_cipher(text, 13)


@tool
def xor_encrypt(text: str, key: str) -> str:
    """
    Simple XOR encryption/decryption.

    XOR is symmetric - applying the same operation with the same key
    decrypts the encrypted text.

    Args:
        text: The text to encrypt/decrypt
        key: The encryption key (must not be empty)

    Returns:
        Hexadecimal string of XOR encrypted bytes
    """
    if not key:
        raise ValueError("Encryption key cannot be empty")

    # Convert text and key to bytes
    text_bytes = text.encode('utf-8')
    key_bytes = key.encode('utf-8')

    # XOR each byte with corresponding key byte (repeating key as needed)
    encrypted = []
    for i, byte in enumerate(text_bytes):
        key_byte = key_bytes[i % len(key_bytes)]
        encrypted.append(byte ^ key_byte)

    # Return as hexadecimal string
    return bytes(encrypted).hex()
