"""Media processing tools for image and color manipulation."""

import base64
import math
from pathlib import Path
from typing import Tuple

from openagentflow import tool


@tool
def image_to_base64(image_path: str) -> str:
    """
    Convert an image file to a base64 encoded string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string of the image

    Example:
        >>> base64_str = image_to_base64("/path/to/image.png")
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        raise Exception(f"Error encoding image to base64: {str(e)}")


@tool
def generate_qr_data(text: str) -> str:
    """
    Generate a simple ASCII art representation of a QR code.

    This is a simplified representation, not a real QR code scanner.
    For production use, consider using the 'qrcode' library.

    Args:
        text: Text to encode in the QR code

    Returns:
        ASCII art representation of a QR code

    Example:
        >>> qr = generate_qr_data("Hello World")
    """
    # Simple hash-based pattern generation for ASCII art
    # This creates a deterministic pattern based on the text
    size = 21  # Standard QR code size (21x21 for version 1)
    pattern = []

    # Add top border
    pattern.append("█" * (size + 2))

    # Generate pattern based on text hash
    text_hash = hash(text)

    for i in range(size):
        row = "█"
        for j in range(size):
            # Create deterministic pattern based on position and text hash
            cell_value = (text_hash + i * size + j) % 2
            row += "█" if cell_value == 0 else " "
        row += "█"
        pattern.append(row)

    # Add bottom border
    pattern.append("█" * (size + 2))

    qr_ascii = "\n".join(pattern)
    return f"QR Code for: '{text}'\n\n{qr_ascii}"


@tool
def color_hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert a hexadecimal color code to RGB tuple.

    Args:
        hex_color: Hex color code (e.g., "#FF5733" or "FF5733")

    Returns:
        Tuple of (red, green, blue) values (0-255)

    Example:
        >>> rgb = color_hex_to_rgb("#FF5733")
        >>> print(rgb)  # (255, 87, 51)
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')

    # Validate hex color format
    if len(hex_color) not in (3, 6):
        raise ValueError(f"Invalid hex color format: {hex_color}")

    # Expand short form (e.g., "F00" -> "FF0000")
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])

    # Convert to RGB
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b)
    except ValueError:
        raise ValueError(f"Invalid hex color: {hex_color}")


@tool
def color_rgb_to_hex(r: int, g: int, b: int) -> str:
    """
    Convert RGB color values to hexadecimal color code.

    Args:
        r: Red value (0-255)
        g: Green value (0-255)
        b: Blue value (0-255)

    Returns:
        Hexadecimal color code with '#' prefix

    Example:
        >>> hex_color = color_rgb_to_hex(255, 87, 51)
        >>> print(hex_color)  # "#FF5733"
    """
    # Validate RGB values
    if not all(0 <= val <= 255 for val in [r, g, b]):
        raise ValueError(f"RGB values must be between 0 and 255. Got: ({r}, {g}, {b})")

    # Convert to hex
    return f"#{r:02X}{g:02X}{b:02X}"


@tool
def generate_placeholder_image_url(width: int, height: int) -> str:
    """
    Generate a placeholder.com URL for a placeholder image.

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        URL string for the placeholder image

    Example:
        >>> url = generate_placeholder_image_url(800, 600)
        >>> print(url)  # "https://via.placeholder.com/800x600"
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Width and height must be positive. Got: {width}x{height}")

    return f"https://via.placeholder.com/{width}x{height}"


@tool
def parse_exif_date(exif_date: str) -> str:
    """
    Parse EXIF date format to ISO 8601 format.

    EXIF date format: "YYYY:MM:DD HH:MM:SS"
    ISO 8601 format: "YYYY-MM-DDTHH:MM:SS"

    Args:
        exif_date: Date string in EXIF format

    Returns:
        Date string in ISO 8601 format

    Example:
        >>> iso_date = parse_exif_date("2024:01:15 14:30:45")
        >>> print(iso_date)  # "2024-01-15T14:30:45"
    """
    try:
        # Split date and time
        date_part, time_part = exif_date.split(' ', 1)

        # Replace colons in date part with hyphens
        iso_date = date_part.replace(':', '-')

        # Combine with 'T' separator
        return f"{iso_date}T{time_part}"
    except Exception as e:
        raise ValueError(f"Invalid EXIF date format: {exif_date}. Expected 'YYYY:MM:DD HH:MM:SS'")


@tool
def estimate_image_size(width: int, height: int, depth: int = 24) -> int:
    """
    Estimate the uncompressed size of an image in bytes.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        depth: Color depth in bits per pixel (default: 24 for RGB)

    Returns:
        Estimated size in bytes

    Example:
        >>> size = estimate_image_size(1920, 1080)
        >>> print(f"{size / (1024**2):.2f} MB")  # "6.22 MB"
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Width and height must be positive. Got: {width}x{height}")

    if depth <= 0:
        raise ValueError(f"Depth must be positive. Got: {depth}")

    # Calculate total bits and convert to bytes
    total_bits = width * height * depth
    total_bytes = total_bits // 8

    return total_bytes


@tool
def aspect_ratio(width: int, height: int) -> str:
    """
    Calculate and simplify the aspect ratio of an image.

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Simplified aspect ratio as a string (e.g., "16:9")

    Example:
        >>> ratio = aspect_ratio(1920, 1080)
        >>> print(ratio)  # "16:9"
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Width and height must be positive. Got: {width}x{height}")

    # Calculate greatest common divisor
    def gcd(a: int, b: int) -> int:
        while b:
            a, b = b, a % b
        return a

    divisor = gcd(width, height)

    # Simplify the ratio
    simplified_width = width // divisor
    simplified_height = height // divisor

    return f"{simplified_width}:{simplified_height}"
