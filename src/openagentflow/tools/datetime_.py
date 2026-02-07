"""Date and time utility tools for working with dates, timestamps, and date formatting."""

from datetime import datetime, timedelta
from typing import Dict
from dateutil import parser as date_parser

from openagentflow.core.decorators import tool


@tool
def parse_date(date_string: str) -> str:
    """
    Parse various date formats to ISO format (YYYY-MM-DD).

    Supports common formats including:
    - YYYY-MM-DD
    - MM/DD/YYYY
    - DD-MM-YYYY
    - Month DD, YYYY
    - And many other natural language formats

    Args:
        date_string: Date string in various formats

    Returns:
        ISO formatted date string (YYYY-MM-DD)

    Example:
        >>> parse_date("12/31/2023")
        "2023-12-31"
    """
    try:
        parsed_date = date_parser.parse(date_string)
        return parsed_date.strftime("%Y-%m-%d")
    except Exception as e:
        raise ValueError(f"Unable to parse date '{date_string}': {str(e)}")


@tool
def format_date(iso_date: str, format: str) -> str:
    """
    Format an ISO date string to a custom format.

    Args:
        iso_date: Date string in ISO format (YYYY-MM-DD)
        format: Python strftime format string (e.g., "%B %d, %Y" for "January 01, 2023")

    Returns:
        Formatted date string

    Common format codes:
        %Y - Year with century (2023)
        %m - Month as number (01-12)
        %d - Day of month (01-31)
        %B - Full month name (January)
        %b - Abbreviated month name (Jan)
        %A - Full weekday name (Monday)
        %a - Abbreviated weekday name (Mon)

    Example:
        >>> format_date("2023-12-31", "%B %d, %Y")
        "December 31, 2023"
    """
    try:
        date_obj = datetime.strptime(iso_date, "%Y-%m-%d")
        return date_obj.strftime(format)
    except ValueError as e:
        raise ValueError(f"Invalid date format. Expected ISO format (YYYY-MM-DD): {str(e)}")


@tool
def date_difference(date1: str, date2: str) -> Dict[str, int]:
    """
    Calculate the difference between two dates.

    Args:
        date1: First date in ISO format (YYYY-MM-DD)
        date2: Second date in ISO format (YYYY-MM-DD)

    Returns:
        Dictionary containing the difference in days, hours, and minutes
        Positive values mean date1 is after date2

    Example:
        >>> date_difference("2023-12-31", "2023-12-25")
        {"days": 6, "hours": 144, "minutes": 8640}
    """
    try:
        dt1 = datetime.strptime(date1, "%Y-%m-%d")
        dt2 = datetime.strptime(date2, "%Y-%m-%d")
        difference = dt1 - dt2

        total_seconds = int(difference.total_seconds())
        days = difference.days
        hours = total_seconds // 3600
        minutes = total_seconds // 60

        return {
            "days": days,
            "hours": hours,
            "minutes": minutes
        }
    except ValueError as e:
        raise ValueError(f"Invalid date format. Expected ISO format (YYYY-MM-DD): {str(e)}")


@tool
def add_days(date: str, days: int) -> str:
    """
    Add or subtract days from a date.

    Args:
        date: Date in ISO format (YYYY-MM-DD)
        days: Number of days to add (negative to subtract)

    Returns:
        New date in ISO format (YYYY-MM-DD)

    Example:
        >>> add_days("2023-12-25", 7)
        "2024-01-01"
        >>> add_days("2023-12-25", -5)
        "2023-12-20"
    """
    try:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        new_date = date_obj + timedelta(days=days)
        return new_date.strftime("%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format. Expected ISO format (YYYY-MM-DD): {str(e)}")


@tool
def get_weekday(date: str) -> str:
    """
    Get the weekday name for a given date.

    Args:
        date: Date in ISO format (YYYY-MM-DD)

    Returns:
        Full weekday name (e.g., "Monday", "Tuesday", etc.)

    Example:
        >>> get_weekday("2023-12-25")
        "Monday"
    """
    try:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        return date_obj.strftime("%A")
    except ValueError as e:
        raise ValueError(f"Invalid date format. Expected ISO format (YYYY-MM-DD): {str(e)}")


@tool
def is_weekend(date: str) -> bool:
    """
    Check if a date falls on a weekend (Saturday or Sunday).

    Args:
        date: Date in ISO format (YYYY-MM-DD)

    Returns:
        True if the date is a Saturday or Sunday, False otherwise

    Example:
        >>> is_weekend("2023-12-23")  # Saturday
        True
        >>> is_weekend("2023-12-25")  # Monday
        False
    """
    try:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        # weekday() returns 5 for Saturday, 6 for Sunday
        return date_obj.weekday() in (5, 6)
    except ValueError as e:
        raise ValueError(f"Invalid date format. Expected ISO format (YYYY-MM-DD): {str(e)}")


@tool
def timestamp_to_date(timestamp: int) -> str:
    """
    Convert a Unix timestamp to an ISO formatted date string.

    Args:
        timestamp: Unix timestamp (seconds since January 1, 1970)

    Returns:
        ISO formatted date string (YYYY-MM-DD HH:MM:SS)

    Example:
        >>> timestamp_to_date(1703548800)
        "2023-12-26 00:00:00"
    """
    try:
        date_obj = datetime.fromtimestamp(timestamp)
        return date_obj.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OSError) as e:
        raise ValueError(f"Invalid timestamp: {str(e)}")


@tool
def date_to_timestamp(date: str) -> int:
    """
    Convert an ISO formatted date string to a Unix timestamp.

    Args:
        date: Date string in ISO format (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)

    Returns:
        Unix timestamp (seconds since January 1, 1970)

    Example:
        >>> date_to_timestamp("2023-12-26")
        1703548800
        >>> date_to_timestamp("2023-12-26 12:30:00")
        1703593800
    """
    try:
        # Try parsing with time first
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            # Fall back to date only
            date_obj = datetime.strptime(date, "%Y-%m-%d")

        return int(date_obj.timestamp())
    except ValueError as e:
        raise ValueError(f"Invalid date format. Expected ISO format (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS): {str(e)}")
