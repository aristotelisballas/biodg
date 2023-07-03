from datetime import datetime


def format_datetime_filename() -> str:
    """
    Create a timestamp based on current time and format it to a string suitable for file-names.

    Use this method to format file-name timestamps instead of formatting them on your own for consistency.
    """

    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
