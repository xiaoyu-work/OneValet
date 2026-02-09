"""
Carrier detection based on tracking number format
"""

import re
from typing import Optional


def detect_carrier(tracking_number: str) -> Optional[str]:
    """
    Detect carrier from tracking number format.

    Args:
        tracking_number: The tracking number to analyze

    Returns:
        Carrier code or None if unknown
    """
    tracking_number = tracking_number.strip().upper()

    # UPS: Starts with 1Z, 18 chars
    if tracking_number.startswith('1Z') and len(tracking_number) == 18:
        return 'ups'

    # FedEx: 12 or 15 digits
    if tracking_number.isdigit():
        if len(tracking_number) == 12:
            return 'fedex'
        elif len(tracking_number) == 15:
            return 'fedex'
        # USPS: 20-22 digits
        elif len(tracking_number) in [20, 22]:
            return 'usps'

    # USPS with letters: 13 chars, starts with letters
    if re.match(r'^[A-Z]{2}\d{9}US$', tracking_number):
        return 'usps'

    # DHL: 10-11 digits or starts with specific patterns
    if tracking_number.isdigit() and len(tracking_number) in [10, 11]:
        return 'dhl'

    # DHL Express: starts with specific numbers
    if tracking_number.isdigit() and len(tracking_number) == 10:
        if tracking_number.startswith(('1', '2', '3', '4', '5', '6', '7', '8', '9')):
            return 'dhl'

    # Amazon: TBA followed by digits
    if tracking_number.startswith('TBA'):
        return 'amazon'

    # OnTrac: C followed by digits, 15 chars
    if tracking_number.startswith('C') and len(tracking_number) == 15:
        return 'ontrac'

    # LaserShip: 1LS or LX
    if tracking_number.startswith(('1LS', 'LX')):
        return 'lasership'

    return None


def get_tracking_url(carrier: str, tracking_number: str) -> Optional[str]:
    """
    Get tracking URL for a carrier.

    Args:
        carrier: Carrier code
        tracking_number: Tracking number

    Returns:
        Tracking URL or None
    """
    tracking_number = tracking_number.strip()

    urls = {
        'usps': f'https://tools.usps.com/go/TrackConfirmAction?tLabels={tracking_number}',
        'ups': f'https://www.ups.com/track?tracknum={tracking_number}',
        'fedex': f'https://www.fedex.com/fedextrack/?trknbr={tracking_number}',
        'dhl': f'https://www.dhl.com/us-en/home/tracking/tracking-express.html?submit=1&tracking-id={tracking_number}',
        'amazon': f'https://www.amazon.com/gp/css/shiptrack/view.html?trackingId={tracking_number}',
        'ontrac': f'https://www.ontrac.com/tracking/?number={tracking_number}',
        'lasership': f'https://www.lasership.com/track/{tracking_number}',
    }

    return urls.get(carrier)


def normalize_tracking_number(tracking_number: str) -> str:
    """
    Normalize tracking number (remove spaces, dashes, uppercase).

    Args:
        tracking_number: Raw tracking number

    Returns:
        Normalized tracking number
    """
    normalized = re.sub(r'[\s\-\.]', '', tracking_number)
    return normalized.upper()
