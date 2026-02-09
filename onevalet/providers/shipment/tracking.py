"""
Unified tracking provider using 17TRACK API v2.4

Supports 3,000+ carriers worldwide via 17TRACK API.
"""

import asyncio
import os
import logging
from typing import Any, Dict, Optional

import httpx

from .carrier_detector import normalize_tracking_number, get_tracking_url

logger = logging.getLogger(__name__)


# 17TRACK carrier codes for common carriers
CARRIER_CODES = {
    "fedex": 100003,
    "ups": 100002,
    "usps": 21051,
    "dhl": 100001,
    "dhl_express": 100001,
    "amazon": 100143,
    "amazon_logistics": 100143,
    "ontrac": 100049,
    "lasership": 100104,
    "china_post": 3011,
}


class TrackingProvider:
    """
    Unified tracking provider using 17TRACK API v2.4.

    Supports 3,000+ carriers including USPS, UPS, FedEx, DHL, Amazon Logistics, etc.
    """

    def __init__(self):
        """Initialize tracking provider with 17TRACK API credentials."""
        self.api_key = os.getenv('TRACK17_API_KEY')
        self.api_base = 'https://api.17track.net/track/v2.4'

    async def track(self, tracking_number: str, carrier: str = None) -> Dict[str, Any]:
        """
        Track a package via 17TRACK API.

        Flow:
        1. Try to get tracking info (if already registered)
        2. If not found, register the tracking number
        3. Return tracking info
        """
        if not self.api_key:
            logger.error("TRACK17_API_KEY not configured")
            return {
                'success': False,
                'error': '17TRACK API key not configured. Please set TRACK17_API_KEY environment variable.',
            }

        tracking_number = normalize_tracking_number(tracking_number)
        carrier_code = self._get_carrier_code(carrier)
        logger.info(f"Tracking {tracking_number} via 17TRACK API")

        try:
            result = await self._get_track_info(tracking_number, carrier_code)

            if result.get('success'):
                return result

            # If not registered, register the tracking number
            if result.get('error_code') == -18019902:
                logger.info(f"Tracking number {tracking_number} not registered, registering...")
                register_result = await self._register(tracking_number, carrier_code)

                if not register_result.get('success'):
                    return register_result

                await asyncio.sleep(5.0)

                registered_carrier = register_result.get('carrier') or carrier_code
                result = await self._get_track_info(tracking_number, registered_carrier)

                if result.get('success'):
                    result['just_added'] = True
                    return result

                carrier_name = self._get_carrier_name(registered_carrier) or carrier or 'unknown'
                return {
                    'success': True,
                    'tracking_number': tracking_number,
                    'carrier': carrier_name,
                    'carrier_code': registered_carrier,
                    'status': 'info_received',
                    'last_update': 'Package added to tracking. Status will update shortly.',
                    'estimated_delivery': None,
                    'events': [],
                    'just_added': True,
                    'tracking_url': get_tracking_url(carrier_name, tracking_number),
                }

            return result

        except httpx.HTTPStatusError as e:
            logger.error(f"17TRACK API HTTP error: {e.response.status_code} - {e.response.text}")
            return {
                'success': False,
                'error': self._get_user_friendly_error('http_error', e.response.status_code),
                'tracking_number': tracking_number,
            }
        except httpx.ConnectError as e:
            logger.error(f"17TRACK API connection error: {e}", exc_info=True)
            return {
                'success': False,
                'error': self._get_user_friendly_error('connection_error'),
                'tracking_number': tracking_number,
            }
        except Exception as e:
            logger.error(f"17TRACK API error: {e}", exc_info=True)
            return {
                'success': False,
                'error': self._get_user_friendly_error('unknown_error'),
                'tracking_number': tracking_number,
            }

    async def _register(self, tracking_number: str, carrier_code: int = None) -> Dict[str, Any]:
        """Register a tracking number with 17TRACK."""
        async with httpx.AsyncClient() as client:
            url = f"{self.api_base}/register"
            headers = {'17token': self.api_key, 'Content-Type': 'application/json'}

            payload = [{"number": tracking_number}]
            if carrier_code:
                payload[0]["carrier"] = carrier_code

            response = await client.post(url, headers=headers, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()

            if data.get('code') != 0:
                return {
                    'success': False,
                    'error': f"API error: {data.get('code')}",
                    'tracking_number': tracking_number,
                }

            accepted = data.get('data', {}).get('accepted', [])
            rejected = data.get('data', {}).get('rejected', [])

            if accepted:
                item = accepted[0]
                logger.info(f"Successfully registered {tracking_number} with carrier {item.get('carrier')}")
                return {
                    'success': True,
                    'tracking_number': tracking_number,
                    'carrier': item.get('carrier'),
                }

            if rejected:
                error = rejected[0].get('error', {})
                error_msg = error.get('message', 'Registration failed')
                logger.error(f"Failed to register {tracking_number}: {error_msg}")
                return {
                    'success': False,
                    'error': self._get_user_friendly_error('register_failed', error_msg),
                    'error_code': error.get('code'),
                    'tracking_number': tracking_number,
                }

            return {'success': False, 'error': 'Unknown registration error', 'tracking_number': tracking_number}

    async def _get_track_info(self, tracking_number: str, carrier_code: int = None) -> Dict[str, Any]:
        """Get tracking info for a registered tracking number."""
        async with httpx.AsyncClient() as client:
            url = f"{self.api_base}/gettrackinfo"
            headers = {'17token': self.api_key, 'Content-Type': 'application/json'}

            payload = [{"number": tracking_number}]
            if carrier_code:
                payload[0]["carrier"] = carrier_code

            response = await client.post(url, headers=headers, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()

            if data.get('code') != 0:
                return {
                    'success': False,
                    'error': f"API error: {data.get('code')}",
                    'tracking_number': tracking_number,
                }

            accepted = data.get('data', {}).get('accepted', [])
            rejected = data.get('data', {}).get('rejected', [])

            if accepted:
                return self._parse_track_info(accepted[0], tracking_number)

            if rejected:
                error = rejected[0].get('error', {})
                return {
                    'success': False,
                    'error': error.get('message', 'Tracking info not found'),
                    'error_code': error.get('code'),
                    'tracking_number': tracking_number,
                }

            return {'success': False, 'error': 'No tracking info found', 'tracking_number': tracking_number}

    async def stop_tracking(self, tracking_number: str, carrier_code: int = None) -> Dict[str, Any]:
        """Stop tracking a package."""
        if not self.api_key:
            return {'success': False, 'error': 'API key not configured'}

        async with httpx.AsyncClient() as client:
            url = f"{self.api_base}/stoptrack"
            headers = {'17token': self.api_key, 'Content-Type': 'application/json'}

            payload = [{"number": tracking_number}]
            if carrier_code:
                payload[0]["carrier"] = carrier_code

            response = await client.post(url, headers=headers, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()

            accepted = data.get('data', {}).get('accepted', [])
            if accepted:
                logger.info(f"Stopped tracking {tracking_number}")
                return {'success': True, 'tracking_number': tracking_number}

            rejected = data.get('data', {}).get('rejected', [])
            if rejected:
                error = rejected[0].get('error', {})
                return {
                    'success': False,
                    'error': error.get('message', 'Stop tracking failed'),
                    'tracking_number': tracking_number,
                }

            return {'success': False, 'error': 'Unknown error'}

    async def delete_tracking(self, tracking_number: str, carrier_code: int = None) -> Dict[str, Any]:
        """Delete a tracking number from 17TRACK."""
        if not self.api_key:
            return {'success': False, 'error': 'API key not configured'}

        async with httpx.AsyncClient() as client:
            url = f"{self.api_base}/deletetrack"
            headers = {'17token': self.api_key, 'Content-Type': 'application/json'}

            payload = [{"number": tracking_number}]
            if carrier_code:
                payload[0]["carrier"] = carrier_code

            response = await client.post(url, headers=headers, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()

            accepted = data.get('data', {}).get('accepted', [])
            if accepted:
                logger.info(f"Deleted tracking {tracking_number}")
                return {'success': True, 'tracking_number': tracking_number}

            rejected = data.get('data', {}).get('rejected', [])
            if rejected:
                error = rejected[0].get('error', {})
                return {
                    'success': False,
                    'error': error.get('message', 'Delete failed'),
                    'tracking_number': tracking_number,
                }

            return {'success': False, 'error': 'Unknown error'}

    def _get_carrier_code(self, carrier: str) -> Optional[int]:
        """Convert carrier name to 17TRACK carrier code."""
        if not carrier:
            return None
        carrier_lower = carrier.lower().replace(' ', '_').replace('-', '_')
        return CARRIER_CODES.get(carrier_lower)

    def _get_carrier_name(self, carrier_code: int) -> Optional[str]:
        """Convert 17TRACK carrier code to carrier name."""
        if not carrier_code:
            return None
        for name, code in CARRIER_CODES.items():
            if code == carrier_code:
                return name
        return None

    def _parse_track_info(self, data: Dict[str, Any], tracking_number: str) -> Dict[str, Any]:
        """Parse 17TRACK gettrackinfo response into standardized format."""
        try:
            track_info = data.get('track_info', {})
            carrier_code = data.get('carrier', 0)

            providers = track_info.get('tracking', {}).get('providers', [])
            carrier_name = 'unknown'
            if providers:
                provider = providers[0].get('provider', {})
                carrier_name = provider.get('name', 'unknown')

            latest_status = track_info.get('latest_status', {})
            status = self._normalize_status(latest_status.get('status', 'NotFound'))
            sub_status = latest_status.get('sub_status', '')

            latest_event = track_info.get('latest_event', {})
            last_update = ''
            if latest_event:
                last_update = latest_event.get('description', '')
                if latest_event.get('location'):
                    last_update = f"{last_update} - {latest_event.get('location')}"

            time_metrics = track_info.get('time_metrics', {})
            estimated_delivery = None
            est_date = time_metrics.get('estimated_delivery_date', {})
            if est_date and est_date.get('from'):
                estimated_delivery = est_date.get('from')[:10]

            events = []
            for provider in providers:
                for event in provider.get('events', []):
                    events.append({
                        'description': event.get('description', ''),
                        'timestamp': event.get('time_utc') or event.get('time_iso', ''),
                        'location': event.get('location', ''),
                        'stage': event.get('stage', ''),
                        'sub_status': event.get('sub_status', ''),
                    })

            carrier_lower = carrier_name.lower()
            tracking_url = get_tracking_url(carrier_lower, tracking_number)

            return {
                'success': True,
                'tracking_number': tracking_number,
                'carrier': carrier_lower,
                'carrier_code': carrier_code,
                'status': status,
                'sub_status': sub_status,
                'last_update': last_update,
                'estimated_delivery': estimated_delivery,
                'events': events,
                'tracking_url': tracking_url,
            }

        except Exception as e:
            logger.error(f"Failed to parse 17TRACK response: {e}", exc_info=True)
            return {
                'success': False,
                'error': f'Failed to parse tracking data: {str(e)}',
                'tracking_number': tracking_number,
            }

    def _normalize_status(self, status: str) -> str:
        """Normalize 17TRACK status to standard status codes."""
        status_map = {
            'NotFound': 'not_found',
            'InfoReceived': 'info_received',
            'InTransit': 'in_transit',
            'Expired': 'expired',
            'AvailableForPickup': 'available_for_pickup',
            'OutForDelivery': 'out_for_delivery',
            'DeliveryFailure': 'delivery_failure',
            'Delivered': 'delivered',
            'Exception': 'exception',
        }
        return status_map.get(status, 'unknown')

    def _get_user_friendly_error(self, error_type: str, details: Any = None) -> str:
        """Convert technical errors to user-friendly messages."""
        if error_type == 'connection_error':
            return "Unable to check tracking status right now. Please try again in a few minutes."

        elif error_type == 'http_error':
            status_code = details
            if status_code == 401:
                return "Tracking service authentication failed. Please contact support."
            elif status_code == 429:
                return "Too many tracking requests. Please try again later."
            elif status_code == 404:
                return "Tracking service temporarily unavailable. Please try again later."
            elif status_code >= 500:
                return "Tracking service is experiencing issues. Please try again later."
            else:
                return "Unable to check tracking status. Please try again later."

        elif error_type == 'register_failed':
            error_msg = str(details).lower() if details else ''
            if 'invalid' in error_msg or 'format' in error_msg:
                return "This doesn't look like a valid tracking number. Please check and try again."
            elif 'carrier' in error_msg and 'detected' in error_msg:
                return "Could not detect the carrier for this tracking number. Please specify the carrier."
            elif 'already registered' in error_msg:
                return "This package is already being tracked."
            elif 'quota' in error_msg or 'limit' in error_msg:
                return "Tracking limit reached. Please try again later."
            else:
                return "Couldn't add this tracking number. Please verify it's correct and try again."

        elif error_type == 'not_found':
            return "No tracking info found yet. The carrier may not have scanned it yet - try again later."

        else:
            return "Something went wrong. Please try again later."
