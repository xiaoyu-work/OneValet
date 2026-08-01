"""
Email importance rules.

The system-default criteria for deciding whether an incoming email warrants an
immediate notification. EmailEventHandler classifies against these (see
triggers/email_handler.py), and EmailPreferenceAgent shows them to the user
alongside whatever custom rules they have set.
"""

SYSTEM_RULES = """
    1. Urgent matters: contains urgent, ASAP, immediate, critical, time-sensitive
    2. Security-related: verification codes, password reset, login alerts, suspicious activity, 2FA codes
    3. Financial: payment confirmation, transfer notification, bill due, invoice, payment failed
    4. Travel changes: flight changes, hotel cancellation, meeting reschedule, booking confirmation
    5. Important notices: interview invitation, job offer, contract signing, delivery confirmation
    6. Action required: approval needed, response required, deadline approaching
    """
