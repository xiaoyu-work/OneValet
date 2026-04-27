"""Credential encryption for at-rest protection.

Uses AES-256-GCM via the cryptography library.
When KOA_CREDENTIAL_KEY is set, credentials are encrypted before storing
in PostgreSQL and decrypted on retrieval.

When no key is configured, operates as a passthrough (no encryption).
This allows gradual migration: existing plaintext credentials remain
readable, and new credentials are encrypted once a key is set.
"""

import base64
import hashlib
import json
import logging
import os
import secrets
from typing import Optional, Union

logger = logging.getLogger(__name__)

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    _HAS_CRYPTO = True
except ImportError:
    _HAS_CRYPTO = False

_ENCRYPTED_PREFIX = "enc:v1:"


class CredentialEncryptionError(RuntimeError):
    """Raised when encryption is required by policy but cannot be performed."""


class CredentialEncryptor:
    """Encrypt/decrypt credential dicts using AES-256-GCM.

    Args:
        key: Encryption key string. Hashed to 32 bytes with SHA-256.
             If None, operates as a no-op passthrough *unless*
             ``require_encryption`` is set.
        require_encryption: When True (production default), the encryptor
            refuses to operate in passthrough mode: :meth:`encrypt` and
            :meth:`decrypt` raise :class:`CredentialEncryptionError` if no
            usable key is configured.  Set via ``KOA_REQUIRE_ENCRYPTION=1``
            or explicitly by :func:`koa.app.Koa` when the environment is
            production.

    Policy summary:
        * ``require_encryption=False`` (legacy default) — passthrough when
          no key: writes plaintext, reads plaintext; keeps compatibility
          with upgrade-in-place deployments.
        * ``require_encryption=True`` — MUST have a working key; startup
          will fail early if ``KOA_CREDENTIAL_KEY`` is missing or the
          ``cryptography`` package is unavailable.
    """

    def __init__(
        self,
        key: Optional[str] = None,
        *,
        require_encryption: Optional[bool] = None,
    ):
        if require_encryption is None:
            # Enable fail-secure automatically in production.
            env = os.environ.get("KOA_ENV", "").lower()
            require_encryption = (
                os.environ.get("KOA_REQUIRE_ENCRYPTION", "").lower() in {"1", "true", "yes"}
                or env == "production"
            )
        self._require = bool(require_encryption)

        if key and _HAS_CRYPTO:
            self._key = hashlib.sha256(key.encode()).digest()
            self._aesgcm = AESGCM(self._key)
            self._enabled = True
        else:
            self._key = None
            self._aesgcm = None
            self._enabled = False
            if self._require:
                if not _HAS_CRYPTO:
                    raise CredentialEncryptionError(
                        "KOA_REQUIRE_ENCRYPTION is set but 'cryptography' "
                        "package is not installed. Install with: "
                        "pip install cryptography"
                    )
                raise CredentialEncryptionError(
                    "Credential encryption is required (KOA_REQUIRE_ENCRYPTION "
                    "or KOA_ENV=production) but KOA_CREDENTIAL_KEY is not set. "
                    "Generate a 32+ byte random key and configure it before "
                    "starting the service."
                )
            if key and not _HAS_CRYPTO:
                logger.warning(
                    "KOA_CREDENTIAL_KEY is set but 'cryptography' package "
                    "is not installed. Credentials will NOT be encrypted. "
                    "Install with: pip install cryptography"
                )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def require_encryption(self) -> bool:
        return self._require

    def encrypt(self, data: Union[dict, str]) -> Union[str, dict]:
        """Encrypt a credential dict to an encrypted string."""
        if not self._enabled:
            if self._require:
                raise CredentialEncryptionError("Encryption required but encryptor is not enabled.")
            return data

        plaintext = json.dumps(data).encode("utf-8")
        nonce = secrets.token_bytes(12)
        ciphertext = self._aesgcm.encrypt(nonce, plaintext, None)
        payload = base64.b64encode(nonce + ciphertext).decode("ascii")
        return f"{_ENCRYPTED_PREFIX}{payload}"

    def decrypt(self, data: Union[str, dict]) -> dict:
        """Decrypt an encrypted string back to a credential dict.

        Behavior:
            * Dict input → returned as-is (legacy unencrypted JSONB).
            * Non-string / non-dict → returned as-is.
            * String without ``enc:v1:`` prefix → JSON-decoded if possible.
              If ``require_encryption`` is True, plaintext strings raise
              :class:`CredentialEncryptionError` — this prevents silent
              acceptance of unencrypted rows in production.
            * String with ``enc:v1:`` prefix → AES-GCM decrypted.  Raises
              if the encryptor is disabled.
        """
        if isinstance(data, dict):
            if self._require:
                # Plaintext JSONB in a required-encryption deployment is
                # a policy violation worth surfacing.
                raise CredentialEncryptionError(
                    "Plaintext credential dict encountered while encryption "
                    "is required. Migrate legacy rows before enabling "
                    "KOA_REQUIRE_ENCRYPTION."
                )
            return data

        if not isinstance(data, str):
            return data

        if not data.startswith(_ENCRYPTED_PREFIX):
            if self._require:
                raise CredentialEncryptionError(
                    "Plaintext credential string encountered while encryption is required."
                )
            try:
                return json.loads(data)
            except (json.JSONDecodeError, TypeError):
                return data

        if not self._enabled:
            raise CredentialEncryptionError(
                "Encrypted credentials found but KOA_CREDENTIAL_KEY is not set."
            )

        payload = data[len(_ENCRYPTED_PREFIX) :]
        raw = base64.b64decode(payload)
        nonce = raw[:12]
        ciphertext = raw[12:]
        plaintext = self._aesgcm.decrypt(nonce, ciphertext, None)
        return json.loads(plaintext.decode("utf-8"))


# ---------------------------------------------------------------------------
# Log sanitization helper
# ---------------------------------------------------------------------------

_SECRET_KEYS = {
    "password",
    "passwd",
    "secret",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "token",
    "authorization",
    "private_key",
    "client_secret",
}


def sanitize_for_log(value):
    """Recursively redact known-secret keys from dicts/lists for logging.

    Leaves non-secret data intact.  Use at log boundaries, never store
    the result — it is lossy by design.
    """
    if isinstance(value, dict):
        return {
            k: ("***REDACTED***" if k.lower() in _SECRET_KEYS else sanitize_for_log(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [sanitize_for_log(v) for v in value]
    return value
